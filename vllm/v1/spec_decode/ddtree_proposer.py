# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTreeProposer: dflash drafter extended with diffusion draft trees.

Staged:
    * C0 — scaffold (introduced earlier). The proposer is only constructed
      when ``speculative_config.ddtree_enabled`` is true; off-path is
      structurally identical to PR #41703.
    * C1/S0 — built a degenerate tree from chosen tokens. Plumbing check
      only; deprecated by S1.
    * **S1 (this revision)** — capture the *real* draft logits from
      ``self.model.compute_logits`` via an override of ``_greedy_sample``,
      build a tree from those logits, and log aggregate tree statistics.
      The returned tokens are still the linear top-1 (= per-position
      argmax), so the output remains bit-identical to dflash. The
      difference vs C1 is that the tree is now meaningful — node spread,
      top-1 path probability mass, etc. give us a real signal for how
      much benefit C2 (tree verify) could deliver.
    * C2 (future) — wire the visibility mask into the target verify pass
      and replace acceptance with ``follow_verified_tree``.

Reference: Liran Ringel, Yaniv Romano,
"Accelerating Speculative Decoding with Block Diffusion Draft Trees",
arXiv:2604.12989, 2026 (MIT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.ddtree_utils import build_ddtree_tree
from vllm.v1.spec_decode.dflash import DFlashProposer

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)


class DDTreeProposer(DFlashProposer):
    """DFlash proposer that also builds a top-k draft tree (S1).

    On every ``propose`` call:
      1. ``super().propose(...)`` runs the standard dflash flow. Inside it,
         ``_greedy_sample`` is the seam that turns hidden states into
         tokens. We override ``_greedy_sample`` to compute the full
         per-position logits via ``self.model.compute_logits`` first,
         stash them on the instance, then return the argmax exactly as
         the base class would. This means token output is unchanged.
      2. After ``super().propose`` returns, we read the stashed logits
         and run ``build_ddtree_tree`` on a sampled subset of the batch
         (S1 logs at most ``ddtree_log_every`` requests per emission to
         keep overhead bounded). Aggregate statistics are emitted via
         ``logger.info`` once enough samples accumulate.
    """

    # Sample one request out of this many calls when computing tree stats.
    # build_ddtree_tree is CPU-bound (~ms per call); throttling keeps
    # serving latency unaffected for production traffic.
    _DDTREE_BUILD_EVERY: int = 8
    # Emit an aggregated INFO log line once this many samples accumulate.
    _DDTREE_LOG_FLUSH_AT: int = 25

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)

        spec_cfg = vllm_config.speculative_config
        assert spec_cfg is not None and spec_cfg.ddtree_enabled, (
            "DDTreeProposer instantiated but ddtree_enabled is not set. "
            "This is a bug in gpu_model_runner instantiation."
        )

        budget = spec_cfg.ddtree_budget
        if budget is None or budget <= 0:
            budget = self.num_speculative_tokens
        self.ddtree_budget: int = int(budget)
        # C2 will use these; S1 only exercises real-logits tree build.
        self._ddtree_verify_enabled: bool = False

        # S1 state.
        self._ddtree_last_logits: torch.Tensor | None = None
        self._ddtree_call_counter: int = 0
        self._ddtree_agg_n: int = 0
        self._ddtree_agg_size: int = 0
        self._ddtree_agg_max_depth: int = 0
        self._ddtree_agg_top1_mass: float = 0.0
        self._ddtree_local_argmax_warned: bool = False

        logger.info(
            "DDTreeProposer enabled (budget=%d, num_speculative_tokens=%d, "
            "verify=tree:%s). S1 path active: tree is built from real "
            "draft logits and aggregate statistics logged every %d sampled "
            "requests; target verify path is unchanged.",
            self.ddtree_budget,
            self.num_speculative_tokens,
            self._ddtree_verify_enabled,
            self._DDTREE_LOG_FLUSH_AT,
        )

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Capture real logits while preserving base argmax behavior.

        Base:
            ``return self.model.compute_logits(hidden_states).argmax(-1)``
        Here:
            we keep the logits tensor on ``self._ddtree_last_logits`` so
            ``propose()`` can build a tree from them. The local-argmax
            reduction path doesn't expose full logits — in that case we
            fall back to base behavior and skip the tree build.
        """
        if self.use_local_argmax_reduction:
            if not self._ddtree_local_argmax_warned:
                logger.warning(
                    "ddtree: use_local_argmax_reduction is enabled — "
                    "full logits are not available, tree build will be "
                    "skipped. Set use_local_argmax_reduction=false to "
                    "exercise the tree build path."
                )
                self._ddtree_local_argmax_warned = True
            self._ddtree_last_logits = None
            return self.model.get_top_tokens(hidden_states)

        logits = self.model.compute_logits(hidden_states)
        # Hold a reference for the duration of this propose() call only.
        # propose() clears it before returning to avoid keeping a large
        # [B*N, vocab] tensor alive across rounds.
        self._ddtree_last_logits = logits
        return logits.argmax(dim=-1)

    @override
    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: "CommonAttentionMetadata",
        sampling_metadata: "SamplingMetadata",
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        self._ddtree_last_logits = None

        draft_token_ids = super().propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            mm_embed_inputs=mm_embed_inputs,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            slot_mappings=slot_mappings,
        )

        try:
            self._ddtree_maybe_build_and_log(draft_token_ids)
        except Exception as exc:  # pragma: no cover - log-only path
            logger.warning(
                "ddtree real-logits tree build failed (silently ignored): %s",
                exc,
            )

        # Drop the logits reference so the next round doesn't hold the
        # previous round's tensor.
        self._ddtree_last_logits = None

        return draft_token_ids

    def _ddtree_maybe_build_and_log(self, draft_token_ids: torch.Tensor) -> None:
        """Build a tree from real logits on a sampled request and aggregate.

        Sampling and aggregation policy:
          * On every call we increment ``_ddtree_call_counter``.
          * Every ``_DDTREE_BUILD_EVERY`` calls, we pick request 0 of the
            current batch and compute (a) tree size, (b) max depth,
            (c) probability mass on the linear top-1 path. These three
            scalars feed an accumulator.
          * When the accumulator has ``_DDTREE_LOG_FLUSH_AT`` samples,
            one INFO log line is emitted and the accumulator resets.

        With the defaults (build_every=8, flush_at=25), one INFO line is
        emitted per ~200 drafter calls, which on a realistic stream is
        well under one log line per second.
        """
        self._ddtree_call_counter += 1
        if (self._ddtree_call_counter % self._DDTREE_BUILD_EVERY) != 0:
            return
        if self._ddtree_last_logits is None:
            return
        if draft_token_ids.numel() == 0:
            return

        # draft_token_ids has been reshaped by super().propose() to
        # [batch_size, num_speculative_tokens]. The stashed logits are
        # the raw output of compute_logits and have shape [B*N, vocab].
        batch_size = int(draft_token_ids.shape[0])
        num_spec = int(self.num_speculative_tokens)
        if batch_size == 0 or num_spec == 0:
            return

        logits = self._ddtree_last_logits
        if logits.dim() != 2 or logits.shape[0] != batch_size * num_spec:
            # Shape didn't match our expectations (e.g., a parent class
            # change). Skip silently rather than risk a misleading stat.
            return

        # Pull one request's slice. detach() is a no-op under inference_mode
        # but explicit for clarity.
        req_logits = logits.view(batch_size, num_spec, -1)[0].detach()

        nti, nd, _parents, _child_maps, _visibility = build_ddtree_tree(
            req_logits, budget=self.ddtree_budget
        )

        size = int(nti.numel())
        max_depth = int(nd.max().item()) if nd.numel() else 0

        # top1_mass = prob(top-1 path) under softmax — proxy for how
        # peaked the draft distribution is at each depth. Lower mass =
        # more spread = more potential benefit from tree expansion.
        f_logits = req_logits.float()
        top1_logits = f_logits.max(dim=-1).values
        lse = torch.logsumexp(f_logits, dim=-1)
        log_top1_path = float((top1_logits - lse).sum().item())
        top1_mass = float(torch.tensor(log_top1_path).exp().item())

        self._ddtree_agg_n += 1
        self._ddtree_agg_size += size
        self._ddtree_agg_max_depth += max_depth
        self._ddtree_agg_top1_mass += top1_mass

        if self._ddtree_agg_n >= self._DDTREE_LOG_FLUSH_AT:
            n = self._ddtree_agg_n
            logger.info(
                "ddtree real-logits: samples=%d avg_size=%.1f "
                "avg_max_depth=%.1f avg_top1_mass=%.4f budget=%d N=%d",
                n,
                self._ddtree_agg_size / n,
                self._ddtree_agg_max_depth / n,
                self._ddtree_agg_top1_mass / n,
                self.ddtree_budget,
                num_spec,
            )
            self._ddtree_agg_n = 0
            self._ddtree_agg_size = 0
            self._ddtree_agg_max_depth = 0
            self._ddtree_agg_top1_mass = 0.0
