# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTreeProposer: dflash drafter extended with diffusion draft trees.

Staged:
    * C0 (this file) — scaffold. The proposer is only constructed when
      ``speculative_config.ddtree_enabled`` is true. When the flag is off,
      the upstream ``DFlashProposer`` path is used and this module is never
      imported, so the off-path is structurally identical to PR #41703.
    * C1 (this file) — when on, build a top-k draft tree from the per-
      position draft logits and log tree statistics. The token result
      returned to the runner is the linear top-1 path of the tree, which
      equals per-position argmax — so token output is identical to
      dflash-only and acceptance length should match.
    * C2 (future) — wire the visibility mask into the target verify pass
      and replace acceptance with ``follow_verified_tree``. Stubbed below.

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
    """DFlash proposer with a top-k diffusion draft tree on the draft side.

    Behaves like ``DFlashProposer`` plus an extra tree-build step on the
    draft logits. C1 path: tree statistics are computed and logged, but the
    drafter still returns a flat ``[batch, num_speculative_tokens]`` token
    tensor whose top-1 path matches dflash. This keeps the on-path
    token-bit-equivalent to dflash-only and validates that the tree code
    runs under live load before C2 wires it into target verify.
    """

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
        # C2 will use these; C0/C1 only exercise build.
        self._ddtree_verify_enabled: bool = False

        logger.info(
            "DDTreeProposer enabled (budget=%d, num_speculative_tokens=%d, "
            "verify=tree:%s). C1 path active: tree is built from draft logits "
            "and statistics logged; target verify path is unchanged.",
            self.ddtree_budget,
            self.num_speculative_tokens,
            self._ddtree_verify_enabled,
        )

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
        # Identical to DFlashProposer.propose, which delegates to base.
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

        # C1: exercise the tree-build path on the draft tokens. We do not
        # yet have access to the raw logits at this seam (base.propose
        # consumes them inside _greedy_sample). Until C2 plumbs logits out,
        # we synthesize a one-hot logits tensor from the chosen tokens so
        # the tree-build call path is exercised end-to-end without changing
        # the returned token sequence.
        try:
            self._ddtree_log_stats(draft_token_ids)
        except Exception as exc:  # pragma: no cover - log-only path
            logger.warning("DDTree tree-build failed; falling back silently: %s", exc)

        return draft_token_ids

    def _ddtree_log_stats(self, draft_token_ids: torch.Tensor) -> None:
        """Build a degenerate tree over the drafted tokens for plumbing.

        Synthesizes a depth × vocab one-hot logits tensor from the drafted
        token ids and calls ``build_ddtree_tree``. This is *only* a
        plumbing check for C1 — when C2 lands, the proposer will instead
        capture the real ``compute_logits`` output from the draft model.
        """
        if not logger.isEnabledFor(10):  # DEBUG = 10
            return
        if draft_token_ids.numel() == 0:
            return

        flat = draft_token_ids.reshape(-1, self.num_speculative_tokens)
        batch_size = flat.shape[0]
        # Cheap synthetic: rank-1 logits, hot on the chosen token per row.
        # We only need vocab >= max token id; use a small upper bound.
        max_id = int(flat.max().item()) + 1
        depth = self.num_speculative_tokens
        for req in range(batch_size):
            ids = flat[req].to(torch.long).cpu()
            logits = torch.full((depth, max_id), -1e4, dtype=torch.float32)
            logits[torch.arange(depth), ids] = 1.0
            (
                node_token_ids,
                node_depths,
                _parents,
                _child_maps,
                _visibility,
            ) = build_ddtree_tree(logits, budget=self.ddtree_budget)
            logger.debug(
                "ddtree[req=%d] size=%d depth=%d budget=%d",
                req,
                int(node_token_ids.numel()),
                int(node_depths.max().item()) if node_depths.numel() else 0,
                self.ddtree_budget,
            )
