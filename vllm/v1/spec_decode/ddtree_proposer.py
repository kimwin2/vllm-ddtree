# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTreeProposer: dflash drafter extended with diffusion draft trees.

Staged:
    * C0 — scaffold (introduced earlier). The proposer is only constructed
      when ``speculative_config.ddtree_enabled`` is true; off-path is
      structurally identical to PR #41703.
    * C1/S0 — built a degenerate tree from chosen tokens. Plumbing check
      only; deprecated by S1.
    * S1 — capture the *real* draft logits from
      ``self.model.compute_logits`` via an override of ``_greedy_sample``,
      build a tree from those logits, and log aggregate tree statistics.
      The returned tokens are still the linear top-1 (= per-position
      argmax), so the output remains bit-identical to dflash.
    * **S2 (this revision)** — at ``initialize_attn_backend`` time,
      probe the active target and draft attention backends to determine
      whether they natively support per-request 2D attention masks
      (required for the future tree-verify pass). Logs a structured
      capability report and stores the verdict on
      ``self._has_tree_mask_support``. No runtime behavior change.
    * S3 (future) — actual tree verify: expand the target query to
      ``1+ddtree_budget`` per request, inject the 2D visibility mask
      into target attention metadata (or fall back when S2 reports no
      native support), and replace linear-cumprod acceptance with
      ``follow_verified_tree``.

Reference: Liran Ringel, Yaniv Romano,
"Accelerating Speculative Decoding with Block Diffusion Draft Trees",
arXiv:2604.12989, 2026 (MIT).
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING, Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.ddtree_utils import build_ddtree_tree
from vllm.v1.spec_decode.dflash import DFlashProposer

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.sample.metadata import SamplingMetadata


# Names that, if present on the metadata dataclass or in a builder's build
# signature, indicate that the backend has some form of native mask
# customization. Heuristic — not a guarantee that the kernel applies an
# arbitrary 2D mask correctly, but a strong signal that it has a slot
# we can plumb the tree visibility mask through.
_MASK_HINT_KEYWORDS = (
    "custom_mask",
    "attn_mask",
    "attention_mask",
    "tree_mask",
    "mask_mod",
    "block_mask",
    "sdpa_attn_masks",
)

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
        # S3 will use these; S1/S2 only measure.
        self._ddtree_verify_enabled: bool = False

        # The base proposer accepts ``runner`` but doesn't store it. We
        # need it for the S2 target-side capability probe to reach
        # ``runner.attn_groups``, so keep a reference ourselves.
        self._ddtree_runner = runner

        # S2 state — set by _ddtree_probe_attention_backends() during
        # initialize_attn_backend. Default False until the probe runs.
        self._has_tree_mask_support: bool = False
        self._ddtree_probe_report: dict[str, Any] | None = None

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
            "verify=tree:%s). S1+S2 path active: real-logits tree stats "
            "every %d sampled requests + backend capability probe at "
            "initialize_attn_backend; target verify path unchanged.",
            self.ddtree_budget,
            self.num_speculative_tokens,
            self._ddtree_verify_enabled,
            self._DDTREE_LOG_FLUSH_AT,
        )

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: "KVCacheConfig",
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """Run dflash's base setup, then probe attention backends (S2).

        The probe runs once at startup and writes its findings to
        ``self._has_tree_mask_support`` plus ``self._ddtree_probe_report``.
        It does not change any runtime behavior — S3 will read the verdict
        to decide whether to attempt native tree-mask injection vs fall
        back to a non-native path.
        """
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        try:
            self._ddtree_probe_attention_backends()
        except Exception as exc:  # pragma: no cover - probe is best-effort
            logger.warning(
                "ddtree backend probe failed (silently ignored): %s", exc
            )
            self._has_tree_mask_support = False
            self._ddtree_probe_report = {"error": repr(exc)}

    def _ddtree_probe_attention_backends(self) -> None:
        """Inspect target & draft attention backends for 2D-mask support.

        Writes ``self._has_tree_mask_support`` based on heuristics that
        look at:
          1. Builder class name / module (e.g. flex_attention, flashinfer)
          2. ``build`` / ``build_for_drafting`` parameter names
          3. Metadata dataclass field names

        Any field/parameter whose name contains one of the mask hint
        keywords (``custom_mask``, ``mask_mod``, ``attn_mask``, …) is
        treated as a positive signal.
        """
        target_groups = self._ddtree_target_attn_groups()
        draft_groups = self._ddtree_draft_attn_groups()

        target_report = self._ddtree_probe_groups("target", target_groups)
        draft_report = self._ddtree_probe_groups("draft", draft_groups)

        # The actual mask gets attached to the *target* verify pass in S3,
        # so target-side support is the binding capability. Draft-side is
        # logged for completeness / future use.
        self._has_tree_mask_support = bool(
            target_report.get("supports_custom_mask", False)
        )
        self._ddtree_probe_report = {
            "target": target_report,
            "draft": draft_report,
        }

        logger.info(
            "ddtree backend probe:\n"
            "  target: %s\n"
            "  draft : %s\n"
            "  verdict: tree_mask_support=%s%s",
            self._ddtree_format_report(target_report),
            self._ddtree_format_report(draft_report),
            self._has_tree_mask_support,
            ""
            if self._has_tree_mask_support
            else "  (S3 will need a fallback or backend swap to FlexAttention/FlashInfer)",
        )

    def _ddtree_target_attn_groups(self) -> list[Any]:
        """Flatten the runner's target attn_groups across KV cache groups.

        ``self._ddtree_runner`` is captured in ``__init__`` because the
        base proposer accepts ``runner`` but does not retain it.
        """
        runner = self._ddtree_runner
        if runner is None:
            return []
        raw = getattr(runner, "attn_groups", None)
        if not raw:
            return []
        flat: list[Any] = []
        for entry in raw:
            if isinstance(entry, list):
                flat.extend(entry)
            else:
                flat.append(entry)
        return flat

    def _ddtree_draft_attn_groups(self) -> list[Any]:
        """The drafter's own attn_groups, populated by base.initialize_attn_backend."""
        groups = getattr(self, "draft_attn_groups", []) or []
        return list(groups)

    def _ddtree_probe_groups(
        self, label: str, groups: list[Any]
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "label": label,
            "n_groups": len(groups),
            "builder_class": None,
            "builder_module": None,
            "metadata_class": None,
            "metadata_fields": [],
            "build_signature": None,
            "build_for_drafting_signature": None,
            "indicators": [],
            "supports_custom_mask": False,
        }
        if not groups:
            info["error"] = "no attention groups"
            return info

        # Probe the first group only; backends within a single role are
        # typically homogeneous for our setup.
        first = groups[0]
        try:
            builder = first.get_metadata_builder()
        except Exception as exc:
            info["error"] = f"get_metadata_builder failed: {exc!r}"
            return info

        builder_cls = type(builder)
        info["builder_class"] = builder_cls.__name__
        info["builder_module"] = builder_cls.__module__

        # Module-name heuristic — flex_attention has a rich mask_mod API,
        # FlashInfer accepts custom_mask in its prefill wrappers.
        mod_lower = (builder_cls.__module__ or "").lower()
        if "flex_attention" in mod_lower or "flexattention" in mod_lower:
            info["indicators"].append("module:flex_attention (mask_mod API)")
            info["supports_custom_mask"] = True
        if "flashinfer" in mod_lower:
            info["indicators"].append("module:flashinfer (custom_mask in prefill)")
            info["supports_custom_mask"] = True

        # Signature heuristic on build() and build_for_drafting().
        for method_name in ("build", "build_for_drafting"):
            method = getattr(builder, method_name, None)
            if method is None:
                continue
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                continue
            sig_str = str(sig)
            info[f"{method_name}_signature"] = sig_str
            for param_name in sig.parameters:
                pn = param_name.lower()
                for hint in _MASK_HINT_KEYWORDS:
                    if hint in pn:
                        info["indicators"].append(
                            f"{method_name} param:{param_name}"
                        )
                        info["supports_custom_mask"] = True

        # Metadata class — extract from build()'s return annotation when
        # available, or from the builder's generic parameter binding.
        meta_cls = self._ddtree_extract_metadata_class(builder)
        if meta_cls is not None:
            info["metadata_class"] = meta_cls.__name__
            if dataclasses.is_dataclass(meta_cls):
                fields = [f.name for f in dataclasses.fields(meta_cls)]
                info["metadata_fields"] = fields
                for fname in fields:
                    fn = fname.lower()
                    for hint in _MASK_HINT_KEYWORDS:
                        if hint in fn:
                            info["indicators"].append(
                                f"metadata field:{fname}"
                            )
                            info["supports_custom_mask"] = True

        return info

    @staticmethod
    def _ddtree_extract_metadata_class(builder: Any) -> type | None:
        """Best-effort: find the metadata dataclass produced by builder.

        Tries:
          1. ``builder.build``'s return annotation
          2. ``builder.__orig_bases__`` generic parameter (M)
        """
        method = getattr(builder, "build", None)
        if method is not None:
            try:
                ret = inspect.signature(method).return_annotation
                if inspect.isclass(ret):
                    return ret
            except (TypeError, ValueError):
                pass
        # AttentionMetadataBuilder[M] generic — pull M from orig_bases.
        cls = type(builder)
        for base in getattr(cls, "__orig_bases__", ()) or ():
            args = getattr(base, "__args__", ()) or ()
            for arg in args:
                if inspect.isclass(arg):
                    return arg
        return None

    @staticmethod
    def _ddtree_format_report(info: dict[str, Any]) -> str:
        parts = [
            f"backend={info.get('builder_module', '?')}.{info.get('builder_class', '?')}",
            f"metadata={info.get('metadata_class', '?')}",
            f"n_groups={info.get('n_groups', 0)}",
            f"supports_custom_mask={info.get('supports_custom_mask', False)}",
        ]
        indicators = info.get("indicators") or []
        if indicators:
            parts.append(f"indicators={indicators}")
        if "error" in info:
            parts.append(f"error={info['error']}")
        return ", ".join(parts)

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
