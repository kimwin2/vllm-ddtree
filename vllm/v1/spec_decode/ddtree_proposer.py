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
    * S2 — at ``initialize_attn_backend`` time, probe the active target
      and draft attention backends to determine whether they natively
      support per-request 2D attention masks. Logs a structured
      capability report and stores the verdict on
      ``self._has_tree_mask_support``. No runtime behavior change.
    * S3a — purely an *out-of-tree* change:
      ``TritonAttentionMetadata`` gains an optional ``tree_attention_mask``
      field (default ``None``). The probe verdict from S2 flipped to
      ``supports_custom_mask=True``. No runtime behavior change.
    * **S3b (this revision)** — the drafter actually builds a per-request
      ``[batch, 1+budget, 1+budget]`` bool visibility mask whenever the
      sampling stride hits, and stashes it on
      ``self._ddtree_pending_mask``. The tensor lives on GPU and is
      ready for the runner/kernel to consume. NOBODY READS IT YET:
      the runner is unchanged, the kernel is unchanged, the metadata
      field still defaults to ``None`` on every build call. Token
      output remains byte-identical to dflash. S3b is the data-shape
      checkpoint: we verify (in the live log) that the mask is the
      right shape, the right dtype, and built without crashes for the
      full batch — not just for the single-sample tree we built in S1.
    * S3c (future) — wire the stashed mask to ``TritonAttentionMetadata``
      on the target verify pass (runner-side plumbing), extend the
      ``unified_attention`` triton kernel to apply it, expand the
      target query to ``1+ddtree_budget`` per request, and replace
      cumprod acceptance with ``follow_verified_tree``. This is where
      token output starts to differ from dflash and where the round
      coupling (mask built at round t for verify at round t+1) is
      formalized.

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

    # The DFlash drafter is trained at a fixed block_size (gemma-4 dflash
    # uses block_size=16 → 15 mask positions). When ``ddtree_verify_tree``
    # is on, we run the drafter with this many internal slots regardless
    # of ``num_speculative_tokens``, so its forward stays in its trained
    # regime. The tree (``ddtree_budget`` nodes) is expanded externally
    # from the 15 logits this forward produces.
    DFLASH_DRAFT_HORIZON: int = 15

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

        # S3c-1: target-side tree verify flag. When True, ``propose`` runs
        # dflash drafter with the fixed ``DFLASH_DRAFT_HORIZON`` window
        # internally and returns ``ddtree_budget`` tree-node tokens
        # (= num_speculative_tokens) by tree expansion of the drafter's
        # 15 logits. Default False keeps the dflash output path intact.
        self._ddtree_verify_enabled: bool = bool(
            getattr(spec_cfg, "ddtree_verify_tree", False)
        )
        if self._ddtree_verify_enabled:
            if self.ddtree_budget != self.num_speculative_tokens:
                raise ValueError(
                    "ddtree_verify_tree=true requires "
                    "ddtree_budget == num_speculative_tokens, got "
                    f"ddtree_budget={self.ddtree_budget}, "
                    f"num_speculative_tokens={self.num_speculative_tokens}"
                )

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

        # S3b state — built by _ddtree_maybe_build_and_log when the
        # sampling stride hits. Shape: [batch, 1+budget, 1+budget], bool,
        # on self.device. Not yet consumed by anything in the runtime
        # path; S3c will plumb it to the target verify metadata.
        self._ddtree_pending_mask: torch.Tensor | None = None

        # S3c-3 state — per-request child_maps from build_ddtree_tree,
        # one list[dict[token_id -> child_node_index]] per request, of
        # length ``1 + budget``. Consumed by gpu_model_runner's
        # tree-follow accept when ddtree_verify_tree=true. Populated by
        # ``_propose_with_tree_expansion`` alongside the mask.
        self._ddtree_pending_child_maps: (
            list[list[dict[int, int]]] | None
        ) = None
        # Per-request tree node token IDs as a list of python lists, so
        # the tree-follow accept can map an accepted node index back to
        # its token. Shape: [batch][budget] (CPU, int).
        self._ddtree_pending_node_tokens: list[list[int]] | None = None

        # NOTE: each branch keeps the stage tag as a literal substring in
        # the format string so that server.sh's
        # ``grep "<stage> path active" ddtree_proposer.py`` patch-detection
        # finds the marker on disk, not just in the runtime log.
        if self._ddtree_verify_enabled:
            logger.info(
                "DDTreeProposer enabled (budget=%d, num_speculative_tokens=%d, "
                "verify=tree:%s, dflash_draft_horizon=%d). "
                "S1+S2+S3b+S3c-1+S3c-2+S3c-3 path active. drafter runs "
                "at horizon=15 internally; propose() returns budget "
                "tree-node tokens; the visibility mask is attached to "
                "target's TritonAttentionMetadata.tree_attention_mask "
                "and applied by the triton kernel as additive qq_bias. "
                "Tree-follow accept is GATED OFF in this build (S3c-3-"
                "quick fallback) pending S3c-4 KV slot compaction — "
                "without compaction, scattered tree-path accepts leave "
                "the next round's drafter with wrong prefix KV and "
                "acceptance length drops below cumprod baseline. So "
                "the verify-side tree path (S3c-2) IS exercised but "
                "the accept side reverts to linear cumprod. Expected "
                "acceptance length: dflash baseline 3.228.",
                self.ddtree_budget,
                self.num_speculative_tokens,
                self._ddtree_verify_enabled,
                self.DFLASH_DRAFT_HORIZON,
            )
        else:
            logger.info(
                "DDTreeProposer enabled (budget=%d, num_speculative_tokens=%d, "
                "verify=tree:%s, dflash_draft_horizon=%d). "
                "S1+S2+S3b path active: drafter and output both at "
                "num_speculative_tokens; real-logits tree stats logged "
                "every %d sampled batches; target verify path unchanged.",
                self.ddtree_budget,
                self.num_speculative_tokens,
                self._ddtree_verify_enabled,
                self.DFLASH_DRAFT_HORIZON,
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

        if self._ddtree_verify_enabled:
            # S3c-1: drafter forward uses dflash's native horizon (16 slots)
            # internally; we temporarily reduce self.num_speculative_tokens
            # so the inherited DFlash setup builds a 1+15 query layout.
            # After super().propose returns, we discard its linear-tokens
            # result and produce a tree of `budget` nodes from the 15
            # logits captured during _greedy_sample.
            return self._propose_with_tree_expansion(
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

        # S1+S2+S3b path (verify_tree off): unchanged from before.
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

    def _propose_with_tree_expansion(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: "CommonAttentionMetadata",
        sampling_metadata: "SamplingMetadata",
        mm_embed_inputs,
        num_rejected_tokens_gpu,
        slot_mappings,
    ) -> torch.Tensor:
        """S3c-1 propose path: drafter at 16 slots, output at budget tokens.

        Mechanics:
          * Temporarily swap ``self.num_speculative_tokens`` to
            ``DFLASH_DRAFT_HORIZON`` (15). This makes the inherited
            ``DFlashProposer.set_inputs_first_pass`` build a
            ``1 + 15 = 16`` query layout for the drafter forward — i.e.
            dflash runs in the regime it was trained on, regardless of
            how many slots the scheduler reserved for the target verify
            pass.
          * After ``super().propose(...)`` completes, the discarded
            return shape would be ``[batch, 15]`` and
            ``self._ddtree_last_logits`` holds the 15 per-position
            logits per request via our S1 ``_greedy_sample`` hook.
          * We then run ``build_ddtree_tree`` per request to expand the
            15 logits into a ``budget``-node tree, stash the visibility
            mask onto ``self._ddtree_pending_mask`` (for S3c-2/3 to
            consume), and return the tree-node token ids reshaped to
            ``[batch, budget]``.

        Note (S3c-1 only): the runner still verifies these tokens with
        non-causal attention and no tree mask. Output tokens will not
        be bit-exact with dflash and may be nonsensical. The mask
        plumbing (S3c-2) and tree-follow accept (S3c-3) close the gap.
        """
        original_num_spec = self.num_speculative_tokens
        try:
            self.num_speculative_tokens = self.DFLASH_DRAFT_HORIZON
            # The inherited propose will sample at DFLASH_DRAFT_HORIZON
            # positions per request and return [batch, 15] — we ignore
            # that. _greedy_sample stashes the [B*15, vocab] logits.
            _ = super().propose(
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
        finally:
            self.num_speculative_tokens = original_num_spec

        # Build the tree from the captured logits and produce
        # [batch, budget] tree-node token ids. The drafter has now
        # written its KV cache for 16 internal slots per request; the
        # remaining (budget - 15) slots reserved by the scheduler are
        # unused by the drafter (the target overwrites them with tree
        # nodes for the next verify).
        return self._ddtree_build_tree_outputs(common_attn_metadata)

    def _ddtree_build_tree_outputs(
        self, common_attn_metadata: "CommonAttentionMetadata"
    ) -> torch.Tensor:
        """Expand drafter logits into a tree and return [batch, budget].

        Also stashes ``self._ddtree_pending_mask`` of shape
        ``[batch, 1+budget, 1+budget]`` for S3c-2 consumption.
        """
        batch_size = int(common_attn_metadata.batch_size())
        budget = self.ddtree_budget
        full_size = 1 + budget
        device = self.device

        logits = self._ddtree_last_logits
        if logits is None:
            # Local-argmax reduction path or other unexpected; return a
            # zero-token fallback that at least preserves shape so the
            # runner doesn't crash. S3c-2/3 will refine.
            logger.warning(
                "ddtree_verify_tree: no logits captured by _greedy_sample; "
                "returning zero-token fallback."
            )
            self._ddtree_pending_mask = None
            self._ddtree_pending_child_maps = None
            self._ddtree_pending_node_tokens = None
            return torch.zeros(
                (batch_size, budget), dtype=torch.int64, device=device
            )

        horizon = self.DFLASH_DRAFT_HORIZON
        expected_rows = batch_size * horizon
        if logits.dim() != 2 or logits.shape[0] != expected_rows:
            logger.warning(
                "ddtree_verify_tree: unexpected logits shape %s "
                "(expected [%d, vocab]). Falling back to zero tokens.",
                tuple(logits.shape),
                expected_rows,
            )
            self._ddtree_pending_mask = None
            self._ddtree_pending_child_maps = None
            self._ddtree_pending_node_tokens = None
            return torch.zeros(
                (batch_size, budget), dtype=torch.int64, device=device
            )

        logits_per_req = logits.view(batch_size, horizon, -1)

        # We build trees on CPU (build_ddtree_tree is CPU-bound), then
        # stack and move the outputs to device once.
        node_tokens_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        # S3c-3: stash per-request child_maps and node-token lists so
        # the runner's tree-follow accept can walk the tree using
        # target's posterior at the next verify pass.
        child_maps_per_req: list[list[dict[int, int]]] = []
        node_tokens_per_req: list[list[int]] = []
        for req_idx in range(batch_size):
            req_logits = logits_per_req[req_idx].detach()
            nti, _nd, _parents, child_maps, visibility = build_ddtree_tree(
                req_logits, budget=budget
            )

            # Pad ``nti`` to fixed length ``budget`` (build_ddtree_tree
            # can return fewer than ``budget`` if depth_limit*topk is
            # small; for our gemma-4 dflash horizon=15 and topk capped
            # at budget=15, it returns exactly budget nodes).
            n_nodes = int(nti.numel())
            if n_nodes < budget:
                pad = torch.zeros(budget - n_nodes, dtype=torch.long)
                nti_padded = torch.cat([nti, pad], dim=0)
            else:
                nti_padded = nti[:budget]
            node_tokens_list.append(nti_padded)

            mask_cpu = torch.zeros(
                (full_size, full_size), dtype=torch.bool
            )
            seen = 1 + n_nodes
            if seen > 0:
                mask_cpu[:seen, :seen] = visibility
            for pad_idx in range(seen, full_size):
                mask_cpu[pad_idx, pad_idx] = True
            mask_list.append(mask_cpu)

            # Normalize child_maps length to ``1 + budget`` (pad with
            # empty dicts when the tree expanded fewer nodes than
            # budget; padding nodes have no children, so the walk will
            # stop at them naturally).
            cm = list(child_maps)
            while len(cm) < full_size:
                cm.append({})
            child_maps_per_req.append(cm)
            node_tokens_per_req.append([int(t) for t in nti_padded.tolist()])

        node_tokens_cpu = torch.stack(node_tokens_list, dim=0)
        mask_cpu_batched = torch.stack(mask_list, dim=0)

        node_tokens = node_tokens_cpu.to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        self._ddtree_pending_mask = mask_cpu_batched.to(
            device=device, non_blocking=True
        )
        self._ddtree_pending_child_maps = child_maps_per_req
        self._ddtree_pending_node_tokens = node_tokens_per_req

        # One-line trace per call when verify is on (small set of calls
        # under a benchmark; vllm logger throttles automatically).
        logger.debug(
            "ddtree_verify_tree: batch=%d budget=%d horizon=%d "
            "mask_shape=%s",
            batch_size,
            budget,
            horizon,
            tuple(self._ddtree_pending_mask.shape),
        )

        # Drop logits ref before returning.
        self._ddtree_last_logits = None
        return node_tokens

    def _ddtree_maybe_build_and_log(self, draft_token_ids: torch.Tensor) -> None:
        """Build a per-batch tree mask, stash it, and aggregate stats.

        Sampling and aggregation policy (unchanged from S1):
          * Every ``_DDTREE_BUILD_EVERY`` propose() calls we run the
            tree-build path. The stats accumulator advances by one each
            time (using the first request of the batch for size/depth/
            top1-mass) and is flushed to an INFO log line once
            ``_DDTREE_LOG_FLUSH_AT`` samples have accumulated.

        S3b addition:
          * On every build cycle we now construct a per-request tree
            and stack the visibility matrices into a
            ``[batch, 1+budget, 1+budget]`` bool tensor on
            ``self.device``. The tensor is stashed at
            ``self._ddtree_pending_mask`` for later consumption (no
            consumer in this revision; S3c will plumb it).
          * Padding rows (when a request's tree has fewer than budget
            nodes) get an identity entry so that, when the mask is
            eventually applied, padded query slots only attend to
            themselves — a benign no-op.
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

        logits_per_req = logits.view(batch_size, num_spec, -1)
        full_size = 1 + self.ddtree_budget
        per_request_masks: list[torch.Tensor] = []

        stats_collected = False
        for req_idx in range(batch_size):
            req_logits = logits_per_req[req_idx].detach()
            nti, nd, _parents, _child_maps, visibility = build_ddtree_tree(
                req_logits, budget=self.ddtree_budget
            )
            node_count = int(nti.numel())
            seen = 1 + node_count

            # Build [full_size, full_size] mask on CPU, copy to device once.
            mask_cpu = torch.zeros(
                (full_size, full_size), dtype=torch.bool
            )
            if seen > 0:
                mask_cpu[:seen, :seen] = visibility
            # Padding rows: each padding slot only attends to itself
            # (identity diagonal). Benign because S3c's query expansion
            # will not address these slots anyway, but keeps the mask
            # well-formed if a kernel iterates over the full grid.
            for pad_idx in range(seen, full_size):
                mask_cpu[pad_idx, pad_idx] = True
            per_request_masks.append(mask_cpu)

            # Collect aggregate stats from request 0 only (cheap).
            if not stats_collected:
                stats_collected = True
                max_depth = int(nd.max().item()) if nd.numel() else 0
                f_logits = req_logits.float()
                top1_logits = f_logits.max(dim=-1).values
                lse = torch.logsumexp(f_logits, dim=-1)
                log_top1_path = float((top1_logits - lse).sum().item())
                top1_mass = float(torch.tensor(log_top1_path).exp().item())

                self._ddtree_agg_n += 1
                self._ddtree_agg_size += node_count
                self._ddtree_agg_max_depth += max_depth
                self._ddtree_agg_top1_mass += top1_mass

        # Stack per-request masks and move once to GPU.
        mask_batched = torch.stack(per_request_masks, dim=0).to(
            self.device, non_blocking=True
        )
        self._ddtree_pending_mask = mask_batched

        if self._ddtree_agg_n >= self._DDTREE_LOG_FLUSH_AT:
            n = self._ddtree_agg_n
            non_masked_ratio = float(mask_batched.float().mean().item())
            logger.info(
                "ddtree real-logits: samples=%d avg_size=%.1f "
                "avg_max_depth=%.1f avg_top1_mass=%.4f budget=%d N=%d | "
                "mask_shape=%s dtype=%s non_masked_ratio=%.3f "
                "(S3b stash; not yet consumed by runner/kernel)",
                n,
                self._ddtree_agg_size / n,
                self._ddtree_agg_max_depth / n,
                self._ddtree_agg_top1_mass / n,
                self.ddtree_budget,
                num_spec,
                tuple(mask_batched.shape),
                mask_batched.dtype,
                non_masked_ratio,
            )
            self._ddtree_agg_n = 0
            self._ddtree_agg_size = 0
            self._ddtree_agg_max_depth = 0
            self._ddtree_agg_top1_mass = 0.0
