# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions of this file (build_ddtree_tree, compile_ddtree_tree,
# follow_verified_tree) are adapted from the reference DDTree implementation
# by Liran Ringel (https://github.com/liranringel/ddtree), used under the MIT
# License. See LICENSE-MIT-DDTREE at the repository root or
# https://github.com/liranringel/ddtree/blob/main/LICENSE for the full text.
#
# Reference: Liran Ringel, Yaniv Romano,
#   "Accelerating Speculative Decoding with Block Diffusion Draft Trees",
#   arXiv:2604.12989, 2026.
"""DDTree primitives, lifted out of the reference HF-Transformers codebase.

These are CPU-side helpers that build, compile, and walk a small top-k
draft tree from a sequence of draft logits produced by a DFlash-style
parallel drafter. They are deliberately framework-agnostic: only torch
tensor input/output, no DynamicCache or HF coupling. They are intended to
be called from the DDTreeProposer inside vLLM's spec decode path.
"""

from __future__ import annotations

import heapq

import numpy as np
import torch


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
]:
    """Heap-expand a top-k draft tree from per-depth draft logits.

    Args:
        draft_logits: tensor of shape ``[depth, vocab]``, one row of draft
            logits per draft depth. Floating point. Lives on any device;
            only ``topk`` and ``logsumexp`` touch the GPU.
        budget: maximum number of tree nodes to expand beyond the root.

    Returns:
        ``node_token_ids``: int64 tensor of shape ``[node_count]``, the
            token id of each expanded child node in expansion order.
        ``node_depths``: int64 tensor of shape ``[node_count]``, depth of
            each expanded child relative to the (implicit) root at depth 0.
        ``parents``: python list of length ``1 + node_count``; ``parents[0]``
            is ``-1`` for the root, ``parents[i]`` is the index of the
            parent node for node ``i``.
        ``child_maps``: list of length ``1 + node_count`` of ``dict[int,
            int]`` mapping ``token_id -> child_index`` for each node.
        ``visibility``: bool tensor of shape ``[1+node_count, 1+node_count]``
            with ``True`` on entries where row's tree path includes the
            column index. The 2D attention visibility mask.
    """
    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw),
            )

    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility


def compile_ddtree_tree(
    root_token_id: torch.Tensor,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Materialize verify-side input ids / positions / attention mask.

    Adapted from the reference DDTree implementation. Buffers are
    pre-allocated and reused across rounds. The resulting attention mask
    uses ``-inf`` for masked positions (HF-style additive mask). The
    output is *not* directly compatible with vLLM's per-layer
    attention metadata; this helper is provided for parity with the
    reference algorithm and for future use in the C2 verify-side path.
    """
    current_length = 1 + int(node_token_ids.numel())

    if previous_tree_length > 0:
        attention_mask_buffer[
            0,
            0,
            :previous_tree_length,
            previous_tree_start : previous_tree_start + previous_tree_length,
        ] = 0

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids, non_blocking=False)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths, non_blocking=False)
        verify_position_ids[0, 1:current_length].add_(start)

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(visibility_cpu, non_blocking=False)

    tree_block = attention_mask_buffer[
        0,
        0,
        :current_length,
        past_length : past_length + current_length,
    ]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)

    attention_mask = attention_mask_buffer[
        :, :, :current_length, : past_length + current_length
    ]
    return (
        verify_input_ids,
        verify_position_ids,
        attention_mask,
        past_length,
        current_length,
    )


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior: torch.Tensor,
) -> tuple[list[int], int]:
    """Greedy walk of the verified tree along the sampled posterior.

    ``posterior`` is the sampled-id tensor returned by the target for the
    full flattened tree (shape ``[1, 1+node_count]``). The walk starts at
    the root, and at each step checks whether the sampled token for the
    current node matches an edge to a child; if so, descend; if not,
    stop and use the sampled token as the "bonus" next token.
    """
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token
