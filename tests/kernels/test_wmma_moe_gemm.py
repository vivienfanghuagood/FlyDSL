#!/usr/bin/env python3
"""Test WMMA MoE GEMM kernels for RDNA4 (gfx12xx).

Tests correctness of the WMMA-based MoE GEMM stage1 and stage2 kernels
against PyTorch reference implementations.
"""

import sys
import os
import time
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from flydsl.runtime.device import get_rocm_arch

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

gpu_arch = get_rocm_arch()
if not gpu_arch.startswith("gfx12"):
    pytest.skip(
        f"WMMA MoE GEMM requires RDNA4 (gfx12xx), got {gpu_arch}",
        allow_module_level=True,
    )

from kernels.wmma_moe_gemm import (
    compile_wmma_moe_gemm1,
    compile_wmma_moe_gemm2,
    preshuffle_w_wmma,
)

device = "cuda"


# =============================================================================
# MoE Routing (simplified torch version)
# =============================================================================


def moe_sorting_torch(topk_ids, topk_weights, num_experts, block_size):
    """Simple MoE token sorting. Returns (sorted_ids, sorted_weights, expert_ids, max_token_ids)."""
    M, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    init_val = (topk << 24) | M
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_val, dtype=torch.int32, device=device
    )
    sorted_weights = torch.zeros(
        (max_num_tokens_padded,), dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=device
    )
    max_token_ids = torch.zeros((2,), dtype=torch.int32, device=device)

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    for e in range(num_experts):
        token_id, topk_id = torch.where(topk_ids == e)
        n = token_id.numel()
        n_blocks = (n + block_size - 1) // block_size
        n_padded = n_blocks * block_size
        sorted_ids[sorted_ids_begin : sorted_ids_begin + n] = (
            topk_id.to(torch.int32) << 24
        ) | token_id.to(torch.int32)
        sorted_weights[sorted_ids_begin : sorted_ids_begin + n] = topk_weights[
            token_id, topk_id
        ].float()
        sorted_ids_begin += n_padded
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + n_blocks
        ] = e
        sorted_expert_ids_begin += n_blocks

    max_token_ids[0] = sorted_ids_begin  # total padded tokens
    max_token_ids[1] = M  # num tokens

    return sorted_ids, sorted_weights, sorted_expert_ids, max_token_ids


# =============================================================================
# Reference implementations
# =============================================================================


def torch_moe_gemm1_ref(x, w1, topk_ids, topk_weights, inter_dim, doweight):
    """PyTorch reference for MoE stage1. Returns [tokens, topk, inter_dim]."""
    tokens, model_dim = x.shape
    topk = topk_ids.shape[1]
    experts = w1.shape[0]
    out = torch.zeros((tokens, topk, inter_dim), device=device, dtype=torch.float32)
    for e in range(experts):
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        t_idx, s_idx = idx[:, 0], idx[:, 1]
        x_in = x[t_idx].float()
        y2 = F.linear(x_in, w1[e].float())  # [n, 2*inter_dim]
        gate = y2[:, :inter_dim]
        up = y2[:, inter_dim:]
        y = F.silu(gate) * up
        if doweight:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1).float()
        out[t_idx, s_idx] = y
    return out


def torch_moe_gemm2_ref(a2, w2, topk_ids, topk_weights, model_dim, doweight):
    """PyTorch reference for MoE stage2. Returns [tokens, model_dim]."""
    tokens, topk, inter_dim = a2.shape
    experts = w2.shape[0]
    out = torch.zeros((tokens, model_dim), device=device, dtype=torch.float32)
    for e in range(experts):
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        t_idx, s_idx = idx[:, 0], idx[:, 1]
        y = F.linear(a2[t_idx, s_idx].float(), w2[e].float())
        if doweight:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1).float()
        out.index_add_(0, t_idx, y)
    return out


# =============================================================================
# Stage 1 Tests
# =============================================================================


@pytest.mark.parametrize(
    "tokens,model_dim,inter_dim,experts,topk",
    [
        (32, 128, 64, 4, 2),
        (64, 256, 128, 4, 2),
        (128, 512, 256, 8, 2),
    ],
    ids=["small", "medium", "larger"],
)
def test_wmma_moe_stage1_correctness(tokens, model_dim, inter_dim, experts, topk):
    """Test WMMA MoE stage1 correctness."""
    tile_m = 16
    tile_n = min(64, inter_dim)  # Ensure tile_n <= inter_dim
    tile_k = 32
    assert inter_dim % tile_n == 0
    assert model_dim % tile_k == 0

    print(
        f"\nMoE Stage1: tokens={tokens}, model={model_dim}, inter={inter_dim}, "
        f"E={experts}, topk={topk}"
    )

    torch.manual_seed(42)
    x = torch.randn(tokens, model_dim, device=device, dtype=torch.bfloat16) * 0.1
    # W1: [E, 2*inter_dim, model_dim]
    w1 = (
        torch.randn(
            experts, 2 * inter_dim, model_dim, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )

    # Generate routing
    logits = torch.randn(tokens, experts, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    sorted_ids, sorted_weights, expert_ids, max_token_ids = moe_sorting_torch(
        topk_ids,
        topk_weights,
        experts,
        tile_m,
    )

    # Preshuffle weights
    w1_shuf_list = []
    for e in range(experts):
        w1_shuf_list.append(preshuffle_w_wmma(w1[e]).flatten())
    w1_shuf = torch.cat(w1_shuf_list)

    # Output
    out = torch.zeros(tokens, topk, inter_dim, device=device, dtype=torch.bfloat16)
    scale_x = torch.empty(0, device=device, dtype=torch.float32)
    scale_w = torch.empty(0, device=device, dtype=torch.float32)

    stream_ptr = torch.cuda.current_stream().cuda_stream
    num_expert_blocks = expert_ids.numel()

    # Compile and run
    exe = compile_wmma_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        in_dtype="bf16",
        out_dtype="bf16",
    )

    exe(
        out.flatten(),
        x.flatten(),
        w1_shuf,
        scale_x,
        scale_w,
        sorted_ids,
        expert_ids,
        sorted_weights,
        max_token_ids,
        tokens,
        inter_dim,
        model_dim,
        num_expert_blocks,
        stream_ptr,
    )
    torch.cuda.synchronize()

    # Reference
    expected = torch_moe_gemm1_ref(
        x, w1, topk_ids, topk_weights, inter_dim, doweight=False
    )

    # Check
    out_f32 = out.float()
    exp_f32 = expected.float()

    # Mask out entries where expected is zero (no routing)
    nonzero_mask = exp_f32.abs() > 1e-6
    if nonzero_mask.any():
        diff = (out_f32[nonzero_mask] - exp_f32[nonzero_mask]).abs()
        max_abs = diff.max().item()
        ref_max = exp_f32[nonzero_mask].abs().max().item()
        rel_error = max_abs / (ref_max + 1e-8)
    else:
        rel_error = 0.0
        max_abs = 0.0

    print(f"  Max abs error: {max_abs:.2e}")
    print(f"  Max rel error: {rel_error:.2e}")
    assert rel_error < 0.15, f"Error too high: rel_error={rel_error:.2e}"
    print("  PASS")


# =============================================================================
# Stage 2 Tests
# =============================================================================


@pytest.mark.parametrize(
    "tokens,model_dim,inter_dim,experts,topk",
    [
        (32, 128, 64, 4, 2),
        (64, 256, 128, 4, 2),
    ],
    ids=["small", "medium"],
)
def test_wmma_moe_stage2_correctness(tokens, model_dim, inter_dim, experts, topk):
    """Test WMMA MoE stage2 correctness."""
    tile_m = 16
    tile_n = min(64, model_dim)
    tile_k = 32
    assert model_dim % tile_n == 0
    assert inter_dim % tile_k == 0

    print(
        f"\nMoE Stage2: tokens={tokens}, model={model_dim}, inter={inter_dim}, "
        f"E={experts}, topk={topk}"
    )

    torch.manual_seed(42)
    # A2: [tokens, topk, inter_dim] - stage1 output
    a2 = torch.randn(tokens, topk, inter_dim, device=device, dtype=torch.bfloat16) * 0.1
    # W2: [E, model_dim, inter_dim]
    w2 = (
        torch.randn(experts, model_dim, inter_dim, device=device, dtype=torch.bfloat16)
        * 0.1
    )

    # Generate routing
    logits = torch.randn(tokens, experts, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    sorted_ids, sorted_weights, expert_ids, max_token_ids = moe_sorting_torch(
        topk_ids,
        topk_weights,
        experts,
        tile_m,
    )

    # Preshuffle W2
    w2_shuf_list = []
    for e in range(experts):
        w2_shuf_list.append(preshuffle_w_wmma(w2[e]).flatten())
    w2_shuf = torch.cat(w2_shuf_list)

    # Output: [tokens, topk, model_dim] (non-accumulate mode)
    out = torch.zeros(tokens, topk, model_dim, device=device, dtype=torch.bfloat16)
    scale_a2 = torch.empty(0, device=device, dtype=torch.float32)
    scale_w = torch.empty(0, device=device, dtype=torch.float32)

    stream_ptr = torch.cuda.current_stream().cuda_stream
    num_expert_blocks = expert_ids.numel()

    exe = compile_wmma_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=True,
        in_dtype="bf16",
        out_dtype="bf16",
        accumulate=False,
    )

    exe(
        out.flatten(),
        a2.flatten(),
        w2_shuf,
        scale_a2,
        scale_w,
        sorted_ids,
        expert_ids,
        sorted_weights,
        max_token_ids,
        tokens,
        model_dim,
        inter_dim,
        num_expert_blocks,
        stream_ptr,
    )
    torch.cuda.synchronize()

    # For non-accumulate mode, compare per-(token,slot) output
    # Then manually reduce and compare with reference
    out_reduced = out.float().sum(dim=1)  # [tokens, model_dim]
    expected = torch_moe_gemm2_ref(
        a2, w2, topk_ids, topk_weights, model_dim, doweight=True
    )

    nonzero_mask = expected.abs() > 1e-6
    if nonzero_mask.any():
        diff = (out_reduced[nonzero_mask] - expected[nonzero_mask]).abs()
        max_abs = diff.max().item()
        ref_max = expected[nonzero_mask].abs().max().item()
        rel_error = max_abs / (ref_max + 1e-8)
    else:
        rel_error = 0.0
        max_abs = 0.0

    print(f"  Max abs error: {max_abs:.2e}")
    print(f"  Max rel error: {rel_error:.2e}")
    assert rel_error < 0.15, f"Error too high: rel_error={rel_error:.2e}"
    print("  PASS")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=== WMMA MoE Stage 1 Tests ===")
    for tokens, model_dim, inter_dim, experts, topk in [
        (32, 128, 64, 4, 2),
        (64, 256, 128, 4, 2),
    ]:
        try:
            test_wmma_moe_stage1_correctness(
                tokens, model_dim, inter_dim, experts, topk
            )
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback

            traceback.print_exc()

    print("\n=== WMMA MoE Stage 2 Tests ===")
    for tokens, model_dim, inter_dim, experts, topk in [
        (32, 128, 64, 4, 2),
        (64, 256, 128, 4, 2),
    ]:
        try:
            test_wmma_moe_stage2_correctness(
                tokens, model_dim, inter_dim, experts, topk
            )
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback

            traceback.print_exc()
