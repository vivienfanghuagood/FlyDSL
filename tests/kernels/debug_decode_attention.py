#!/usr/bin/env python3
"""Debug FlyDSL decode attention kernel — minimal test."""

import math
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from kernels.wmma_decode_attention import compile_decode_attention


def reference_gqa_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads
    o = torch.zeros_like(q)
    for b in range(batch):
        kv_start = kv_indptr[b].item()
        kv_end = kv_indptr[b + 1].item()
        indices = kv_indices[kv_start:kv_end]
        for h in range(num_heads):
            kv_h = h // kv_group_num
            q_vec = q[b, h].float()
            k_mat = k_buffer[indices, kv_h].float()
            v_mat = v_buffer[indices, kv_h].float()
            scores = (q_vec @ k_mat.T) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v_mat
            o[b, h] = out.to(q.dtype)
    return o


def test_single():
    """Test with simplest possible config."""
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    bs = 1
    kv_len = 256  # test with larger KV

    print(f"=== Debug: BS={bs}, KV={kv_len}, 1 wave ===")
    exe = compile_decode_attention(
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_kv_len=1024,
        num_waves=4,  # test multi-wave (production config)
    )

    torch.manual_seed(42)
    q = torch.randn(bs, num_heads, head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    total_kv = bs * kv_len
    k_buffer = (
        torch.randn(
            total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.1
    )
    v_buffer = (
        torch.randn(
            total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.1
    )
    kv_indptr = torch.arange(
        0, (bs + 1) * kv_len, kv_len, device="cuda", dtype=torch.int32
    )
    kv_indices = torch.arange(0, total_kv, device="cuda", dtype=torch.int32)
    o = torch.zeros(bs, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Run
    q_flat = q.contiguous().view(-1)
    k_flat = k_buffer.contiguous().view(-1)
    v_flat = v_buffer.contiguous().view(-1)
    o_flat = o.contiguous().view(-1)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    exe(
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        kv_indptr,
        kv_indices,
        sm_scale,
        bs,
        num_heads,
        head_dim,
        num_kv_heads,
        stream_ptr,
    )
    torch.cuda.synchronize()

    o_ref = reference_gqa_attention(
        q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
    )

    # Compare head 0 only
    fly_h0 = o[0, 0].float().cpu()
    ref_h0 = o_ref[0, 0].float().cpu()

    cos = torch.nn.functional.cosine_similarity(fly_h0, ref_h0, dim=0).item()
    print(f"  Head 0 cos_sim: {cos:.6f}")
    print(f"  Head 0 max_diff: {(fly_h0 - ref_h0).abs().max().item():.4e}")
    print(f"  FlyDSL head 0 first 16: {fly_h0[:16].tolist()}")
    print(f"  Ref    head 0 first 16: {ref_h0[:16].tolist()}")
    print(f"  FlyDSL head 0 [16:32]:  {fly_h0[16:32].tolist()}")
    print(f"  Ref    head 0 [16:32]:  {ref_h0[16:32].tolist()}")

    # Check if output is all zeros or garbage
    print(f"  FlyDSL norm: {fly_h0.norm():.6f}")
    print(f"  Ref norm: {ref_h0.norm():.6f}")

    # Also test Q@K^T correctness manually
    q_h0 = q[0, 0].float().cpu()  # [128]
    kv_h = 0  # head 0 maps to kv_head 0
    k_tokens = k_buffer[:, kv_h].float().cpu()  # [16, 128]
    scores_ref = (q_h0 @ k_tokens.T) * sm_scale
    print(f"\n  Ref Q@K^T scores (head 0): {scores_ref.tolist()}")
    attn_ref = torch.softmax(scores_ref, dim=-1)
    print(f"  Ref attn weights: {attn_ref.tolist()}")

    # Test with identity-like V to isolate P@V vs Q@K^T issues
    print("\n=== Test with V = identity-like (isolate P@V) ===")
    # Set V to be an identity-like pattern
    v_test = torch.zeros_like(v_buffer)
    for t in range(kv_len):
        v_test[t, :, :] = 0
        # Set just one element per token to 1.0
        for kh in range(num_kv_heads):
            v_test[t, kh, t * (head_dim // kv_len)] = 1.0

    o2 = torch.zeros_like(o)
    o2_flat = o2.contiguous().view(-1)
    v_test_flat = v_test.contiguous().view(-1)
    exe(
        q_flat,
        k_flat,
        v_test_flat,
        o2_flat,
        kv_indptr,
        kv_indices,
        sm_scale,
        bs,
        num_heads,
        head_dim,
        num_kv_heads,
        stream_ptr,
    )
    torch.cuda.synchronize()

    o2_ref = reference_gqa_attention(
        q, k_buffer, v_test, kv_indptr, kv_indices, sm_scale
    )

    fly2_h0 = o2[0, 0].float().cpu()
    ref2_h0 = o2_ref[0, 0].float().cpu()
    cos2 = torch.nn.functional.cosine_similarity(fly2_h0, ref2_h0, dim=0).item()
    print(f"  V=identity Head 0 cos_sim: {cos2:.6f}")
    print(f"  FlyDSL first 16: {fly2_h0[:16].tolist()}")
    print(f"  Ref    first 16: {ref2_h0[:16].tolist()}")


if __name__ == "__main__":
    test_single()
