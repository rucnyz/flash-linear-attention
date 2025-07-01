# -*- coding: utf-8 -*-

import os
from typing import List

import pytest
import torch
from einops import rearrange

from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.path_attn_poly.parallel import triton_kernel_path_attn
from fla.ops.utils import solve_tril
from fla.utils import (
    COMPILER_MODE,
    assert_close,
    check_shared_mem,
    device,
    is_intel_alchemist,
)

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [1024]
    test_d_list = [64]
else:
    test_b_list = [1]
    test_t_list = [192, 400]
    test_d_list = [64]
test_fgate_logit_range_list = [(0.95, 1), (1, 1)]
test_hq_list = [2, 8]
test_h_list = [1, 2]


def naive_parallel_kernel(q, k, q2, k2, v, beta, g, scale, BT=64):
    original_dtype = q.dtype
    HQ = q.shape[1]
    H = k.shape[1]

    k = k.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    k2 = k2.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2).contiguous()
    beta = beta.unsqueeze(2).expand(-1, -1, HQ // H, -1).flatten(1, 2)
    b, h, original_seq_length, d_k = q.shape
    k_beta = k * beta[..., None]

    A = chunk_scaled_dot_kkt_fwd(
        k=k.transpose(1, 2).contiguous(),
        beta=beta.transpose(1, 2).contiguous(),
        cu_seqlens=None,
        chunk_size=BT,
        output_dtype=torch.float32,
    )
    A2 = chunk_scaled_dot_kkt_fwd(
        k=k2.transpose(1, 2).contiguous(),
        beta=beta.transpose(1, 2).contiguous(),
        cu_seqlens=None,
        chunk_size=BT,
        output_dtype=torch.float32,
    )
    T = solve_tril(A, output_dtype=k.dtype).transpose(1, 2)
    T2 = solve_tril(A2, output_dtype=k.dtype).transpose(1, 2)

    if original_seq_length % BT != 0:
        padding_size = BT - original_seq_length % BT
        pad_seq_length = original_seq_length + padding_size
        q, k, k_beta, q2, k2, T, T2 = map(
            lambda x: torch.nn.functional.pad(x, (0, 0, 0, padding_size)),
            [q, k, k_beta, q2, k2, T, T2],
        )
    else:
        pad_seq_length = original_seq_length

    # T_n = -(k @ k.transpose(-1, -2)).tril(-1)
    # for i in range(1, pad_seq_length):
    #     T_n[..., i, :i] = T_n[..., i, :i].clone() + (T_n[..., i, :, None].clone() * T_n[..., :, :i].clone()).sum(-2)
    # T_n = T_n + torch.eye(pad_seq_length, dtype=q.dtype, device=q.device)

    q, k, q2, k2 = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=BT), [q, k, q2, k2]
    )
    num_steps = q.shape[2]
    A_logit = cal_A(
        q,
        k,
        T,
        b,
        h,
        pad_seq_length,
        original_seq_length,
        num_steps,
        original_dtype,
        BT=BT,
    )
    A_logit2 = cal_A(
        q2,
        k2,
        T2,
        b,
        h,
        pad_seq_length,
        original_seq_length,
        num_steps,
        original_dtype,
        BT=BT,
    )
    A_logit_2nd = A_logit * A_logit2

    # g_cumsum shape: [batch, heads, seq_len]
    g_cumsum = g.cumsum(dim=-1)
    g_diff = g_cumsum[:, :, :, None] - g_cumsum[:, :, None, :]
    # shape: [batch, heads, seq_len, seq_len]

    causal_mask = (
        torch.arange(original_seq_length)[:, None]
        < torch.arange(original_seq_length)[None, :]
    )
    causal_mask = causal_mask.to(q.device)

    g_diff_masked = g_diff.masked_fill_(causal_mask, float("-inf"))

    exp_g_diff = torch.exp(g_diff_masked)
    A_logit_2nd = A_logit_2nd * exp_g_diff
    o = A_logit_2nd.to(original_dtype) @ v
    return o.to(original_dtype)


def compute_T_mj(T, k, m, j, BT, computed_T=None):
    """
    递归计算 T_{m,j}
    computed_T 用于缓存已经计算过的 T_{i,j} 避免重复计算
    """
    if computed_T is None:
        computed_T = {}

    if m == j:
        # return T[:, :, m * BT : (m + 1) * BT, :]
        return T[:, :, m * BT : (m + 1) * BT, :]

    if (m, j) in computed_T:
        return computed_T[(m, j)]

    sum_term = torch.zeros(
        T.shape[0], T.shape[1], BT, BT, device=T.device, dtype=T.dtype
    )

    # 计算累加和
    k_m = k[:, :, m]
    for f in range(j, m):
        k_f = k[:, :, f]

        A_mf = k_m @ k_f.transpose(-1, -2)

        T_fj = compute_T_mj(T, k, f, j, BT, computed_T)

        sum_term += A_mf @ T_fj

    T_mm = T[:, :, m * BT : (m + 1) * BT, :]
    T_mj = -T_mm @ sum_term
    # cache the computed T_{m,j}
    computed_T[(m, j)] = T_mj
    return T_mj


def cal_A(
    q, k, T, b, h, pad_seq_length, original_seq_length, num_steps, original_dtype, BT=64
):
    A = torch.zeros(b, h, pad_seq_length, pad_seq_length, device=q.device).to(
        original_dtype
    )
    computed_T = {}
    for i in range(num_steps):
        q_i = q[:, :, i]
        for j in range(i + 1):
            for m in range(j, i + 1):
                k_m = k[:, :, m]
                q_i_k_m = q_i @ k_m.transpose(-1, -2)
                if m == i:
                    q_i_k_m = q_i_k_m.tril()

                T_mj = compute_T_mj(T, k, m, j, BT, computed_T)

                A[:, :, i * BT : (i + 1) * BT, j * BT : (j + 1) * BT] += q_i_k_m @ T_mj
    A = A[:, :, :original_seq_length, :original_seq_length]
    return A


def solve_T(k, BT):
    # calculate (1+kk^t)^-1
    T = -(k @ k.transpose(-1, -2)).tril(-1)
    for i in range(1, BT):
        T[..., i, :i] = T[..., i, :i].clone() + (
            T[..., i, :, None].clone() * T[..., :, :i].clone()
        ).sum(-2)
    T = T + torch.eye(BT, dtype=k.dtype, device=k.device)
    return T


def naive_kernel(q, k, q2, k2, v, beta, g, scale, BT=64):
    original_dtype = q.dtype
    HQ = q.shape[1]
    H = k.shape[1]
    k = k.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    k2 = k2.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)

    g_cumsum = g.cumsum(dim=-1)
    qk = (q @ k.transpose(-1, -2)).tril()
    qk2 = (q2 @ k2.transpose(-1, -2)).tril()
    ikk = torch.eye(BT).to(q.device) + (k @ k.transpose(-1, -2)).tril(-1)
    ikk2 = torch.eye(BT).to(q.device) + (k2 @ k2.transpose(-1, -2)).tril(-1)
    A_logit = qk @ torch.inverse(ikk).to(original_dtype)
    A_logit2 = qk2 @ torch.inverse(ikk2).to(original_dtype)

    A_logit_2nd = A_logit * A_logit2

    # g_cumsum shape: [batch, heads, seq_len]
    g_diff = g_cumsum[:, :, :, None] - g_cumsum[:, :, None, :]
    # shape: [batch, heads, seq_len, seq_len]

    causal_mask = torch.arange(BT)[:, None] < torch.arange(BT)[None, :]
    causal_mask = causal_mask.to(q.device)

    g_diff_masked = g_diff.masked_fill_(causal_mask, float("-inf"))

    exp_g_diff = torch.exp(g_diff_masked)
    A_logit_2nd = A_logit_2nd * exp_g_diff
    o = A_logit_2nd.to(original_dtype) @ v
    return o.to(original_dtype)


def naive_chunk_kernel(q, k, q2, k2, v, beta, g, scale, BT=64):
    original_dtype = q.dtype
    HQ = q.shape[1]
    H = k.shape[1]
    k = k.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    k2 = k2.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    d = q.shape[-1]
    S = torch.zeros(q.shape[0], q.shape[1], d, d, d).to(q.device)
    o2 = torch.zeros_like(q)
    # 对第一个维度来做 (I-kk^T) transition.
    for i in range(BT):
        # 先对第一个维度来做
        k_i = k[:, :, i]
        k2_i = k2[:, :, i]
        v_i = v[:, :, i]
        q_i = q[:, :, i]
        q2_i = q2[:, :, i]
        S = torch.einsum("efabc, ef -> efabc", S, g[:, :, i].exp()).to(original_dtype)
        S = S - torch.einsum("efabc, efa, efd -> efdbc", S, k_i, k_i)
        S = S - torch.einsum("efabc, efb, efd -> efadc", S, k2_i, k2_i)
        S += torch.einsum("efa, efb, efc -> efabc", k_i, k2_i, v_i)
        o_i = torch.einsum("efabc, efa, efb -> efc", S, q_i, q2_i)
        o2[:, :, i] = o_i
    return o2.to(original_dtype)


def naive_kernel_path_attn(q, k, q2, k2, v, beta, g, scale, BT=64):
    original_dtype = q.dtype
    HQ = q.shape[1]
    H = k.shape[1]
    g_cumsum = g.cumsum(-1)
    #
    q = q.unsqueeze(2).expand(-1, -1, HQ // HQ, -1, -1).flatten(1, 2)
    q2 = q2.unsqueeze(2).expand(-1, -1, HQ // HQ, -1, -1).flatten(1, 2)
    k = k.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    k2 = k2.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, HQ // H, -1, -1).flatten(1, 2)
    beta = beta.unsqueeze(2).expand(-1, -1, HQ // H, -1).flatten(1, 2)
    g_cumsum = g_cumsum.unsqueeze(2).expand(-1, -1, HQ // HQ, -1).flatten(1, 2)
    b, h, l, d_k = q.shape
    if l % BT != 0:
        padding_size = BT - l % BT
        q, k, q2, k2 = map(
            lambda x: torch.nn.functional.pad(x, (0, 0, 0, padding_size)),
            [q, k, q2, k2],
        )
        beta = torch.nn.functional.pad(beta, (0, padding_size))
    seq_len = q.shape[2]
    k_beta = k * beta[..., None]
    k_beta2 = k2 * beta[..., None]

    q, k, q2, k2, k_beta, k_beta2 = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=BT),
        [q, k, q2, k2, k_beta, k_beta2],
    )
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    T2 = -(k_beta2 @ k2.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BT):
        T[..., i, :i] = T[..., i, :i].clone() + (
            T[..., i, :, None].clone() * T[..., :, :i].clone()
        ).sum(-2)
    T = T + torch.eye(BT, dtype=q.dtype, device=q.device)

    for i in range(1, BT):
        T2[..., i, :i] = T2[..., i, :i].clone() + (
            T2[..., i, :, None].clone() * T2[..., :, :i].clone()
        ).sum(-2)
    T2 = T2 + torch.eye(BT, dtype=q.dtype, device=q.device)

    Tkbk = T @ (k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    Tkbk2 = T2 @ (k_beta2 @ k2.transpose(-1, -2)).masked_fill(mask, 0)
    qk = (q @ k.transpose(-1, -2)).tril()
    qk2 = (q2 @ k2.transpose(-1, -2)).tril()
    Tkb = T @ k_beta
    Tkb2 = T2 @ k_beta2
    A_local = (q @ k.transpose(-1, -2)).tril() - qk @ Tkbk
    A_local2 = (q2 @ k2.transpose(-1, -2)).tril() - qk2 @ Tkbk2

    H = k.transpose(-1, -2) @ Tkb
    H2 = k2.transpose(-1, -2) @ Tkb2

    q = q - qk @ Tkb
    k = k - Tkbk.transpose(-1, -2) @ k

    q2 = q2 - qk2 @ Tkb2
    k2 = k2 - Tkbk2.transpose(-1, -2) @ k2


    A = torch.zeros(b, h, seq_len, seq_len, device=q.device)
    A2 = torch.zeros(b, h, seq_len, seq_len, device=q.device)
    q, k, q2, k2, k_beta, k_beta2 = map(
        lambda x: rearrange(x, "b h n c d -> b h (n c) d"),
        [q, k, q2, k2, k_beta, k_beta2],
    )
    for i in range(0, seq_len, BT):
        q_i = q[:, :, i : i + BT].clone()
        for j in range(i - BT, -BT, -BT):
            k_j = k[:, :, j : j + BT]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i : i + BT, j : j + BT] = A_ij
            q_i = q_i - q_i @ H[:, :, j // BT]
    for i in range(0, seq_len // BT):
        A[:, :, i * BT : i * BT + BT, i * BT : i * BT + BT] = A_local[:, :, i]
    A = A.masked_fill_(
        ~torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)), 0
    )
    A = A[:, :, :l, :l]
    A = A + g_cumsum[..., None] - g_cumsum[..., None, :]

    for i in range(0, seq_len, BT):
        q_i2 = q2[:, :, i : i + BT].clone()
        for j in range(i - BT, -BT, -BT):
            k_j2 = k2[:, :, j : j + BT]
            A_ij2 = q_i2 @ k_j2.transpose(-1, -2)
            A2[:, :, i : i + BT, j : j + BT] = A_ij2
            q_i2 = q_i2 - q_i2 @ H2[:, :, j // BT]
    for i in range(0, seq_len // BT):
        A2[:, :, i * BT : i * BT + BT, i * BT : i * BT + BT] = A_local2[:, :, i]
    A2 = A2.masked_fill_(
        ~torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)), 0
    )
    A2 = A2[:, :, :l, :l]
    A2 = A2 + g_cumsum[..., None] - g_cumsum[..., None, :]
    ref_o = (A.to(v) * A2.to(v)) @ v
    return ref_o.to(original_dtype)


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("HQ", test_hq_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("use_forget_gate", [False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled",
)
@pytest.mark.skipif(is_intel_alchemist, reason="Intel Triton Failure")
def test_parallel(
    B: int, H: int, HQ: int, T: int, D: int, use_forget_gate: bool, dtype: torch.dtype
):
    if not check_shared_mem("hopper") and D > 128:
        # maybe we can enable this test on Triton 3.3.0
        pytest.skip("Skipping test because global shared memory is not available")
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    require_grad = False

    q = torch.randn((B, HQ, T, D), dtype=dtype, device=device).requires_grad_(
        require_grad
    )
    q2 = torch.randn((B, HQ, T, D), dtype=dtype, device=device).requires_grad_(
        require_grad
    )
    k = torch.nn.functional.normalize(
        torch.randn((B, H, T, D), dtype=dtype, device=device), dim=-1, p=2
    ).requires_grad_(require_grad)
    k2 = torch.nn.functional.normalize(
        torch.randn((B, H, T, D), dtype=dtype, device=device), dim=-1, p=2
    ).requires_grad_(require_grad)
    v = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_(
        require_grad
    )
    beta = (
        torch.rand((B, H, T), dtype=dtype, device=device)
        .sigmoid()
        .requires_grad_(require_grad)
    )

    if use_forget_gate:
        g = (
            torch.empty((B, HQ, T), dtype=torch.float, device=device)
            .uniform_(0.8, 1)
            .log()
            .requires_grad_(require_grad)
        )
    else:
        g = None
    scale = D**-0.5

    beta = torch.ones_like(beta)
    # naive_ref = naive_kernel(
    #     q,
    #     k,
    #     q2,
    #     k2,
    #     v,
    #     beta,
    #     torch.zeros(B, HQ, T, device=device, dtype=torch.float) if g is None else g,
    #     scale,
    #     BT=T,
    # )
    naive_path_attn_ref = naive_kernel_path_attn(
        q,
        k,
        q2,
        k2,
        v,
        beta,
        torch.zeros(B, HQ, T, device=device, dtype=torch.float) if g is None else g,
        scale,
    )
    triton_path_attn_ref = triton_kernel_path_attn(
        q,
        k,
        q2,
        k2,
        v,
        beta,
        torch.zeros(B, HQ, T, device=device, dtype=torch.float) if g is None else g,
        scale,
    )
    # naive_parallel_ref = naive_parallel_kernel(
    #     q,
    #     k,
    #     q2,
    #     k2,
    #     v,
    #     beta,
    #     torch.zeros(B, HQ, T, device=device, dtype=torch.float) if g is None else g,
    #     scale,
    # )
    # naive_chunk_ref = naive_chunk_kernel(
    #     q,
    #     k,
    #     q2,
    #     k2,
    #     v,
    #     beta,
    #     torch.zeros(B, HQ, T, device=device, dtype=torch.float) if g is None else g,
    #     scale,
    # )
    # triton_parallel_ref = triton_parallel_kernel(
    #     q,
    #     k,
    #     q2,
    #     k2,
    #     v,
    #     beta,
    #     torch.zeros(B, HQ, T, device=device, dtype=torch.float) if g is None else g,
    #     scale,
    # )
    # ref.backward(do)
    # ref_dq, q.grad = q.grad.clone(), None
    # ref_dk, k.grad = k.grad.clone(), None
    # ref_dv, v.grad = v.grad.clone(), None
    # if use_forget_gate:
    #     ref_dg, g.grad = g.grad.clone(), None
    # ref_dw, w.grad = w.grad.clone(), None
    # ref_db, beta.grad = beta.grad.clone(), None
    #
    # tri, _ = parallel_kernel_path_attention(q = q, k = k, v = v, w = w, beta = beta, g = g, scale = scale)
    # tri.backward(do)
    # tri_dq, q.grad = q.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dv, v.grad = v.grad.clone(), None
    # if use_forget_gate:
    #     tri_dg, g.grad = g.grad.clone(), None
    # tri_dw, w.grad = w.grad.clone(), None
    # tri_db, beta.grad = beta.grad.clone(), None

    # assert_close(" o", naive_ref, naive_parallel_ref, 0.005)
    # assert_close(" o", naive_ref, naive_chunk_ref, 0.005)
    assert_close(" o", triton_path_attn_ref, naive_path_attn_ref, 0.005)
    # assert_close("dq", ref_dq, tri_dq, 0.005)
    # assert_close("dk", ref_dk, tri_dk, 0.005)
    # assert_close("dv", ref_dv, tri_dv, 0.005)
    # if use_forget_gate:
    #     assert_close("dg", ref_dg, tri_dg, 0.005)
    # assert_close("dw", ref_dw, tri_dw, 0.005)
    # assert_close("db", ref_db, tri_db, 0.005)


@pytest.mark.parametrize(
    "cu_seqlens", [[0, 19, 321, 394, 1111, 2048], [0, 621, 1024, 4222]]
)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("HQ", test_hq_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("use_forget_gate", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled",
)
@pytest.mark.skipif(is_intel_alchemist, reason="Intel Triton Failure")
def test_parallel_varlen(
    cu_seqlens: List[int],
    H: int,
    HQ: int,
    D: int,
    use_forget_gate: bool,
    dtype: torch.dtype,
):
    if not check_shared_mem("hopper") and D > 128:
        # maybe we can enable this test on Triton 3.3.0
        pytest.skip("Skipping test because global shared memory is not available")
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    w = torch.nn.functional.normalize(
        torch.randn((1, T, H, D), dtype=dtype, device=device), dim=-1, p=2
    ).requires_grad_(True)
    beta = (
        torch.rand((1, T, H), dtype=dtype, device=device).sigmoid().requires_grad_(True)
    )
    if use_forget_gate:
        g = (
            torch.empty((1, T, HQ), dtype=torch.float, device=device)
            .uniform_(0.95, 1)
            .log()
            .requires_grad_(True)
        )
    else:
        g = None
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)
    scale = D**-0.5
    ref = torch.zeros(1, T, HQ, D, device=device, dtype=dtype)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        g_segment = (
            torch.zeros(1, eos - bos, HQ, device=device, dtype=torch.float)
            if g is None
            else g[:, bos:eos]
        )
        ref[:, bos:eos] = naive_chunk_kernel(
            q[:, bos:eos],
            k[:, bos:eos],
            v[:, bos:eos],
            w[:, bos:eos],
            beta[:, bos:eos],
            g_segment,
            scale,
        )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    if use_forget_gate:
        ref_dg, g.grad = g.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_db, beta.grad = beta.grad.clone(), None
    tri, _ = triton_kernel_path_attn(
        q=q, k=k, v=v, w=w, beta=beta, g=g, scale=scale, cu_seqlens=cu_seqlens
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    if use_forget_gate:
        tri_dg, g.grad = g.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_db, beta.grad = beta.grad.clone(), None
    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)
    if use_forget_gate:
        assert_close("dg", ref_dg, tri_dg, 0.005)
    assert_close("dw", ref_dw, tri_dw, 0.005)
    assert_close("db", ref_db, tri_db, 0.005)
