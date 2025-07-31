# -*- coding: utf-8 -*-

import os
from typing import List

import pytest
import torch
from einops import rearrange

from fla.ops.path_attn.parallel import parallel_path_attention
from fla.utils import COMPILER_MODE, assert_close, check_shared_mem, device, is_intel_alchemist

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [1024]
    test_d_list = [64]
else:
    test_b_list = [2]
    test_t_list = [300, 4095]
    test_d_list = [64]
test_fgate_logit_range_list = [(0.95, 1), (1, 1)]
test_hq_list = [8, 16]
test_h_list = [2]


def naive_path_attn(q, k, v, w, beta, g, scale, BT=64):
    original_dtype = q.dtype
    HQ = q.shape[2]
    H = k.shape[2]
    q, k, v, w, beta, g = map(lambda x: x.to(torch.float32).transpose(1, 2), [q, k, v, w, beta, g])
    g_cumsum = g.cumsum(-1)
    #
    q = q.unsqueeze(2).expand(-1, -1, HQ//HQ, -1, -1).flatten(1, 2)
    k = k.unsqueeze(2).expand(-1, -1, HQ//H, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, HQ//H, -1, -1).flatten(1, 2)
    w = w.unsqueeze(2).expand(-1, -1, HQ//H, -1, -1).flatten(1, 2)
    beta = beta.unsqueeze(2).expand(-1, -1, HQ//H, -1).flatten(1, 2)
    g_cumsum = g_cumsum.unsqueeze(2).expand(-1, -1, HQ//HQ, -1).flatten(1, 2)
    b, h, l, d_k = q.shape
    if l % BT != 0:
        padding_size = BT - l % BT
        q, k, w = map(lambda x: torch.nn.functional.pad(x, (0, 0, 0, padding_size)), [q, k, w])
        beta = torch.nn.functional.pad(beta, (0, padding_size))
    seq_len = q.shape[2]
    w_beta = w * beta[..., None]
    q, k, w, w_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT), [q, k, w, w_beta])
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(w_beta @ w.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BT):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BT, dtype=q.dtype, device=q.device)
    Twbk = T @ (w_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    qw = (q @ w.transpose(-1, -2)).tril()
    Twb = T @ w_beta
    A_local = (q @ k.transpose(-1, -2)).tril() - qw @ Twbk
    q = q - qw @ Twb
    k = k - Twbk.transpose(-1, -2) @ w
    H = w.transpose(-1, -2) @ Twb
    A = torch.zeros(b, h, seq_len, seq_len, device=q.device)
    q, k, w, w_beta = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, w, w_beta])
    for i in range(0, seq_len, BT):
        q_i = q[:, :, i:i+BT].clone()
        for j in range(i - BT, -BT, -BT):
            k_j = k[:, :, j:j+BT]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BT, j:j+BT] = A_ij
            q_i = q_i - q_i @ H[:, :, j // BT]
    for i in range(0, seq_len//BT):
        A[:, :, i*BT:i*BT+BT, i*BT:i*BT+BT] = A_local[:, :, i]
    A = A.masked_fill_(~torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)), float("-inf"))
    A = A[:, :, :l, :l]
    A = A + g_cumsum[..., None] - g_cumsum[..., None, :]
    ref_o = (A * scale).softmax(-1).to(v) @ v
    return ref_o.to(original_dtype).transpose(1, 2)


@pytest.mark.parametrize("B", test_b_list)
@pytest.mark.parametrize("T", test_t_list)
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("HQ", test_hq_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("use_forget_gate", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Intel Triton Failure"
)
def test_parallel(
    B: int,
    H: int,
    HQ: int,
    T: int,
    D: int,
    use_forget_gate: bool,
    dtype: torch.dtype
):
    if not check_shared_mem('hopper') and D > 128:
        # maybe we can enable this test on Triton 3.3.0
        pytest.skip("Skipping test because global shared memory is not available")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    w = torch.nn.functional.normalize(torch.randn((B, T, H, D), dtype=dtype, device=device), dim=-1, p=2).requires_grad_(True)
    beta = torch.rand((B, T, H), dtype=dtype, device=device).sigmoid().requires_grad_(True)
    if use_forget_gate:
        g = torch.empty((B, T, HQ), dtype=torch.float, device=device).uniform_(
            0.95, 1).log().requires_grad_(True)
    else:
        g = None
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    scale = D ** -0.5
    ref = naive_path_attn(q, k, v, w, beta, torch.zeros(B, T, HQ, device=device, dtype=torch.float) if g is None else g, scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    if use_forget_gate:
        ref_dg, g.grad = g.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_db, beta.grad = beta.grad.clone(), None

    tri, _ = parallel_path_attention(q=q, k=k, v=v, w=w, beta=beta, g=g, scale=scale)
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


@pytest.mark.parametrize("cu_seqlens", [[0, 19, 321, 394, 1111, 2048], [0, 621, 1024, 4222]])
@pytest.mark.parametrize("H", test_h_list)
@pytest.mark.parametrize("HQ", test_hq_list)
@pytest.mark.parametrize("D", test_d_list)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("use_forget_gate", [True, False])
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "0",
    reason="Skipping test because TEST_CHUNK_VARLEN is enabled"
)
@pytest.mark.skipif(
    is_intel_alchemist,
    reason="Intel Triton Failure"
)
def test_parallel_varlen(
    cu_seqlens: List[int],
    H: int,
    HQ: int,
    D: int,
    use_forget_gate: bool,
    dtype: torch.dtype
):
    if not check_shared_mem('hopper') and D > 128:
        # maybe we can enable this test on Triton 3.3.0
        pytest.skip("Skipping test because global shared memory is not available")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    w = torch.nn.functional.normalize(torch.randn((1, T, H, D), dtype=dtype, device=device), dim=-1, p=2).requires_grad_(True)
    beta = torch.rand((1, T, H), dtype=dtype, device=device).sigmoid().requires_grad_(True)
    if use_forget_gate:
        g = torch.empty((1, T, HQ), dtype=torch.float, device=device).uniform_(
            0.95, 1).log().requires_grad_(True)
    else:
        g = None
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)
    scale = D ** -0.5
    ref = torch.zeros(1, T, HQ, D, device=device, dtype=dtype)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        g_segment = torch.zeros(1, eos - bos, HQ, device=device, dtype=torch.float) if g is None else g[:, bos:eos]
        ref[:, bos:eos] = naive_path_attn(
            q[:, bos:eos], k[:, bos:eos], v[:, bos:eos],
            w[:, bos:eos], beta[:, bos:eos], g_segment, scale
        )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    if use_forget_gate:
        ref_dg, g.grad = g.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_db, beta.grad = beta.grad.clone(), None
    tri, _ = parallel_path_attention(q=q, k=k, v=v, w=w, beta=beta, g=g, scale=scale, cu_seqlens=cu_seqlens)
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
