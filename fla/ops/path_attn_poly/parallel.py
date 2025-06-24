# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import reduce

from fla.ops.attn.parallel import parallel_attn_bwd_preprocess
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.path_attn_poly.cumprod_householder_bwd import chunk_cumprod_householder_bwd_fn
from fla.ops.path_attn_poly.cumprod_householder_fwd import chunk_cumprod_householder_fwd_fn
from fla.ops.path_attn_poly.intra_chunk_preprocess_bwd import intra_chunk_preprocess_bwd_fn
from fla.ops.path_attn_poly.intra_chunk_preprocess_bwd_prepare import intra_chunk_preprocess_bwd_prepare_fn
from fla.ops.path_attn_poly.parallel_path_bwd_inter_dkv import parallel_path_bwd_dkv_fn
from fla.ops.path_attn_poly.parallel_path_bwd_inter_dqh import parallel_path_bwd_dq_fn
from fla.ops.path_attn_poly.parallel_path_bwd_intra import parallel_path_bwd_intra_chunk_fn
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.solve_tril import solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_G': lambda args: args['g_cumsum'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_qk_dot_fwd_kernel(
        q,
        q2,
        k,
        k2,
        A,
        A2,
        out,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = tl.arange(0, BT)

    # p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # b_beta = tl.load(p_beta, boundary_check=(0,))




def chunk_qk_matmul_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        q2: torch.Tensor,
        k2: torch.Tensor,
        A: torch.Tensor,
        A2: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        output_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    r"""
    Compute (Q @ K @ A) * (Q2 @ K2 @ A2)

    Args:
        q (torch.Tensor):
            The query tensor of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        q2 (torch.Tensor):
            The query tensor of shape `[B, T, HQ, K]`.
        k2 (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        A (torch.Tensor):
            The precomputed attention matrix of shape `[B, T, H, BT]`.
        A2 (torch.Tensor):
            The precomputed attention matrix of shape `[B, T, H, BT]`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    out = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    chunk_qk_dot_fwd_kernel[(NT, B * H)](
        q=q,
        q2=q2,
        k=k,
        k2=k2,
        A=A,
        A2=A2,
        out=out,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    return out

class ParallelKernelPATHAttentionFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, q2, k2, v, beta, g, scale, cu_seqlens, use_cache=False):
        g_cumsum = (
            chunk_global_cumsum(g, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
            if g is not None
            else None
        )
        BS = 64
        A = chunk_scaled_dot_kkt_fwd(
            k=k.transpose(2, 1),
            beta=torch.ones_like(beta).transpose(2, 1),
            cu_seqlens=cu_seqlens,
            chunk_size=BS,
            output_dtype=torch.float32,
        )
        # triangular value of A
        A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)

        A2 = chunk_scaled_dot_kkt_fwd(
            k=k2.transpose(2, 1),
            beta=torch.ones_like(beta).transpose(2, 1),
            cu_seqlens=cu_seqlens,
            chunk_size=BS,
            output_dtype=torch.float32,
        )
        # triangular value of A2
        A2 = solve_tril(A=A2, cu_seqlens=cu_seqlens, output_dtype=k2.dtype)
        A3 = chunk_qk_matmul_fwd(
            q=q.transpose(2, 1),
            k=k.transpose(2, 1),
            q2=q2.transpose(2, 1),
            k2=k2.transpose(2, 1),
            A=A,
            A2=A2,
            cu_seqlens=cu_seqlens,
            chunk_size=BS,
            output_dtype=torch.float32,
        )

        # q_new, k_new, h, o, L, M = intra_chunk_preprocess_fwd_fn(
        #     q=q,
        #     k=k,
        #     v=v,
        #     w=w,
        #     beta=beta,
        #     g_cumsum=g_cumsum,
        #     A=A,
        #     scale=scale,
        #     BT=BS,
        #     cu_seqlens=cu_seqlens,
        # )
        # o, L = parallel_path_fwd_fn(
        #     q=q_new,
        #     k=k_new,
        #     v=v,
        #     L=L,
        #     h=h,
        #     M=M,
        #     o=o,
        #     g_cumsum=g_cumsum,
        #     scale=scale,
        #     cu_seqlens=cu_seqlens,
        #     BT=BT,
        #     BS=BS,
        # )
        # k_cache = prepare_k_cache_fn(
        #     k=k_new, h=h, cu_seqlens=cu_seqlens, BS=BS, use_cache=use_cache
        # )
        # ctx.save_for_backward(q, k, q2, k2, v, g_cumsum, o, beta, L)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return None

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dk_new):
        q, k, v, w, g_cumsum, o, beta, L = ctx.saved_tensors
        BT = 64
        BS = 64
        S = 512
        cu_seqlens = ctx.cu_seqlens
        A, _ = chunk_scaled_dot_kkt_fwd(
            k=w,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_size=BS,
            output_dtype=torch.float32,
        )
        A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
        delta = parallel_attn_bwd_preprocess(o, do)
        q_new, k_new, h, dA_local, dv, dg_cumsum = (
            intra_chunk_preprocess_bwd_prepare_fn(
                q=q,
                k=k,
                v=v,
                w=w,
                beta=beta,
                g_cumsum=g_cumsum,
                A=A,
                L=L,
                D=delta,
                do=do,
                scale=ctx.scale,
                cu_seqlens=cu_seqlens,
            )
        )
        q_new_large, k_new_large, hc_suffix, hc_prefix, hc_whole = (
            chunk_cumprod_householder_fwd_fn(
                q=q_new, k=k_new, h=h, S=S, BT=BS, cu_seqlens=cu_seqlens
            )
        )
        dq, dhc_whole, dg_cumsum = parallel_path_bwd_dq_fn(
            q=q_new_large,
            k=k_new_large,
            v=v,
            g_cumsum=g_cumsum,
            do=do,
            dg_cumsum=dg_cumsum,
            hc_whole=hc_whole,
            scale=ctx.scale,
            L=L,
            D=delta,
            cu_seqlens=cu_seqlens,
            S=S,
            BT=BT,
            BS=BS,
        )
        dk, dv, dg_cumsum3 = parallel_path_bwd_dkv_fn(
            q=q_new_large,
            k=k_new_large,
            v=v,
            g_cumsum=g_cumsum,
            do=do,
            dv=dv,
            dg_cumsum=dg_cumsum,
            hc_whole=hc_whole,
            scale=ctx.scale,
            L=L,
            D=delta,
            cu_seqlens=cu_seqlens,
            S=S,
            BT=BT,
            BS=BS,
        )
        dh, dk = chunk_cumprod_householder_bwd_fn(
            h=h,
            hc_suffix=hc_suffix,
            k=k_new,
            dk=dk,
            dhc_whole=dhc_whole,
            cu_seqlens=cu_seqlens,
            S=S,
            BT=BS,
        )
        dq, dk_new, dv, dh, dg_cumsum = parallel_path_bwd_intra_chunk_fn(
            q=q_new,
            k=k_new,
            v=v,
            g_cumsum=g_cumsum,
            h=h,
            L=L,
            D=delta,
            scale=ctx.scale,
            dq=dq,
            dk=dk,
            dv=dv,
            dh=dh,
            do=do,
            dg_cumsum=dg_cumsum,
            cu_seqlens=cu_seqlens,
            S=S,
            BT=BT,
        )
        dq, dk, dbeta, dw = intra_chunk_preprocess_bwd_fn(
            q=q,
            k=k,
            w=w,
            beta=beta,
            dq=dq,
            dk=dk,
            dh=dh,
            dA_local=dA_local,
            A=A,
            L=L,
            D=delta,
            do=do,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
        )
        G = q.shape[-2] // k.shape[-2]
        if G > 1:
            assert dk.dtype == dv.dtype == dw.dtype == dbeta.dtype == torch.float32, (
                "reduction requires float32"
            )
            dk = reduce(dk, "b t (h g) k -> b t h k", g=G, reduction="sum")
            dv = reduce(dv, "b t (h g) k -> b t h k", g=G, reduction="sum")
            dw = reduce(dw, "b t (h g) k -> b t h k", g=G, reduction="sum")
            dbeta = reduce(dbeta, "b t (h g) -> b t h", g=G, reduction="sum")
        if dg_cumsum is not None:
            dg_cumsum = chunk_global_cumsum(
                dg_cumsum, cu_seqlens=cu_seqlens, reverse=True
            )
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dw.to(w.dtype),
            dbeta.to(beta.dtype),
            dg_cumsum.to(g_cumsum.dtype) if g_cumsum is not None else None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def parallel_kernel_path_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    q2: torch.Tensor,
    k2: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, T, H, V]`
        beta (torch.Tensor):
            beta of shape `[B, T, H]`
        g (torch.Tensor):
            g of shape `[B, T, HQ]`
        scale (float):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        use_cache (bool):
            Whether to transform and cache the key values for decoding. Default: `False`.
    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, V]`
        k_cache (torch.Tensor):
            k_cache of shape `[B, T, H, K]`
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    assert q.shape[-1] in [16, 32, 64], (
        "only support head_dim in [16, 32, 64] for now. Stay tuned!"
    )
    assert v.shape[-1] in [16, 32, 64], (
        "only support head_dim in [16, 32, 64] for now. Stay tuned!"
    )
    assert q.shape[-1] == k.shape[-1], "q, k should have the same head_dim."
    assert beta.shape[:3] == k.shape[:3], (
        "beta should have the same number of heads as k"
    )
    if g is not None:
        assert g.shape[:3] == q.shape[:3], "g should have the same number of heads as q"
    assert q.shape[-2] % k.shape[-2] == 0, (
        "the number of query heads should be divisible by the number of key heads"
    )
    o, k_cache = ParallelKernelPATHAttentionFunction.apply(
        q, k, q2, k2, v, beta, g, scale, cu_seqlens, use_cache
    )
    return o, k_cache
