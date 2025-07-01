import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
    'USE_GATE': lambda args: args['g_cumsum'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=num_stages, num_warps=4)
        for num_stages in [2, 3, 4, 5, 6]
    ],
    key=['BK', 'BS', 'BT', 'USE_GATE', 'IS_VARLEN']
)
@triton.jit(do_not_specialize=['T'])
def parallel_path_fwd_kernel(
        q,q2, k,k2, v, o, o_new, g_cumsum, h,h2, scale,
        offsets, indices, chunk_offsets,
        T,
        G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
        K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
        BV: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        USE_GATE: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
        boh = i_n * tl.cdiv(T, BS)


    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_q2 = tl.make_block_ptr(q2 + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q2 = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))
    b_q2 += tl.load(p_q2, boundary_check=(0, 1))
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))


    if USE_GATE:
        p_g_cumsum_q = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0,))
        b_g_cumsum_q = tl.load(p_g_cumsum_q, boundary_check=(0,))
    else:
        b_g_cumsum_q = None

    for offset in range((i_t + 1) * BT - 2 * BS, i_t*BT-BS, -BS):
        i_tk = tl.cdiv(offset, BS)
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k2 + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (V*H, 1), (offset, 0), (BS, BV), (1, 0))
        p_h = tl.make_block_ptr(h + ((boh+i_tk) * H + i_h) * K*K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        p_h2 = tl.make_block_ptr(h2 + ((boh+i_tk) * H + i_h) * K*K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_h2 = tl.load(p_h2, boundary_check=(0, 1))

        m_s = i_t * BT + tl.arange(0, BT) >= (offset + BS)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        b_s2 = tl.dot(b_q2.to(b_k2.dtype), b_k2)
        b_q_minus = tl.dot(b_q.to(b_h.dtype), b_h)
        b_q_minus2 = tl.dot(b_q2.to(b_h2.dtype), b_h2)
        b_q = tl.where(m_s[:, None], b_q - b_q_minus, b_q)
        b_q2 = tl.where(m_s[:, None], b_q2 - b_q_minus2, b_q2)

        if USE_GATE:
            p_g_cumsum_k = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq), (T, ), (HQ, ), (offset, ), (BS, ), (0,))
            b_g_cumsum_k = tl.load(p_g_cumsum_k, boundary_check=(0,))
            b_s = b_s + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]

        b_s = tl.where(m_s[:, None], b_s * b_s2, 0)

        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    tl.debug_barrier()

    # 类间更新
    for offset in range(i_t * BT - BS, -BS, -BS):
        i_tk = tl.cdiv(offset, BS)
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k2 + (bos * H + i_h) * K, (K, T), (1, K*H), (0, offset), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (V*H, 1), (offset, 0), (BS, BV), (1, 0))
        p_h = tl.make_block_ptr(h + ((boh+i_tk) * H + i_h) * K*K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
        p_h2 = tl.make_block_ptr(h2 + ((boh+i_tk) * H + i_h) * K*K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_h2 = tl.load(p_h2, boundary_check=(0, 1))

        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        b_s2 = tl.dot(b_q2.to(b_k.dtype), b_k2)
        b_q -= tl.dot(b_q.to(b_h.dtype), b_h)
        b_q2 -= tl.dot(b_q2.to(b_h.dtype), b_h2)

        if USE_GATE:
            p_g_cumsum_k = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq), (T, ), (HQ, ), (offset, ), (BS, ), (0,))
            b_g_cumsum_k = tl.load(p_g_cumsum_k, boundary_check=(0,))
            b_s = b_s + b_g_cumsum_q[:, None] - b_g_cumsum_k[None, :]

        b_s_2nd = b_s * b_s2

        b_o += tl.dot(b_s_2nd.to(b_v.dtype), b_v)

    p_o_new = tl.make_block_ptr(o_new + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t*BT, 0), (BT, BV), (1, 0))
    tl.store(p_o_new, b_o.to(p_o_new.dtype.element_ty), boundary_check=(0, 1))



def parallel_path_fwd_fn(
    q,q2, k,k2, v, o, g_cumsum, h,h2, scale,
    cu_seqlens, BT, BS,
):
    B, T, HQ, K = q.shape
    V = v.shape[-1]
    H = k.shape[-2]
    G = HQ // H
    indices_BT = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices_BT)
    grid = (NT, B * HQ)
    o_new = torch.empty_like(o)

    parallel_path_fwd_kernel[grid](
        q=q,
        q2=q2,
        k=k,
        k2=k2,
        v=v,
        o=o,
        o_new=o_new,
        h=h,
        h2=h2,
        g_cumsum=g_cumsum,
        scale=scale,
        chunk_offsets=chunk_offsets,
        offsets=cu_seqlens,
        indices=indices_BT,
        T=T,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        G=G,
        HQ=HQ,
        H=H,
        BS=BS,
        BT=BT,
    )
    return o_new
