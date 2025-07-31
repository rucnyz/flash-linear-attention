import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    'IS_VARLEN': lambda args: args['offsets'] is not None,
    'USE_GATE': lambda args: args['g_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_path_bwd_intra_chunk_kernel(
    q, k, v, g_cumsum,
    h, L, D,
    dq, dq_new, dk, dv, dh, do, dg_cumsum,
    offsets, indices, chunk_offsets,
    T, scale,
    G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,  BV: tl.constexpr,
    BT: tl.constexpr, S: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_GATE: tl.constexpr,
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
        boh = i_n * tl.cdiv(T, BT)

    # offset calculations
    k += (bos * H + i_h) * K  # GQA when H!=HQ
    v += (bos * H + i_h) * V  # GQA when H!=HQ
    h += (boh * H + i_h) * K * K

    q += (bos * HQ + i_hq) * K
    dq += (bos * HQ + i_hq) * K
    dq_new += (bos * HQ + i_hq) * K
    dk += (bos * HQ + i_hq) * K
    dv += (bos * HQ + i_hq) * V
    do += (bos * HQ + i_hq) * V
    dh += (boh * HQ + i_hq) * K * K
    L += (bos * HQ + i_hq)
    D += (bos * HQ + i_hq)
    if USE_GATE:
        g_cumsum += (bos * HQ + i_hq)
        dg_cumsum += (bos * HQ + i_hq)

    # constants
    sm_scale = scale * 1.44269504

    p_do = tl.make_block_ptr(do, (T, V), (HQ*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))

    p_delta = tl.make_block_ptr(D, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    b_delta = tl.load(p_delta, boundary_check=(0, ))
    p_l = tl.make_block_ptr(L, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
    b_l = tl.load(p_l, boundary_check=(0, ))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    p_dq = tl.make_block_ptr(dq, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    p_q = tl.make_block_ptr(q, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))

    if USE_GATE:
        p_gq_cumsum = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (i_t * BT, ), (BT, ), (0, ))
        b_gq_cumsum = tl.load(p_gq_cumsum, boundary_check=(0, ))
        b_dgq = tl.zeros([BT, ], dtype=tl.float32)
    else:
        b_dgq = None

    curr_start = (tl.floor(i_t * BT / S).to(tl.int32) * S).to(tl.int32)

    for offset in range(curr_start, i_t * BT, BT):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (offset, 0), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q_tmp = tl.zeros([BT, BK], dtype=tl.float32)
        b_q_tmp += b_q

        for i_t_small in range(i_t * BT - BT, offset, -BT):
            p_h = tl.make_block_ptr(h + tl.cdiv(i_t_small, BT) * H*K*K, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_q_tmp -= tl.dot(b_q_tmp.to(b_h.dtype), b_h)

        b_q2 = b_q_tmp.to(b_k.dtype)
        b_dh = -tl.dot(tl.trans(b_q2), b_dq.to(b_q2.dtype))
        tl.atomic_add(dh + tl.cdiv(offset, BT) * HQ*K*K + tl.arange(0, K)
                      [:, None] * K + tl.arange(0, K)[None, :], b_dh, sem='relaxed')

        b_A = tl.dot(b_q2, tl.trans(b_k))
        if USE_GATE:
            p_gk_cumsum = tl.make_block_ptr(g_cumsum, (T, ), (HQ, ), (offset, ), (BT, ), (0, ))
            b_gk_cumsum = tl.load(p_gk_cumsum, boundary_check=(0, ))
            b_A = b_A + b_gq_cumsum[:, None] - b_gk_cumsum[None, :]
            b_A = tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], b_A, float("-inf"))  # avoid nan

        b_A_softmax = tl.math.exp2(b_A * sm_scale - b_l[:, None])
        b_dv = tl.dot(tl.trans(b_A_softmax.to(b_do.dtype)), b_do)

        tl.atomic_add(dv + ((offset + tl.arange(0, BT)) * HQ * V)
                      [:, None] + tl.arange(0, BV)[None, :], b_dv.to(dv.dtype.element_ty), sem='relaxed')

        p_v = tl.make_block_ptr(v, (T, V), (V*H, 1), (offset, 0), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dp = tl.dot(b_do, tl.trans(b_v))
        b_dA = ((b_dp - b_delta[:, None]) * b_A_softmax * scale)
        if USE_GATE:
            b_dgk = -tl.sum(b_dA, axis=0)
            tl.atomic_add(dg_cumsum + (offset + tl.arange(0, BT)) * HQ, b_dgk, sem='relaxed')
            b_dgq += tl.sum(b_dA, axis=1)

        b_dA = b_dA.to(b_k.dtype)
        p_h = tl.make_block_ptr(h + tl.cdiv(offset, BT) * H*K*K, (K, K), (1, K), (0, 0), (BK, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dk = tl.dot(tl.trans(b_dA), b_q2)
        tl.atomic_add(dk + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dk, sem='relaxed')
        b_dq -= tl.dot(b_dq.to(b_h.dtype), b_h)
        b_dq += tl.dot(b_dA, b_k)

    p_dq_new = tl.make_block_ptr(dq_new, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_dq_new, b_dq.to(dq_new.dtype.element_ty), boundary_check=(0, 1))
    if USE_GATE:
        tl.atomic_add(dg_cumsum + (i_t * BT + tl.arange(0, BT)) * HQ, b_dgq, sem='relaxed')


def parallel_path_bwd_intra_chunk_fn(
    q, k, v, g_cumsum, h,
    dq, dk, dv, dg_cumsum, dh, do,
    scale, L, D,
    cu_seqlens,
    S, BT
):
    assert dk.dtype == dv.dtype == dh.dtype == torch.float32, 'atomic_add requires float32'
    B, T, HQ, K = q.shape
    assert dk.shape == dq.shape

    V = v.shape[-1]
    H = k.shape[-2]
    G = HQ // H
    indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(indices)
    dq_new = torch.empty_like(dq, dtype=q.dtype)
    parallel_path_bwd_intra_chunk_kernel[(NT, B*HQ)](
        q=q, k=k, v=v, g_cumsum=g_cumsum,
        h=h, L=L, D=D,
        dq=dq, dq_new=dq_new, dk=dk, dv=dv, dh=dh, do=do, dg_cumsum=dg_cumsum,
        offsets=cu_seqlens, indices=indices, chunk_offsets=chunk_offsets,
        T=T, S=S, BT=BT, scale=scale,
        G=G, HQ=HQ, H=H, K=K, V=V,
        BK=triton.next_power_of_2(K), BV=triton.next_power_of_2(V),
    )
    return dq_new, dk, dv, dh, dg_cumsum
