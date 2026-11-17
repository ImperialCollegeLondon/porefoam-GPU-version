from __future__ import annotations
import os
import os as _os
from typing import Any, Dict, List
import numpy as np
import scipy.sparse as sp
_SRC = r'''
extern "C" {
__global__ void gpu_csrmv(
    const long long* __restrict__ indptr, const int* __restrict__ indices,
    const double* __restrict__ data, const double* __restrict__ x,
    double* __restrict__ y, const long long nRows)
{
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nRows) return;
    double s = 0.0;
    for (long long k = indptr[r]; k < indptr[r+1]; ++k)
        s += data[k] * x[indices[k]];
    y[r] = s;
}
#define GPU_WPR_CSRMV(NAME, LPR)                                                \
__global__ void NAME(                                                          \
    const long long* __restrict__ indptr,                                      \
    const int* __restrict__ indices,                                     \
    const double* __restrict__ data, const double* __restrict__ x,             \
    double* __restrict__ y, const long long nRows)                             \
{                                                                              \
    const int lane = (int)(threadIdx.x & (unsigned)((LPR) - 1));               \
    const long long r =                                                        \
        ((long long)blockIdx.x * blockDim.x + threadIdx.x) / (LPR);            \
    const bool ok = (r < nRows);                                               \
    double v = 0.0;                                                            \
    if (ok) {                                                                  \
        const long long kend = indptr[r + 1];                                  \
        for (long long k = indptr[r] + lane; k < kend; k += (LPR))             \
            v += data[k] * x[indices[k]];                                      \
    }                                                                          \
    for (int off = (LPR) >> 1; off > 0; off >>= 1)                             \
        v += __shfl_down_sync(0xffffffffu, v, off, (LPR));                     \
    if (ok && lane == 0) y[r] = v;                                             \
}
GPU_WPR_CSRMV(gpu_csrmv_w4,  4)
GPU_WPR_CSRMV(gpu_csrmv_w8,  8)
GPU_WPR_CSRMV(gpu_csrmv_w16, 16)
GPU_WPR_CSRMV(gpu_csrmv_w32, 32)
__global__ void gpu_sub(const double* __restrict__ a, const double* __restrict__ b,
                       double* __restrict__ y, const long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] - b[i];
}
__global__ void gpu_jac_fused(
    const long long* __restrict__ indptr, const int* __restrict__ indices,
    const double* __restrict__ data, const double* __restrict__ idiag,
    const double* __restrict__ rhs, double* __restrict__ x,
    const double omega, const long long n, const int sweeps)
{
    extern __shared__ double xs[];
    const long long i = (long long)threadIdx.x;
    if (i < n) xs[i] = x[i];
    __syncthreads();
    for (int sw = 0; sw < sweeps; ++sw) {
        double t = 0.0;
        if (i < n) {
            for (long long k = indptr[i]; k < indptr[i+1]; ++k)
                t += data[k] * xs[indices[k]];
            t = rhs[i] - t;
            t = omega * idiag[i] * t;
        }
        __syncthreads();
        if (i < n) xs[i] += t;
        __syncthreads();
    }
    if (i < n) x[i] = xs[i];
}

__global__ void gpu_jac(const double* __restrict__ idiag,
                       const double* __restrict__ r, double* __restrict__ x,
                       const double omega, const long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += omega * idiag[i] * r[i];
}
__global__ void gpu_af_to_diag(
    const double* __restrict__ af, const long long* __restrict__ own,
    const long long* __restrict__ nei, double* __restrict__ dgl,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    atomicAdd(&dgl[own[f]], af[f]);
    atomicAdd(&dgl[nei[f]], af[f]);
}

__global__ void gpu_fill_csr_from_af(
    const double* __restrict__ dgl, const double* __restrict__ af,
    const long long* __restrict__ src_kind,
    const long long* __restrict__ src_idx,
    double* __restrict__ data, const long long nnz)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nnz) return;
    data[t] = (src_kind[t] == 0) ? dgl[src_idx[t]] : -af[src_idx[t]];
}

__global__ void gpu_idiag_from_csr(
    const long long* __restrict__ indptr, const int* __restrict__ indices,
    const double* __restrict__ data, double* __restrict__ idiag,
    const long long n)
{
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    double d = 0.0;
    for (long long k = indptr[r]; k < indptr[r+1]; ++k)
        if (indices[k] == r) { d = data[k]; break; }
    idiag[r] = (d != 0.0) ? 1.0 / d : 1.0;
}

__global__ void gpu_add(const double* __restrict__ z, double* __restrict__ x,
                       const double s, const long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += s * z[i];
}
__global__ void gpu_bnd_dirichlet_diag(
    const double* __restrict__ rau, const long long* __restrict__ bown,
    const double* __restrict__ bmag, const double* __restrict__ bdelta,
    const long long* __restrict__ bPt, double* __restrict__ dgl,
    const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    if (bPt[f] != 1) return;
    const long long o = bown[f];
    atomicAdd(&dgl[o], rau[o] * bmag[f] * bdelta[f]);
}
}
'''
_MOD = None
_FN: dict = {}
KERNELS = ('gpu_csrmv', 'gpu_sub', 'gpu_jac', 'gpu_add', 'gpu_af_to_diag', 'gpu_fill_csr_from_af', 'gpu_idiag_from_csr')

def _fn(cp, name):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))
    f = _FN.get(name)
    if f is None:
        f = _MOD.get_function(name)
        _FN[name] = f
    return f

def _grid(n, tpb=256):
    return ((int((n + tpb - 1) // tpb),), (int(tpb),))
_WPR = True

def _wpr_plan(nrows, nnz, tpb=256):
    n = int(nrows)
    per = float(int(nnz)) / max(n, 1)
    if n >= 100000 and per < 8.0:
        return ('gpu_csrmv',) + _grid(n, tpb)
    if n < 4096 or per >= 32.0:
        lpr = 32
    elif per >= 16.0:
        lpr = 16
    elif per >= 8.0:
        lpr = 8
    else:
        lpr = 4
    return ('gpu_csrmv_w%d' % lpr,) + _grid(n * lpr, tpb)
_MVPLAN: Dict[Any, Any] = {}

def _csrmv(cp, args):
    k = (int(args[5]), int(args[2].size))
    p = _MVPLAN.get(k)
    if p is None:
        name, g, b = _wpr_plan(k[0], k[1])
        p = (_fn(cp, name), g, b)
        _MVPLAN[k] = p
    return p[0](p[1], p[2], args)

def ldu_to_csr(n_cells, own, nei, diag, upper):
    n = int(n_cells)
    rows = np.r_[np.arange(n), np.asarray(own), np.asarray(nei)]
    cols = np.r_[np.arange(n), np.asarray(nei), np.asarray(own)]
    vals = np.r_[np.asarray(diag, float), np.asarray(upper, float), np.asarray(upper, float)]
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
_AGG_C_SO = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'amg_native.so')
_agg_c_fn = None
_agg_c_tried = False

def _agg_c_load():
    global _agg_c_fn, _agg_c_tried
    if _agg_c_tried:
        return _agg_c_fn
    _agg_c_tried = True
    try:
        import ctypes
        lib = ctypes.CDLL(_AGG_C_SO)
        f = lib.gpu_aggregate
        f.restype = ctypes.c_int64
        f.argtypes = [np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS'), ctypes.c_int64, ctypes.c_double, np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')]
        _agg_c_fn = f
    except Exception:
        _agg_c_fn = None
    return _agg_c_fn

def aggregate(A, theta: float=0.4):
    n = A.shape[0]
    Ao = (A - sp.diags(A.diagonal())).tocsr()
    Ao.data = np.abs(Ao.data)
    rowmax = np.maximum(np.asarray(Ao.max(axis=1).todense()).ravel(), 1e-300)
    agg = np.full(n, -1, dtype=np.int64)
    indptr, indices, data = (Ao.indptr, Ao.indices, Ao.data)
    k = 0
    _cf = _agg_c_load()
    if _cf is not None:
        import ctypes as _ct
        _cf(np.ascontiguousarray(indptr, dtype=np.int32), np.ascontiguousarray(indices, dtype=np.int32), np.ascontiguousarray(data, dtype=np.float64), np.ascontiguousarray(rowmax, dtype=np.float64), np.int64(n), _ct.c_double(float(theta)), agg)
        return agg
    for i in range(n):
        if agg[i] != -1:
            continue
        lo, hi = (indptr[i], indptr[i + 1])
        nb = indices[lo:hi][data[lo:hi] >= theta * rowmax[i]]
        if nb.size and (agg[nb] != -1).any():
            continue
        agg[i] = k
        agg[nb] = k
        k += 1
    left = np.nonzero(agg == -1)[0]
    for i in left:
        lo, hi = (indptr[i], indptr[i + 1])
        nb = indices[lo:hi]
        if nb.size:
            cand = nb[agg[nb] != -1]
            if cand.size:
                agg[i] = agg[cand[np.argmax(data[lo:hi][agg[nb] != -1])]]
                continue
        agg[i] = k
        k += 1
    return agg

def build(A, *, omega: float=0.67, min_cells: int=40, max_levels: int=24, verbose: bool=False, max_coarsest: int=600) -> List[Dict[str, Any]]:
    levels = [{'A': A.tocsr(), 'P': None}]
    while levels[-1]['A'].shape[0] > min_cells and len(levels) < max_levels:
        Af = levels[-1]['A']
        n = Af.shape[0]
        agg = aggregate(Af)
        nCo = int(agg.max()) + 1
        if nCo >= n:
            break
        Pt = sp.coo_matrix((np.ones(n), (np.arange(n), agg)), shape=(n, nCo)).tocsr()
        d = Af.diagonal().copy()
        d[d == 0.0] = 1.0
        Dinv = sp.diags(1.0 / d)
        P = ((sp.eye(n, format='csr') - omega * (Dinv @ Af)) @ Pt).tocsr()
        levels[-1]['P'] = P
        Ac = (P.T @ Af @ P).tocsr()
        levels.append({'A': Ac, 'P': None})
    ncoarse = levels[-1]['A'].shape[0]
    if ncoarse > max_coarsest:
        raise RuntimeError(f'coarsening stalled at {ncoarse:,} cells (limit {max_coarsest:,})')
    return levels

def spectral_radius(A, iters: int=40) -> float:
    n = A.shape[0]
    d = A.diagonal().copy()
    d[d == 0.0] = 1.0
    Dinv = sp.diags(1.0 / d)
    M = Dinv @ A
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= max(np.linalg.norm(v), 1e-300)
    lam = 1.0
    for _ in range(int(iters)):
        w = M @ v
        nw = np.linalg.norm(w)
        if nw <= 0.0:
            return 1.0
        v = w / nw
        lam = nw
    return float(max(lam, 1e-300))

class AmgWorkspace:

    def __init__(self, cp, levels):
        self.cp = cp
        self.n = len(levels)
        self.L = []
        for L in levels:
            A = L['A'].tocsr()
            d = A.diagonal().copy()
            d[d == 0.0] = 1.0
            e = {'n': A.shape[0], 'ip': cp.asarray(A.indptr.astype(np.int64)), 'ix': cp.asarray(A.indices.astype(np.int32)), 'dv': cp.asarray(A.data.astype(np.float64)), 'idiag': cp.asarray((1.0 / d).astype(np.float64)), 'g': _grid(A.shape[0]), 'omega': 1.0 / spectral_radius(A)}
            if L['P'] is not None:
                P = L['P'].tocsr()
                R = L['P'].T.tocsr()
                e.update({'Pip': cp.asarray(P.indptr.astype(np.int64)), 'Pix': cp.asarray(P.indices.astype(np.int32)), 'Pdv': cp.asarray(P.data.astype(np.float64)), 'Rip': cp.asarray(R.indptr.astype(np.int64)), 'Rix': cp.asarray(R.indices.astype(np.int32)), 'Rdv': cp.asarray(R.data.astype(np.float64)), 'Rn': R.shape[0], 'Rg': _grid(R.shape[0])})
            self.L.append(e)

        def _f(k):
            return cp.zeros(int(k), dtype=cp.float64)
        Ac = levels[-1]['A'].toarray().astype(np.float64)
        Pinv = dense_as_csr(coarse_pinv(Ac))
        self.cpi_ip = cp.asarray(Pinv.indptr.astype(np.int64))
        self.cpi_ix = cp.asarray(Pinv.indices.astype(np.int32))
        self.cpi_dv = cp.asarray(Pinv.data.astype(np.float64))
        self.cpi_n = Pinv.shape[0]
        self.cpi_g = _grid(Pinv.shape[0])
        self.x = [_f(e['n']) for e in self.L]
        self.r = [_f(e['n']) for e in self.L]
        self.t = [_f(e['n']) for e in self.L]
        for k in KERNELS:
            _fn(cp, k)


def _mv(cp, e, x, y):
    _csrmv(cp, (e['ip'], e['ix'], e['dv'], x, y, e['n']))

def _smooth(cp, ws, l, x, rhs, sweeps, omega=None):
    e = ws.L[l]
    g, b = e['g']
    if omega is None:
        omega = e['omega']
    for _ in range(int(sweeps)):
        _mv(cp, e, x, ws.t[l])
        _fn(cp, 'gpu_sub')(g, b, (rhs, ws.t[l], ws.t[l], e['n']))
        _fn(cp, 'gpu_jac')(g, b, (e['idiag'], ws.t[l], x, cp.float64(omega), e['n']))

def vcycle(cp, ws, *, omega=0.67, pre=2, post=2, coarsest=30):
    top = ws.n - 1
    for l in range(top):
        e = ws.L[l]
        g, b = e['g']
        ws.x[l].fill(0.0)
        _smooth(cp, ws, l, ws.x[l], ws.r[l], pre)
        _mv(cp, e, ws.x[l], ws.t[l])
        _fn(cp, 'gpu_sub')(g, b, (ws.r[l], ws.t[l], ws.t[l], e['n']))
        _csrmv(cp, (e['Rip'], e['Rix'], e['Rdv'], ws.t[l], ws.r[l + 1], e['Rn']))
    if getattr(ws, 'coarse_jacobi', 0):
        ws.x[top].fill(0.0)
        _e = ws.L[top]
        _n = int(_e['n'])
        if _n <= 1024:
            _blk = min(1024, max(32, (_n + 31) // 32 * 32))
            _fn(cp, 'gpu_jac_fused')((1,), (_blk,), (_e['ip'], _e['ix'], _e['dv'], _e['idiag'], ws.r[top], ws.x[top], cp.float64(_e['omega']), _e['n'], np.int32(int(ws.coarse_jacobi))), shared_mem=8 * _n)
        else:
            _smooth(cp, ws, top, ws.x[top], ws.r[top], int(ws.coarse_jacobi))
    else:
        cg, cb = ws.cpi_g
        _fn(cp, 'gpu_csrmv')(cg, cb, (ws.cpi_ip, ws.cpi_ix, ws.cpi_dv, ws.r[top], ws.x[top], ws.cpi_n))
    for l in range(top - 1, -1, -1):
        e = ws.L[l]
        g, b = e['g']
        _csrmv(cp, (e['Pip'], e['Pix'], e['Pdv'], ws.x[l + 1], ws.t[l], e['n']))
        _fn(cp, 'gpu_add')(g, b, (ws.t[l], ws.x[l], cp.float64(1.0), e['n']))
        _smooth(cp, ws, l, ws.x[l], ws.r[l], post)
_SOLVE_BUFFERS = {}
_SOLVE_STATS = {}

def solve(cp, ws, *, b, x, cycles, omega=0.67, pre=2, post=2, coarsest=30):
    e = ws.L[0]
    g, bl = e['g']
    nn = int(e['n'])
    key = id(ws)
    buf = _SOLVE_BUFFERS.get(key)
    if buf is None or int(buf[0].size) != nn:
        buf = (cp.ones(nn, dtype=cp.float64), cp.zeros(nn, dtype=cp.float64))
        _SOLVE_BUFFERS[key] = buf
    one, a1 = buf
    _mv(cp, e, one, a1)
    _mv(cp, e, x, ws.t[0])
    axr = cp.mean(x) * a1
    nf = float(cp.sum(cp.abs(ws.t[0] - axr) + cp.abs(b - axr))) + 1e-20
    _fn(cp, 'gpu_sub')(g, bl, (b, ws.t[0], ws.r[0], e['n']))
    r0 = float(cp.sum(cp.abs(ws.r[0]))) / nf
    rk = r0
    it = 0
    while not (it >= 1000 or rk < 1e-07 or False):
        vcycle(cp, ws, omega=omega, pre=pre, post=post, coarsest=coarsest)
        _fn(cp, 'gpu_add')(g, bl, (ws.x[0], x, cp.float64(1.0), e['n']))
        _mv(cp, e, x, ws.t[0])
        _fn(cp, 'gpu_sub')(g, bl, (b, ws.t[0], ws.r[0], e['n']))
        rk = float(cp.sum(cp.abs(ws.r[0]))) / nf
        it += 1
    st = _SOLVE_STATS.setdefault(key, [0, 0, 0, 0])
    st[0] += 1
    st[1] += it
    st[2] = max(st[2], it)
    st[3] += int(it >= 1000)
    return it

def coarse_pinv(Ac):
    import numpy as _np
    A = _np.asarray(Ac, dtype=_np.float64)
    s = float(_np.abs(A).max())
    if not _np.isfinite(s) or s <= 0.0:
        return _np.zeros_like(A)
    return _np.linalg.pinv(A / s, rcond=1e-12) / s

def dense_as_csr(A):
    import numpy as _np
    A = _np.asarray(A, dtype=_np.float64)
    n = A.shape[0]
    indptr = _np.arange(0, n * n + 1, n, dtype=_np.int64)
    indices = _np.tile(_np.arange(n, dtype=_np.int64), n)
    return sp.csr_matrix((A.reshape(-1).copy(), indices, indptr), shape=(n, n))

def csr_source_map(nC, own, nei, A):
    n = int(nC)
    own = np.asarray(own, dtype=np.int64)
    nei = np.asarray(nei, dtype=np.int64)
    A = A.tocsr()
    kind = np.zeros(A.nnz, dtype=np.int64)
    idx = np.zeros(A.nnz, dtype=np.int64)
    face_of = {}
    for f in range(own.size):
        face_of[int(own[f]), int(nei[f])] = f
        face_of[int(nei[f]), int(own[f])] = f
    for r in range(n):
        for t in range(A.indptr[r], A.indptr[r + 1]):
            c = int(A.indices[t])
            if c == r:
                kind[t] = 0
                idx[t] = r
            else:
                kind[t] = 1
                idx[t] = face_of[r, c]
    return (kind, idx)

def attach_device_refresh(cp, ws, levels, maps, nC, own_dev, nei_dev, nIF, kind, idx):
    ws.rf = {'own': own_dev, 'nei': nei_dev, 'nIF': int(nIF), 'nC': int(nC), 'dgl': cp.zeros(int(nC), dtype=cp.float64), 'kind': cp.asarray(np.asarray(kind, dtype=np.int64)), 'idx': cp.asarray(np.asarray(idx, dtype=np.int64)), 'gf': _grid(int(nIF)), 'gc': _grid(int(nC)), 'maps': [], 'bnd': None}
    for M in maps:
        M = M.tocsr()
        ws.rf['maps'].append({'ip': cp.asarray(M.indptr.astype(np.int64)), 'ix': cp.asarray(M.indices.astype(np.int32)), 'dv': cp.asarray(M.data.astype(np.float64)), 'n': M.shape[0], 'g': _grid(M.shape[0])})

def attach_boundary(cp, ws, *, bown, bmag, bdelta, bPt):
    n = int(bown.size)
    g, b = _grid(n)
    ws.rf['bnd'] = {'own': bown, 'mag': bmag, 'delta': bdelta, 'typ': bPt, 'dgl_n': n, 'g': g, 'b': b}

def refresh_device(cp, ws, af, rau_v=None):
    rf = ws.rf
    g, b = rf['gf']
    rf['dgl'].fill(0.0)
    _fn(cp, 'gpu_af_to_diag')(g, b, (af, rf['own'], rf['nei'], rf['dgl'], rf['nIF']))
    bn = rf.get('bnd')
    if bn is not None and rau_v is not None:
        _fn(cp, 'gpu_bnd_dirichlet_diag')(bn['g'], bn['b'], (rau_v, bn['own'], bn['mag'], bn['delta'], bn['typ'], rf['dgl'], bn['dgl_n']))
    e0 = ws.L[0]
    gn, bn = _grid(int(e0['dv'].size))
    _fn(cp, 'gpu_fill_csr_from_af')(gn, bn, (rf['dgl'], af, rf['kind'], rf['idx'], e0['dv'], int(e0['dv'].size)))
    for i, M in enumerate(rf['maps'], start=1):
        _csrmv(cp, (M['ip'], M['ix'], M['dv'], ws.L[i - 1]['dv'], ws.L[i]['dv'], M['n']))
    for e in ws.L:
        eg, eb = e['g']
        _fn(cp, 'gpu_idiag_from_csr')(eg, eb, (e['ip'], e['ix'], e['dv'], e['idiag'], e['n']))
try:
    import os.path as _gpu_op
    import sys as _gpu_sys
    _gpu_d = _gpu_op.dirname(_gpu_op.abspath(__file__))
    if _gpu_d not in _gpu_sys.path:
        _gpu_sys.path.insert(0, _gpu_d)
    import fused_vcycle_levels as _LEVELS
    _LEVELS.install()
except Exception:
    pass
