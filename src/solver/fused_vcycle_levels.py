from __future__ import annotations
import os
import re
import sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import amg_solver as AMG
_OPTS = ('--fmad=false', '-std=c++14')
_SRC = r'''
extern "C" {

__global__ void fused_jac_sweep(
    const long long* __restrict__ ip, const int* __restrict__ ix,
    const double* __restrict__ dv, const double* __restrict__ idiag,
    const double* __restrict__ b, const double* __restrict__ xo,
    double* __restrict__ xn, const double omega, const long long n)
{
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    double s = 0.0;
    for (long long k = ip[r]; k < ip[r+1]; ++k)
        s += dv[k] * xo[ix[k]];
    const double t = b[r] - s;
    xn[r] = xo[r] + omega * idiag[r] * t;
}

#define FUSED_WSWEEP(NAME, LPR)                                                 \
__global__ void NAME(                                                          \
    const long long* __restrict__ ip, const int* __restrict__ ix,        \
    const double* __restrict__ dv, const double* __restrict__ idiag,           \
    const double* __restrict__ b, const double* __restrict__ xo,               \
    double* __restrict__ xn, const double omega, const long long n)            \
{                                                                              \
    const int lane = (int)(threadIdx.x & (unsigned)((LPR) - 1));               \
    const long long r =                                                        \
        ((long long)blockIdx.x * blockDim.x + threadIdx.x) / (LPR);            \
    const bool ok = (r < n);                                                   \
    double v = 0.0;                                                            \
    if (ok) {                                                                  \
        const long long kend = ip[r + 1];                                      \
        for (long long k = ip[r] + lane; k < kend; k += (LPR))                 \
            v += dv[k] * xo[ix[k]];                                            \
    }                                                                          \
    for (int off = (LPR) >> 1; off > 0; off >>= 1)                             \
        v += __shfl_down_sync(0xffffffffu, v, off, (LPR));                     \
    if (ok && lane == 0) {                                                     \
        const double t = b[r] - v;                                             \
        xn[r] = xo[r] + omega * idiag[r] * t;                                  \
    }                                                                          \
}
FUSED_WSWEEP(fused_jac_sweep_w4,  4)
FUSED_WSWEEP(fused_jac_sweep_w8,  8)
FUSED_WSWEEP(fused_jac_sweep_w16, 16)
FUSED_WSWEEP(fused_jac_sweep_w32, 32)

__global__ void fused_prolong_add(
    const long long* __restrict__ ip, const int* __restrict__ ix,
    const double* __restrict__ dv, const double* __restrict__ xc,
    double* __restrict__ x, const long long n)
{
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    double s = 0.0;
    for (long long k = ip[r]; k < ip[r+1]; ++k)
        s += dv[k] * xc[ix[k]];
    x[r] += 1.0 * s;
}

#define FUSED_WPA(NAME, LPR)                                                    \
__global__ void NAME(                                                          \
    const long long* __restrict__ ip, const int* __restrict__ ix,        \
    const double* __restrict__ dv, const double* __restrict__ xc,              \
    double* __restrict__ x, const long long n)                                 \
{                                                                              \
    const int lane = (int)(threadIdx.x & (unsigned)((LPR) - 1));               \
    const long long r =                                                        \
        ((long long)blockIdx.x * blockDim.x + threadIdx.x) / (LPR);            \
    const bool ok = (r < n);                                                   \
    double v = 0.0;                                                            \
    if (ok) {                                                                  \
        const long long kend = ip[r + 1];                                      \
        for (long long k = ip[r] + lane; k < kend; k += (LPR))                 \
            v += dv[k] * xc[ix[k]];                                            \
    }                                                                          \
    for (int off = (LPR) >> 1; off > 0; off >>= 1)                             \
        v += __shfl_down_sync(0xffffffffu, v, off, (LPR));                     \
    if (ok && lane == 0) x[r] += 1.0 * v;                                      \
}
FUSED_WPA(fused_prolong_add_w4,  4)
FUSED_WPA(fused_prolong_add_w8,  8)
FUSED_WPA(fused_prolong_add_w16, 16)
FUSED_WPA(fused_prolong_add_w32, 32)
}
'''
_MOD = None
_K: dict = {}
SWEEP = True
PA = True

def fn(cp, name):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=_OPTS)
    f = _K.get(name)
    if f is None:
        f = _MOD.get_function(name)
        _K[name] = f
    return f
NAMES = tuple(re.findall('__global__\\s+void\\s+([A-Za-z_0-9]+)\\s*\\(', _SRC)) + ('fused_jac_sweep_w4', 'fused_jac_sweep_w8', 'fused_jac_sweep_w16', 'fused_jac_sweep_w32', 'fused_prolong_add_w4', 'fused_prolong_add_w8', 'fused_prolong_add_w16', 'fused_prolong_add_w32')
_PLAN: dict = {}

def plan(cp, n, nnz, base):
    key = (int(n), int(nnz), base, True)
    p = _PLAN.get(key)
    if p is None:
        name, g, b = AMG._wpr_plan(int(n), int(nnz))
        if name == 'gpu_csrmv':
            nm = base
        else:
            nm = base + name[8:]
        if nm not in NAMES:
            raise AssertionError('fused: no kernel %r (from %r)' % (nm, name))
        p = (fn(cp, nm), g, b, nm)
        _PLAN[key] = p
    return p

def _scratch(cp, ws, l, n):
    return ws.t[l]
    d = getattr(ws, '_fused_scratch', None)
    if d is None:
        d = {}
        ws._fused_scratch = d
    s = d.get(l)
    if s is None:
        s = cp.zeros(int(n), dtype=cp.float64)
        d[l] = s
    return s

def smooth_fused(cp, ws, l, x, rhs, sweeps, omega=None):
    e = ws.L[l]
    if omega is None:
        omega = e['omega']
    ns = int(sweeps)
    if ns <= 0:
        return
    n = int(e['n'])
    scr = _scratch(cp, ws, l, n)
    if rhs is scr or x is scr:
        raise AssertionError("fused: the fused sweep's scratch aliases %s at level %d" % ('rhs' if rhs is scr else 'x', l))
    k, g, b, _nm = plan(cp, n, int(e['dv'].size), 'fused_jac_sweep')
    om = cp.float64(omega)
    ip, ix, dv, idg, nn = (e['ip'], e['ix'], e['dv'], e['idiag'], e['n'])
    src, dst = (x, scr)
    for _ in range(ns):
        k(g, b, (ip, ix, dv, idg, rhs, src, dst, om, nn))
        src, dst = (dst, src)
    if src is not x:
        cp.copyto(x, src)

def prolong_add(cp, ws, l):
    e = ws.L[l]
    k, g, b, _nm = plan(cp, int(e['n']), int(e['Pdv'].size), 'fused_prolong_add')
    k(g, b, (e['Pip'], e['Pix'], e['Pdv'], ws.x[l + 1], ws.x[l], e['n']))
_UNFUSED_VCYCLE = AMG.vcycle
_UNFUSED_SMOOTH = AMG._smooth
_FB = None

def _fb():
    global _FB
    if _FB is None:
        import fused_vcycle as FUSED
        _FB = FUSED
    return _FB

def _smooth(cp, ws, l, x, rhs, sweeps, omega=None):
    return smooth_fused(cp, ws, l, x, rhs, sweeps, omega)
    return _UNFUSED_SMOOTH(cp, ws, l, x, rhs, sweeps, omega)

def _up(cp, ws, l):
    e = ws.L[l]
    prolong_add(cp, ws, l)
    return
    g, bl = e['g']
    AMG._csrmv(cp, (e['Pip'], e['Pix'], e['Pdv'], ws.x[l + 1], ws.t[l], e['n']))
    AMG._fn(cp, 'gpu_add')(g, bl, (ws.t[l], ws.x[l], cp.float64(1.0), e['n']))

def _down(cp, ws, l, pre):
    e = ws.L[l]
    g, bl = e['g']
    ws.x[l].fill(0.0)
    _smooth(cp, ws, l, ws.x[l], ws.r[l], pre)
    AMG._mv(cp, e, ws.x[l], ws.t[l])
    AMG._fn(cp, 'gpu_sub')(g, bl, (ws.r[l], ws.t[l], ws.t[l], e['n']))
    AMG._csrmv(cp, (e['Rip'], e['Rix'], e['Rdv'], ws.t[l], ws.r[l + 1], e['Rn']))

def vcycle_levels(cp, ws, *, omega=0.67, pre=2, post=2, coarsest=30):
    top = ws.n - 1
    K = top - 2
    b = _fb().bind(cp, ws, K) if K >= 1 else None
    if b is not None:
        for l in range(K):
            _down(cp, ws, l, pre)
        _fb().launch(cp, ws, K, rF=ws.r[K], xF=ws.x[K], pre=pre, post=post, sweepsC=int(getattr(ws, 'coarse_jacobi', 0) or 0))
        for l in range(K - 1, -1, -1):
            _up(cp, ws, l)
            _smooth(cp, ws, l, ws.x[l], ws.r[l], post)
        return
    for l in range(top):
        _down(cp, ws, l, pre)
    if getattr(ws, 'coarse_jacobi', 0):
        ws.x[top].fill(0.0)
        _e = ws.L[top]
        _n = int(_e['n'])
        if _n <= 1024:
            _blk = min(1024, max(32, (_n + 31) // 32 * 32))
            AMG._fn(cp, 'gpu_jac_fused')((1,), (_blk,), (_e['ip'], _e['ix'], _e['dv'], _e['idiag'], ws.r[top], ws.x[top], cp.float64(_e['omega']), _e['n'], np.int32(int(ws.coarse_jacobi))), shared_mem=8 * _n)
        else:
            _smooth(cp, ws, top, ws.x[top], ws.r[top], int(ws.coarse_jacobi))
    else:
        cg, cb = ws.cpi_g
        AMG._fn(cp, 'gpu_csrmv')(cg, cb, (ws.cpi_ip, ws.cpi_ix, ws.cpi_dv, ws.r[top], ws.x[top], ws.cpi_n))
    for l in range(top - 1, -1, -1):
        _up(cp, ws, l)
        _smooth(cp, ws, l, ws.x[l], ws.r[l], post)

def install():
    if getattr(AMG.vcycle, '_installed', False):
        return True
    vcycle_levels._installed = True
    AMG.vcycle = vcycle_levels
    return True
