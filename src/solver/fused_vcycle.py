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
_SRC = '''
extern "C" {

__device__ __forceinline__ void fused_spmv(
    const long long* __restrict__ ip, const int* __restrict__ ix,
    const double* __restrict__ dv, const double* xs, double* ys,
    const long long n, const long long i)
{
    if (i < n) {
        double s = 0.0;
        for (long long k = ip[i]; k < ip[i+1]; ++k)
            s += dv[k] * xs[ix[k]];
        ys[i] = s;
    }
}

__device__ __forceinline__ void fused_smooth(
    const long long* __restrict__ ip, const int* __restrict__ ix,
    const double* __restrict__ dv, const double* __restrict__ idiag,
    const double* rhs, double* xs, const double omega,
    const long long n, const int sweeps, const long long i)
{
    for (int sw = 0; sw < sweeps; ++sw) {
        double t = 0.0;
        if (i < n) {
            for (long long k = ip[i]; k < ip[i+1]; ++k)
                t += dv[k] * xs[ix[k]];
            t = rhs[i] - t;
            t = omega * idiag[i] * t;
        }
        __syncthreads();
        if (i < n) xs[i] += t;
        __syncthreads();
    }
}

__global__ void fused_sub3(
    const long long* __restrict__ ipF, const int* __restrict__ ixF,
    const double* __restrict__ dvF, const double* __restrict__ idF,
    const long long* __restrict__ PipF, const int* __restrict__ PixF,
    const double* __restrict__ PdvF,
    const long long* __restrict__ RipF, const int* __restrict__ RixF,
    const double* __restrict__ RdvF,
    const long long* __restrict__ ipM, const int* __restrict__ ixM,
    const double* __restrict__ dvM, const double* __restrict__ idM,
    const long long* __restrict__ PipM, const int* __restrict__ PixM,
    const double* __restrict__ PdvM,
    const long long* __restrict__ RipM, const int* __restrict__ RixM,
    const double* __restrict__ RdvM,
    const long long* __restrict__ ipC, const int* __restrict__ ixC,
    const double* __restrict__ dvC, const double* __restrict__ idC,
    const long long* __restrict__ QipC, const int* __restrict__ QixC,
    const double* __restrict__ QdvC,
    const double* __restrict__ rF, double* __restrict__ xFout,
    const double omF, const double omM, const double omC,
    const long long nF, const long long nM, const long long nC,
    const int pre, const int post, const int sweepsC, const int narrow)
{
    extern __shared__ double sh[];
    double* xsF = sh;
    double* tsF = xsF + nF;
    double* rsM = tsF + nF;
    double* xsM = rsM + nM;
    double* tsM = xsM + nM;
    double* rsC = tsM + nM;
    double* xsC = rsC + nC;
    const long long i = (long long)threadIdx.x;

    if (i < nF) xsF[i] = 0.0;
    __syncthreads();
    fused_smooth(ipF, ixF, dvF, idF, rF, xsF, omF, nF, pre, i);
    fused_spmv(ipF, ixF, dvF, xsF, tsF, nF, i);
    if (i < nF) tsF[i] = rF[i] - tsF[i];
    __syncthreads();
    fused_spmv(RipF, RixF, RdvF, tsF, rsM, nM, i);
    __syncthreads();

    if (i < nM) xsM[i] = 0.0;
    __syncthreads();
    fused_smooth(ipM, ixM, dvM, idM, rsM, xsM, omM, nM, pre, i);
    fused_spmv(ipM, ixM, dvM, xsM, tsM, nM, i);
    if (i < nM) tsM[i] = rsM[i] - tsM[i];
    __syncthreads();
    fused_spmv(RipM, RixM, RdvM, tsM, rsC, nC, i);
    __syncthreads();

    if (i < nC) xsC[i] = 0.0;
    __syncthreads();
    if (sweepsC > 0) {
        if (narrow != 0 && nC <= 32) {
            if (i < 32) {
                for (int sw = 0; sw < sweepsC; ++sw) {
                    double t = 0.0;
                    if (i < nC) {
                        for (long long k = ipC[i]; k < ipC[i+1]; ++k)
                            t += dvC[k] * xsC[ixC[k]];
                        t = rsC[i] - t;
                        t = omC * idC[i] * t;
                    }
                    __syncwarp();
                    if (i < nC) xsC[i] += t;
                    __syncwarp();
                }
            }
            __syncthreads();
        } else {
            fused_smooth(ipC, ixC, dvC, idC, rsC, xsC, omC, nC, sweepsC, i);
        }
    } else {
        fused_spmv(QipC, QixC, QdvC, rsC, xsC, nC, i);
        __syncthreads();
    }

    fused_spmv(PipM, PixM, PdvM, xsC, tsM, nM, i);
    __syncthreads();
    if (i < nM) xsM[i] += 1.0 * tsM[i];
    __syncthreads();
    fused_smooth(ipM, ixM, dvM, idM, rsM, xsM, omM, nM, post, i);

    fused_spmv(PipF, PixF, PdvF, xsM, tsF, nF, i);
    __syncthreads();
    if (i < nF) xsF[i] += 1.0 * tsF[i];
    __syncthreads();
    fused_smooth(ipF, ixF, dvF, idF, rF, xsF, omF, nF, post, i);

    if (i < nF) xFout[i] = xsF[i];
}
}
'''
ARGS = ('ipF', 'ixF', 'dvF', 'idF', 'PipF', 'PixF', 'PdvF', 'RipF', 'RixF', 'RdvF', 'ipM', 'ixM', 'dvM', 'idM', 'PipM', 'PixM', 'PdvM', 'RipM', 'RixM', 'RdvM', 'ipC', 'ixC', 'dvC', 'idC', 'QipC', 'QixC', 'QdvC', 'rF', 'xFout', 'omF', 'omM', 'omC', 'nF', 'nM', 'nC', 'pre', 'post', 'sweepsC', 'narrow')

def _check_signature():
    m = re.search('__global__\\s+void\\s+fused_sub3\\s*\\((.*?)\\)\\s*\\{', _SRC, re.S)
    if m is None:
        raise ImportError("fused_vcycle: cannot find fused_sub3's declaration")
    names = []
    for p in m.group(1).split(','):
        ids = re.findall('[A-Za-z_][A-Za-z_0-9]*', p)
        if ids:
            names.append(ids[-1])
    if tuple(names) != ARGS:
        raise ImportError('''fused_vcycle: kernel signature does not match ARGS.
  kernel: %r
  ARGS  : %r''' % (tuple(names), ARGS))
    return len(ARGS)
NARG = _check_signature()
_MOD = None
_K = {}

def kernel(cp, name='fused_sub3'):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=_OPTS)
    k = _K.get(name)
    if k is None:
        k = _MOD.get_function(name)
        _K[name] = k
    return k
_SLOT = {n: ARGS.index(n) for n in ('rF', 'xFout', 'pre', 'post', 'sweepsC', 'narrow')}

def bind(cp, ws, K):
    got = getattr(ws, '_fused_tag', None)
    if got is not None and got.get('K') == K:
        return got
    if ws.n - 1 != K + 2:
        return None
    F, M, C = (ws.L[K], ws.L[K + 1], ws.L[K + 2])
    nF, nM, nC = (int(F['n']), int(M['n']), int(C['n']))
    if nF > 1024:
        return None
    if 'Pip' not in F or 'Pip' not in M:
        return None
    shared = 8 * (2 * nF + 3 * nM + 2 * nC)
    blk = int(min(1024, max(32, (nF + 31) // 32 * 32)))
    k = kernel(cp)
    if shared > 49152:
        try:
            k.max_dynamic_shared_size_bytes = shared
        except Exception:
            return None
    d = {'ipF': F['ip'], 'ixF': F['ix'], 'dvF': F['dv'], 'idF': F['idiag'], 'PipF': F['Pip'], 'PixF': F['Pix'], 'PdvF': F['Pdv'], 'RipF': F['Rip'], 'RixF': F['Rix'], 'RdvF': F['Rdv'], 'ipM': M['ip'], 'ixM': M['ix'], 'dvM': M['dv'], 'idM': M['idiag'], 'PipM': M['Pip'], 'PixM': M['Pix'], 'PdvM': M['Pdv'], 'RipM': M['Rip'], 'RixM': M['Rix'], 'RdvM': M['Rdv'], 'ipC': C['ip'], 'ixC': C['ix'], 'dvC': C['dv'], 'idC': C['idiag'], 'QipC': ws.cpi_ip, 'QixC': ws.cpi_ix, 'QdvC': ws.cpi_dv, 'omF': cp.float64(F['omega']), 'omM': cp.float64(M['omega']), 'omC': cp.float64(C['omega']), 'nF': np.int64(nF), 'nM': np.int64(nM), 'nC': np.int64(nC)}
    d.update({'rF': None, 'xFout': None, 'pre': None, 'post': None, 'sweepsC': None, 'narrow': None})
    got = {'K': K, 'd': d, 'argl': [d[n] for n in ARGS], 'blk': (blk,), 'grid': (1,), 'shared': shared, 'nF': nF, 'nM': nM, 'nC': nC, 'k': k, 'key': None, 'tup': None}
    ws._fused_tag = got
    return got

def launch(cp, ws, K, *, rF, xF, pre, post, sweepsC, narrow=None):
    b = bind(cp, ws, K)
    if b is None:
        return False
    nw = 0 if narrow is None else int(narrow)
    key = (id(rF), id(xF), int(pre), int(post), int(sweepsC), nw)
    if b['key'] != key:
        al = b['argl']
        al[_SLOT['rF']] = rF
        al[_SLOT['xFout']] = xF
        al[_SLOT['pre']] = np.int32(int(pre))
        al[_SLOT['post']] = np.int32(int(post))
        al[_SLOT['sweepsC']] = np.int32(int(sweepsC))
        al[_SLOT['narrow']] = np.int32(nw)
        if any((a is None for a in al)):
            raise AssertionError('fused_vcycle: unbound argument in %r' % [n for n, a in zip(ARGS, al) if a is None])
        b['tup'] = tuple(al)
        b['key'] = key
    b['k'](b['grid'], b['blk'], b['tup'], shared_mem=b['shared'])
    return True
import fused_vcycle_levels as _LEVELS_PRE
_UNFUSED_FALLBACK = getattr(_LEVELS_PRE, '_UNFUSED_VCYCLE', AMG.vcycle)

def vcycle_fused(cp, ws, *, omega=0.67, pre=2, post=2, coarsest=30):
    top = ws.n - 1
    K = top - 2
    b = bind(cp, ws, K) if K >= 1 else None
    if b is None:
        return _UNFUSED_FALLBACK(cp, ws, omega=omega, pre=pre, post=post, coarsest=coarsest)
    sweepsC = int(getattr(ws, 'coarse_jacobi', 0) or 0)
    for l in range(K):
        e = ws.L[l]
        g, bl = e['g']
        ws.x[l].fill(0.0)
        AMG._smooth(cp, ws, l, ws.x[l], ws.r[l], pre)
        AMG._mv(cp, e, ws.x[l], ws.t[l])
        AMG._fn(cp, 'gpu_sub')(g, bl, (ws.r[l], ws.t[l], ws.t[l], e['n']))
        AMG._csrmv(cp, (e['Rip'], e['Rix'], e['Rdv'], ws.t[l], ws.r[l + 1], e['Rn']))
    launch(cp, ws, K, rF=ws.r[K], xF=ws.x[K], pre=pre, post=post, sweepsC=sweepsC)
    for l in range(K - 1, -1, -1):
        e = ws.L[l]
        g, bl = e['g']
        AMG._csrmv(cp, (e['Pip'], e['Pix'], e['Pdv'], ws.x[l + 1], ws.t[l], e['n']))
        AMG._fn(cp, 'gpu_add')(g, bl, (ws.t[l], ws.x[l], cp.float64(1.0), e['n']))
        AMG._smooth(cp, ws, l, ws.x[l], ws.r[l], post)

def install():
    if getattr(AMG.vcycle, '_fused_tag', False):
        return True
    vcycle_fused._fused_tag = True
    AMG.vcycle = vcycle_fused
    return True
