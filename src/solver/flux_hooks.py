from __future__ import annotations
import numpy as np
_SRC = '''
extern "C" {

__device__ __forceinline__ double warp_sum(double v)
{
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__global__ void gpu_flux_hooks_q(
    const double* __restrict__ alpha, double* __restrict__ q1,
    double* __restrict__ q2, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double a = alpha[c];
    const double ar = 2.0 * fmin(fmax(a, 0.0), 1.0) - a;
    q1[c] = sqrt(fabs(ar));
    q2[c] = sqrt(fabs(1.0 - ar));
}

__global__ void gpu_flux_hooks_grad_acc(
    const double* __restrict__ q1, const double* __restrict__ q2,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ Sfx, const double* __restrict__ Sfy,
    const double* __restrict__ Sfz, const double* __restrict__ w,
    double* __restrict__ g1x, double* __restrict__ g1y,
    double* __restrict__ g1z, double* __restrict__ g2x,
    double* __restrict__ g2y, double* __restrict__ g2z,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double wf = w[f];
    const double sx = Sfx[f], sy = Sfy[f], sz = Sfz[f];
    const double f1 = wf * q1[o] + (1.0 - wf) * q1[n];
    const double f2 = wf * q2[o] + (1.0 - wf) * q2[n];
    atomicAdd(&g1x[o],  f1 * sx);  atomicAdd(&g1x[n], -f1 * sx);
    atomicAdd(&g1y[o],  f1 * sy);  atomicAdd(&g1y[n], -f1 * sy);
    atomicAdd(&g1z[o],  f1 * sz);  atomicAdd(&g1z[n], -f1 * sz);
    atomicAdd(&g2x[o],  f2 * sx);  atomicAdd(&g2x[n], -f2 * sx);
    atomicAdd(&g2y[o],  f2 * sy);  atomicAdd(&g2y[n], -f2 * sy);
    atomicAdd(&g2z[o],  f2 * sz);  atomicAdd(&g2z[n], -f2 * sz);
}

__global__ void gpu_flux_hooks_grad_acc_bnd(
    const double* __restrict__ q1, const double* __restrict__ q2,
    const long long* __restrict__ bown, const double* __restrict__ bSfx,
    const double* __restrict__ bSfy, const double* __restrict__ bSfz,
    double* __restrict__ g1x, double* __restrict__ g1y,
    double* __restrict__ g1z, double* __restrict__ g2x,
    double* __restrict__ g2y, double* __restrict__ g2z,
    const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    const long long o = bown[f];
    const double v1 = q1[o], v2 = q2[o];
    const double sx = bSfx[f], sy = bSfy[f], sz = bSfz[f];
    atomicAdd(&g1x[o], v1 * sx);
    atomicAdd(&g1y[o], v1 * sy);
    atomicAdd(&g1z[o], v1 * sz);
    atomicAdd(&g2x[o], v2 * sx);
    atomicAdd(&g2y[o], v2 * sy);
    atomicAdd(&g2z[o], v2 * sz);
}

__global__ void gpu_flux_hooks_grad_div(
    double* __restrict__ g1x, double* __restrict__ g1y,
    double* __restrict__ g1z, double* __restrict__ g2x,
    double* __restrict__ g2y, double* __restrict__ g2z,
    const double* __restrict__ vol, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double iv = 1.0 / vol[c];
    g1x[c] *= iv; g1y[c] *= iv; g1z[c] *= iv;
    g2x[c] *= iv; g2y[c] *= iv; g2z[c] *= iv;
}

__global__ void gpu_flux_hooks_flux_split(
    const double* __restrict__ af_up, const double* __restrict__ af_hi,
    const double* __restrict__ phi,   const double* __restrict__ phir,
    const double* __restrict__ q1,    const double* __restrict__ q2,
    const double* __restrict__ g1x,   const double* __restrict__ g1y,
    const double* __restrict__ g1z,   const double* __restrict__ g2x,
    const double* __restrict__ g2y,   const double* __restrict__ g2z,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ dxv,   const double* __restrict__ dyv,
    const double* __restrict__ dzv,   const double* __restrict__ w,
    double* __restrict__ flux_lo,     double* __restrict__ flux_corr,
    double* __restrict__ stats,
    const int form, const int do_stats, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int valid = (f < nIF) ? 1 : 0;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    double s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;
    if (valid) {
        const long long o = own[f], n = nei[f];
        const double lo = af_up[f] * phi[f];
        const double a1o = q1[o], a1n = q1[n];
        const double a2o = q2[o], a2n = q2[n];
        const double pr  = phir[f];
        const double cdw = w[f];
        const double dxf = dxv[f], dyf = dyv[f], dzf = dzv[f];

        const double a2f_dw  = (-pr >= 0.0) ? a2n : a2o;
        const double vc_dw   = pr * a2f_dw;
        const double a1f_dw  = (vc_dw >= 0.0) ? a1n : a1o;
        const double comp_dw = vc_dw * a1f_dw;

        double gfx = -pr;
        double gdf = a2n - a2o;
        double gcf = (gfx > 0.0) ? (dxf * g2x[o] + dyf * g2y[o] + dzf * g2z[o])
                                 : (dxf * g2x[n] + dyf * g2y[n] + dzf * g2z[n]);
        gdf = (gdf >= 0.0) ? (gdf + 1.0e-15) : (gdf - 1.0e-15);
        double r = 2.0 * (gcf / gdf) - 1.0;
        if (!(r > 0.0)) r = 0.0;
        const double psi2 = (r + fabs(r)) / (1.0 + fabs(r));
        const double W2 = psi2 * cdw
                        + (1.0 - psi2) * ((gfx >= 0.0) ? 1.0 : 0.0);
        const double a2f_vl = W2 * a2o + (1.0 - W2) * a2n;
        const double vc_vl  = pr * a2f_vl;
        gfx = vc_vl;
        gdf = a1n - a1o;
        gcf = (gfx > 0.0) ? (dxf * g1x[o] + dyf * g1y[o] + dzf * g1z[o])
                          : (dxf * g1x[n] + dyf * g1y[n] + dzf * g1z[n]);
        gdf = (gdf >= 0.0) ? (gdf + 1.0e-15) : (gdf - 1.0e-15);
        r = 2.0 * (gcf / gdf) - 1.0;
        if (!(r > 0.0)) r = 0.0;
        const double psi1 = (r + fabs(r)) / (1.0 + fabs(r));
        const double W1 = psi1 * cdw
                        + (1.0 - psi1) * ((gfx >= 0.0) ? 1.0 : 0.0);
        const double a1f_vl = W1 * a1o + (1.0 - W1) * a1n;
        const double comp_vl = vc_vl * a1f_vl;

        const double comp = (form == 1) ? comp_vl : comp_dw;
        flux_lo[f]   = lo;
        flux_corr[f] = af_hi[f] * phi[f] - lo + comp;

        if (do_stats) {
            const double act = (pr != 0.0) ? 1.0 : 0.0;
            s0 = fabs(comp_vl);
            s1 = fabs(comp_dw);
            s2 = act;
            s3 = (act != 0.0 && psi2 == 0.0) ? 1.0 : 0.0;
            s4 = (act != 0.0 && psi1 == 0.0) ? 1.0 : 0.0;
            s5 = act * psi2;
            s6 = act * psi1;
            s7 = fabs(comp_vl - comp_dw);
        }
    }
    if (do_stats) {
        s0 = warp_sum(s0); s1 = warp_sum(s1);
        s2 = warp_sum(s2); s3 = warp_sum(s3);
        s4 = warp_sum(s4); s5 = warp_sum(s5);
        s6 = warp_sum(s6); s7 = warp_sum(s7);
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&stats[0], s0); atomicAdd(&stats[1], s1);
            atomicAdd(&stats[2], s2); atomicAdd(&stats[3], s3);
            atomicAdd(&stats[4], s4); atomicAdd(&stats[5], s5);
            atomicAdd(&stats[6], s6); atomicAdd(&stats[7], s7);
        }
    }
}

}
'''
_MOD = None
_FNC: dict = {}
STATS = 0
_CP = None
_MESH: dict = {}
_AREA_ROWS: list = []
_AREA_SEEN: set = set()
_AH: dict = {}
_ORIG_GATHER = None
_BUF: dict = {}
_STATS_DEV = None
_NCALLS = 0
_ORIG_FN = None
_ORIG_STEP = None

def module(cp):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))
    return _MOD

def fn(cp, name):
    f = _FNC.get(name)
    if f is None:
        f = module(cp).get_function(name)
        _FNC[name] = f
    return f

def grid(n, tpb=256):
    return ((int((n + tpb - 1) // tpb),), (int(tpb),))

def _buffers(cp, nC):
    b = _BUF.get(nC)
    if b is None:
        pool = cp.empty(8 * nC, dtype=cp.float64)
        b = dict(pool=pool, q1=pool[0 * nC:1 * nC], q2=pool[1 * nC:2 * nC], g=pool[2 * nC:8 * nC], g1x=pool[2 * nC:3 * nC], g1y=pool[3 * nC:4 * nC], g1z=pool[4 * nC:5 * nC], g2x=pool[5 * nC:6 * nC], g2y=pool[6 * nC:7 * nC], g2z=pool[7 * nC:8 * nC])
        _BUF[nC] = b
    return b

def _stats(cp):
    global _STATS_DEV
    if _STATS_DEV is None:
        _STATS_DEV = cp.zeros(8, dtype=cp.float64)
    return _STATS_DEV

def calls():
    return _NCALLS

def _flux_split_shim(g, b, args):
    global _NCALLS, _FIRST
    af_up, af_hi, phi, phir, alpha, own, nei, w, flux_lo, flux_corr, nIF = args
    cp = _CP
    m = _MESH.get('m')
    if m is None:
        raise RuntimeError('flux_hooks: mesh never captured; install() did not wrap time_step.step, or step was called differently')
    nC = int(m['vol'].size)
    nIF = int(nIF)
    buf = _buffers(cp, nC)
    gc = grid(nC)
    gf = grid(nIF)
    fn(cp, 'gpu_flux_hooks_q')(gc[0], gc[1], (alpha, buf['q1'], buf['q2'], nC))
    buf['g'].fill(0.0)
    fn(cp, 'gpu_flux_hooks_grad_acc')(gf[0], gf[1], (buf['q1'], buf['q2'], own, nei, m['Sfx'], m['Sfy'], m['Sfz'], w, buf['g1x'], buf['g1y'], buf['g1z'], buf['g2x'], buf['g2y'], buf['g2z'], nIF))
    bown = m.get('bown')
    nBF = int(bown.size) if bown is not None else 0
    if nBF > 0:
        gb = grid(nBF)
        fn(cp, 'gpu_flux_hooks_grad_acc_bnd')(gb[0], gb[1], (buf['q1'], buf['q2'], bown, m['bSfx'], m['bSfy'], m['bSfz'], buf['g1x'], buf['g1y'], buf['g1z'], buf['g2x'], buf['g2y'], buf['g2z'], nBF))
    fn(cp, 'gpu_flux_hooks_grad_div')(gc[0], gc[1], (buf['g1x'], buf['g1y'], buf['g1z'], buf['g2x'], buf['g2y'], buf['g2z'], m['vol'], nC))
    fn(cp, 'gpu_flux_hooks_flux_split')(gf[0], gf[1], (af_up, af_hi, phi, phir, buf['q1'], buf['q2'], buf['g1x'], buf['g1y'], buf['g1z'], buf['g2x'], buf['g2y'], buf['g2z'], own, nei, m['dx'], m['dy'], m['dz'], w, flux_lo, flux_corr, _stats(cp), np.int32(1), np.int32(0), nIF))
    _NCALLS += 1
    if _NCALLS == 1 and 0:
        _FIRST = _STATS_DEV.get().copy()

def _area_of(a_h):
    own = _AH['own']
    nei = _AH['nei']
    mag = _AH['mag']
    a_h = np.asarray(a_h).ravel()
    return float((np.abs(a_h[nei] - a_h[own]) * mag).sum())

def _gather_hook(case, tgt, name, *a, **kw):
    out = _ORIG_GATHER(case, tgt, name, *a, **kw)
    return out
    key = str(tgt)
    if key in _AREA_SEEN:
        return out
    m = _MESH.get('m')
    f = _MESH.get('fields')
    if m is None or f is None:
        return out
    if 'own' not in _AH:
        _AH['own'] = m['own'].get()
        _AH['nei'] = m['nei'].get()
        _AH['mag'] = m['magSf'].get()
    _AREA_SEEN.add(key)
    try:
        a6 = f['alpha'].get()
    except Exception:
        return out
    _AREA_ROWS.append((float(tgt), _area_of(a6), _area_of(out)))
    return out

def install():
    global _ORIG_FN, _ORIG_STEP, _ORIG_GATHER
    import alpha_kernels as K
    import time_step as STEP
    import foam_fields as FF
    if _ORIG_GATHER is None and 0:
        _ORIG_GATHER = FF.gather
        FF.gather = _gather_hook
    if _ORIG_FN is None:
        _ORIG_FN = K.fn

        def _fn(cp, name):
            global _CP
            _CP = cp
            if name == 'gpu_flux_split':
                for _k in ('gpu_flux_hooks_q', 'gpu_flux_hooks_grad_acc', 'gpu_flux_hooks_grad_acc_bnd', 'gpu_flux_hooks_grad_div', 'gpu_flux_hooks_flux_split'):
                    fn(cp, _k)
                return _flux_split_shim
            return _ORIG_FN(cp, name)
        K.fn = _fn
    if _ORIG_STEP is None:
        _ORIG_STEP = STEP.step

        def _step(ws, *, mesh, fields, control):
            if 'm' not in _MESH:
                _MESH['m'] = mesh
            _MESH['fields'] = fields
            return _ORIG_STEP(ws, mesh=mesh, fields=fields, control=control)
        STEP.step = _step
    return True
