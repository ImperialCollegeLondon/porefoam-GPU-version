from __future__ import annotations
_SRC = '''
extern "C" {

__global__ void gpu_alpha_face_upwind(
    const double* __restrict__ alpha, const double* __restrict__ phi,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    double* __restrict__ af, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    af[f] = (phi[f] >= 0.0) ? alpha[own[f]] : alpha[nei[f]];
}

__global__ void gpu_alpha_face_vanleer_w(
    const double* __restrict__ alpha, const double* __restrict__ phi,
    const double* __restrict__ gcx, const double* __restrict__ gcy,
    const double* __restrict__ gcz,
    const double* __restrict__ dx, const double* __restrict__ dy,
    const double* __restrict__ dz,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ w, double* __restrict__ af, double* __restrict__ psi_out,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const int up = (phi[f] >= 0.0);
    const double aU = up ? alpha[o] : alpha[n];
    const double aD = up ? alpha[n] : alpha[o];
    const double d  = aD - aU;
    double psi = 0.0;
    if (d != 0.0) {
        const long long u = up ? o : n;
        const double s = up ? 1.0 : -1.0;
        const double gd = s * (gcx[u] * dx[f] + gcy[u] * dy[f]
                               + gcz[u] * dz[f]);
        const double r  = 2.0 * gd / d - 1.0;
        const double ar = fabs(r);
        psi = (r + ar) / (1.0 + ar);
        if (!isfinite(psi)) {
            psi = ((gd > 0.0) == (d > 0.0)) ? 2.0 : 0.0;
        }
    }
    psi_out[f] = psi;
    af[f] = aU + (up ? (1.0 - w[f]) : w[f]) * psi * d;
}

__global__ void gpu_flux_split(
    const double* __restrict__ af_up, const double* __restrict__ af_hi,
    const double* __restrict__ phi, const double* __restrict__ phir,
    const double* __restrict__ alpha, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ w,
    double* __restrict__ flux_lo,
    double* __restrict__ flux_corr, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double lo = af_up[f] * phi[f];
    const long long o = own[f], n = nei[f];
    const double ar_o = 2.0 * fmin(fmax(alpha[o], 0.0), 1.0) - alpha[o];
    const double ar_n = 2.0 * fmin(fmax(alpha[n], 0.0), 1.0) - alpha[n];
    const double a1o = sqrt(fabs(ar_o)), a1n = sqrt(fabs(ar_n));
    const double a2o = sqrt(fabs(1.0 - ar_o)), a2n = sqrt(fabs(1.0 - ar_n));
    const double pr = phir[f];
    const double a2f = (-pr >= 0.0) ? a2n : a2o;
    const double vcomp = pr * a2f;
    const double a1f = (vcomp >= 0.0) ? a1n : a1o;
    const double comp = vcomp * a1f;
    flux_lo[f]   = lo;
    flux_corr[f] = af_hi[f] * phi[f] - lo + comp;
}

__global__ void gpu_mules_sums(
    const double* __restrict__ fc, const double* __restrict__ lam,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    double* __restrict__ sp, double* __restrict__ sm, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double c = fc[f] * lam[f];
    const long long o = own[f], n = nei[f];
    if (c > 0.0)      { atomicAdd(&sp[o], c);  atomicAdd(&sm[n], c); }
    else if (c < 0.0) { atomicAdd(&sm[o], -c); atomicAdd(&sp[n], -c); }
}

__global__ void gpu_mules_cell_ratio_ref(
    const double* __restrict__ a_lo, const double* __restrict__ vol,
    const double* __restrict__ sp,  const double* __restrict__ sm,
    const double* __restrict__ spu, const double* __restrict__ smu,
    double* __restrict__ rp, double* __restrict__ rm,
    const double dt, const double* __restrict__ pmin,
    const double* __restrict__ pmax, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double V = vol[c];
    const double up = (pmax[c] - a_lo[c]) * V / dt;
    const double dn = (a_lo[c] - pmin[c]) * V / dt;
    rp[c] = fmin(fmax((sp[c] + up) / smu[c], 0.0), 1.0);
    rm[c] = fmin(fmax((sm[c] + dn) / spu[c], 0.0), 1.0);
}

__global__ void gpu_mules_bounds_init_ref(
    double* __restrict__ pmax, double* __restrict__ pmin,
    const double amin, const double amax, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    pmax[c] = amin;
    pmin[c] = amax;
}

__global__ void gpu_mules_face_lambda(
    const double* __restrict__ fc, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ rp,
    const double* __restrict__ rm, double* __restrict__ lam,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double c = fc[f];
    double l = (c > 0.0) ? fmin(rm[o], rp[n]) : fmin(rp[o], rm[n]);
    l = fmin(1.0, fmax(0.0, l));
    lam[f] = fmin(lam[f], l);
}

__device__ __forceinline__ double gpu_atomicMaxD(double* addr, double val)
{
    unsigned long long* p = (unsigned long long*)addr;
    unsigned long long old = *p, assumed;
    do {
        assumed = old;
        if (__longlong_as_double((long long)assumed) >= val) break;
        old = atomicCAS(p, assumed,
                        (unsigned long long)__double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double((long long)old);
}
__device__ __forceinline__ double gpu_atomicMinD(double* addr, double val)
{
    unsigned long long* p = (unsigned long long*)addr;
    unsigned long long old = *p, assumed;
    do {
        assumed = old;
        if (__longlong_as_double((long long)assumed) <= val) break;
        old = atomicCAS(p, assumed,
                        (unsigned long long)__double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double((long long)old);
}

__global__ void gpu_mules_bounds_face(
    const double* __restrict__ a0, const long long* __restrict__ own,
    const long long* __restrict__ nei, double* __restrict__ pmax,
    double* __restrict__ pmin, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double ao = a0[o], an = a0[n];
    gpu_atomicMaxD(&pmax[o], an);  gpu_atomicMinD(&pmin[o], an);
    gpu_atomicMaxD(&pmax[n], ao);  gpu_atomicMinD(&pmin[n], ao);
}

__global__ void gpu_mules_bounds_bnd_ref(
    const double* __restrict__ a0, const double* __restrict__ bphi,
    const long long* __restrict__ bown, const long long* __restrict__ bUt,
    const double a_inlet, double* __restrict__ pmax,
    double* __restrict__ pmin, const long long nBF)
{
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBF) return;
    const long long o = bown[b];
    const double af = (bUt[b] == 1) ? a_inlet : a0[o];
    gpu_atomicMaxD(&pmax[o], af);  gpu_atomicMinD(&pmin[o], af);
}

__global__ void gpu_mules_bounds_bnd_ref_field(
    const double* __restrict__ a0, const double* __restrict__ bphi,
    const long long* __restrict__ bown, const long long* __restrict__ bUt,
    const double* __restrict__ a_inlet, double* __restrict__ pmax,
    double* __restrict__ pmin, const long long nBF)
{
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBF) return;
    const long long o = bown[b];
    const double af = (bUt[b] == 1) ? a_inlet[b] : a0[o];
    gpu_atomicMaxD(&pmax[o], af);  gpu_atomicMinD(&pmin[o], af);
}

__global__ void gpu_mules_bounds_clamp(
    double* __restrict__ pmax, double* __restrict__ pmin,
    const double amin, const double amax, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    pmax[c] = fmin(pmax[c], amax);
    pmin[c] = fmax(pmin[c], amin);
}

__global__ void gpu_div_accumulate(
    const double* __restrict__ flux_lo, const double* __restrict__ flux_corr,
    const double* __restrict__ lam, const long long* __restrict__ own,
    const long long* __restrict__ nei, double* __restrict__ div,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double F = flux_lo[f] + lam[f] * flux_corr[f];
    atomicAdd(&div[own[f]],  F);
    atomicAdd(&div[nei[f]], -F);
}

__global__ void gpu_alpha_bnd_div(
    const double* __restrict__ a0, const double* __restrict__ bphi,
    const long long* __restrict__ bown, const long long* __restrict__ bUt,
    const double a_inlet, double* __restrict__ div, const long long nBF)
{
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBF) return;
    const double q = bphi[b];
    if (q == 0.0) return;
    const long long o = bown[b];
    const double af = (q >= 0.0) ? a0[o]
                                 : ((bUt[b] == 1) ? a_inlet : a0[o]);
    atomicAdd(&div[o], q * af);
}

__global__ void gpu_alpha_bnd_div_field(
    const double* __restrict__ a0, const double* __restrict__ bphi,
    const long long* __restrict__ bown, const long long* __restrict__ bUt,
    const double* __restrict__ a_inlet, double* __restrict__ div, const long long nBF)
{
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBF) return;
    const double q = bphi[b];
    if (q == 0.0) return;
    const long long o = bown[b];
    const double af = (q >= 0.0) ? a0[o]
                                 : ((bUt[b] == 1) ? a_inlet[b] : a0[o]);
    atomicAdd(&div[o], q * af);
}

__global__ void gpu_alpha_apply(
    const double* __restrict__ a0, const double* __restrict__ div,
    const double* __restrict__ vol, double* __restrict__ a1,
    const double dt, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    a1[c] = a0[c] - dt * div[c] / vol[c];
}

__global__ void gpu_a_alphasm(
    const double* __restrict__ alpha, const double* __restrict__ avg_a,
    double* __restrict__ asm_, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double q = fmin(fmax(alpha[c], 0.03), 0.97);
    asm_[c] = 0.7 * ((1.0 / 0.94) * (q - 0.03)) + 0.3 * avg_a[c];
}

__global__ void gpu_a_grad_relax(
    const double* __restrict__ ax, const double* __restrict__ ay,
    const double* __restrict__ az, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ w,
    double* __restrict__ gx, double* __restrict__ gy,
    double* __restrict__ gz, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double a = w[f], b = 1.0 - w[f];
    gx[f] += 0.1 * (a * ax[o] + b * ax[n]);
    gy[f] += 0.1 * (a * ay[o] + b * ay[n]);
    gz[f] += 0.1 * (a * az[o] + b * az[n]);
}

__global__ void gpu_a_nhatf(
    const double* __restrict__ gx, const double* __restrict__ gy,
    const double* __restrict__ gz, const double* __restrict__ Sfx,
    const double* __restrict__ Sfy, const double* __restrict__ Sfz,
    double* __restrict__ nHatf, const double deltaN, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double m = sqrt(gx[f]*gx[f] + gy[f]*gy[f] + gz[f]*gz[f]) + deltaN;
    nHatf[f] = (gx[f]*Sfx[f] + gy[f]*Sfy[f] + gz[f]*Sfz[f]) / m;
}

__global__ void gpu_a_phir(
    const double* __restrict__ phic, const double* __restrict__ asm_,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ dmag, const double* __restrict__ nHatf,
    const double* __restrict__ phic_max, double* __restrict__ phir,
    const double cAlpha, const double deltaN, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double so = fmin(fmax(asm_[own[f]], 0.48), 0.52);
    const double sn = fmin(fmax(asm_[nei[f]], 0.48), 0.52);
    const double isS = fabs((sn - so) / dmag[f]);
    const double dni = 1.0 - 0.9 * isS / (isS + deltaN);
    phir[f] = fmin(cAlpha * dni * phic[f], phic_max[0]) * nHatf[f];
}

__global__ void gpu_face_localmax(
    const double* __restrict__ q, const long long* __restrict__ own,
    const long long* __restrict__ nei, double* __restrict__ qf,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    qf[f] = fmax(q[own[f]], q[nei[f]]);
}

}
'''
_MOD = None
_FN: dict = {}
KERNELS = ('gpu_alpha_face_upwind', 'gpu_alpha_face_vanleer_w', 'gpu_flux_split', 'gpu_mules_sums', 'gpu_mules_bounds_face', 'gpu_mules_bounds_bnd_ref', 'gpu_mules_bounds_bnd_ref_field', 'gpu_mules_bounds_clamp', 'gpu_mules_cell_ratio_ref', 'gpu_mules_bounds_init_ref', 'gpu_mules_face_lambda', 'gpu_div_accumulate', 'gpu_alpha_bnd_div', 'gpu_alpha_bnd_div_field', 'gpu_alpha_apply', 'gpu_a_alphasm', 'gpu_a_grad_relax', 'gpu_a_nhatf', 'gpu_a_phir', 'gpu_face_localmax')

def module(cp):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))
    return _MOD

def fn(cp, name):
    if name not in KERNELS:
        raise KeyError('alpha kernels: unknown kernel %r' % name)
    f = _FN.get(name)
    if f is None:
        f = module(cp).get_function(name)
        _FN[name] = f
    return f

def grid(n, tpb=256):
    return ((int((n + tpb - 1) // tpb),), (int(tpb),))

def compile_all(cp) -> int:
    for k in KERNELS:
        fn(cp, k)
    return len(KERNELS)
