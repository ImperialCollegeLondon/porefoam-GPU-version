from __future__ import annotations
_SRC = '''
extern "C" {

__global__ void gpu_alpha_gradf(
    const double* __restrict__ alpha, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ dmag,
    double* __restrict__ gradf, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double d = dmag[f];
    gradf[f] = (d > 0.0) ? (alpha[nei[f]] - alpha[own[f]]) / d : 0.0;
}

__global__ void gpu_grad_accumulate(
    const double* __restrict__ alpha, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ Sfx,
    const double* __restrict__ Sfy, const double* __restrict__ Sfz,
    const double* __restrict__ w,
    double* __restrict__ gx, double* __restrict__ gy, double* __restrict__ gz,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double af = w[f] * alpha[o] + (1.0 - w[f]) * alpha[n];
    atomicAdd(&gx[o],  af * Sfx[f]);  atomicAdd(&gx[n], -af * Sfx[f]);
    atomicAdd(&gy[o],  af * Sfy[f]);  atomicAdd(&gy[n], -af * Sfy[f]);
    atomicAdd(&gz[o],  af * Sfz[f]);  atomicAdd(&gz[n], -af * Sfz[f]);
}

__global__ void gpu_grad_divide_volume(
    double* __restrict__ gx, double* __restrict__ gy, double* __restrict__ gz,
    const double* __restrict__ vol, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double iv = 1.0 / vol[c];
    gx[c] *= iv; gy[c] *= iv; gz[c] *= iv;
}

__global__ void gpu_nhat(
    const double* __restrict__ gx, const double* __restrict__ gy,
    const double* __restrict__ gz, double* __restrict__ nx,
    double* __restrict__ ny, double* __restrict__ nz,
    const double deltaN, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double a = gx[c], b = gy[c], d = gz[c];
    const double m = sqrt(a*a + b*b + d*d + deltaN*deltaN);
    nx[c] = a / m; ny[c] = b / m; nz[c] = d / m;
}

__global__ void gpu_kappa_accumulate(
    const double* __restrict__ nx, const double* __restrict__ ny,
    const double* __restrict__ nz, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ Sfx,
    const double* __restrict__ Sfy, const double* __restrict__ Sfz,
    const double* __restrict__ w, double* __restrict__ kap,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double wf = w[f], wm = 1.0 - wf;
    const double fx = wf*nx[o] + wm*nx[n];
    const double fy = wf*ny[o] + wm*ny[n];
    const double fz = wf*nz[o] + wm*nz[n];
    const double flux = fx*Sfx[f] + fy*Sfy[f] + fz*Sfz[f];
    atomicAdd(&kap[o],  flux);
    atomicAdd(&kap[n], -flux);
}

__global__ void gpu_kappa_finish(
    double* __restrict__ kap, const double* __restrict__ vol,
    const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    kap[c] = -kap[c] / vol[c];
}

__global__ void gpu_surface_tension_face(
    const double* __restrict__ kap, const double* __restrict__ gradf,
    const double* __restrict__ magSf, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ w,
    double* __restrict__ Fsig, const double sigma, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double kf = w[f]*kap[own[f]] + (1.0 - w[f])*kap[nei[f]];
    Fsig[f] = sigma * kf * gradf[f] * magSf[f];
}

__global__ void gpu_contact_angle(
    double* __restrict__ nx, double* __restrict__ ny, double* __restrict__ nz,
    const long long* __restrict__ wall_cells,
    const double* __restrict__ wnx, const double* __restrict__ wny,
    const double* __restrict__ wnz, const double* __restrict__ costheta,
    unsigned long long* __restrict__ degenerate,
    const double tan_tol, const double* __restrict__ gx,
    const double* __restrict__ gy, const double* __restrict__ gz,
    const double contrast, const long long nW)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nW) return;
    const long long c = wall_cells[i];
    const double gm = sqrt(gx[c]*gx[c] + gy[c]*gy[c]
                           + gz[c]*gz[c]);
    if (gm <= contrast) return;
    const double ax = nx[c], ay = ny[c], az = nz[c];
    const double bx = wnx[i], by = wny[i], bz = wnz[i];
    const double ct = costheta[i];
    const double st = sqrt(fmax(0.0, 1.0 - ct*ct));
    const double ab = ax*bx + ay*by + az*bz;
    double tx = ax - ab*bx, ty = ay - ab*by, tz = az - ab*bz;
    const double tm = sqrt(tx*tx + ty*ty + tz*tz);
    if (tm <= tan_tol) {
        atomicAdd(degenerate, 1ULL);
        return;
    }
    tx /= tm; ty /= tm; tz /= tm;
    const double am = sqrt(ax*ax + ay*ay + az*az);
    nx[c] = am * (bx*ct + tx*st);
    ny[c] = am * (by*ct + ty*st);
    nz[c] = am * (bz*ct + tz*st);
}

__global__ void gpu_grad_accumulate_bnd(
    const double* __restrict__ alpha, const long long* __restrict__ bown,
    const double* __restrict__ bSfx, const double* __restrict__ bSfy,
    const double* __restrict__ bSfz, double* __restrict__ gx,
    double* __restrict__ gy, double* __restrict__ gz, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    const long long o = bown[f];
    const double af = alpha[o];
    atomicAdd(&gx[o], af * bSfx[f]);
    atomicAdd(&gy[o], af * bSfy[f]);
    atomicAdd(&gz[o], af * bSfz[f]);
}

__global__ void gpu_grad_accumulate_bnd_field(
    const double* __restrict__ alpha, const long long* __restrict__ bown,
    const double* __restrict__ bSfx, const double* __restrict__ bSfy,
    const double* __restrict__ bSfz, double* __restrict__ gx,
    double* __restrict__ gy, double* __restrict__ gz, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    const long long o = bown[f];
    const double af = alpha[f];
    atomicAdd(&gx[o], af * bSfx[f]);
    atomicAdd(&gy[o], af * bSfy[f]);
    atomicAdd(&gz[o], af * bSfz[f]);
}

__global__ void gpu_grad_accumulate_pbnd(
    const double* __restrict__ p, const long long* __restrict__ bown,
    const long long* __restrict__ bPt, const double* __restrict__ bSfx,
    const double* __restrict__ bSfy, const double* __restrict__ bSfz,
    double* __restrict__ gx, double* __restrict__ gy,
    double* __restrict__ gz, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    const long long o = bown[f];
    const double pf = (bPt[f] == 1) ? 0.0 : p[o];
    atomicAdd(&gx[o], pf * bSfx[f]);
    atomicAdd(&gy[o], pf * bSfy[f]);
    atomicAdd(&gz[o], pf * bSfz[f]);
}

__global__ void gpu_kappa_accumulate_bnd(
    const double* __restrict__ nx, const double* __restrict__ ny,
    const double* __restrict__ nz, const long long* __restrict__ bown,
    const double* __restrict__ bSfx, const double* __restrict__ bSfy,
    const double* __restrict__ bSfz, double* __restrict__ kap,
    const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    const long long o = bown[f];
    atomicAdd(&kap[o], nx[o]*bSfx[f] + ny[o]*bSfy[f] + nz[o]*bSfz[f]);
}

__global__ void gpu_nhat_face(
    const double* __restrict__ nx, const double* __restrict__ ny,
    const double* __restrict__ nz, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ w,
    const double* __restrict__ Sfx, const double* __restrict__ Sfy,
    const double* __restrict__ Sfz, double* __restrict__ nHatf,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double a = w[f], b = 1.0 - w[f];
    nHatf[f] = (a * nx[o] + b * nx[n]) * Sfx[f]
             + (a * ny[o] + b * ny[n]) * Sfy[f]
             + (a * nz[o] + b * nz[n]) * Sfz[f];
}

__global__ void gpu_ccf_alpha_sh(
    const double* __restrict__ alpha, double* __restrict__ ash,
    const double pcT, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double v = (1.0 / pcT) * alpha[c] - (0.5 / pcT - 0.5);
    ash[c] = fmin(fmax(v, 0.0), 1.0);
}

__global__ void gpu_ccf_delta_s(
    const double* __restrict__ ash, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ dmag,
    double* __restrict__ dS, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double d = dmag[f];
    dS[f] = (d > 0.0) ? (ash[nei[f]] - ash[own[f]]) / d : 0.0;
}

__global__ void gpu_ccf_avg_acc(
    const double* __restrict__ qf, const double* __restrict__ magSf,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    double* __restrict__ num, double* __restrict__ den, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double w = magSf[f];
    const double n = w * qf[f];
    atomicAdd(&num[own[f]], n);  atomicAdd(&num[nei[f]], n);
    atomicAdd(&den[own[f]], w);  atomicAdd(&den[nei[f]], w);
}

__global__ void gpu_ccf_avg_acc_bnd(
    const double* __restrict__ qb, const double* __restrict__ bmagSf,
    const long long* __restrict__ bown, double* __restrict__ num,
    double* __restrict__ den, const double if1, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    const double w = bmagSf[f] * if1;
    atomicAdd(&num[bown[f]], w * qb[f]);
    atomicAdd(&den[bown[f]], w);
}

__global__ void gpu_ccf_avg_div(
    const double* __restrict__ num, const double* __restrict__ den,
    double* __restrict__ out, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    out[c] = (den[c] != 0.0) ? num[c] / den[c] : 0.0;
}

__global__ void gpu_ccf_blend1(
    const double* __restrict__ alpha, const double* __restrict__ avg1,
    double* __restrict__ x, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    double v = 1.02 * alpha[c] - 0.01;
    v = fmin(fmax(v, 1.0e-3), 1.0 - 1.0e-3);
    const double w = 0.95 + 0.1 * sqrt(v * (1.0 - v));
    x[c] = w * v + (1.0 - w) * avg1[c];
}

__global__ void gpu_ccf_blend2(
    const double* __restrict__ avgx, double* __restrict__ x,
    const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double v = x[c];
    const double w = 2.0 * sqrt(v * (1.0 - v));
    x[c] = w * v + (1.0 - w) * avgx[c];
}

__global__ void gpu_ccf_blend3(
    const double* __restrict__ avgx, double* __restrict__ x,
    const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    x[c] = 0.99 * x[c] + 0.01 * avgx[c];
}

__global__ void gpu_ccf_atanh(
    const double* __restrict__ x, double* __restrict__ a1s,
    const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    a1s[c] = atanh(1.8 * x[c] - 0.9);
}

}
'''
_MOD = None
_FN: dict = {}
KERNELS = ('gpu_alpha_gradf', 'gpu_grad_accumulate', 'gpu_grad_divide_volume', 'gpu_nhat', 'gpu_kappa_accumulate', 'gpu_kappa_finish', 'gpu_surface_tension_face', 'gpu_contact_angle', 'gpu_grad_accumulate_bnd', 'gpu_grad_accumulate_bnd_field', 'gpu_grad_accumulate_pbnd', 'gpu_kappa_accumulate_bnd', 'gpu_nhat_face', 'gpu_ccf_alpha_sh', 'gpu_ccf_delta_s', 'gpu_ccf_avg_acc', 'gpu_ccf_avg_acc_bnd', 'gpu_ccf_avg_div', 'gpu_ccf_blend1', 'gpu_ccf_blend2', 'gpu_ccf_blend3', 'gpu_ccf_atanh')

def module(cp):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))
    return _MOD

def fn(cp, name):
    if name not in KERNELS:
        raise KeyError('curvature kernels: unknown kernel %r' % name)
    f = _FN.get(name)
    if f is None:
        f = module(cp).get_function(name)
        _FN[name] = f
    return f

def grid(n, tpb=256):
    return ((int((n + tpb - 1) // tpb),), (int(tpb),))

def delta_n(mean_cell_volume: float) -> float:
    return 1e-12 / float(mean_cell_volume) ** 0.3333333333333333

def compile_all(cp) -> int:
    for k in KERNELS:
        fn(cp, k)
    return len(KERNELS)
