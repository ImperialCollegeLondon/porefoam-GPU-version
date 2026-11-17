from __future__ import annotations
_SRC = '''
extern "C" {

__global__ void gpu_blend_props(
    const double* __restrict__ alpha, double* __restrict__ rho,
    double* __restrict__ mu, const double rho1, const double rho2,
    const double mu1, const double mu2, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double a = alpha[c];
    rho[c] = a * rho1 + (1.0 - a) * rho2;
    mu[c]  = a * mu1  + (1.0 - a) * mu2;
}

__global__ void gpu_prop_face(
    const double* __restrict__ x, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ w,
    double* __restrict__ xf, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    xf[f] = w[f] * x[own[f]] + (1.0 - w[f]) * x[nei[f]];
}

__global__ void gpu_nonorth_correct(
    const double* __restrict__ gx, const double* __restrict__ gy,
    const double* __restrict__ gz, const double* __restrict__ Sfx,
    const double* __restrict__ Sfy, const double* __restrict__ Sfz,
    const double* __restrict__ dx, const double* __restrict__ dy,
    const double* __restrict__ dz, const double* __restrict__ dmag,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ w, const double* __restrict__ rauf,
    const double* __restrict__ p, const double* __restrict__ af,
    const double limit, double* __restrict__ corr, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double wf = w[f], wm = 1.0 - wf;
    const double fx = wf*gx[o] + wm*gx[n];
    const double fy = wf*gy[o] + wm*gy[n];
    const double fz = wf*gz[o] + wm*gz[n];
    const double L = dmag[f];
    if (L <= 0.0) { corr[f] = 0.0; return; }
    const double ux = dx[f]/L, uy = dy[f]/L, uz = dz[f]/L;
    const double SdotU = Sfx[f]*ux + Sfy[f]*uy + Sfz[f]*uz;
    const double kx = Sfx[f] - SdotU*ux;
    const double ky = Sfy[f] - SdotU*uy;
    const double kz = Sfz[f] - SdotU*uz;
    double c = rauf[f] * (fx*kx + fy*ky + fz*kz);
    const double ortho = af[f] * (p[n] - p[o]);
    const double cap = limit * fabs(ortho);
    if (c >  cap) c =  cap;
    if (c < -cap) c = -cap;
    corr[f] = c;
}


__global__ void gpu_add_face_force(
    double* __restrict__ phiHbyA, const double* __restrict__ rauf,
    const double* __restrict__ Fface, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    phiHbyA[f] += rauf[f] * Fface[f];
}


__global__ void gpu_add_ddt_diag(
    double* __restrict__ diag, const double* __restrict__ rho,
    const double* __restrict__ vol, const double dt, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    diag[c] += rho[c] * vol[c] / dt;
}


__global__ void gpu_rau_per_volume(
    const double* __restrict__ diag, const double* __restrict__ vol,
    double* __restrict__ rau_v, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double d = diag[c];
    rau_v[c] = (d != 0.0) ? vol[c] / d : 0.0;
}

}
'''
_MOD = None
_FN: dict = {}
KERNELS = ('gpu_blend_props', 'gpu_prop_face', 'gpu_nonorth_correct', 'gpu_add_face_force', 'gpu_add_ddt_diag', 'gpu_rau_per_volume')

def module(cp):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))
    return _MOD

def fn(cp, name):
    if name not in KERNELS:
        raise KeyError('fluid property kernels: unknown kernel %r' % name)
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
