from __future__ import annotations
_SRC = '''
extern "C" {

__global__ void gpu_mom_assemble(
    const double* __restrict__ phi, const double* __restrict__ mu_f,
    const double* __restrict__ magSf, const double* __restrict__ dmag,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    double* __restrict__ upper, double* __restrict__ lower,
    double* __restrict__ diag, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double d = dmag[f];
    const double diff = (d > 0.0) ? mu_f[f] * magSf[f] / d : 0.0;
    const double q = phi[f];
    const double pos = (q > 0.0) ? q : 0.0;
    const double neg = (q < 0.0) ? q : 0.0;
    upper[f] = -diff + neg;
    lower[f] = -diff - pos;
    atomicAdd(&diag[o], diff + pos);
    atomicAdd(&diag[n], diff - neg);
}

__global__ void gpu_rau(const double* __restrict__ diag, double* __restrict__ rau,
                       const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double d = diag[c];
    rau[c] = (d != 0.0) ? 1.0 / d : 0.0;
}

__global__ void gpu_face_interp(
    const double* __restrict__ x, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ w,
    double* __restrict__ xf, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    xf[f] = w[f] * x[own[f]] + (1.0 - w[f]) * x[nei[f]];
}

__global__ void gpu_face_minmod(
    const double* __restrict__ x, const long long* __restrict__ own,
    const long long* __restrict__ nei, double* __restrict__ xf,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double a = x[own[f]], b = x[nei[f]];
    xf[f] = (fabs(a) < fabs(b)) ? a : b;
}

__global__ void gpu_offdiag_gather3(
    const double* __restrict__ upper, const double* __restrict__ lower,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ Ux, const double* __restrict__ Uy,
    const double* __restrict__ Uz,
    double* __restrict__ Hx, double* __restrict__ Hy, double* __restrict__ Hz,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double au = upper[f], al = lower[f];
    atomicAdd(&Hx[o], -au * Ux[n]);  atomicAdd(&Hx[n], -al * Ux[o]);
    atomicAdd(&Hy[o], -au * Uy[n]);  atomicAdd(&Hy[n], -al * Uy[o]);
    atomicAdd(&Hz[o], -au * Uz[n]);  atomicAdd(&Hz[n], -al * Uz[o]);
}

__global__ void gpu_hbya3(
    const double* __restrict__ rau,
    const double* __restrict__ Hx, const double* __restrict__ Hy,
    const double* __restrict__ Hz,
    const double* __restrict__ sx, const double* __restrict__ sy,
    const double* __restrict__ sz,
    double* __restrict__ hx, double* __restrict__ hy, double* __restrict__ hz,
    const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double r = rau[c];
    hx[c] = r * (sx[c] + Hx[c]);
    hy[c] = r * (sy[c] + Hy[c]);
    hz[c] = r * (sz[c] + Hz[c]);
}

__global__ void gpu_ddt_phi_corr(
    const double* __restrict__ phio, const double* __restrict__ ux,
    const double* __restrict__ uy, const double* __restrict__ uz,
    const double* __restrict__ rau, const double* __restrict__ rho,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ w, const double* __restrict__ Sfx,
    const double* __restrict__ Sfy, const double* __restrict__ Sfz,
    double* __restrict__ phiHbyA, const double rdt, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double wf = w[f], wn = 1.0 - w[f];
    const double sx = Sfx[f], sy = Sfy[f], sz = Sfz[f];
    const double phiU = (wf * ux[o] + wn * ux[n]) * sx
                      + (wf * uy[o] + wn * uy[n]) * sy
                      + (wf * uz[o] + wn * uz[n]) * sz;
    const double po = phio[f];
    const double coeff = 1.0 - fmin(fabs(po - phiU) / (fabs(po) + 1.0e-300),
                                    1.0);
    const double ao = rau[o] * rho[o], an = rau[n] * rho[n];
    const double arf = wf * ao + wn * an;
    const double phiRU = (wf * ao * ux[o] + wn * an * ux[n]) * sx
                       + (wf * ao * uy[o] + wn * an * uy[n]) * sy
                       + (wf * ao * uz[o] + wn * an * uz[n]) * sz;
    phiHbyA[f] += rdt * coeff * (arf * po - phiRU);
}

__global__ void gpu_phi_hbya(
    const double* __restrict__ hx, const double* __restrict__ hy,
    const double* __restrict__ hz, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ w,
    const double* __restrict__ Sfx, const double* __restrict__ Sfy,
    const double* __restrict__ Sfz, double* __restrict__ phiHbyA,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double a = w[f], b = 1.0 - w[f];
    phiHbyA[f] = (a * hx[o] + b * hx[n]) * Sfx[f]
               + (a * hy[o] + b * hy[n]) * Sfy[f]
               + (a * hz[o] + b * hz[n]) * Sfz[f];
}

__global__ void gpu_u_correct3(
    const double* __restrict__ hx, const double* __restrict__ hy,
    const double* __restrict__ hz, const double* __restrict__ rau_v,
    const double* __restrict__ gx, const double* __restrict__ gy,
    const double* __restrict__ gz,
    double* __restrict__ Ux, double* __restrict__ Uy, double* __restrict__ Uz,
    const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double r = rau_v[c];
    Ux[c] = hx[c] - r * gx[c];
    Uy[c] = hy[c] - r * gy[c];
    Uz[c] = hz[c] - r * gz[c];
}

__global__ void gpu_ddt_src3(
    const double* __restrict__ rho, const double* __restrict__ vol,
    const double* __restrict__ Ux0, const double* __restrict__ Uy0,
    const double* __restrict__ Uz0, const double rdt,
    double* __restrict__ sx, double* __restrict__ sy, double* __restrict__ sz,
    const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double a = rho[c] * vol[c] * rdt;
    sx[c] = a * Ux0[c];
    sy[c] = a * Uy0[c];
    sz[c] = a * Uz0[c];
}

__global__ void gpu_pressure_face_coeff(
    const double* __restrict__ rauf, const double* __restrict__ magSf,
    const double* __restrict__ dmag, double* __restrict__ af,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double d = dmag[f];
    af[f] = (d > 0.0) ? rauf[f] * magSf[f] / d : 0.0;
}

__global__ void gpu_div_face(
    const double* __restrict__ ff, const long long* __restrict__ own,
    const long long* __restrict__ nei, double* __restrict__ div,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    atomicAdd(&div[own[f]],  ff[f]);
    atomicAdd(&div[nei[f]], -ff[f]);
}

__global__ void gpu_flux_correct(
    const double* __restrict__ phiHbyA, const double* __restrict__ af,
    const double* __restrict__ p, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ nocorr,
    double* __restrict__ phi, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    phi[f] = phiHbyA[f] + nocorr[f] - af[f] * (p[nei[f]] - p[own[f]]);
}

__global__ void gpu_pd_filter(
    const double* __restrict__ pfc, double* __restrict__ p,
    const double fac, unsigned long long* __restrict__ nclip,
    const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double b = fabs(fac * pfc[c]);
    const double v = p[c];
    const double q = fmin(fmax(v, -b), b);
    if (q != v) atomicAdd(nclip, 1ULL);
    p[c] = q;
}

__global__ void gpu_u_filter(
    const double* __restrict__ fx, const double* __restrict__ fy,
    const double* __restrict__ fz, double* __restrict__ ux,
    double* __restrict__ uy, double* __restrict__ uz,
    const double fac, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double bx = fabs(fac * fx[c]);
    const double by = fabs(fac * fy[c]);
    const double bz = fabs(fac * fz[c]);
    ux[c] = fmin(fmax(ux[c], -bx), bx);
    uy[c] = fmin(fmax(uy[c], -by), by);
    uz[c] = fmin(fmax(uz[c], -bz), bz);
}

__global__ void gpu_bnd_div_add(
    const double* __restrict__ bphi, const long long* __restrict__ bown,
    double* __restrict__ src, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    atomicAdd(&src[bown[f]], bphi[f]);
}

__global__ void gpu_bnd_flux_correct(
    const double* __restrict__ rau, const double* __restrict__ p,
    const long long* __restrict__ bown, const double* __restrict__ bmag,
    const double* __restrict__ bdelta, const long long* __restrict__ bPt,
    double* __restrict__ bphi, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    if (bPt[f] != 1) return;
    const long long o = bown[f];
    bphi[f] += rau[o] * bmag[f] * bdelta[f] * p[o];
}

__global__ void gpu_u_face_resid(
    const double* __restrict__ p, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ magSf,
    const double* __restrict__ dmag, const double* __restrict__ force,
    const double fc, double* __restrict__ out, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    out[f] = fc * force[f] - (magSf[f] / dmag[f]) * (p[nei[f]] - p[own[f]]);
}

__global__ void gpu_u_add_recon(
    const double* __restrict__ hx, const double* __restrict__ hy,
    const double* __restrict__ hz, const double* __restrict__ rau,
    const double* __restrict__ cx, const double* __restrict__ cy,
    const double* __restrict__ cz,
    const double* __restrict__ gx, const double* __restrict__ gy,
    const double* __restrict__ gz, double* __restrict__ ux,
    double* __restrict__ uy, double* __restrict__ uz, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double ax = rau[c] * cx[c], ay = rau[c] * cy[c], az = rau[c] * cz[c];
    const int ok = isfinite(ax) && isfinite(ay) && isfinite(az);
    ux[c] = ok ? (hx[c] + ax) : (hx[c] - rau[c] * gx[c]);
    uy[c] = ok ? (hy[c] + ay) : (hy[c] - rau[c] * gy[c]);
    uz[c] = ok ? (hz[c] + az) : (hz[c] - rau[c] * gz[c]);
}

__global__ void gpu_bnd_u_resid(
    const double* __restrict__ p, const long long* __restrict__ bown,
    const double* __restrict__ bmag, const double* __restrict__ bdelta,
    const long long* __restrict__ bPt, double* __restrict__ out,
    const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    out[f] = (bPt[f] == 1) ? (bmag[f] * bdelta[f] * p[bown[f]]) : 0.0;
}

__global__ void gpu_mom_bnd_diag(
    const long long* __restrict__ bown, const double* __restrict__ bmag,
    const double* __restrict__ bdelta, const long long* __restrict__ bUt,
    const double mu_b, double* __restrict__ diag, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    if (bUt[f] != 0) return;
    atomicAdd(&diag[bown[f]], mu_b * bmag[f] * bdelta[f]);
}

__global__ void gpu_mom_bnd_cbc(
    const long long* __restrict__ bown, const double* __restrict__ bmag,
    const double* __restrict__ bdelta, const long long* __restrict__ bUt,
    const double* __restrict__ bddot,
    const double* __restrict__ gpx, const double* __restrict__ gpy,
    const double* __restrict__ gpz,
    const double* __restrict__ qcx, const double* __restrict__ qcy,
    const double* __restrict__ qcz,
    const double* __restrict__ ux, const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double mu_b, const double cbc, const double cap,
    double* __restrict__ diag, const long long nBF,
    double* __restrict__ probe)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    if (bUt[f] != 0) return;
    const long long o = bown[f];
    const double ex = gpx[o] - qcx[o];
    const double ey = gpy[o] - qcy[o];
    const double ez = gpz[o] - qcz[o];
    const double e = sqrt(ex*ex + ey*ey + ez*ez);
    const double u = sqrt(ux[o]*ux[o] + uy[o]*uy[o] + uz[o]*uz[o]);
    double fac = 1.0 + cbc * e * bddot[f] / ((u + 1e-16) * 1000.0 * mu_b);
    if (cap > 0.0 && fac > cap) fac = cap;
    atomicAdd(&diag[o], fac * mu_b * bmag[f] * bdelta[f]);
    if (probe != 0) {
        probe[f] = fac;
        probe[nBF + f] = u;
        probe[2*nBF + f] = sqrt(gpx[o]*gpx[o] + gpy[o]*gpy[o] + gpz[o]*gpz[o]);
        probe[3*nBF + f] = sqrt(qcx[o]*qcx[o] + qcy[o]*qcy[o] + qcz[o]*qcz[o]);
    }
}

__global__ void gpu_mom_bnd_inlet(
    const long long* __restrict__ bown, const double* __restrict__ bmag,
    const double* __restrict__ bdelta, const long long* __restrict__ bUt,
    const double* __restrict__ bSfx, const double* __restrict__ bSfy,
    const double* __restrict__ bSfz, const double* __restrict__ bphi,
    const double* __restrict__ bddot,
    const double* __restrict__ gpx, const double* __restrict__ gpy,
    const double* __restrict__ gpz,
    const double* __restrict__ qcx, const double* __restrict__ qcy,
    const double* __restrict__ qcz,
    const double* __restrict__ ux, const double* __restrict__ uy,
    const double* __restrict__ uz,
    const double mu_b, const double cbc, const double cap,
    double* __restrict__ diag,
    double* __restrict__ sx, double* __restrict__ sy,
    double* __restrict__ sz, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    if (bUt[f] != 1) return;
    const double s = bmag[f];
    if (!(s > 0.0)) return;
    double fac = 1.0;
    if (cbc > 0.0) {
        const long long oc = bown[f];
        const double ex = gpx[oc] - qcx[oc];
        const double ey = gpy[oc] - qcy[oc];
        const double ez = gpz[oc] - qcz[oc];
        const double e = sqrt(ex*ex + ey*ey + ez*ez);
        const double u = sqrt(ux[oc]*ux[oc] + uy[oc]*uy[oc] + uz[oc]*uz[oc]);
        fac = 1.0 + cbc * e * bddot[f] / ((u + 1e-16) * 1000.0 * mu_b);
        if (cap > 0.0 && fac > cap) fac = cap;
    }
    const double a = fac * mu_b * s * bdelta[f];
    const long long o = bown[f];
    atomicAdd(&diag[o], a);
    const double ub = bphi[f] / (s * s);
    atomicAdd(&sx[o], a * ub * bSfx[f]);
    atomicAdd(&sy[o], a * ub * bSfy[f]);
    atomicAdd(&sz[o], a * ub * bSfz[f]);
}

__global__ void gpu_bnd_flux_nb(
    const double* __restrict__ hx, const double* __restrict__ hy,
    const double* __restrict__ hz, const long long* __restrict__ bown,
    const double* __restrict__ bSfx, const double* __restrict__ bSfy,
    const double* __restrict__ bSfz, const long long* __restrict__ bUt,
    double* __restrict__ bphi, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    const long long ty = bUt[f];
    if (ty == 1) return;
    const long long o = bown[f];
    if (ty == 2) bphi[f] = hx[o]*bSfx[f] + hy[o]*bSfy[f] + hz[o]*bSfz[f];
    else         bphi[f] = 0.0;
}

}
'''
_MOD = None
_FN: dict = {}
KERNELS = ('gpu_mom_assemble', 'gpu_rau', 'gpu_face_interp', 'gpu_pressure_face_coeff', 'gpu_div_face', 'gpu_flux_correct', 'gpu_offdiag_gather3', 'gpu_hbya3', 'gpu_phi_hbya', 'gpu_ddt_phi_corr', 'gpu_u_correct3', 'gpu_ddt_src3', 'gpu_u_filter', 'gpu_pd_filter', 'gpu_face_minmod', 'gpu_bnd_div_add', 'gpu_bnd_flux_correct', 'gpu_u_face_resid', 'gpu_u_add_recon', 'gpu_bnd_u_resid', 'gpu_mom_bnd_diag', 'gpu_mom_bnd_inlet', 'gpu_mom_bnd_cbc', 'gpu_bnd_flux_nb')

def module(cp):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))
    return _MOD

def fn(cp, name):
    if name not in KERNELS:
        raise KeyError('momentum kernels: unknown kernel %r' % name)
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
