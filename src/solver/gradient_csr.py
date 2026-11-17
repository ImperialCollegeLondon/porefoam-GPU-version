from __future__ import annotations
import numpy as np
DEV = None

def install(d):
    global DEV
    DEV = d


_SRC = '''
extern "C" {

__global__ void gpu_gradient_ls_acc(
    const double* __restrict__ psi, const long long* __restrict__ own,
    const long long* __restrict__ nei,
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ pz,
    const double* __restrict__ nx, const double* __restrict__ ny,
    const double* __restrict__ nz,
    double* __restrict__ gx, double* __restrict__ gy, double* __restrict__ gz,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double dp = psi[n] - psi[o];
    atomicAdd(&gx[o], px[f] * dp);  atomicAdd(&gx[n], -nx[f] * dp);
    atomicAdd(&gy[o], py[f] * dp);  atomicAdd(&gy[n], -ny[f] * dp);
    atomicAdd(&gz[o], pz[f] * dp);  atomicAdd(&gz[n], -nz[f] * dp);
}

__global__ void gpu_gradient_ls_acc_bnd(
    const double* __restrict__ psi, const long long* __restrict__ bown,
    const long long* __restrict__ bPt,
    const double* __restrict__ bvalues,
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ pz,
    double* __restrict__ gx, double* __restrict__ gy, double* __restrict__ gz,
    const int bmode, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    const long long o = bown[f];
    const double pb = (bmode == 1 && bPt[f] == 1) ? bvalues[f] : psi[o];
    const double dp = pb - psi[o];
    if (dp == 0.0) return;
    atomicAdd(&gx[o], px[f] * dp);
    atomicAdd(&gy[o], py[f] * dp);
    atomicAdd(&gz[o], pz[f] * dp);
}

__device__ __forceinline__ int gpu_gradient_limitface(
    double& gx, double& gy, double& gz,
    const double maxDelta, const double minDelta,
    const double dx, const double dy, const double dz)
{
    const double ex = dx * gx + dy * gy + dz * gz;
    if (fabs(ex) < 1.0e-15) return 0;
    const double m2 = dx * dx + dy * dy + dz * dz;
    double s;
    if (ex > maxDelta)      s = (maxDelta - ex) / m2;
    else if (ex < minDelta) s = (minDelta - ex) / m2;
    else                    return 0;
    gx += dx * s; gy += dy * s; gz += dz * s;
    return 1;
}

__global__ void gpu_gradient_facemd(
    const double* __restrict__ psi, const long long* __restrict__ bown,
    const long long* __restrict__ bPt,
    const double* __restrict__ bvalues,
    const long long* __restrict__ ir, const long long* __restrict__ nbr,
    const double* __restrict__ dcx, const double* __restrict__ dcy,
    const double* __restrict__ dcz,
    const long long* __restrict__ br, const long long* __restrict__ bface,
    const double* __restrict__ bdx, const double* __restrict__ bdy,
    const double* __restrict__ bdz,
    double* __restrict__ gx, double* __restrict__ gy, double* __restrict__ gz,
    const double rk, const int bmode,
    long long* __restrict__ ncell, long long* __restrict__ nface,
    const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double pc = psi[c];
    const double b0x = gx[c], b0y = gy[c], b0z = gz[c];
    double ax = b0x, ay = b0y, az = b0z;
    long long moved = 0;
    for (long long k = ir[c]; k < ir[c + 1]; ++k)
    {
        const double pn = psi[nbr[k]];
        double mx = fmax(pc, pn), mn = fmin(pc, pn);
        const double mm = rk * (mx - mn);
        mx += mm;  mn -= mm;
        moved += gpu_gradient_limitface(ax, ay, az, mx - pc, mn - pc,
                                   dcx[k], dcy[k], dcz[k]);
    }
    if (bmode == 1)
    {
        for (long long k = br[c]; k < br[c + 1]; ++k)
        {
            const long long bf = bface[k];
            if (bPt[bf] != 1) continue;
            const double pn = bvalues[bf];
            double mx = fmax(pc, pn), mn = fmin(pc, pn);
            const double mm = rk * (mx - mn);
            mx += mm;  mn -= mm;
            moved += gpu_gradient_limitface(ax, ay, az, mx - pc, mn - pc,
                                       bdx[k], bdy[k], bdz[k]);
        }
    }
    gx[c] = ax; gy[c] = ay; gz[c] = az;
    if (moved)
    {
        atomicAdd((unsigned long long*)nface, (unsigned long long)moved);
        if (ax != b0x || ay != b0y || az != b0z)
            atomicAdd((unsigned long long*)ncell, 1ull);
    }
}

}
'''
KERNELS = ('gpu_gradient_ls_acc', 'gpu_gradient_ls_acc_bnd', 'gpu_gradient_facemd')
_MOD = None
_FN = {}

def module(cp):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=_SRC, options=('--fmad=false', '-std=c++14'))
    return _MOD

def fn(cp, name):
    if name not in KERNELS:
        raise KeyError('gradient kernels: unknown kernel %r' % name)
    f = _FN.get(name)
    if f is None:
        f = module(cp).get_function(name)
        _FN[name] = f
    return f

def grid(n, tpb=256):
    return ((int((n + tpb - 1) // tpb),), (int(tpb),))

def build_host(own, nei, vol, Sf, ctr, bown, bSf, bCf, Cfi, plo, verbose=True):
    nC = int(vol.size)
    nIF = int(own.size)
    nBF = int(bown.size)
    assert np.all(np.diff(np.asarray(plo)) > 0), "patches are not in ascending startFace order; the CSR's 'ascending global face index' claim would be false"
    d = ctr[nei] - ctr[own]
    im2 = (d * d).sum(axis=1)
    bd = bCf - ctr[bown]
    bm2 = (bd * bd).sum(axis=1)
    dd = np.zeros((nC, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i, 3):
            v = np.bincount(own, weights=d[:, i] * d[:, j] / im2, minlength=nC) + np.bincount(nei, weights=d[:, i] * d[:, j] / im2, minlength=nC) + np.bincount(bown, weights=bd[:, i] * bd[:, j] / bm2, minlength=nC)
            dd[:, i, j] = v
            dd[:, j, i] = v
    det = np.linalg.det(dd)
    sing = det <= 1e-15
    invDd = np.empty_like(dd)
    invDd[~sing] = np.linalg.inv(dd[~sing])
    if sing.any():
        ev, evec = np.linalg.eigh(dd[sing])
        sub = dd[sing]
        zi = np.zeros_like(sub)
        ez = np.abs(ev[:, 2])
        for cmpt in (0, 1):
            msk = ez > 10000000000.0 * np.abs(ev[:, cmpt])
            if msk.any():
                q = evec[msk][:, :, cmpt]
                zi[msk] += q[:, :, None] * q[:, None, :]
        allz = ez == 10000000000.0 * ez
        s2 = sub + zi
        good = (np.linalg.det(s2) >= 1e-300) & ~allz
        out = np.zeros_like(sub)
        if good.any():
            out[good] = np.linalg.inv(s2[good]) - zi[good]
        invDd[sing] = out
    lsP = np.einsum('fij,fj->fi', invDd[own], d) / im2[:, None]
    lsN = -np.einsum('fij,fj->fi', invDd[nei], d) / im2[:, None]
    blsP = np.einsum('fij,fj->fi', invDd[bown], bd) / bm2[:, None]
    magSf = np.sqrt((Sf * Sf).sum(axis=1))
    bmagSf = np.sqrt((bSf * bSf).sum(axis=1))
    cP = (lsP * Sf).sum(axis=1) / (np.sqrt((lsP * lsP).sum(axis=1)) * magSf)
    cN = (lsN * Sf).sum(axis=1) / (np.sqrt((lsN * lsN).sum(axis=1)) * magSf)
    cB = (blsP * bSf).sum(axis=1) / (np.sqrt((blsP * blsP).sum(axis=1)) * bmagSf)
    ugg = np.zeros(nC, dtype=bool)
    ugg[own[cP < 0.1]] = True
    ugg[nei[cN > -0.1]] = True
    ugg[bown[cB < 0.1]] = True
    n_fb = int(ugg.sum())
    zdo, zdn = (Cfi - ctr[own], ctr[nei] - Cfi)
    w = np.abs((Sf * zdn).sum(axis=1)) / np.maximum(np.abs((Sf * zdo).sum(axis=1)) + np.abs((Sf * zdn).sum(axis=1)), 1e-300)
    if n_fb:
        mo, mn_ = (ugg[own], ugg[nei])
        lsP[mo] = ((1.0 - w)[:, None] * Sf / vol[own][:, None])[mo]
        lsN[mn_] = (-w[:, None] * Sf / vol[nei][:, None])[mn_]
        mb = ugg[bown]
        blsP[mb] = (bSf / vol[bown][:, None])[mb]
    cellkey = np.concatenate([own, nei])
    faceid = np.concatenate([np.arange(nIF), np.arange(nIF)])
    sidearr = np.concatenate([np.zeros(nIF, np.int8), np.ones(nIF, np.int8)])
    o2 = np.lexsort((faceid, cellkey))
    face_of, side, ck = (faceid[o2], sidearr[o2], cellkey[o2])
    ir = np.zeros(nC + 1, dtype=np.int64)
    np.cumsum(np.bincount(ck, minlength=nC), out=ir[1:])
    nbr = np.where(side == 0, nei[face_of], own[face_of]).astype(np.int64)
    dcf = Cfi[face_of] - ctr[ck]
    bord = np.lexsort((np.arange(nBF), bown))
    bface = np.arange(nBF)[bord].astype(np.int64)
    br = np.zeros(nC + 1, dtype=np.int64)
    np.cumsum(np.bincount(bown, minlength=nC), out=br[1:])
    bdcf = bCf[bface] - ctr[bown[bface]]
    same = np.diff(ck) == 0
    viol = int(((np.diff(face_of) <= 0) & same).sum())
    assert viol == 0, 'CSR is not in ascending face order (%d violations)' % viol
    rep = dict(n_gauss_fallback=n_fb, n_hinv_singular=int(sing.sum()), det_min=float(det.min()), det_max=float(det.max()), cos_min_internal=float(min(cP.min(), -cN.max())), cos_min_boundary=float(cB.min()), tol=0.1, nnz_int=int(face_of.size), nnz_bnd=int(bface.size))
    return (dict(lsP=lsP, lsN=lsN, blsP=blsP, ir=ir, nbr=nbr, dcf=dcf, br=br, bface=bface, bdcf=bdcf, ugg=ugg), rep)

def to_device(cp, h):

    def f(a):
        return cp.asarray(np.ascontiguousarray(a, dtype=np.float64))

    def i(a):
        return cp.asarray(np.ascontiguousarray(a, dtype=np.int64))
    return dict(lsPx=f(h['lsP'][:, 0]), lsPy=f(h['lsP'][:, 1]), lsPz=f(h['lsP'][:, 2]), lsNx=f(h['lsN'][:, 0]), lsNy=f(h['lsN'][:, 1]), lsNz=f(h['lsN'][:, 2]), blsPx=f(h['blsP'][:, 0]), blsPy=f(h['blsP'][:, 1]), blsPz=f(h['blsP'][:, 2]), ir=i(h['ir']), nbr=i(h['nbr']), br=i(h['br']), bface=i(h['bface']), dcx=f(h['dcf'][:, 0]), dcy=f(h['dcf'][:, 1]), dcz=f(h['dcf'][:, 2]), bdx=f(h['bdcf'][:, 0]), bdy=f(h['bdcf'][:, 1]), bdz=f(h['bdcf'][:, 2]), bzero=cp.zeros(int(h['bface'].size), dtype=cp.int64), ncell=cp.zeros(1, dtype=cp.int64), nface=cp.zeros(1, dtype=cp.int64), calls=cp.zeros(1, dtype=cp.int64))

def calc_grad(cp, Z, own, nei, bown, bPt, psi, gx, gy, gz, bmode, nC, nIF, nBF, bvalues=None):
    if bmode == 1 and bvalues is None:
        raise ValueError('Actual pd boundary values required')
    gc, bc = grid(nC)
    gf, bf = grid(nIF)
    gx.fill(0.0)
    gy.fill(0.0)
    gz.fill(0.0)
    fn(cp, 'gpu_gradient_ls_acc')(gf, bf, (psi, own, nei, Z['lsPx'], Z['lsPy'], Z['lsPz'], Z['lsNx'], Z['lsNy'], Z['lsNz'], gx, gy, gz, nIF))
    if bmode == 1 and nBF:
        gb, bb = grid(nBF)
        fn(cp, 'gpu_gradient_ls_acc_bnd')(gb, bb, (psi, bown, bPt, bvalues, Z['blsPx'], Z['blsPy'], Z['blsPz'], gx, gy, gz, cp.int32(bmode), nBF))
    fn(cp, 'gpu_gradient_facemd')(gc, bc, (psi, bown, bPt if bPt is not None else Z['bzero'], bvalues if bvalues is not None else psi, Z['ir'], Z['nbr'], Z['dcx'], Z['dcy'], Z['dcz'], Z['br'], Z['bface'], Z['bdx'], Z['bdy'], Z['bdz'], gx, gy, gz, cp.float64(1.0), cp.int32(bmode), Z['ncell'], Z['nface'], nC))
    Z['calls'] += 1
    return (gx, gy, gz)
