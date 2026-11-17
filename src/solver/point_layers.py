from __future__ import annotations
import hashlib
import os
import numpy as np
SCHEMA = 'p1:points,facePoints,faceCentres,nPoints'
KERNELS = '''
extern "C" {

__global__ void gpu_p_interp_acc(
    const long long* __restrict__ fp,
    const double* __restrict__ cfx, const double* __restrict__ cfy,
    const double* __restrict__ cfz,
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ pz,
    const double* __restrict__ af,
    const double wscale,
    double* __restrict__ num, double* __restrict__ den,
    const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    const long long f = f0 + i;
    const double a = af[i];
    for (int k = 0; k < 4; ++k) {
        const long long p = fp[4 * f + k];
        const double dx = cfx[f] - px[p];
        const double dy = cfy[f] - py[p];
        const double dz = cfz[f] - pz[p];
        const double w = wscale * sqrt(dx * dx + dy * dy + dz * dz);
        atomicAdd(&num[p], w * a);
        atomicAdd(&den[p], w);
    }
}

__global__ void gpu_p_interp_div(
    const double* __restrict__ num, const double* __restrict__ den,
    double* __restrict__ ap, const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    ap[p] = num[p] / (den[p] + 1.0e-37);
}

__global__ void gpu_p_interf_mark(
    const long long* __restrict__ fp, const double* __restrict__ magDelS,
    double* __restrict__ mark, const double contrast,
    const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    if (magDelS[i] <= contrast) return;
    const long long f = f0 + i;
    for (int k = 0; k < 4; ++k) mark[fp[4 * f + k]] = 1.0;
}

__global__ void gpu_p_gg_acc(
    const long long* __restrict__ fp,
    const double* __restrict__ cfx, const double* __restrict__ cfy,
    const double* __restrict__ cfz,
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ pz, const double* __restrict__ magSf,
    const double wscale, double* __restrict__ gg,
    const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    const long long f = f0 + i;
    const double wf = wscale * magSf[i] * magSf[i];
    for (int k = 0; k < 4; ++k) {
        const long long p = fp[4 * f + k];
        const double dx = cfx[f] - px[p];
        const double dy = cfy[f] - py[p];
        const double dz = cfz[f] - pz[p];
        const double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= 0.0) continue;
        const double w = wf / d2;
        atomicAdd(&gg[6 * p + 0], w * dx * dx);
        atomicAdd(&gg[6 * p + 1], w * dx * dy);
        atomicAdd(&gg[6 * p + 2], w * dx * dz);
        atomicAdd(&gg[6 * p + 3], w * dy * dy);
        atomicAdd(&gg[6 * p + 4], w * dy * dz);
        atomicAdd(&gg[6 * p + 5], w * dz * dz);
    }
}

__global__ void gpu_p_gg_inv(
    double* __restrict__ gg, const double* __restrict__ mark,
    double* __restrict__ inv, const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    for (int k = 0; k < 6; ++k) inv[6 * p + k] = 0.0;
    if (mark[p] <= 0.0) return;
    const double a = gg[6*p+0] + 1.0e-36, b = gg[6*p+1], c = gg[6*p+2];
    const double d = gg[6*p+3] + 1.0e-36, e = gg[6*p+4];
    const double g = gg[6*p+5] + 1.0e-36;
    const double c00 = d*g - e*e, c01 = c*e - b*g, c02 = b*e - c*d;
    const double det = a*c00 + b*c01 + c*c02;
    if (det == 0.0) return;
    const double r = 1.0 / det;
    inv[6*p+0] = c00 * r;
    inv[6*p+1] = c01 * r;
    inv[6*p+2] = c02 * r;
    inv[6*p+3] = (a*g - c*c) * r;
    inv[6*p+4] = (b*c - a*e) * r;
    inv[6*p+5] = (a*d - b*b) * r;
}

__global__ void gpu_p_grad_acc(
    const long long* __restrict__ fp,
    const double* __restrict__ cfx, const double* __restrict__ cfy,
    const double* __restrict__ cfz,
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ pz, const double* __restrict__ magSf,
    const double* __restrict__ af, const double* __restrict__ ap,
    const double* __restrict__ inv, const double wscale,
    double* __restrict__ gx, double* __restrict__ gy,
    double* __restrict__ gz, const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    const long long f = f0 + i;
    const double wf = wscale * magSf[i] * magSf[i];
    const double a = af[i];
    for (int k = 0; k < 4; ++k) {
        const long long p = fp[4 * f + k];
        const double dx = cfx[f] - px[p];
        const double dy = cfy[f] - py[p];
        const double dz = cfz[f] - pz[p];
        const double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= 0.0) continue;
        const double w = wf / d2;
        const double da = a - ap[p];
        const double i0 = inv[6*p+0], i1 = inv[6*p+1], i2 = inv[6*p+2];
        const double i3 = inv[6*p+3], i4 = inv[6*p+4], i5 = inv[6*p+5];
        atomicAdd(&gx[p], w * da * (i0*dx + i1*dy + i2*dz));
        atomicAdd(&gy[p], w * da * (i1*dx + i3*dy + i4*dz));
        atomicAdd(&gz[p], w * da * (i2*dx + i4*dy + i5*dz));
    }
}

__global__ void gpu_p_smooth_acc(
    const long long* __restrict__ fp, const long long* __restrict__ fe,
    const double* __restrict__ marks, const double* __restrict__ magDelS,
    const double* __restrict__ cfx, const double* __restrict__ cfy,
    const double* __restrict__ cfz,
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ pz,
    const double* __restrict__ nx, const double* __restrict__ ny,
    const double* __restrict__ nz,
    double* __restrict__ ax, double* __restrict__ ay, double* __restrict__ az,
    const double contrast, const double wscale,
    const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    const double ds = magDelS[i];
    if (ds <= contrast) return;
    const long long f = f0 + i;
    for (int k = 0; k < 4; ++k) {
        const long long p1 = fp[4 * f + k];
        const long long p2 = fp[4 * f + ((k + 1) & 3)];
        const double bp1 = marks[fe[4 * f + ((k + 3) & 3)]];
        const double bp2 = marks[fe[4 * f + ((k + 1) & 3)]];
        const double w1 = 1.0 + 2.0 * bp1;
        const double w2 = 1.0 + 2.0 * bp2;
        double ex = 0.5 * (nx[p2] * w2 + nx[p1] * w1);
        double ey = 0.5 * (ny[p2] * w2 + ny[p1] * w1);
        double ez = 0.5 * (nz[p2] * w2 + nz[p1] * w1);
        const double em = sqrt(ex * ex + ey * ey + ez * ez);
        if (em <= 0.0) continue;
        ex /= em; ey /= em; ez /= em;
        const double e1x = px[p2] - px[p1];
        const double e1y = py[p2] - py[p1];
        const double e1z = pz[p2] - pz[p1];
        const double elen = sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
        if (elen <= 0.0) continue;
        const double area = wscale * ds / elen;
        const double d2x = cfx[f] - px[p2], d2y = cfy[f] - py[p2],
                     d2z = cfz[f] - pz[p2];
        const double d1x = cfx[f] - px[p1], d1y = cfy[f] - py[p1],
                     d1z = cfz[f] - pz[p1];
        const double r2 = sqrt(d2x * d2x + d2y * d2y + d2z * d2z);
        const double r1 = sqrt(d1x * d1x + d1y * d1y + d1z * d1z);
        const double s1 = area * r2, s2 = area * r1;
        atomicAdd(&ax[p1], s1 * ex);  atomicAdd(&ay[p1], s1 * ey);
        atomicAdd(&az[p1], s1 * ez);
        atomicAdd(&ax[p2], s2 * ex);  atomicAdd(&ay[p2], s2 * ey);
        atomicAdd(&az[p2], s2 * ez);
    }
}

__global__ void gpu_p_smooth_update(
    double* __restrict__ nx, double* __restrict__ ny, double* __restrict__ nz,
    const double* __restrict__ ax, const double* __restrict__ ay,
    const double* __restrict__ az, const double* __restrict__ mark,
    const double relax, const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    if (mark[p] <= 0.0) return;
    const double am = sqrt(ax[p]*ax[p] + ay[p]*ay[p] + az[p]*az[p]) + 1.0e-18;
    double vx = (1.0 - relax) * nx[p] + relax * ax[p] / am;
    double vy = (1.0 - relax) * ny[p] + relax * ay[p] / am;
    double vz = (1.0 - relax) * nz[p] + relax * az[p] / am;
    const double vm = sqrt(vx*vx + vy*vy + vz*vz) + 1.0e-14;
    nx[p] = vx / vm;  ny[p] = vy / vm;  nz[p] = vz / vm;
}

__global__ void gpu_p_renorm(
    double* __restrict__ nx, double* __restrict__ ny, double* __restrict__ nz,
    const double eps, const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    const double m = sqrt(nx[p]*nx[p] + ny[p]*ny[p] + nz[p]*nz[p]) + eps;
    nx[p] /= m;  ny[p] /= m;  nz[p] /= m;
}

__global__ void gpu_p_interface_dist(
    const double* __restrict__ ap, const double* __restrict__ gmag,
    const double* __restrict__ nx, const double* __restrict__ ny,
    const double* __restrict__ nz, const double* __restrict__ mark,
    double* __restrict__ dx, double* __restrict__ dy, double* __restrict__ dz,
    const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    if (mark[p] <= 0.0) { dx[p] = 0.0; dy[p] = 0.0; dz[p] = 0.0; return; }
    const double s = (0.0 - ap[p]) / gmag[p];
    dx[p] = s * nx[p];  dy[p] = s * ny[p];  dz[p] = s * nz[p];
}

__global__ void gpu_p_interface_sf(
    const long long* __restrict__ fp, const double* __restrict__ px,
    const double* __restrict__ py, const double* __restrict__ pz,
    const double* __restrict__ dx, const double* __restrict__ dy,
    const double* __restrict__ dz, const double* __restrict__ deltaS,
    double* __restrict__ sfx, double* __restrict__ sfy,
    double* __restrict__ sfz, const double contrast,
    const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    const double ds = deltaS[i];
    const double m = fabs(ds);
    if (m <= contrast) { sfx[i] = 0.0; sfy[i] = 0.0; sfz[i] = 0.0; return; }
    const double rhc = ds / m;
    const long long f = f0 + i;
    double X[4], Y[4], Z[4], cx = 0.0, cy = 0.0, cz = 0.0;
    for (int k = 0; k < 4; ++k) {
        const long long q = fp[4 * f + k];
        X[k] = px[q] + dx[q];  Y[k] = py[q] + dy[q];  Z[k] = pz[q] + dz[q];
        cx += X[k];  cy += Y[k];  cz += Z[k];
    }
    cx *= 0.25;  cy *= 0.25;  cz *= 0.25;
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int k = 0; k < 4; ++k) {
        const int n = (k + 1) & 3;
        const double ux = X[n] - X[k], uy = Y[n] - Y[k], uz = Z[n] - Z[k];
        const double vx = cx - X[k],   vy = cy - Y[k],   vz = cz - Z[k];
        ax += 0.5 * (uy * vz - uz * vy);
        ay += 0.5 * (uz * vx - ux * vz);
        az += 0.5 * (ux * vy - uy * vx);
    }
    sfx[i] = rhc * ax;  sfy[i] = rhc * ay;  sfz[i] = rhc * az;
}

__global__ void gpu_p_contact_point(
    const double* __restrict__ wpx, const double* __restrict__ wpy,
    const double* __restrict__ wpz,
    const double* __restrict__ wmark,
    const double* __restrict__ mark,
    double* __restrict__ nx, double* __restrict__ ny, double* __restrict__ nz,
    const double sinth, const double costh, const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    if (wmark[p] <= 0.0 || mark[p] <= 0.0) return;
    const double gx = nx[p], gy = ny[p], gz = nz[p];
    const double gm = sqrt(gx*gx + gy*gy + gz*gz);
    if (gm <= 0.0) return;
    const double hx = gx / (gm + 1.0e-37);
    const double hy = gy / (gm + 1.0e-37);
    const double hz = gz / (gm + 1.0e-37);
    const double wx = wpx[p], wy = wpy[p], wz = wpz[p];
    const double hw = hx*wx + hy*wy + hz*wz;
    double sx = hx - hw*wx, sy = hy - hw*wy, sz = hz - hw*wz;
    const double sm = sqrt(sx*sx + sy*sy + sz*sz) + 1.0e-37;
    sx /= sm; sy /= sm; sz /= sm;
    nx[p] = gm * (sinth*sx + costh*wx);
    ny[p] = gm * (sinth*sy + costh*wy);
    nz[p] = gm * (sinth*sz + costh*wz);
}

__global__ void gpu_p_w_f2p(
    const long long* __restrict__ fp, const long long* __restrict__ wf,
    const double* __restrict__ cfx, const double* __restrict__ cfy,
    const double* __restrict__ cfz,
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ pz,
    const double* __restrict__ vx, const double* __restrict__ vy,
    const double* __restrict__ vz,
    double* __restrict__ nux, double* __restrict__ nuy,
    double* __restrict__ nuz, double* __restrict__ den, const long long nW)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nW) return;
    const long long f = wf[i];
    for (int k = 0; k < 4; ++k) {
        const long long p = fp[4 * f + k];
        const double dx = cfx[f] - px[p], dy = cfy[f] - py[p],
                     dz = cfz[f] - pz[p];
        const double r = sqrt(dx*dx + dy*dy + dz*dz);
        const double w = 1.0 / (r + 1.0e-300);
        atomicAdd(&nux[p], w * vx[i]);  atomicAdd(&nuy[p], w * vy[i]);
        atomicAdd(&nuz[p], w * vz[i]);  atomicAdd(&den[p], w);
    }
}

__global__ void gpu_p_w_p2f(
    const long long* __restrict__ fp, const long long* __restrict__ wf,
    const double* __restrict__ nux, const double* __restrict__ nuy,
    const double* __restrict__ nuz,
    double* __restrict__ vx, double* __restrict__ vy,
    double* __restrict__ vz, const long long nW)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nW) return;
    const long long f = wf[i];
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int k = 0; k < 4; ++k) {
        const long long p = fp[4 * f + k];
        ax += nux[p]; ay += nuy[p]; az += nuz[p];
    }
    vx[i] = 0.25 * ax; vy[i] = 0.25 * ay; vz[i] = 0.25 * az;
}

__global__ void gpu_p_w_div(
    const double* __restrict__ nux, const double* __restrict__ nuy,
    const double* __restrict__ nuz, const double* __restrict__ den,
    double* __restrict__ ox, double* __restrict__ oy,
    double* __restrict__ oz, double* __restrict__ wmark, const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    const double d = den[p];
    if (d <= 0.0) { ox[p] = 0.0; oy[p] = 0.0; oz[p] = 0.0; wmark[p] = 0.0;
                    return; }
    ox[p] = nux[p] / d; oy[p] = nuy[p] / d; oz[p] = nuz[p] / d;
    wmark[p] = 1.0;
}

__global__ void gpu_p_w_renorm(
    double* __restrict__ vx, double* __restrict__ vy,
    double* __restrict__ vz, const long long nW)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nW) return;
    const double m = sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]) + 1.0e-15;
    vx[i] /= m; vy[i] /= m; vz[i] /= m;
}

__global__ void gpu_p_curv_force(
    const long long* __restrict__ fp, const double* __restrict__ px,
    const double* __restrict__ py, const double* __restrict__ pz,
    const double* __restrict__ dx, const double* __restrict__ dy,
    const double* __restrict__ dz,
    const double* __restrict__ nx, const double* __restrict__ ny,
    const double* __restrict__ nz,
    const double* __restrict__ gx, const double* __restrict__ gy,
    const double* __restrict__ gz, const double* __restrict__ deltaS,
    double* __restrict__ cfx, double* __restrict__ cfy,
    double* __restrict__ cfz, double* __restrict__ nvx,
    double* __restrict__ nvy, double* __restrict__ nvz,
    const double contrast, const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    cfx[i] = 0.0; cfy[i] = 0.0; cfz[i] = 0.0;
    nvx[i] = 0.0; nvy[i] = 0.0; nvz[i] = 0.0;
    const double ds = deltaS[i];
    if (fabs(ds) <= contrast) return;
    const double sgn = (ds >= 0.0) ? 1.0 : -1.0;
    const long long f = f0 + i;
    double ax = 0.0, ay = 0.0, az = 0.0;
    double bx = 0.0, by = 0.0, bz = 0.0;
    for (int k = 0; k < 4; ++k) {
        const long long p1 = fp[4 * f + k];
        const long long p2 = fp[4 * f + ((k + 1) & 3)];
        const double lx = (px[p2] - px[p1]) + (dx[p2] - dx[p1]);
        const double ly = (py[p2] - py[p1]) + (dy[p2] - dy[p1]);
        const double lz = (pz[p2] - pz[p1]) + (dz[p2] - dz[p1]);
        double ex = 0.5 * (nx[p2] + nx[p1]);
        double ey = 0.5 * (ny[p2] + ny[p1]);
        double ez = 0.5 * (nz[p2] + nz[p1]);
        const double em = sqrt(ex*ex + ey*ey + ez*ez) + 1.0e-15;
        ex /= em; ey /= em; ez /= em;
        ax += sgn * (ly * ez - lz * ey);
        ay += sgn * (lz * ex - lx * ez);
        az += sgn * (lx * ey - ly * ex);
        bx += gx[p1];  by += gy[p1];  bz += gz[p1];
    }
    cfx[i] = ax; cfy[i] = ay; cfz[i] = az;
    const double bm = sqrt(bx*bx + by*by + bz*bz) + 1.0e-15;
    nvx[i] = bx / bm; nvy[i] = by / bm; nvz[i] = bz / bm;
}

__global__ void gpu_p_stf(
    const double* __restrict__ cfx, const double* __restrict__ cfy,
    const double* __restrict__ cfz, const double* __restrict__ nvx,
    const double* __restrict__ nvy, const double* __restrict__ nvz,
    const double* __restrict__ isx, const double* __restrict__ isy,
    const double* __restrict__ isz, const double* __restrict__ deltaS,
    const double* __restrict__ magSf, double* __restrict__ stf,
    const double sigma, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    const double m = fabs(isx[i]*nvx[i] + isy[i]*nvy[i] + isz[i]*nvz[i])
                     + 1.0e-14 * magSf[i];
    const double dp = cfx[i]*nvx[i] + cfy[i]*nvy[i] + cfz[i]*nvz[i];
    stf[i] = sigma * dp / m * deltaS[i] * magSf[i];
}

__global__ void gpu_p_sgpc_init(
    const double* __restrict__ stf, const double* __restrict__ err,
    double* __restrict__ sgpc, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    sgpc[i] = stf[i] - err[i];
}

__global__ void gpu_p_sgpc_subtract_flux(
    const double* __restrict__ pc, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ dmag,
    const double* __restrict__ magSf, double* __restrict__ sgpc,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double d = dmag[f];
    const double fl = (d > 0.0) ? (pc[nei[f]] - pc[own[f]]) / d * magSf[f]
                                : 0.0;
    sgpc[f] -= fl;
}

__global__ void gpu_p_sgpc_err_update(
    const double* __restrict__ sgpc, const double* __restrict__ deltaS,
    const double* __restrict__ magSf,
    double* __restrict__ err, const double sigma, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double typ = fabs(0.1 * sigma * deltaS[f] * deltaS[f] * magSf[f]);
    if (sgpc[f] * err[f] < -1.0e-6 * typ) err[f] *= 0.99;
    if (fabs(sgpc[f]) < 0.5 * fabs(err[f]))      err[f] = 0.5 * sgpc[f];
}

__global__ void gpu_p_fcsmooth_acc(
    const long long* __restrict__ fp, const double* __restrict__ magDelS,
    const double* __restrict__ misf,
    const double* __restrict__ cfx, const double* __restrict__ cfy,
    const double* __restrict__ cfz,
    double* __restrict__ pfx, double* __restrict__ pfy,
    double* __restrict__ pfz, double* __restrict__ parea,
    const double contrast, const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    const double ds = magDelS[i];
    if (ds <= contrast) return;
    const long long f = f0 + i;
    const double q = 0.25 * ds;
    for (int k = 0; k < 4; ++k) {
        const long long p = fp[4 * f + k];
        atomicAdd(&pfx[p], q * cfx[i]);
        atomicAdd(&pfy[p], q * cfy[i]);
        atomicAdd(&pfz[p], q * cfz[i]);
        atomicAdd(&parea[p], ds * misf[i]);
    }
}

__global__ void gpu_p_fcsmooth_apply(
    const long long* __restrict__ fp, const double* __restrict__ magDelS,
    const double* __restrict__ misf,
    const double* __restrict__ pfx, const double* __restrict__ pfy,
    const double* __restrict__ pfz, const double* __restrict__ parea,
    double* __restrict__ cfx, double* __restrict__ cfy,
    double* __restrict__ cfz, const double relax, const double contrast,
    const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    if (magDelS[i] <= contrast) return;
    const long long f = f0 + i;
    double ax = (1.0 - relax) * cfx[i];
    double ay = (1.0 - relax) * cfy[i];
    double az = (1.0 - relax) * cfz[i];
    for (int k = 0; k < 4; ++k) {
        const long long p = fp[4 * f + k];
        const double w = relax * misf[i] / (fabs(parea[p]) + 1.0e-36);
        ax += w * pfx[p];  ay += w * pfy[p];  az += w * pfz[p];
    }
    cfx[i] = ax;  cfy[i] = ay;  cfz[i] = az;
}

__global__ void gpu_p_mag_interface_sf(
    const double* __restrict__ isx, const double* __restrict__ isy,
    const double* __restrict__ isz, const double* __restrict__ nvx,
    const double* __restrict__ nvy, const double* __restrict__ nvz,
    const double* __restrict__ magSf, double* __restrict__ misf,
    const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    misf[i] = fabs(isx[i]*nvx[i] + isy[i]*nvy[i] + isz[i]*nvz[i])
              + 1.0e-14 * magSf[i];
}

__global__ void gpu_p_cellsmooth_acc(
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ magDelS, const double* __restrict__ misf,
    const double* __restrict__ cw,
    const double* __restrict__ cfx, const double* __restrict__ cfy,
    const double* __restrict__ cfz,
    double* __restrict__ Fx, double* __restrict__ Fy, double* __restrict__ Fz,
    double* __restrict__ area, const double contrast, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double ds = magDelS[f];
    if (ds <= contrast) return;
    const long long o = own[f], n = nei[f];
    const double wo = 0.5 * cw[o] * ds, wn = 0.5 * cw[n] * ds;
    atomicAdd(&Fx[o], wo * cfx[f]);  atomicAdd(&Fx[n], wn * cfx[f]);
    atomicAdd(&Fy[o], wo * cfy[f]);  atomicAdd(&Fy[n], wn * cfy[f]);
    atomicAdd(&Fz[o], wo * cfz[f]);  atomicAdd(&Fz[n], wn * cfz[f]);
    atomicAdd(&area[o], cw[o] * ds * misf[f]);
    atomicAdd(&area[n], cw[n] * ds * misf[f]);
}

__global__ void gpu_p_cellsmooth_apply(
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ magDelS, const double* __restrict__ misf,
    const double* __restrict__ Fx, const double* __restrict__ Fy,
    const double* __restrict__ Fz, const double* __restrict__ area,
    double* __restrict__ cfx, double* __restrict__ cfy,
    double* __restrict__ cfz, const double contrast, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    if (magDelS[f] <= contrast) return;
    const long long o = own[f], n = nei[f];
    const double ao = area[o], an = area[n];
    if (ao == 0.0 || an == 0.0) return;
    const double wo = 0.99 * misf[f] / ao, wn = 0.99 * misf[f] / an;
    cfx[f] = 0.01 * cfx[f] + wo * Fx[o] + wn * Fx[n];
    cfy[f] = 0.01 * cfy[f] + wo * Fy[o] + wn * Fy[n];
    cfz[f] = 0.01 * cfz[f] + wo * Fz[o] + wn * Fz[n];
}

__global__ void gpu_p_cellweight(
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ magDelS, double* __restrict__ cw,
    const double contrast, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    if (magDelS[f] <= contrast) return;
    atomicAdd(&cw[own[f]], 100.0);
    atomicAdd(&cw[nei[f]], 100.0);
}

__global__ void gpu_p_sgpc_clip(
    const double* __restrict__ stf, const double* __restrict__ deltaS,
    const double* __restrict__ magSf, double* __restrict__ sgpc,
    double* __restrict__ errn, const double fcd, const double sigma,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double typ = fabs(0.1 * sigma * deltaS[f] * deltaS[f] * magSf[f]);
    const double thr = fcd * (fabs(stf[f]) + typ);
    const double e = fmin(fmax(sgpc[f], -thr), thr);
    errn[f] = e;
    sgpc[f] -= e;
}

__global__ void gpu_p_bnd_incidence(
    const long long* __restrict__ fp, double* __restrict__ cnt,
    const long long f0, const long long nF)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;
    const long long f = f0 + i;
    for (int k = 0; k < 4; ++k) atomicAdd(&cnt[fp[4 * f + k]], 1.0);
}

__global__ void gpu_p_annihilate(
    const double* __restrict__ cnt, double* __restrict__ num,
    double* __restrict__ den, const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    const double c = cnt[p];
    if (c <= 0.0) return;
    const double s = pow(1.0e-32, c);
    num[p] *= s;
    den[p] *= s;
}

__global__ void gpu_p_recon_solve(
    const double* __restrict__ T, const double* __restrict__ bx,
    const double* __restrict__ by, const double* __restrict__ bz,
    double* __restrict__ gx, double* __restrict__ gy,
    double* __restrict__ gz, const long long nC)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const double a = T[6*c+0], b = T[6*c+1], e = T[6*c+2];
    const double d = T[6*c+3], f2 = T[6*c+4], g = T[6*c+5];
    const double c00 = d*g - f2*f2, c01 = e*f2 - b*g, c02 = b*f2 - e*d;
    const double det = a*c00 + b*c01 + e*c02;
    if (det == 0.0) { gx[c] = 0.0; gy[c] = 0.0; gz[c] = 0.0; return; }
    const double r = 1.0 / det;
    const double i0 = c00*r, i1 = c01*r, i2 = c02*r;
    const double i3 = (a*g - e*e)*r, i4 = (b*e - a*f2)*r;
    const double i5 = (a*d - b*b)*r;
    gx[c] = i0*bx[c] + i1*by[c] + i2*bz[c];
    gy[c] = i1*bx[c] + i3*by[c] + i4*bz[c];
    gz[c] = i2*bx[c] + i4*by[c] + i5*bz[c];
}

__global__ void gpu_p_recon_acc2(
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ Sfx, const double* __restrict__ Sfy,
    const double* __restrict__ Sfz, const double* __restrict__ magSf,
    const double* __restrict__ ssf,
    double* __restrict__ T, double* __restrict__ bx,
    double* __restrict__ by, double* __restrict__ bz, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double s = magSf[f];
    if (s <= 0.0) return;
    const double ax = Sfx[f], ay = Sfy[f], az = Sfz[f];
    const double t0 = ax*ax/(s*s), t1 = ax*ay/(s*s), t2 = ax*az/(s*s);
    const double t3 = ay*ay/(s*s), t4 = ay*az/(s*s), t5 = az*az/(s*s);
    const double v = ssf[f] / (s*s);
    const long long o = own[f], n = nei[f];
    for (int r = 0; r < 2; ++r) {
        const long long c = (r == 0) ? o : n;
        atomicAdd(&T[6*c+0], t0);  atomicAdd(&T[6*c+1], t1);
        atomicAdd(&T[6*c+2], t2);  atomicAdd(&T[6*c+3], t3);
        atomicAdd(&T[6*c+4], t4);  atomicAdd(&T[6*c+5], t5);
        atomicAdd(&bx[c], v * ax);
        atomicAdd(&by[c], v * ay);
        atomicAdd(&bz[c], v * az);
    }
}

__global__ void gpu_p_recon_acc_bnd2(
    const long long* __restrict__ bown, const double* __restrict__ bSfx,
    const double* __restrict__ bSfy, const double* __restrict__ bSfz,
    const double* __restrict__ bmag, const double* __restrict__ bssf,
    double* __restrict__ T, double* __restrict__ bx,
    double* __restrict__ by, double* __restrict__ bz, const long long nBF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nBF) return;
    const double s = bmag[f];
    if (s <= 0.0) return;
    const double ax = bSfx[f], ay = bSfy[f], az = bSfz[f];
    const long long c = bown[f];
    atomicAdd(&T[6*c+0], ax*ax/(s*s));  atomicAdd(&T[6*c+1], ax*ay/(s*s));
    atomicAdd(&T[6*c+2], ax*az/(s*s));  atomicAdd(&T[6*c+3], ay*ay/(s*s));
    atomicAdd(&T[6*c+4], ay*az/(s*s));  atomicAdd(&T[6*c+5], az*az/(s*s));
    const double v = (bssf == 0) ? 0.0 : (bssf[f] / (s*s));
    atomicAdd(&bx[c], v * ax);
    atomicAdd(&by[c], v * ay);
    atomicAdd(&bz[c], v * az);
}

__global__ void gpu_p_clip2(
    const double* __restrict__ gx, const double* __restrict__ gy,
    const double* __restrict__ gz, const long long* __restrict__ own,
    const long long* __restrict__ nei, const double* __restrict__ w,
    const double* __restrict__ Sfx, const double* __restrict__ Sfy,
    const double* __restrict__ Sfz, double* __restrict__ sgpc,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double a = w[f], b = 1.0 - w[f];
    const double thr = 2.0 * fabs((a*gx[o] + b*gx[n]) * Sfx[f]
                                + (a*gy[o] + b*gy[n]) * Sfy[f]
                                + (a*gz[o] + b*gz[n]) * Sfz[f]);
    sgpc[f] = fmin(fmax(sgpc[f], -thr), thr);
}

__global__ void gpu_p_nhat_gated(
    const double* __restrict__ gx, const double* __restrict__ gy,
    const double* __restrict__ gz, const double* __restrict__ mark,
    double* __restrict__ nx, double* __restrict__ ny,
    double* __restrict__ nz, double* __restrict__ gmag,
    const double deltaN, const double cut, const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    const double g = sqrt(gx[p]*gx[p] + gy[p]*gy[p] + gz[p]*gz[p]);
    gmag[p] = g + 0.1 * deltaN;
    if (mark[p] <= 0.0 || g <= cut * deltaN) {
        nx[p] = 0.0; ny[p] = 0.0; nz[p] = 0.0;
        return;
    }
    nx[p] = gx[p] / gmag[p];
    ny[p] = gy[p] / gmag[p];
    nz[p] = gz[p] / gmag[p];
}

__global__ void gpu_p_sgpc_err_accum(
    const double* __restrict__ gpx, const double* __restrict__ gpy,
    const double* __restrict__ gpz,
    const double* __restrict__ gax, const double* __restrict__ gay,
    const double* __restrict__ gaz, const double deltaN,
    const long long* __restrict__ own, const long long* __restrict__ nei,
    const double* __restrict__ w,
    const double* __restrict__ Sfx, const double* __restrict__ Sfy,
    const double* __restrict__ Sfz, const double* __restrict__ magSf,
    const double* __restrict__ dinterf,
    double* __restrict__ sgpc_err, const double fcCorrectTangent,
    const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const long long o = own[f], n = nei[f];
    const double a = w[f], b = 1.0 - w[f];

    double vx, vy, vz;
    {
        const double m1 = sqrt(gax[o]*gax[o] + gay[o]*gay[o] + gaz[o]*gaz[o])
                          + 1.0e-4 * deltaN;
        const double m2 = sqrt(gax[n]*gax[n] + gay[n]*gay[n] + gaz[n]*gaz[n])
                          + 1.0e-4 * deltaN;
        const double s1x = gax[o]/m1, s1y = gay[o]/m1, s1z = gaz[o]/m1;
        const double s2x = gax[n]/m2, s2y = gay[n]/m2, s2z = gaz[n]/m2;
        const double d1 = gpx[o]*s1x + gpy[o]*s1y + gpz[o]*s1z;
        const double d2 = gpx[n]*s2x + gpy[n]*s2y + gpz[n]*s2z;
        vx = a*(gpx[o] - d1*s1x) + b*(gpx[n] - d2*s2x);
        vy = a*(gpy[o] - d1*s1y) + b*(gpy[n] - d2*s2y);
        vz = a*(gpz[o] - d1*s1z) + b*(gpz[n] - d2*s2z);
    }
    double nhf = 0.0;
    {
        double hx = a*gax[o] + b*gax[n];
        double hy = a*gay[o] + b*gay[n];
        double hz = a*gaz[o] + b*gaz[n];
        const double hm = sqrt(hx*hx + hy*hy + hz*hz) + 1.0e-15;
        hx /= hm; hy /= hm; hz /= hm;
        const double dd = vx*hx + vy*hy + vz*hz;
        vx -= dd*hx; vy -= dd*hy; vz -= dd*hz;
        nhf = hx*Sfx[f] + hy*Sfy[f] + hz*Sfz[f];
    }
    const double s = magSf[f];
    const double q = (s > 0.0) ? fabs(nhf) / s : 0.0;
    const double sc = 1.0 - q;
    vx *= sc; vy *= sc; vz *= sc;
    const double add = fcCorrectTangent * (vx*Sfx[f] + vy*Sfy[f] + vz*Sfz[f]);
    const double damp = fmin(2.0 * (1.4 - q), 1.0);
    sgpc_err[f] = dinterf[f] * (damp * (sgpc_err[f] + add));
}

__global__ void gpu_p_dinterf(
    const double* __restrict__ deltaS, double* __restrict__ dint,
    const double deltaN, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    const double m = fabs(deltaS[f]);
    dint[f] = m / (m + 1.0e-4 * deltaN);
}

__global__ void gpu_p_sgpc_err_decay(
    const double* __restrict__ dint, double* __restrict__ sgpc_err,
    const double relax, const long long nIF)
{
    long long f = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIF) return;
    sgpc_err[f] *= relax * dint[f];
}

__global__ void gpu_p_contact_point_th(
    const double* __restrict__ wpx, const double* __restrict__ wpy,
    const double* __restrict__ wpz,
    const double* __restrict__ wmark,
    const double* __restrict__ mark,
    double* __restrict__ nx, double* __restrict__ ny, double* __restrict__ nz,
    const double* __restrict__ sinth, const double* __restrict__ costh,
    const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    if (wmark[p] <= 0.0 || mark[p] <= 0.0) return;
    const double gx = nx[p], gy = ny[p], gz = nz[p];
    const double gm = sqrt(gx*gx + gy*gy + gz*gz);
    if (gm <= 0.0) return;
    const double hx = gx / (gm + 1.0e-37);
    const double hy = gy / (gm + 1.0e-37);
    const double hz = gz / (gm + 1.0e-37);
    const double wx = wpx[p], wy = wpy[p], wz = wpz[p];
    const double hw = hx*wx + hy*wy + hz*wz;
    double sx = hx - hw*wx, sy = hy - hw*wy, sz = hz - hw*wz;
    const double sm = sqrt(sx*sx + sy*sy + sz*sz) + 1.0e-37;
    sx /= sm; sy /= sm; sz /= sm;
    const double st = sinth[p], ct = costh[p];
    nx[p] = gm * (st*sx + ct*wx);
    ny[p] = gm * (st*sy + ct*wy);
    nz[p] = gm * (st*sz + ct*wz);
}

__global__ void gpu_p_dist_patch(
    const long long* __restrict__ cls,
    const double* __restrict__ wpx, const double* __restrict__ wpy,
    const double* __restrict__ wpz,
    const double* __restrict__ nx, const double* __restrict__ ny,
    const double* __restrict__ nz,
    double* __restrict__ dx, double* __restrict__ dy,
    double* __restrict__ dz, const long long nP)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nP) return;
    const long long c = cls[p];
    if (c == 0) return;
    if (c == 2) { dx[p] = 0.0; dy[p] = 0.0; dz[p] = 0.0; return; }
    double wx = wpx[p], wy = wpy[p], wz = wpz[p];
    const double wm = sqrt(wx*wx + wy*wy + wz*wz) + 1.0e-15;
    wx /= wm; wy /= wm; wz /= wm;
    const double sxn = nx[p], syn = ny[p], szn = nz[p];
    const double d = sxn*wx + syn*wy + szn*wz;
    double sx = sxn - d*wx, sy = syn - d*wy, sz = szn - d*wz;
    const double sm = sqrt(sx*sx + sy*sy + sz*sz) + 1.0e-6;
    sx /= sm; sy /= sm; sz /= sm;
    const double dd = dx[p]*sx + dy[p]*sy + dz[p]*sz;
    const double cv = sxn*sx + syn*sy + szn*sz;
    const double s = 1.1 * dd / (cv + 0.1);
    dx[p] = s*sx; dy[p] = s*sy; dz[p] = s*sz;
}

}
'''
_MOD = None
_FN: dict = {}

def module(cp):
    global _MOD
    if _MOD is None:
        _MOD = cp.RawModule(code=KERNELS, options=('--fmad=false', '-std=c++14'))
    return _MOD

def fn(cp, name):
    if name not in _FN:
        _FN[name] = module(cp).get_function(name)
    return _FN[name]

def grid(n, block=256):
    n = max(int(n), 1)
    return (((n + block - 1) // block,), (block,))

def point_layer(case: str):
    pm = os.path.join(case, 'constant', 'polyMesh')
    key = []
    for fnm in sorted(os.listdir(pm)) if os.path.isdir(pm) else []:
        fp_ = os.path.join(pm, fnm)
        if os.path.isfile(fp_):
            st = os.stat(fp_)
            key.append('%s:%d:%d' % (fnm, st.st_size, int(st.st_mtime)))
    h = hashlib.sha256(('|'.join(key) + '||' + 'p1:points,facePoints,faceCentres,nPoints').encode()).hexdigest()[:16]
    cdir = os.environ.get('GPU_MESH_CACHE', os.path.join(os.path.expanduser('~'), '.mesh_cache'))
    os.makedirs(cdir, exist_ok=True)
    cf = os.path.join(cdir, 'p_%s.npz' % h)
    if os.path.exists(cf):
        z = np.load(cf)
        return (z['pts'], z['fp'], z['cf'])
    from pyporefoam import poly_mesh
    mesh = poly_mesh.read_poly_mesh(case)
    pts = np.ascontiguousarray(np.asarray(mesh.points), dtype=np.float64)
    faces = mesh.faces
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    assert (sizes == 4).all(), 'this layer assumes 4 points per face; found %r' % np.unique(sizes).tolist()
    fp = np.concatenate([np.asarray(f, dtype=np.int64) for f in faces])
    ctr = np.ascontiguousarray(np.asarray(mesh.face_centres()), dtype=np.float64)
    tmp = cf[:-4] + '.%d.npz' % os.getpid()
    np.savez(tmp, pts=pts, fp=fp, cf=ctr)
    os.replace(tmp, cf)
    return (pts, fp, ctr)

def edge_layer(case: str, fp: 'np.ndarray', n_points: int, n_internal: int, wall_faces: 'np.ndarray'):
    pm = os.path.join(case, 'constant', 'polyMesh')
    key = []
    for fnm in sorted(os.listdir(pm)) if os.path.isdir(pm) else []:
        q = os.path.join(pm, fnm)
        if os.path.isfile(q):
            st = os.stat(q)
            key.append('%s:%d:%d' % (fnm, st.st_size, int(st.st_mtime)))
    key.append('wall:%d' % int(wall_faces.size))
    h = hashlib.sha256(('|'.join(key) + '||' + 'e1:edges,faceEdges,edgemarks').encode()).hexdigest()[:16]
    cdir = os.environ.get('GPU_MESH_CACHE', os.path.join(os.path.expanduser('~'), '.mesh_cache'))
    os.makedirs(cdir, exist_ok=True)
    cf = os.path.join(cdir, 'e_%s.npz' % h)
    if os.path.exists(cf):
        z = np.load(cf)
        return (z['edges'], z['fe'], z['marks'])
    nF = int(fp.size // 4)
    a = fp.reshape(nF, 4)
    b = np.roll(a, -1, axis=1)
    lo = np.minimum(a, b).ravel()
    hi = np.maximum(a, b).ravel()
    key2 = lo.astype(np.int64) * np.int64(n_points) + hi.astype(np.int64)
    uk, fe = np.unique(key2, return_inverse=True)
    edges = np.stack([uk // np.int64(n_points), uk % np.int64(n_points)], axis=1).astype(np.int64)
    fe = fe.astype(np.int64)
    fe4 = fe.reshape(nF, 4)
    assert (np.sort(fe4, axis=1)[:, 1:] != np.sort(fe4, axis=1)[:, :-1]).all(), 'a face has a repeated edge'
    e0, e1 = (edges[fe4[:, 0], 0], edges[fe4[:, 0], 1])
    p0, p1 = (a[:, 0], a[:, 1])
    assert ((e0 == np.minimum(p0, p1)) & (e1 == np.maximum(p0, p1))).all(), 'edge 0 does not join point 0 to point 1'
    marks = np.zeros(edges.shape[0], dtype=np.float64)
    if wall_faces.size:
        gf = wall_faces.astype(np.int64) + int(n_internal)
        marks[fe4[gf].ravel()] = 1.0
    tmp = cf[:-4] + '.%d.npz' % os.getpid()
    np.savez(tmp, edges=edges, fe=fe, marks=marks)
    os.replace(tmp, cf)
    return (edges, fe, marks)
