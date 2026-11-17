from __future__ import annotations
import alpha_kernels as K
import halo_decomp as HALO
import gradient_csr as GRAD

class AlphaWorkspace:

    def __init__(self, cp, n_cells: int, n_internal_faces: int, n_boundary_faces: int=0):
        self.cp = cp
        self.nC = int(n_cells)
        self.nIF = int(n_internal_faces)
        self.nBF = int(n_boundary_faces)

        def _f(n):
            return cp.empty(int(n), dtype=cp.float64)
        self.af_up = _f(self.nIF)
        self.af_hi = _f(self.nIF)
        self.phir = _f(self.nIF)
        self.psi = _f(self.nIF)
        self.asm_ = _f(self.nC)
        self.acell = _f(self.nC)
        self.anum, self.aden = (_f(self.nC), _f(self.nC))
        self.cgx, self.cgy, self.cgz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.gfx, self.gfy, self.gfz = (_f(self.nIF), _f(self.nIF), _f(self.nIF))
        self.phic = _f(self.nIF)
        self.nhf = _f(self.nIF)
        self.sf1 = _f(self.nIF)
        self.phic_max = _f(1)
        self.flux_lo = _f(self.nIF)
        self.flux_corr = _f(self.nIF)
        self.lam = _f(self.nIF)
        self.sp = _f(self.nC)
        self.sm = _f(self.nC)
        self.rp = _f(self.nC)
        self.rm = _f(self.nC)
        self.div = _f(self.nC)
        self.a_lo = _f(self.nC)
        self.pmax = _f(self.nC)
        self.pmin = _f(self.nC)
        self.spu = _f(self.nC)
        self.smu = _f(self.nC)
        self.lam1 = _f(self.nIF)
        self.gmax = _f(self.nC)
        self.gmin = _f(self.nC)
        self._gf = K.grid(self.nIF)
        self._gc = K.grid(self.nC)
        self._gb = K.grid(self.nBF) if self.nBF > 0 else None
        self.dd = HALO.Decomp.serial(self.nC, self.nIF, self.nBF)
        K.compile_all(cp)


def alpha_step(ws: AlphaWorkspace, *, alpha0, alpha1, phi, own, nei, vol, magSf, nHatf, gradf, w, gcx, gcy, gcz, dx, dy, dz, Sfx, Sfy, Sfz, dmag, delta_n, dt: float, c_alpha: float=1.0, limiter_iterations: int=3, subcycles: int=1, alpha_min: float=0.0, alpha_max: float=1.0, bown=None, bSfx=None, bSfy=None, bSfz=None, bphi=None, bUt=None, a_inlet: float=1.0, a_inlet_field=None) -> None:
    gf, bf = ws._gf
    gc, bc = ws._gc
    ns = max(1, int(subcycles))
    if ns > 1:
        sub_dt = dt / ns
        src_a, dst_a = (alpha0, alpha1)
        for k in range(ns):
            _alpha_substep(ws, w=w, Sfx=Sfx, Sfy=Sfy, Sfz=Sfz, dmag=dmag, delta_n=delta_n, gcx=gcx, gcy=gcy, gcz=gcz, dx=dx, dy=dy, dz=dz, alpha0=src_a, alpha1=dst_a, phi=phi, own=own, nei=nei, vol=vol, magSf=magSf, nHatf=nHatf, gradf=gradf, dt=sub_dt, c_alpha=c_alpha, limiter_iterations=limiter_iterations, alpha_min=alpha_min, alpha_max=alpha_max, bown=bown, bSfx=bSfx, bSfy=bSfy, bSfz=bSfz, bphi=bphi, bUt=bUt, a_inlet=a_inlet, a_inlet_field=a_inlet_field)
            if k + 1 < ns:
                src_a, dst_a = (dst_a, src_a)
        if ns % 2 == 0:
            alpha1[...] = dst_a
        return
    _alpha_substep(ws, w=w, Sfx=Sfx, Sfy=Sfy, Sfz=Sfz, dmag=dmag, delta_n=delta_n, gcx=gcx, gcy=gcy, gcz=gcz, dx=dx, dy=dy, dz=dz, alpha0=alpha0, alpha1=alpha1, phi=phi, own=own, nei=nei, vol=vol, magSf=magSf, nHatf=nHatf, gradf=gradf, dt=dt, c_alpha=c_alpha, limiter_iterations=limiter_iterations, alpha_min=alpha_min, alpha_max=alpha_max, bown=bown, bSfx=bSfx, bSfy=bSfy, bSfz=bSfz, bphi=bphi, bUt=bUt, a_inlet=a_inlet, a_inlet_field=a_inlet_field)

def _alpha_substep(ws, *, alpha0, alpha1, phi, own, nei, vol, magSf, nHatf, gradf, w, gcx, gcy, gcz, dx, dy, dz, Sfx, Sfy, Sfz, dmag, delta_n, dt, c_alpha, limiter_iterations, alpha_min, alpha_max, bown=None, bSfx=None, bSfy=None, bSfz=None, bphi=None, bUt=None, a_inlet: float=1.0, a_inlet_field=None) -> None:
    cp = ws.cp
    dd = ws.dd
    gf, bf = ws._gf
    gc, bc = ws._gc
    nIF = ws.nIF
    nC = ws.nC
    dd.halo(alpha0)
    dd.halo(gcx)
    dd.halo(gcy)
    dd.halo(gcz)
    K.fn(cp, 'gpu_alpha_face_upwind')(gf, bf, (alpha0, phi, own, nei, ws.af_up, nIF))
    K.fn(cp, 'gpu_alpha_face_vanleer_w')(gf, bf, (alpha0, phi, gcx, gcy, gcz, dx, dy, dz, own, nei, w, ws.af_hi, ws.psi, nIF))
    import curvature_kernels as _CURV
    import momentum_kernels as _MOM

    def _avg_cell(qf):
        ws.anum.fill(0.0)
        ws.aden.fill(0.0)
        _CURV.fn(cp, 'gpu_ccf_avg_acc')(gf, bf, (qf, magSf, own, nei, ws.anum, ws.aden, nIF))
        _CURV.fn(cp, 'gpu_ccf_avg_div')(gc, bc, (ws.anum, ws.aden, ws.acell, nC))
        dd.halo(ws.acell)
        return ws.acell

    def _interp(qc, out):
        _MOM.fn(cp, 'gpu_face_interp')(gf, bf, (qc, own, nei, w, out, nIF))
        return out
    _interp(alpha0, ws.sf1)
    K.fn(cp, 'gpu_a_alphasm')(gc, bc, (alpha0, _avg_cell(ws.sf1), ws.asm_, nC))
    dd.halo(ws.asm_)
    ws.cgx.fill(0.0)
    ws.cgy.fill(0.0)
    ws.cgz.fill(0.0)
    _zg_s5 = GRAD.DEV is not None
    if _zg_s5:
        GRAD.calc_grad(cp, GRAD.DEV, own, nei, bown, None, ws.asm_, ws.cgx, ws.cgy, ws.cgz, 0, nC, nIF, int(ws.nBF))
    else:
        _CURV.fn(cp, 'gpu_grad_accumulate')(gf, bf, (ws.asm_, own, nei, Sfx, Sfy, Sfz, w, ws.cgx, ws.cgy, ws.cgz, nIF))
    if not _zg_s5 and bown is not None and (ws.nBF > 0) and (ws._gb is not None):
        gb, bb = ws._gb
        _CURV.fn(cp, 'gpu_grad_accumulate_bnd')(gb, bb, (ws.asm_, bown, bSfx, bSfy, bSfz, ws.cgx, ws.cgy, ws.cgz, ws.nBF))
    if not _zg_s5:
        _CURV.fn(cp, 'gpu_grad_divide_volume')(gc, bc, (ws.cgx, ws.cgy, ws.cgz, vol, nC))
    dd.halo(ws.cgx)
    dd.halo(ws.cgy)
    dd.halo(ws.cgz)
    _interp(ws.cgx, ws.gfx)
    _interp(ws.cgy, ws.gfy)
    _interp(ws.cgz, ws.gfz)
    for _ in range(2):
        for src, dst in ((ws.gfx, ws.cgx), (ws.gfy, ws.cgy), (ws.gfz, ws.cgz)):
            dst[...] = _avg_cell(src)
        dd.halo(ws.cgx)
        dd.halo(ws.cgy)
        dd.halo(ws.cgz)
        K.fn(cp, 'gpu_a_grad_relax')(gf, bf, (ws.cgx, ws.cgy, ws.cgz, own, nei, w, ws.gfx, ws.gfy, ws.gfz, nIF))
    K.fn(cp, 'gpu_a_nhatf')(gf, bf, (ws.gfx, ws.gfy, ws.gfz, Sfx, Sfy, Sfz, ws.nhf, cp.float64(delta_n), nIF))
    cp.divide(cp.abs(phi, out=ws.sf1), magSf, out=ws.phic)
    for _ in range(2):
        K.fn(cp, 'gpu_face_localmax')(gf, bf, (_avg_cell(ws.phic), own, nei, ws.sf1, nIF))
        ws.phic += ws.sf1
    cp.amax(ws.phic, keepdims=True, out=ws.phic_max)
    dd.allmax_(ws.phic_max)
    K.fn(cp, 'gpu_a_phir')(gf, bf, (ws.phic, ws.asm_, own, nei, dmag, ws.nhf, ws.phic_max, ws.phir, cp.float64(c_alpha), cp.float64(delta_n), nIF))
    K.fn(cp, 'gpu_flux_split')(gf, bf, (ws.af_up, ws.af_hi, phi, ws.phir, alpha0, own, nei, w, ws.flux_lo, ws.flux_corr, nIF))
    dd.own_faces(ws.flux_lo)
    dd.own_faces(ws.flux_corr)
    ws.div.fill(0.0)
    ws.lam.fill(0.0)
    K.fn(cp, 'gpu_div_accumulate')(gf, bf, (ws.flux_lo, ws.flux_corr, ws.lam, own, nei, ws.div, nIF))
    if bphi is not None and bUt is not None and (ws.nBF > 0) and (ws._gb is not None):
        K.fn(cp, 'gpu_alpha_bnd_div' + ('_field' if a_inlet_field is not None else ''))(ws._gb[0], ws._gb[1], (alpha0, bphi, bown, bUt, a_inlet_field if a_inlet_field is not None else cp.float64(a_inlet), ws.div, ws.nBF))
    K.fn(cp, 'gpu_alpha_apply')(gc, bc, (alpha0, ws.div, vol, ws.a_lo, cp.float64(dt), nC))
    K.fn(cp, 'gpu_mules_bounds_init_ref')(gc, bc, (ws.pmax, ws.pmin, cp.float64(alpha_min), cp.float64(alpha_max), nC))
    K.fn(cp, 'gpu_mules_bounds_face')(gf, bf, (alpha0, own, nei, ws.pmax, ws.pmin, nIF))
    if bphi is not None and bUt is not None and (ws.nBF > 0) and (ws._gb is not None):
        K.fn(cp, 'gpu_mules_bounds_bnd_ref' + ('_field' if a_inlet_field is not None else ''))(ws._gb[0], ws._gb[1], (alpha0, bphi, bown, bUt, a_inlet_field if a_inlet_field is not None else cp.float64(a_inlet), ws.pmax, ws.pmin, ws.nBF))
    K.fn(cp, 'gpu_mules_bounds_clamp')(gc, bc, (ws.pmax, ws.pmin, cp.float64(alpha_min), cp.float64(alpha_max), nC))
    ws.gmax.fill(alpha_max)
    ws.gmin.fill(alpha_min)
    ws.lam1.fill(1.0)
    ws.spu.fill(1e-300)
    ws.smu.fill(1e-300)
    K.fn(cp, 'gpu_mules_sums')(gf, bf, (ws.flux_corr, ws.lam1, own, nei, ws.spu, ws.smu, nIF))
    ws.lam.fill(1.0)
    for _iter in range(int(limiter_iterations)):
        ws.sp.fill(0.0)
        ws.sm.fill(0.0)
        K.fn(cp, 'gpu_mules_sums')(gf, bf, (ws.flux_corr, ws.lam, own, nei, ws.sp, ws.sm, nIF))
        K.fn(cp, 'gpu_mules_cell_ratio_ref')(gc, bc, (ws.a_lo, vol, ws.sp, ws.sm, ws.spu, ws.smu, ws.rp, ws.rm, cp.float64(dt), ws.pmin, ws.pmax, nC))
        dd.halo(ws.rp)
        dd.halo(ws.rm)
        K.fn(cp, 'gpu_mules_face_lambda')(gf, bf, (ws.flux_corr, own, nei, ws.rp, ws.rm, ws.lam, nIF))
        dd.own_faces(ws.lam)
    ws.div.fill(0.0)
    K.fn(cp, 'gpu_div_accumulate')(gf, bf, (ws.flux_lo, ws.flux_corr, ws.lam, own, nei, ws.div, nIF))
    if bphi is not None and bUt is not None and (ws.nBF > 0) and (ws._gb is not None):
        K.fn(cp, 'gpu_alpha_bnd_div' + ('_field' if a_inlet_field is not None else ''))(ws._gb[0], ws._gb[1], (alpha0, bphi, bown, bUt, a_inlet_field if a_inlet_field is not None else cp.float64(a_inlet), ws.div, ws.nBF))
    K.fn(cp, 'gpu_alpha_apply')(gc, bc, (alpha0, ws.div, vol, alpha1, cp.float64(dt), nC))
    dd.halo(alpha1)
