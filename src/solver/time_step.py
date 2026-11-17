from __future__ import annotations
import alpha_transport as ALPHA
import curvature_driver as CURV
import amg_solver as AMG
import linear_solver as LSOLVE
import gradient_csr as GRAD
import momentum_kernels as MOMK
import curvature_kernels as CURVK
import fluid_props as PROPS

class StepWorkspace:

    def __init__(self, cp, *, n_cells, n_internal_faces, mean_cell_volume, pressure_levels, n_boundary_faces=0, n_wall: int=0):
        self.cp = cp
        self.nC = int(n_cells)
        self.nIF = int(n_internal_faces)
        self.s1 = ALPHA.AlphaWorkspace(cp, self.nC, self.nIF, n_boundary_faces=n_boundary_faces)
        self.s2 = CURV.CurvatureWorkspace(cp, self.nC, self.nIF, mean_cell_volume, n_wall=n_wall, n_boundary_faces=n_boundary_faces)
        self.s3 = AMG.AmgWorkspace(cp, pressure_levels)
        self.psolver = LSOLVE.SAAMG(cp, self.s3, name='p')

        def _f(n):
            return cp.zeros(int(n), dtype=cp.float64)
        self.upper = _f(self.nIF)
        self.diag = _f(self.nC)
        self.rau = _f(self.nC)
        self.rauf = _f(self.nIF)
        self.af = _f(self.nIF)
        self.phiHbyA = _f(self.nIF)
        self.src = _f(self.nC)
        self.p = _f(self.nC)
        self.lower = _f(self.nIF)
        self.Hx, self.Hy, self.Hz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.hx, self.hy, self.hz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.sx, self.sy, self.sz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.alpha_next = _f(self.nC)
        self.rho = _f(self.nC)
        self.mu = _f(self.nC)
        self.mu_f = _f(self.nIF)
        self.rau_v = _f(self.nC)
        self.rauf_v = _f(self.nIF)
        self.pdff = _f(self.nIF)
        self.pdfn = _f(self.nC)
        self.pdfd = _f(self.nC)
        self.pdfc = _f(self.nC)
        self.pdclip = cp.zeros(1, dtype=cp.uint64)
        self.pgx = _f(self.nC)
        self.pgy = _f(self.nC)
        self.pgz = _f(self.nC)
        self.nocorr = _f(self.nIF)
        self.ufx, self.ufy, self.ufz = (_f(self.nIF), _f(self.nIF), _f(self.nIF))
        self.ufcx, self.ufcy, self.ufcz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.ufn, self.ufd = (_f(self.nC), _f(self.nC))
        self.nBF = int(n_boundary_faces)
        self.bphi = _f(max(1, self.nBF))
        self.pdfb = _f(max(1, self.nBF))
        self.ufd = _f(self.nIF)
        self.bres = _f(max(1, int(n_boundary_faces)))
        self.urT = _f(6 * self.nC)
        self.urx, self.ury, self.urz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.ucx, self.ucy, self.ucz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.cbgx, self.cbgy, self.cbgz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.cbT = _f(6 * self.nC)
        self.cbc_probe = None
        self.cbbx, self.cbby, self.cbbz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.cbqx, self.cbqy, self.cbqz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.cbzero = _f(max(1, self.nBF))
        self._gb = MOMK.grid(max(1, self.nBF))
        self.ccf = None
        self.ccf_s3 = None
        self._gc = MOMK.grid(self.nC)
        self._gf = MOMK.grid(self.nIF)
        MOMK.compile_all(cp)
        PROPS.compile_all(cp)
        CURVK.compile_all(cp)
        self.steps = 0


def step(ws: StepWorkspace, *, mesh, fields, control) -> None:
    cp = ws.cp
    gf, bf = ws._gf
    gc, bc = ws._gc
    m, F, C = (mesh, fields, control)
    CURV.curvature_step(ws.s2, alpha=F['alpha'], own=m['own'], nei=m['nei'], dmag=m['dmag'], Sfx=m['Sfx'], Sfy=m['Sfy'], Sfz=m['Sfz'], w=m['w'], vol=m['vol'], magSf=m['magSf'], sigma=C['sigma'], bown=m.get('bown'), bSfx=m.get('bSfx'), bSfy=m.get('bSfy'), bSfz=m.get('bSfz'), wall_cells=m.get('wall_cells'), wnx=m.get('wnx'), wny=m.get('wny'), wnz=m.get('wnz'), costheta=m.get('costheta'), contrast=float(C.get('contrast', 0.0)), alpha_boundary=cp.where(m['alpha_fixed_mask'], m['alpha_fixed'], F['alpha'][m['bown']]) if m.get('alpha_fixed') is not None else None)
    if getattr(ws, 'ccf', None) is not None:
        import ccf as _CCF_MOD
        _CCF_MOD.correct(ws.ccf, alpha=F['alpha'], mesh=m, control=C, s3=ws.ccf_s3)
    PROPS.fn(cp, 'gpu_blend_props')(gc, bc, (F['alpha'], ws.rho, ws.mu, cp.float64(C['rho1']), cp.float64(C['rho2']), cp.float64(C['mu1']), cp.float64(C['mu2']), ws.nC))
    PROPS.fn(cp, 'gpu_prop_face')(gf, bf, (ws.mu, m['own'], m['nei'], m['w'], ws.mu_f, ws.nIF))
    MOMK.fn(cp, 'gpu_ddt_src3')(gc, bc, (ws.rho, m['vol'], F['Ux'], F['Uy'], F['Uz'], cp.float64(1.0 / C['dt']), ws.sx, ws.sy, ws.sz, ws.nC))
    ws.upper.fill(0.0)
    ws.lower.fill(0.0)
    ws.diag.fill(0.0)
    _mass_flux = F['phi']
    if getattr(ws, '_rho_flux', None) is None:
        ws._rho_flux = cp.empty_like(F['phi'])
    cp.multiply(F['phi'], float(C.get('rho2', 1000.0)), out=ws._rho_flux)
    _mass_flux = ws._rho_flux
    MOMK.fn(cp, 'gpu_mom_assemble')(gf, bf, (_mass_flux, ws.mu_f, m['magSf'], m['dmag'], m['own'], m['nei'], ws.upper, ws.lower, ws.diag, ws.nIF))
    if float(C.get('wall_drag', 1.0)) > 0.0 and int(getattr(ws, 'nBF', 0)) and (m.get('bUt') is not None):
        _cbc = float(C.get('cbc', 0.0))
        _cbb = 0.0
        if _cbb > 0.0:
            _cbn = 0
            if int(getattr(ws, 'steps', 0)) < _cbn:
                _cbc = _cbb
        if _cbc > 0.0 and m.get('bddot') is not None:
            ws.cbgx.fill(0.0)
            ws.cbgy.fill(0.0)
            ws.cbgz.fill(0.0)
            if GRAD.DEV is not None:
                cp.take(ws.p, m['bown'], out=ws.pdfb)
                if getattr(ws, '_outlet_faces', None) is not None:
                    ws.pdfb[ws._outlet_faces] = ws._outlet_pressure
                GRAD.calc_grad(cp, GRAD.DEV, m['own'], m['nei'], m['bown'], m['bPt'], ws.p, ws.cbgx, ws.cbgy, ws.cbgz, 1, ws.nC, ws.nIF, ws.nBF, bvalues=ws.pdfb)
            else:
                CURVK.fn(cp, 'gpu_grad_accumulate')(gf, bf, (ws.p, m['own'], m['nei'], m['Sfx'], m['Sfy'], m['Sfz'], m['w'], ws.cbgx, ws.cbgy, ws.cbgz, ws.nIF))
                CURVK.fn(cp, 'gpu_grad_accumulate_pbnd')(ws._gb[0], ws._gb[1], (ws.p, m['bown'], m['bPt'], m['bSfx'], m['bSfy'], m['bSfz'], ws.cbgx, ws.cbgy, ws.cbgz, ws.nBF))
                if getattr(ws, '_outlet_faces', None) is not None and int(ws._outlet_faces.size):
                    _kpv = ws.p[ws._outlet_owner]
                    _kwt = cp.abs(ws.bphi[ws._outlet_faces])
                    _kden = float(_kwt.sum())
                    _kmean = float((_kwt * _kpv).sum()) / _kden if _kden > 0.0 else float(_kpv.mean())
                    _kpb = 0.95 * (_kpv - _kmean)
                    for _kk, _kg in enumerate((ws.cbgx, ws.cbgy, ws.cbgz)):
                        _kg += cp.bincount(ws._outlet_owner, weights=_kpb * ws._outlet_weight[_kk], minlength=int(ws.nC))[:int(ws.nC)]
                CURVK.fn(cp, 'gpu_grad_divide_volume')(gc, bc, (ws.cbgx, ws.cbgy, ws.cbgz, m['vol'], ws.nC))
            ws.cbqx.fill(0.0)
            ws.cbqy.fill(0.0)
            ws.cbqz.fill(0.0)
            if getattr(ws, 'ccf', None) is not None:
                import point_layers as _P
                ws.cbT.fill(0.0)
                ws.cbbx.fill(0.0)
                ws.cbby.fill(0.0)
                ws.cbbz.fill(0.0)
                _P.fn(cp, 'gpu_p_recon_acc2')(gf, bf, (m['own'], m['nei'], m['Sfx'], m['Sfy'], m['Sfz'], m['magSf'], ws.ccf.sgpc, ws.cbT, ws.cbbx, ws.cbby, ws.cbbz, ws.nIF))
                _P.fn(cp, 'gpu_p_recon_acc_bnd2')(ws._gb[0], ws._gb[1], (m['bown'], m['bSfx'], m['bSfy'], m['bSfz'], m['bmagSf'], ws.cbzero, ws.cbT, ws.cbbx, ws.cbby, ws.cbbz, ws.nBF))
                _P.fn(cp, 'gpu_p_recon_solve')(gc, bc, (ws.cbT, ws.cbbx, ws.cbby, ws.cbbz, ws.cbqx, ws.cbqy, ws.cbqz, ws.nC))
            MOMK.fn(cp, 'gpu_mom_bnd_cbc')(ws._gb[0], ws._gb[1], (m['bown'], m['bmagSf'], m['bdelta'], m['bUt'], m['bddot'], ws.cbgx, ws.cbgy, ws.cbgz, ws.cbqx, ws.cbqy, ws.cbqz, F['Ux'], F['Uy'], F['Uz'], cp.float64(float(C.get('mu1', 0.001))), cp.float64(_cbc), cp.float64(float(C.get('cbc_cap', 0.0))), ws.diag, ws.nBF, ws.cbc_probe if getattr(ws, 'cbc_probe', None) is not None else cp.zeros(0, dtype=cp.float64)))
        else:
            MOMK.fn(cp, 'gpu_mom_bnd_diag')(ws._gb[0], ws._gb[1], (m['bown'], m['bmagSf'], m['bdelta'], m['bUt'], cp.float64(float(C.get('mu1', 0.001))), ws.diag, ws.nBF))
        MOMK.fn(cp, 'gpu_mom_bnd_inlet')(ws._gb[0], ws._gb[1], (m['bown'], m['bmagSf'], m['bdelta'], m['bUt'], m['bSfx'], m['bSfy'], m['bSfz'], ws.bphi, m['bddot'], ws.cbgx, ws.cbgy, ws.cbgz, ws.cbqx, ws.cbqy, ws.cbqz, F['Ux'], F['Uy'], F['Uz'], cp.float64(float(C.get('mu1', 0.001))), cp.float64(_cbc), cp.float64(0.0), ws.diag, ws.sx, ws.sy, ws.sz, ws.nBF))
    if C.get('case_mixed_outlet', 0):
        import case_outlet as BO
        BO.momentum(ws, m, F, C, cp)
    PROPS.fn(cp, 'gpu_add_ddt_diag')(gc, bc, (ws.diag, ws.rho, m['vol'], cp.float64(C['dt']), ws.nC))
    MOMK.fn(cp, 'gpu_rau')(gc, bc, (ws.diag, ws.rau, ws.nC))
    MOMK.fn(cp, 'gpu_face_interp')(gf, bf, (ws.rau, m['own'], m['nei'], m['w'], ws.rauf, ws.nIF))
    PROPS.fn(cp, 'gpu_rau_per_volume')(gc, bc, (ws.diag, m['vol'], ws.rau_v, ws.nC))
    MOMK.fn(cp, 'gpu_face_interp')(gf, bf, (ws.rau_v, m['own'], m['nei'], m['w'], ws.rauf_v, ws.nIF))
    MOMK.fn(cp, 'gpu_pressure_face_coeff')(gf, bf, (ws.rauf_v, m['magSf'], m['dmag'], ws.af, ws.nIF))
    if C.get('refresh') is not None:
        C['refresh']()
    MOMK.fn(cp, 'gpu_face_interp')(gf, bf, (ws.p, m['own'], m['nei'], m['w'], ws.pdff, ws.nIF))
    ws.pdfn.fill(0.0)
    ws.pdfd.fill(0.0)
    CURVK.fn(cp, 'gpu_ccf_avg_acc')(gf, bf, (ws.pdff, m['magSf'], m['own'], m['nei'], ws.pdfn, ws.pdfd, ws.nIF))
    cp.take(ws.p, m['bown'], out=ws.pdfb)
    _zo = getattr(ws, '_outlet_faces', None)
    if _zo is not None and int(_zo.size) and (getattr(ws, '_outlet_pressure', None) is not None):
        ws.pdfb[_zo] = ws._outlet_pressure
    CURVK.fn(cp, 'gpu_ccf_avg_acc_bnd')(ws._gb[0], ws._gb[1], (ws.pdfb, m['bmagSf'], m['bown'], ws.pdfn, ws.pdfd, cp.float64(1.0), ws.nBF))
    CURVK.fn(cp, 'gpu_ccf_avg_div')(gc, bc, (ws.pdfn, ws.pdfd, ws.pdfc, ws.nC))
    MOMK.fn(cp, 'gpu_pd_filter')(gc, bc, (ws.pdfc, ws.p, cp.float64(float(C.get('pd_filter', 1.3))), ws.pdclip, ws.nC))
    ws.pdfi = getattr(ws, 'pdfi', 0) + 1
    ws.Hx.fill(0.0)
    ws.Hy.fill(0.0)
    ws.Hz.fill(0.0)
    MOMK.fn(cp, 'gpu_offdiag_gather3')(gf, bf, (ws.upper, ws.lower, m['own'], m['nei'], F['Ux'], F['Uy'], F['Uz'], ws.Hx, ws.Hy, ws.Hz, ws.nIF))
    MOMK.fn(cp, 'gpu_hbya3')(gc, bc, (ws.rau, ws.Hx, ws.Hy, ws.Hz, ws.sx, ws.sy, ws.sz, ws.hx, ws.hy, ws.hz, ws.nC))
    import case_predictor as BP
    BP.prepare(ws, m, F, C, cp)
    if float(C.get('filter_u_on', 1.0)) != 0.0:
        for _u, _uf, _uc in ((ws.hx, ws.ufx, ws.ufcx), (ws.hy, ws.ufy, ws.ufcy), (ws.hz, ws.ufz, ws.ufcz)):
            MOMK.fn(cp, 'gpu_face_minmod')(gf, bf, (_u, m['own'], m['nei'], _uf, ws.nIF))
            ws.ufn.fill(0.0)
            ws.ufd.fill(0.0)
            CURVK.fn(cp, 'gpu_ccf_avg_acc')(gf, bf, (_uf, m['magSf'], m['own'], m['nei'], ws.ufn, ws.ufd, ws.nIF))
            CURVK.fn(cp, 'gpu_ccf_avg_div')(gc, bc, (ws.ufn, ws.ufd, _uc, ws.nC))
        MOMK.fn(cp, 'gpu_u_filter')(gc, bc, (ws.ufcx, ws.ufcy, ws.ufcz, ws.hx, ws.hy, ws.hz, cp.float64(float(C.get('u_filter', 2.0))), ws.nC))
    ws.Hx.fill(0.0)
    ws.Hy.fill(0.0)
    ws.Hz.fill(0.0)
    MOMK.fn(cp, 'gpu_offdiag_gather3')(gf, bf, (ws.upper, ws.lower, m['own'], m['nei'], ws.hx, ws.hy, ws.hz, ws.Hx, ws.Hy, ws.Hz, ws.nIF))
    MOMK.fn(cp, 'gpu_hbya3')(gc, bc, (ws.rau, ws.Hx, ws.Hy, ws.Hz, ws.sx, ws.sy, ws.sz, ws.hx, ws.hy, ws.hz, ws.nC))
    MOMK.fn(cp, 'gpu_phi_hbya')(gf, bf, (ws.hx, ws.hy, ws.hz, m['own'], m['nei'], m['w'], m['Sfx'], m['Sfy'], m['Sfz'], ws.phiHbyA, ws.nIF))
    MOMK.fn(cp, 'gpu_ddt_phi_corr')(gf, bf, (F['phi'], F['Ux'], F['Uy'], F['Uz'], ws.rau_v, ws.rho, m['own'], m['nei'], m['w'], m['Sfx'], m['Sfy'], m['Sfz'], ws.phiHbyA, cp.float64(1.0 / float(C['dt'])), ws.nIF))
    if getattr(ws, 'ccf', None) is not None:
        force = ws.ccf.sgpc
    else:
        force = ws.s2.Fsig
    PROPS.fn(cp, 'gpu_add_face_force')(gf, bf, (ws.phiHbyA, ws.rauf_v, force, ws.nIF))
    ws.nocorr.fill(0.0)
    for _corr in range(int(C['n_corr'])):
        for _no in range(int(C['n_non_orth']) + 1):
            if m.get('bUt') is not None and ws.nBF > 0:
                MOMK.fn(cp, 'gpu_bnd_flux_nb')(ws._gb[0], ws._gb[1], (ws.hx, ws.hy, ws.hz, m['bown'], m['bSfx'], m['bSfy'], m['bSfz'], m['bUt'], ws.bphi, ws.nBF))
            if C.get('case_mixed_outlet', 0):
                BO.apply_hbya_boundary(ws, cp)
            if getattr(ws, '_outlet_faces', None) is None:
                _zi = cp.nonzero(m['bPt'] == 1)[0]
                ws._outlet_faces = _zi
                ws._outlet_owner = m['bown'][_zi]
                ws._outlet_area = m['bmagSf'][_zi] * m['bdelta'][_zi]
                ws._outlet_weight = (m['bSfx'][_zi], m['bSfy'][_zi], m['bSfz'][_zi])
                ws._outlet_pressure = cp.zeros(int(_zi.size))
            if int(ws._outlet_faces.size):
                _zpv = ws.p[ws._outlet_owner]
                _zwt = cp.abs(ws.bphi[ws._outlet_faces])
                _zden = float(_zwt.sum())
                _zmean = float((_zwt * _zpv).sum()) / _zden if _zden > 0.0 else float(_zpv.mean())
                ws._outlet_pressure[...] = 0.95 * (_zpv - _zmean)
            ws.src.fill(0.0)
            MOMK.fn(cp, 'gpu_div_face')(gf, bf, (ws.phiHbyA, m['own'], m['nei'], ws.src, ws.nIF))
            if m.get('bUt') is not None and ws.nBF > 0:
                MOMK.fn(cp, 'gpu_bnd_div_add')(ws._gb[0], ws._gb[1], (ws.bphi, m['bown'], ws.src, ws.nBF))
            if _no > 0 and False:
                ws.pgx.fill(0.0)
                ws.pgy.fill(0.0)
                ws.pgz.fill(0.0)
                CURVK.fn(cp, 'gpu_grad_accumulate')(gf, bf, (ws.p, m['own'], m['nei'], m['Sfx'], m['Sfy'], m['Sfz'], m['w'], ws.pgx, ws.pgy, ws.pgz, ws.nIF))
                if C.get('p_close', 1) and m.get('bPt') is not None and (ws.nBF > 0):
                    CURVK.fn(cp, 'gpu_grad_accumulate_pbnd')(ws._gb[0], ws._gb[1], (ws.p, m['bown'], m['bPt'], m['bSfx'], m['bSfy'], m['bSfz'], ws.pgx, ws.pgy, ws.pgz, ws.nBF))
                    if int(ws._outlet_faces.size):
                        _zpb = ws._outlet_pressure
                        ws.pgx += cp.bincount(ws._outlet_owner, weights=_zpb * ws._outlet_weight[0], minlength=int(ws.nC))[:int(ws.nC)]
                        ws.pgy += cp.bincount(ws._outlet_owner, weights=_zpb * ws._outlet_weight[1], minlength=int(ws.nC))[:int(ws.nC)]
                        ws.pgz += cp.bincount(ws._outlet_owner, weights=_zpb * ws._outlet_weight[2], minlength=int(ws.nC))[:int(ws.nC)]
                CURVK.fn(cp, 'gpu_grad_divide_volume')(gc, bc, (ws.pgx, ws.pgy, ws.pgz, m['vol'], ws.nC))
                PROPS.fn(cp, 'gpu_nonorth_correct')(gf, bf, (ws.pgx, ws.pgy, ws.pgz, m['Sfx'], m['Sfy'], m['Sfz'], m['dx'], m['dy'], m['dz'], m['dmag'], m['own'], m['nei'], m['w'], ws.rauf_v, ws.p, ws.af, cp.float64(C.get('nonorth_limit', 0.33)), ws.nocorr, ws.nIF))
                MOMK.fn(cp, 'gpu_div_face')(gf, bf, (ws.nocorr, m['own'], m['nei'], ws.src, ws.nIF))
            cp.negative(ws.src, out=ws.src)
            if int(ws._outlet_faces.size):
                ws.src += cp.bincount(ws._outlet_owner, weights=ws.rau_v[ws._outlet_owner] * ws._outlet_area * ws._outlet_pressure, minlength=int(ws.nC))[:int(ws.nC)]
            ws.psolver.solve(b=ws.src, x=ws.p, cycles=C['pressure_cycles'])
            if m.get('bUt') is not None and ws.nBF > 0:
                MOMK.fn(cp, 'gpu_bnd_flux_correct')(ws._gb[0], ws._gb[1], (ws.rau_v, ws.p, m['bown'], m['bmagSf'], m['bdelta'], m['bPt'], ws.bphi, ws.nBF))
                if int(ws._outlet_faces.size):
                    ws.bphi[ws._outlet_faces] -= ws.rau_v[ws._outlet_owner] * ws._outlet_area * ws._outlet_pressure
            MOMK.fn(cp, 'gpu_flux_correct')(gf, bf, (ws.phiHbyA, ws.af, ws.p, m['own'], m['nei'], ws.nocorr, F['phi'], ws.nIF))
        ws.pgx.fill(0.0)
        ws.pgy.fill(0.0)
        ws.pgz.fill(0.0)
        CURVK.fn(cp, 'gpu_grad_accumulate')(gf, bf, (ws.p, m['own'], m['nei'], m['Sfx'], m['Sfy'], m['Sfz'], m['w'], ws.pgx, ws.pgy, ws.pgz, ws.nIF))
        if C.get('p_close', 1) and m.get('bPt') is not None and (ws.nBF > 0):
            CURVK.fn(cp, 'gpu_grad_accumulate_pbnd')(ws._gb[0], ws._gb[1], (ws.p, m['bown'], m['bPt'], m['bSfx'], m['bSfy'], m['bSfz'], ws.pgx, ws.pgy, ws.pgz, ws.nBF))
            if int(ws._outlet_faces.size):
                _zpb = ws._outlet_pressure
                ws.pgx += cp.bincount(ws._outlet_owner, weights=_zpb * ws._outlet_weight[0], minlength=int(ws.nC))[:int(ws.nC)]
                ws.pgy += cp.bincount(ws._outlet_owner, weights=_zpb * ws._outlet_weight[1], minlength=int(ws.nC))[:int(ws.nC)]
                ws.pgz += cp.bincount(ws._outlet_owner, weights=_zpb * ws._outlet_weight[2], minlength=int(ws.nC))[:int(ws.nC)]
        CURVK.fn(cp, 'gpu_grad_divide_volume')(gc, bc, (ws.pgx, ws.pgy, ws.pgz, m['vol'], ws.nC))
        if C.get('u_recon', 1):
            import point_layers as _P
            MOMK.fn(cp, 'gpu_u_face_resid')(gf, bf, (ws.p, m['own'], m['nei'], m['magSf'], m['dmag'], force, cp.float64(float(C.get('u_cap', 1.0))), ws.ufd, ws.nIF))
            ws.urT.fill(0.0)
            ws.urx.fill(0.0)
            ws.ury.fill(0.0)
            ws.urz.fill(0.0)
            _P.fn(cp, 'gpu_p_recon_acc2')(gf, bf, (m['own'], m['nei'], m['Sfx'], m['Sfy'], m['Sfz'], m['magSf'], ws.ufd, ws.urT, ws.urx, ws.ury, ws.urz, ws.nIF))
            if m.get('bmagSf') is not None and ws.nBF > 0:
                MOMK.fn(cp, 'gpu_bnd_u_resid')(ws._gb[0], ws._gb[1], (ws.p, m['bown'], m['bmagSf'], m['bdelta'], m['bPt'], ws.bres, ws.nBF))
                if C.get('case_right_recon', 0) and True:
                    ws.bres[ws._outlet_faces] -= ws._outlet_area * ws._outlet_pressure
                _P.fn(cp, 'gpu_p_recon_acc_bnd2')(ws._gb[0], ws._gb[1], (m['bown'], m['bSfx'], m['bSfy'], m['bSfz'], m['bmagSf'], ws.bres, ws.urT, ws.urx, ws.ury, ws.urz, ws.nBF))
            _P.fn(cp, 'gpu_p_recon_solve')(gc, bc, (ws.urT, ws.urx, ws.ury, ws.urz, ws.ucx, ws.ucy, ws.ucz, ws.nC))
            MOMK.fn(cp, 'gpu_u_add_recon')(gc, bc, (ws.hx, ws.hy, ws.hz, ws.rau_v, ws.ucx, ws.ucy, ws.ucz, ws.pgx, ws.pgy, ws.pgz, F['Ux'], F['Uy'], F['Uz'], ws.nC))
        else:
            MOMK.fn(cp, 'gpu_u_correct3')(gc, bc, (ws.hx, ws.hy, ws.hz, ws.rau_v, ws.pgx, ws.pgy, ws.pgz, F['Ux'], F['Uy'], F['Uz'], ws.nC))
    _nsub = int(C.get('alpha_subcycles', 1))
    ALPHA.alpha_step(ws.s1, alpha0=F['alpha'], alpha1=ws.alpha_next, phi=F['phi'], own=m['own'], nei=m['nei'], vol=m['vol'], magSf=m['magSf'], nHatf=ws.s2.nHatf, w=m['w'], gradf=ws.s2.gradf, dt=C['dt'], gcx=ws.s2.gx, gcy=ws.s2.gy, gcz=ws.s2.gz, dx=m['dx'], dy=m['dy'], dz=m['dz'], Sfx=m['Sfx'], Sfy=m['Sfy'], Sfz=m['Sfz'], dmag=m['dmag'], delta_n=float(C.get('deltaN', 1e-12)), bown=m.get('bown'), bSfx=m.get('bSfx'), bSfy=m.get('bSfy'), bSfz=m.get('bSfz'), bphi=ws.bphi if float(C.get('alpha_bnd', 1.0)) > 0.0 else None, bUt=m.get('bUt'), a_inlet=float(C.get('alpha_inlet', 1.0)), a_inlet_field=m.get('alpha_fixed'), c_alpha=C['c_alpha'], limiter_iterations=C['limiter_iterations'], subcycles=_nsub)
    F['alpha'], ws.alpha_next = (ws.alpha_next, F['alpha'])
    ws.steps += 1
