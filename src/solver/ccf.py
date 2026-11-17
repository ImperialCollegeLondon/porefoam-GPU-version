from __future__ import annotations
from typing import Any, Dict
import numpy as np
import point_layers as P
import curvature_kernels as K
import momentum_kernels as MOMK
import linear_solver as LSOLVE
import face_select as SELECT
_ORDER = True
_CCFT_CALLS = [0]


def _owner_boundary_values(m, cellf):
    if m.get('bown') is None:
        return None
    return cellf[m['bown']]

def _transition_boundary_clip(ws, m, cp, gsrc):
    import cupyx as _cpx
    tf, hit = (m['transition_factor'], m['transition_mask'])
    ab = gsrc[m['bown']]
    cl = cp.clip(1.0 / tf * ab - (0.5 / tf - 0.5), 0.0, 1.0)
    d = cp.where(hit, cl - ab, 0.0)
    if m.get('alpha_fixed') is not None:
        d = cp.where(m['alpha_fixed_mask'], m['alpha_fixed'] - ab, d)
    _cpx.scatter_add(ws.gax, m['bown'], d * m['bSfx'])
    _cpx.scatter_add(ws.gay, m['bown'], d * m['bSfy'])
    _cpx.scatter_add(ws.gaz, m['bown'], d * m['bSfz'])
    return (int((d != 0.0).sum()), float(cp.abs(d).max()))

class CCFWorkspace:

    def __init__(self, cp, n_cells, n_internal_faces, n_points, n_faces, pc_levels=None, n_boundary_faces=0):
        self.cp = cp
        self.nC, self.nIF = (int(n_cells), int(n_internal_faces))
        self.nP, self.nF = (int(n_points), int(n_faces))
        self.nBF = int(n_boundary_faces)
        f = lambda n: cp.zeros(int(n), dtype=cp.float64)
        self.ash, self.x = (f(self.nC), f(self.nC))
        self.aclip = f(self.nC)
        self.a1s = f(self.nC)
        self.num, self.den = (f(self.nC), f(self.nC))
        self.deltaS = f(self.nIF)
        self.af = f(self.nIF)
        self.ap = f(self.nP)
        self.pnum, self.pden = (f(self.nP), f(self.nP))
        self.mark = f(self.nP)
        self.gg, self.ggi = (f(6 * self.nP), f(6 * self.nP))
        self.gx, self.gy, self.gz = (f(self.nP), f(self.nP), f(self.nP))
        self.nx, self.ny, self.nz = (f(self.nP), f(self.nP), f(self.nP))
        self.ax, self.ay, self.az = (f(self.nP), f(self.nP), f(self.nP))
        self.dx, self.dy, self.dz = (f(self.nP), f(self.nP), f(self.nP))
        self.isx, self.isy, self.isz = (f(self.nIF), f(self.nIF), f(self.nIF))
        self.cfx, self.cfy, self.cfz = (f(self.nIF), f(self.nIF), f(self.nIF))
        self.nvx, self.nvy, self.nvz = (f(self.nIF), f(self.nIF), f(self.nIF))
        self.stf = f(self.nIF)
        self.sgpc = f(self.nIF)
        self.sgpc_err = f(self.nIF)
        self.dint = f(self.nIF)
        self.gax, self.gay, self.gaz = (f(self.nC), f(self.nC), f(self.nC))
        self.pc = f(self.nC)
        self.src = f(self.nC)
        self.cavg = f(self.nC)
        self.mds = f(self.nIF)
        self.gm, self.gm2 = (f(self.nP), f(self.nP))
        self.pfx, self.pfy = (f(self.nP), f(self.nP))
        self.pfz, self.parea = (f(self.nP), f(self.nP))
        self.bcnt = f(self.nP)
        self.Fx, self.Fy, self.Fz = (f(self.nC), f(self.nC), f(self.nC))
        self.carea, self.cw = (f(self.nC), f(self.nC))
        self.misf, self.errn = (f(self.nIF), f(self.nIF))
        self.sgpc_preclip = f(self.nIF)
        self.sgpc_bnd = f(max(int(n_boundary_faces), 1))
        self.rT = f(6 * self.nC)
        self.rbx, self.rby = (f(self.nC), f(self.nC))
        self.rbz = f(self.nC)
        self.gpx, self.gpy = (f(self.nC), f(self.nC))
        self.gpz = f(self.nC)
        self.sx, self.sy, self.sz = (f(self.nC), f(self.nC), f(self.nC))
        self.dels_tmp = f(self.nIF)
        self.pc_levels = pc_levels
        self.pc_solver = LSOLVE.SAAMG(cp, pc_levels, name='pc', updatable=False) if pc_levels is not None else None

def _wall_point_normal(ws, m, C):
    cp = ws.cp
    if getattr(ws, '_wnp_ready', False):
        return True
    wf = m.get('wall_faces')
    if wf is None or int(wf.size) == 0 or m.get('wall_nfx') is None:
        return False
    nW = int(wf.size)
    gw, bw = P.grid(nW)
    gp, bp = P.grid(ws.nP)
    f = lambda n: cp.zeros(int(n), dtype=cp.float64)
    ws.wpx, ws.wpy, ws.wpz = (f(ws.nP), f(ws.nP), f(ws.nP))
    ws.wmark = f(ws.nP)
    nux, nuy, nuz, den = (f(ws.nP), f(ws.nP), f(ws.nP), f(ws.nP))
    vx = m['wall_nfx'].copy()
    vy = m['wall_nfy'].copy()
    vz = m['wall_nfz'].copy()
    for _ in range(int(C.get('wallSmoothingKernel', 2))):
        nux.fill(0.0)
        nuy.fill(0.0)
        nuz.fill(0.0)
        den.fill(0.0)
        P.fn(cp, 'gpu_p_w_f2p')(gw, bw, (m['fp'], wf, m['cfx'], m['cfy'], m['cfz'], m['px'], m['py'], m['pz'], vx, vy, vz, nux, nuy, nuz, den, nW))
        P.fn(cp, 'gpu_p_w_div')(gp, bp, (nux, nuy, nuz, den, ws.wpx, ws.wpy, ws.wpz, ws.wmark, ws.nP))
        P.fn(cp, 'gpu_p_w_p2f')(gw, bw, (m['fp'], wf, ws.wpx, ws.wpy, ws.wpz, vx, vy, vz, nW))
    P.fn(cp, 'gpu_p_w_renorm')(gw, bw, (vx, vy, vz, nW))
    nux.fill(0.0)
    nuy.fill(0.0)
    nuz.fill(0.0)
    den.fill(0.0)
    P.fn(cp, 'gpu_p_w_f2p')(gw, bw, (m['fp'], wf, m['cfx'], m['cfy'], m['cfz'], m['px'], m['py'], m['pz'], vx, vy, vz, nux, nuy, nuz, den, nW))
    P.fn(cp, 'gpu_p_w_div')(gp, bp, (nux, nuy, nuz, den, ws.wpx, ws.wpy, ws.wpz, ws.wmark, ws.nP))
    ws._wnp_ready = True
    return True

def _contact(ws, m, C, normal=False):
    ws.contact_calls = getattr(ws, 'contact_calls', 0) + 1
    if not _wall_point_normal(ws, m, C):
        return
    cp = ws.cp
    gp, bp = P.grid(ws.nP)
    th = float(C.get('theta_deg', 140.0)) * np.pi / 180.0
    tgt = (ws.nx, ws.ny, ws.nz) if normal else (ws.gx, ws.gy, ws.gz)
    if m.get('point_patch_sin') is not None:
        P.fn(cp, 'gpu_p_contact_point_th')(gp, bp, (ws.wpx, ws.wpy, ws.wpz, ws.wmark, ws.mark, tgt[0], tgt[1], tgt[2], m['point_patch_sin'], m['point_patch_cos'], ws.nP))
    else:
        P.fn(cp, 'gpu_p_contact_point')(gp, bp, (ws.wpx, ws.wpy, ws.wpz, ws.wmark, ws.mark, tgt[0], tgt[1], tgt[2], cp.float64(np.sin(th)), cp.float64(np.cos(th)), ws.nP))

def correct(ws, *, alpha, mesh, control, s3=None):
    cp = ws.cp
    m, C = (mesh, control)
    gc, bc = K.grid(ws.nC)
    gf, bf = K.grid(ws.nIF)
    gp, bp = P.grid(ws.nP)
    pct = float(C['pcThicknessFactor'])
    contrast = 0.0001 * float(C['deltaN'])
    _CCFT_CALLS[0] += 1
    K.fn(cp, 'gpu_ccf_alpha_sh')(gc, bc, (alpha, ws.ash, cp.float64(pct), ws.nC))
    _relax = float(C.get('relax_dels', 0.0))
    if _relax < 0.0001:
        K.fn(cp, 'gpu_ccf_delta_s')(gf, bf, (ws.ash, m['own'], m['nei'], m['dmag'], ws.deltaS, ws.nIF))
    else:
        K.fn(cp, 'gpu_ccf_delta_s')(gf, bf, (ws.ash, m['own'], m['nei'], m['dmag'], ws.dels_tmp, ws.nIF))
        ws.deltaS *= 0.5
        ws.deltaS += 0.5 * ws.dels_tmp

    def avg(qf, qb=None, if1=1e-09):
        ws.num.fill(0.0)
        ws.den.fill(0.0)
        K.fn(cp, 'gpu_ccf_avg_acc')(gf, bf, (qf, m['magSf'], m['own'], m['nei'], ws.num, ws.den, ws.nIF))
        if qb is not None and ws.nBF and (m.get('bmagSf') is not None):
            _gb, _bb = K.grid(ws.nBF)
            K.fn(cp, 'gpu_ccf_avg_acc_bnd')(_gb, _bb, (qb, m['bmagSf'], m['bown'], ws.num, ws.den, cp.float64(if1), ws.nBF))
        K.fn(cp, 'gpu_ccf_avg_div')(gc, bc, (ws.num, ws.den, ws.cavg, ws.nC))
        return ws.cavg

    def interp(cellf):
        MOMK.fn(cp, 'gpu_face_interp')(gf, bf, (cellf, m['own'], m['nei'], m['w'], ws.af, ws.nIF))
        return ws.af
    cp.clip(alpha, 0.0, 1.0, out=ws.aclip)
    _a1f = ws.aclip
    K.fn(cp, 'gpu_ccf_blend1')(gc, bc, (alpha, avg(interp(_a1f), cp.where(m['alpha_fixed_mask'], m['alpha_fixed'], _a1f[m['bown']]) if m.get('alpha_fixed') is not None else _owner_boundary_values(m, _a1f)), ws.x, ws.nC))
    K.fn(cp, 'gpu_ccf_blend2')(gc, bc, (avg(interp(ws.x), _owner_boundary_values(m, ws.x)), ws.x, ws.nC))
    K.fn(cp, 'gpu_ccf_blend3')(gc, bc, (avg(interp(ws.x), _owner_boundary_values(m, ws.x)), ws.x, ws.nC))
    K.fn(cp, 'gpu_ccf_atanh')(gc, bc, (ws.x, ws.a1s, ws.nC))
    af1s = interp(ws.a1s)
    ws.pnum.fill(0.0)
    ws.pden.fill(0.0)
    gfp, bfp = P.grid(ws.nIF)
    gbp, bbp = P.grid(max(ws.nBF, 1))
    if m.get('bown') is not None and ws.nBF and (m.get('a1s_bnd') is not None):
        SELECT.gather(ws.a1s, m['bown'], out=m['a1s_bnd'])
        if m.get('alpha_fixed_mask') is not None:
            _fm = m['alpha_fixed_mask']
            m['a1s_bnd'][_fm] = m['alpha_fixed'][_fm]
    P.fn(cp, 'gpu_p_interp_acc')(gfp, bfp, (m['fp'], m['cfx'], m['cfy'], m['cfz'], m['px'], m['py'], m['pz'], af1s, cp.float64(1.0), ws.pnum, ws.pden, np.int64(0), np.int64(ws.nIF)))
    if ws.nBF:
        if not getattr(ws, '_bcnt_ready', False):
            ws.bcnt.fill(0.0)
            P.fn(cp, 'gpu_p_bnd_incidence')(gbp, bbp, (m['fp'], ws.bcnt, np.int64(ws.nIF), np.int64(ws.nBF)))
            ws._bcnt_ready = True
        P.fn(cp, 'gpu_p_annihilate')(gp, bp, (ws.bcnt, ws.pnum, ws.pden, ws.nP))
        P.fn(cp, 'gpu_p_interp_acc')(gbp, bbp, (m['fp'], m['cfx'], m['cfy'], m['cfz'], m['px'], m['py'], m['pz'], m['a1s_bnd'], cp.float64(0.0001), ws.pnum, ws.pden, np.int64(ws.nIF), np.int64(ws.nBF)))
    P.fn(cp, 'gpu_p_interp_div')(gp, bp, (ws.pnum, ws.pden, ws.ap, ws.nP))
    cp.abs(ws.deltaS, out=ws.mds)
    ws.mark.fill(0.0)
    P.fn(cp, 'gpu_p_interf_mark')(gfp, bfp, (m['fp'], ws.mds, ws.mark, cp.float64(contrast), np.int64(0), np.int64(ws.nIF)))
    _plsb = m.get('bmagSf') is not None and ws.nBF and (m.get('a1s_bnd') is not None)
    ws.gg.fill(0.0)
    P.fn(cp, 'gpu_p_gg_acc')(gfp, bfp, (m['fp'], m['cfx'], m['cfy'], m['cfz'], m['px'], m['py'], m['pz'], m['magSf'], cp.float64(1.0), ws.gg, np.int64(0), np.int64(ws.nIF)))
    if _plsb:
        P.fn(cp, 'gpu_p_gg_acc')(gbp, bbp, (m['fp'], m['cfx'], m['cfy'], m['cfz'], m['px'], m['py'], m['pz'], m['bmagSf'], cp.float64(1.0), ws.gg, np.int64(ws.nIF), np.int64(ws.nBF)))
    P.fn(cp, 'gpu_p_gg_inv')(gp, bp, (ws.gg, ws.mark, ws.ggi, ws.nP))
    ws.gx.fill(0.0)
    ws.gy.fill(0.0)
    ws.gz.fill(0.0)
    P.fn(cp, 'gpu_p_grad_acc')(gfp, bfp, (m['fp'], m['cfx'], m['cfy'], m['cfz'], m['px'], m['py'], m['pz'], m['magSf'], af1s, ws.ap, ws.ggi, cp.float64(1.0), ws.gx, ws.gy, ws.gz, np.int64(0), np.int64(ws.nIF)))
    if _plsb:
        P.fn(cp, 'gpu_p_grad_acc')(gbp, bbp, (m['fp'], m['cfx'], m['cfy'], m['cfz'], m['px'], m['py'], m['pz'], m['bmagSf'], m['a1s_bnd'], ws.ap, ws.ggi, cp.float64(1.0), ws.gx, ws.gy, ws.gz, np.int64(ws.nIF), np.int64(ws.nBF)))
    _contact(ws, m, C)
    P.fn(cp, 'gpu_p_nhat_gated')(gp, bp, (ws.gx, ws.gy, ws.gz, ws.mark, ws.nx, ws.ny, ws.nz, ws.gm, cp.float64(C['deltaN']), cp.float64(C.get('nhat_cut', 1.0)), ws.nP))
    ws.ax.fill(0.0)
    ws.ay.fill(0.0)
    ws.az.fill(0.0)
    for _ in range(int(C['smoothingKernel'])):
        P.fn(cp, 'gpu_p_smooth_acc')(gfp, bfp, (m['fp'], m['fe'], m['marks'], ws.mds, m['cfx'], m['cfy'], m['cfz'], m['px'], m['py'], m['pz'], ws.nx, ws.ny, ws.nz, ws.ax, ws.ay, ws.az, cp.float64(contrast), cp.float64(1.0), np.int64(0), np.int64(ws.nIF)))
        P.fn(cp, 'gpu_p_smooth_update')(gp, bp, (ws.nx, ws.ny, ws.nz, ws.ax, ws.ay, ws.az, ws.mark, cp.float64(C['smoothingRelaxFactor']), ws.nP))
        _contact(ws, m, C, normal=True)
    P.fn(cp, 'gpu_p_renorm')(gp, bp, (ws.nx, ws.ny, ws.nz, cp.float64(1e-12), ws.nP))
    P.fn(cp, 'gpu_p_interface_dist')(gp, bp, (ws.ap, ws.gm, ws.nx, ws.ny, ws.nz, ws.mark, ws.dx, ws.dy, ws.dz, ws.nP))
    if m.get('point_patch_class') is not None and getattr(ws, 'wpx', None) is not None:
        P.fn(cp, 'gpu_p_dist_patch')(gp, bp, (m['point_patch_class'], ws.wpx, ws.wpy, ws.wpz, ws.nx, ws.ny, ws.nz, ws.dx, ws.dy, ws.dz, ws.nP))
    P.fn(cp, 'gpu_p_interface_sf')(gfp, bfp, (m['fp'], m['px'], m['py'], m['pz'], ws.dx, ws.dy, ws.dz, ws.deltaS, ws.isx, ws.isy, ws.isz, cp.float64(contrast), np.int64(0), np.int64(ws.nIF)))
    P.fn(cp, 'gpu_p_curv_force')(gfp, bfp, (m['fp'], m['px'], m['py'], m['pz'], ws.dx, ws.dy, ws.dz, ws.nx, ws.ny, ws.nz, ws.gx, ws.gy, ws.gz, ws.deltaS, ws.cfx, ws.cfy, ws.cfz, ws.nvx, ws.nvy, ws.nvz, cp.float64(contrast), np.int64(0), np.int64(ws.nIF)))
    P.fn(cp, 'gpu_p_mag_interface_sf')(gf, bf, (ws.isx, ws.isy, ws.isz, ws.nvx, ws.nvy, ws.nvz, m['magSf'], ws.misf, ws.nIF))
    _smk_fc = int(C['smoothingKernel']) % 10
    for _ in range(_smk_fc if C.get('sm_fc', 1) else 0):
        ws.pfx.fill(0.0)
        ws.pfy.fill(0.0)
        ws.pfz.fill(0.0)
        ws.parea.fill(0.0)
        P.fn(cp, 'gpu_p_fcsmooth_acc')(gfp, bfp, (m['fp'], ws.mds, ws.misf, ws.cfx, ws.cfy, ws.cfz, ws.pfx, ws.pfy, ws.pfz, ws.parea, cp.float64(contrast), np.int64(0), np.int64(ws.nIF)))
        P.fn(cp, 'gpu_p_fcsmooth_apply')(gfp, bfp, (m['fp'], ws.mds, ws.misf, ws.pfx, ws.pfy, ws.pfz, ws.parea, ws.cfx, ws.cfy, ws.cfz, cp.float64(C['smoothingRelaxFactor']), cp.float64(contrast), np.int64(0), np.int64(ws.nIF)))
    ws.cw.fill(1.0)
    P.fn(cp, 'gpu_p_cellweight')(gf, bf, (m['own'], m['nei'], ws.mds, ws.cw, cp.float64(contrast), ws.nIF))
    for _ in range(2 if C.get('sm_cell', 1) else 0):
        ws.Fx.fill(0.0)
        ws.Fy.fill(0.0)
        ws.Fz.fill(0.0)
        ws.carea.fill(0.0)
        P.fn(cp, 'gpu_p_cellsmooth_acc')(gf, bf, (m['own'], m['nei'], ws.mds, ws.misf, ws.cw, ws.cfx, ws.cfy, ws.cfz, ws.Fx, ws.Fy, ws.Fz, ws.carea, cp.float64(contrast), ws.nIF))
        P.fn(cp, 'gpu_p_cellsmooth_apply')(gf, bf, (m['own'], m['nei'], ws.mds, ws.misf, ws.Fx, ws.Fy, ws.Fz, ws.carea, ws.cfx, ws.cfy, ws.cfz, cp.float64(contrast), ws.nIF))
    P.fn(cp, 'gpu_p_stf')(gf, bf, (ws.cfx, ws.cfy, ws.cfz, ws.nvx, ws.nvy, ws.nvz, ws.isx, ws.isy, ws.isz, ws.deltaS, m['magSf'], ws.stf, cp.float64(C['sigma']), ws.nIF))
    _sge = float(C.get('sgpc_err', 1.0)) > 0.0
    if _sge:
        P.fn(cp, 'gpu_p_dinterf')(gf, bf, (ws.deltaS, ws.dint, cp.float64(C['deltaN']), ws.nIF))
        P.fn(cp, 'gpu_p_sgpc_err_decay')(gf, bf, (ws.dint, ws.sgpc_err, cp.float64(C.get('fcCorrectTangentRelax', 1.0)), ws.nIF))
    else:
        ws.sgpc_err.fill(0.0)
    P.fn(cp, 'gpu_p_sgpc_init')(gf, bf, (ws.stf, ws.sgpc_err, ws.sgpc, ws.nIF))
    _pcs = avg(interp(ws.pc), _owner_boundary_values(m, ws.pc), 1.0)
    ws.pc *= 0.9
    ws.pc += 0.1 * _pcs
    if s3 is not None and ws.pc_levels is not None:
        ws.src.fill(0.0)
        MOMK.fn(cp, 'gpu_div_face')(gf, bf, (ws.sgpc, m['own'], m['nei'], ws.src, ws.nIF))
        cp.negative(ws.src, out=ws.src)
        ws.pc_solver.solve(b=ws.src, x=ws.pc, cycles=int(C.get('pc_cycles', 4)))
        P.fn(cp, 'gpu_p_sgpc_subtract_flux')(gf, bf, (ws.pc, m['own'], m['nei'], m['dmag'], m['magSf'], ws.sgpc, ws.nIF))

    def _regpc():
        ws.rT.fill(0.0)
        ws.rbx.fill(0.0)
        ws.rby.fill(0.0)
        ws.rbz.fill(0.0)
        P.fn(cp, 'gpu_p_recon_acc2')(gf, bf, (m['own'], m['nei'], m['Sfx'], m['Sfy'], m['Sfz'], m['magSf'], ws.sgpc, ws.rT, ws.rbx, ws.rby, ws.rbz, ws.nIF))
        if ws.nBF and m.get('bmagSf') is not None:
            P.fn(cp, 'gpu_p_recon_acc_bnd2')(gbp, bbp, (m['bown'], m['bSfx'], m['bSfy'], m['bSfz'], m['bmagSf'], ws.sgpc_bnd, ws.rT, ws.rbx, ws.rby, ws.rbz, ws.nBF))
        P.fn(cp, 'gpu_p_recon_solve')(gc, bc, (ws.rT, ws.rbx, ws.rby, ws.rbz, ws.gpx, ws.gpy, ws.gpz, ws.nC))

    def _deadband():
        ws.sgpc_preclip[...] = ws.sgpc
        P.fn(cp, 'gpu_p_sgpc_clip')(gf, bf, (ws.stf, ws.deltaS, m['magSf'], ws.sgpc, ws.errn, cp.float64(C['fcdFilter']), cp.float64(C['sigma']), ws.nIF))

    def _clip2():
        for src, dst in ((ws.gpx, ws.sx), (ws.gpy, ws.sy), (ws.gpz, ws.sz)):
            dst[...] = avg(interp(src), _owner_boundary_values(m, src))
        P.fn(cp, 'gpu_p_clip2')(gf, bf, (ws.sx, ws.sy, ws.sz, m['own'], m['nei'], m['w'], m['Sfx'], m['Sfy'], m['Sfz'], ws.sgpc, ws.nIF))

    def _err_chain():
        ws.gax.fill(0.0)
        ws.gay.fill(0.0)
        ws.gaz.fill(0.0)
        cp.clip(alpha, 0.0, 1.0, out=ws.aclip)
        _gsrc = ws.aclip
        K.fn(cp, 'gpu_grad_accumulate')(gf, bf, (_gsrc, m['own'], m['nei'], m['Sfx'], m['Sfy'], m['Sfz'], m['w'], ws.gax, ws.gay, ws.gaz, ws.nIF))
        if ws.nBF and m.get('bown') is not None:
            K.fn(cp, 'gpu_grad_accumulate_bnd')(gbp, bbp, (_gsrc, m['bown'], m['bSfx'], m['bSfy'], m['bSfz'], ws.gax, ws.gay, ws.gaz, ws.nBF))
            if m.get('transition_factor') is not None:
                _n6, _m6 = _transition_boundary_clip(ws, m, cp, _gsrc)
        K.fn(cp, 'gpu_grad_divide_volume')(gc, bc, (ws.gax, ws.gay, ws.gaz, m['vol'], ws.nC))
        P.fn(cp, 'gpu_p_sgpc_err_accum')(gf, bf, (ws.gpx, ws.gpy, ws.gpz, ws.gax, ws.gay, ws.gaz, cp.float64(C['deltaN']), m['own'], m['nei'], m['w'], m['Sfx'], m['Sfy'], m['Sfz'], m['magSf'], ws.dint, ws.sgpc_err, cp.float64(C.get('fcCorrectTangent', 0.05)), ws.nIF))
        P.fn(cp, 'gpu_p_sgpc_err_update')(gf, bf, (ws.sgpc, ws.deltaS, m['magSf'], ws.sgpc_err, cp.float64(C['sigma']), ws.nIF))
    if _sge:
        _regpc()
        _err_chain()
    _deadband()
    _regpc()
    _clip2()
    return ws.sgpc
    _deadband()
    _regpc()
    if _sge:
        _err_chain()
    _clip2()
    return ws.sgpc

def stats(ws) -> Dict[str, Any]:
    cp = ws.cp
    return {'interface_points': int(ws.mark.sum()), 'max_stf': float(cp.abs(ws.stf).max()), 'max_sgpc': float(cp.abs(ws.sgpc).max())}
