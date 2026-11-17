from __future__ import annotations
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import cupy as cp
import numpy as np
import case_manifest as MANIFEST
import amg_solver as AMG
import face_select as SELECT
import time_step as STEP
import gradient_csr as GRAD
from pyporefoam import poly_mesh

def _manifest_scalar(mf, key, default_value=None):
    got = mf.case_scalar(key) if mf is not None else None
    if got is not None:
        return (float(got), mf.case_scalar_source(key))
    if default_value is not None:
        return (float(default_value), 'default')
    raise SystemExit('ABORT: %s is absent from the case.' % key)

def _wall_contact_angles(mf):
    ang = mf.contact_angles() or {}
    return tuple((n for n in ang if not isinstance(ang[n], list)))

def _wall_theta(mf):
    ang = mf.contact_angles() or {}
    return {n: float(v) for n, v in ang.items() if not isinstance(v, list)}

def _boundary_patch_ranges(case, nIF):
    import re as _re
    _b = open(os.path.join(case, 'constant', 'polyMesh', 'boundary'), 'r', errors='replace').read()
    _pt = _re.compile('(\\w+)\\s*\\{[^{}]*?type\\s+(\\w+)\\s*;[^{}]*?nFaces\\s+(\\d+)\\s*;[^{}]*?startFace\\s+(\\d+)\\s*;[^{}]*?\\}', _re.S)
    out = []
    for _n, _ty, _nf, _sf in _pt.findall(_b):
        if int(_nf) > 0:
            out.append((_n, int(_sf) - int(nIF), int(_nf), _ty))
    return out
import flux_hooks as _FLUX_HOOKS
_FLUX_HOOKS.install()
CASE = os.environ.get('GPU_CASE', '/path/to/case')

def _zng_control_dict(case):
    import re as _re
    for _p in (os.path.join(case, 'system', 'controlDict'), os.path.join(case, 'processor0', 'system', 'controlDict')):
        if os.path.exists(_p):
            _txt = open(_p, encoding='utf-8', errors='replace').read()
            _out = {}
            for _k in ('maxCo', 'maxAlphaCo', 'maxDeltaT', 'deltaT'):
                _m = _re.search(_k + '\\s+([0-9eE.+-]+)\\s*;', _txt)
                if _m is None:
                    raise SystemExit('ABORT: %s is absent from %s and reads it with no default' % (_k, _p))
                _out[_k] = float(_m.group(1))
            return _out
    raise SystemExit('ABORT: no system/controlDict under ' + case)
_TRANSITION_FACTORS = {}

def _wall_faces(case, nIF, nBF, mf=None):
    if mf is None:
        mf = MANIFEST.discover(case)
    sel_mask, names = (np.zeros(int(nBF), dtype=bool), [])
    for nm, lo, nf in mf.wall_patch_ranges(nIF):
        assert 0 <= lo and lo + nf <= nBF, 'patch %s: faces [%d,%d) fall outside the %d boundary faces' % (nm, lo, lo + nf, nBF)
        sel_mask[lo:lo + nf] = True
        names.append('%s(%d)' % (nm, nf))
    sel_mask.fill(False)
    names.clear()
    _ca = _wall_contact_angles(mf)
    for _zn, _zlo, _znf, _zty in _boundary_patch_ranges(case, nIF):
        if _zn in _ca and (not bool(sel_mask[_zlo])):
            sel_mask[_zlo:_zlo + _znf] = True
            names.append('%s(%d)' % (_zn, _znf))
    loc = SELECT.select(sel_mask)
    mf.theta_deg()
    _ang = mf.contact_angles()
    _cf = np.full(int(nBF), np.nan)
    for _n2, _lo2, _nf2, _ty2 in _boundary_patch_ranges(case, nIF):
        if not sel_mask[_lo2:_lo2 + _nf2].any():
            continue
        _a2 = _ang.get(_n2)
        if _a2 is None or isinstance(_a2, list):
            raise SystemExit('ABORT: patch %s is in the contact-angle set but the case gives %r for its angle' % (_n2, _a2))
        _cf[_lo2:_lo2 + _nf2] = float(_a2)
    _sel = _cf[loc]
    if not np.isfinite(_sel).all():
        raise SystemExit('ABORT: %d selected wall faces have no contact angle' % int((~np.isfinite(_sel)).sum()))
    return (loc, np.cos(np.radians(_sel)))

def _mesh_arrays(case: str):
    pm = os.path.join(case, 'constant', 'polyMesh')
    key = []
    for fn in sorted(os.listdir(pm)) if os.path.isdir(pm) else []:
        fp = os.path.join(pm, fn)
        if os.path.isfile(fp):
            st = os.stat(fp)
            key.append(f'{fn}:{st.st_size}:{int(st.st_mtime)}')
    import hashlib
    SCHEMA = 'mesh:nC,nIF,own,nei,vol,mag,dmag,Sf,ctr,bown,bSf,bCf,pnm,plo,pnf'
    SCHEMA = SCHEMA + '|Cfi,dl'
    h = hashlib.sha256(('|'.join(key) + '||' + SCHEMA).encode()).hexdigest()[:16]
    cdir = os.environ.get('GPU_MESH_CACHE', os.path.join(os.path.expanduser('~'), '.mesh_cache'))
    os.makedirs(cdir, exist_ok=True)
    cf = os.path.join(cdir, f'm_{h}.npz')
    if os.path.exists(cf):
        z = np.load(cf)
        out = (int(z['nC']), int(z['nIF']), z['own'], z['nei'], z['vol'], z['mag'], z['dmag'], z['Sf'], z['ctr'], z['bown'], z['bSf'], z['bCf'], z['pnm'], z['plo'], z['pnf'], z['Cfi'], z['dl'])
        return out
    mesh = poly_mesh.read_poly_mesh(case)
    nC = int(mesh.nCells)
    nIF = int(mesh.nInternalFaces)
    own_h = np.ascontiguousarray(mesh.owner[:nIF], dtype=np.int64)
    nei_h = np.ascontiguousarray(mesh.neighbour[:nIF], dtype=np.int64)
    vol_h = np.ascontiguousarray(mesh.cell_volumes(), dtype=np.float64)
    Sf_all = np.asarray(mesh.face_area_vectors(), dtype=np.float64)
    Sf = Sf_all[:nIF]
    bown = np.ascontiguousarray(mesh.owner[nIF:], dtype=np.int64)
    bSf = np.ascontiguousarray(Sf_all[nIF:], dtype=np.float64)
    _pts = np.asarray(mesh.points, dtype=np.float64)
    _fc = []
    for _f in mesh.faces[nIF:]:
        _v = _pts[np.asarray(_f, dtype=np.int64)]
        _fc.append(_v.mean(axis=0))
    bCf = np.ascontiguousarray(np.asarray(_fc, dtype=np.float64))
    _pp = [pp for pp in mesh.boundary if int(pp.nFaces) > 0]
    pnm = np.array([str(pp.name) for pp in _pp])
    plo = np.array([int(pp.startFace) - nIF for pp in _pp], dtype=np.int64)
    pnf = np.array([int(pp.nFaces) for pp in _pp], dtype=np.int64)
    mag = np.sqrt((Sf * Sf).sum(axis=1))
    ctr = np.asarray(mesh.cell_centres(), dtype=np.float64)
    dvec = ctr[nei_h] - ctr[own_h]
    dmag_h = np.sqrt((dvec * dvec).sum(axis=1))
    _zu = Sf / np.maximum(mag, 1e-300)[:, None]
    dmag_h = np.maximum((_zu * dvec).sum(axis=1), 0.05 * dmag_h)
    dmag_h[dmag_h == 0.0] = 1.0
    dl_h = np.sqrt((dvec * dvec).sum(axis=1))
    dl_h[dl_h == 0.0] = 1.0
    Cfi = np.ascontiguousarray(np.asarray(mesh.face_centres(), dtype=np.float64)[:nIF])
    tmp = cf[:-4] + f'.{os.getpid()}.npz'
    np.savez(tmp, nC=nC, nIF=nIF, own=own_h, nei=nei_h, vol=vol_h, mag=mag, dmag=dmag_h, Sf=Sf, ctr=ctr, bown=bown, bSf=bSf, bCf=bCf, pnm=pnm, plo=plo, pnf=pnf, Cfi=Cfi, dl=dl_h)
    os.replace(tmp, cf)
    return (nC, nIF, own_h, nei_h, vol_h, mag, dmag_h, Sf, ctr, bown, bSf, bCf, pnm, plo, pnf, Cfi, dl_h)

def main() -> int:
    MF = MANIFEST.discover(CASE)
    from case_driver import transition_factors
    _TRANSITION_FACTORS.clear()
    _TRANSITION_FACTORS.update(transition_factors(CASE, MF))
    nC, nIF, own_h, nei_h, vol_h, mag, dmag_h, Sf, ctr, bown_h, bSf_h, bCf_h, pnm, plo, pnf, Cfi_h, dl_h = _mesh_arrays(CASE)
    nBF = int(bown_h.size)
    U_BC = MF.u_bc()
    A_BC = MF.a_bc()
    P_BC = MF.p_bc()
    bUt_h = np.zeros(nBF, dtype=np.int64)
    bPt_h = np.zeros(nBF, dtype=np.int64)
    bAt_h = np.zeros(nBF, dtype=np.int64)
    for _nm, _lo, _n in zip(pnm, plo, pnf):
        bUt_h[_lo:_lo + _n] = int(U_BC.get(str(_nm), 0))
        bPt_h[_lo:_lo + _n] = int(P_BC.get(str(_nm), 0))
        bAt_h[_lo:_lo + _n] = int(A_BC.get(str(_nm), 0))
    bmag_h = np.sqrt((bSf_h * bSf_h).sum(axis=1))
    bnh_h = bSf_h / np.maximum(bmag_h, 1e-300)[:, None]
    bd_h = ((bCf_h - ctr[bown_h]) * bnh_h).sum(axis=1)
    assert bd_h.min() > 0.0, 'boundary face centre is not outside its owner cell: min %.3e' % bd_h.min()
    bdelta_h = 1.0 / bd_h
    _P = {str(a): (int(b), int(c)) for a, b, c in zip(pnm, plo, pnf)}
    _lo, _n = _P[MF.inlet_patch()]
    _area = float(bmag_h[_lo:_lo + _n].sum())
    MF.attach_mesh(nC, nIF, nBF, bmag_h, plo, pnf, pnm)
    _nx = MF.left_n()
    _ax = MF.left_a()
    assert _n == _nx, 'Left has %d faces, expected %d' % (_n, _nx)
    assert abs(_area / _ax - 1.0) < 1e-06, 'Left area %.6e' % _area
    wall_local, costh = _wall_faces(CASE, nIF, nBF, MF)
    _wcell_face = bown_h[wall_local]
    _wcells, _inv = np.unique(_wcell_face, return_inverse=True)
    _acc = np.zeros((_wcells.size, 3), dtype=np.float64)
    for _k in range(3):
        _acc[:, _k] = np.bincount(_inv, weights=bSf_h[wall_local][:, _k], minlength=_wcells.size)
    _wm_ = np.sqrt((_acc * _acc).sum(axis=1))
    _keep = _wm_ > 0.0
    _wcells, _acc, _wm_ = (_wcells[_keep], _acc[_keep], _wm_[_keep])
    _wn = _acc / _wm_[:, None]
    _am = np.sqrt((bSf_h[wall_local] ** 2).sum(axis=1))
    _cnum = np.bincount(_inv, weights=_am * costh, minlength=_keep.size)[_keep]
    _cden = np.bincount(_inv, weights=_am, minlength=_keep.size)[_keep]
    costh = _cnum / np.maximum(_cden, 1e-300)
    nW = int(_wcells.size)
    dvec = ctr[nei_h] - ctr[own_h]
    _zdo = Cfi_h - ctr[own_h]
    _zdn = ctr[nei_h] - Cfi_h
    _zso = np.abs((Sf * _zdo).sum(axis=1))
    _zsn = np.abs((Sf * _zdn).sum(axis=1))
    w_h = _zsn / np.maximum(_zso + _zsn, 1e-300)
    assert float(np.abs(w_h - 0.5).max()) > 1e-12, 'all face weights equal 0.5'

    def f(a):
        return cp.asarray(np.ascontiguousarray(a, dtype=np.float64))
    _cfcsr = {}
    m = dict(own=cp.asarray(own_h), nei=cp.asarray(nei_h), vol=f(vol_h), magSf=f(mag), dmag=f(dmag_h), Sfx=f(Sf[:, 0]), Sfy=f(Sf[:, 1]), Sfz=f(Sf[:, 2]), w=f(w_h), mu_f=f(np.full(nIF, 0.001)), dl=f(dl_h), dx=f(dvec[:, 0]), dy=f(dvec[:, 1]), dz=f(dvec[:, 2]), bown=cp.asarray(bown_h), bSfx=f(bSf_h[:, 0]), bSfy=f(bSf_h[:, 1]), bSfz=f(bSf_h[:, 2]), bmagSf=cp.asarray(bmag_h), bdelta=cp.asarray(bdelta_h), bddot=cp.asarray(((bCf_h - ctr[bown_h]) ** 2).sum(axis=1)), bUt=cp.asarray(bUt_h), bPt=cp.asarray(bPt_h), **_cfcsr, **dict(bAt=cp.asarray(bAt_h)), **{}, wall_cells=cp.asarray(_wcells), wnx=f(_wn[:, 0]), wny=f(_wn[:, 1]), wnz=f(_wn[:, 2]), costheta=f(costh), wall_faces=cp.asarray(np.ascontiguousarray(nIF + wall_local, dtype=np.int64)), wall_nfx=f(bSf_h[wall_local][:, 0] / np.maximum(bmag_h[wall_local], 1e-300)), wall_nfy=f(bSf_h[wall_local][:, 1] / np.maximum(bmag_h[wall_local], 1e-300)), wall_nfz=f(bSf_h[wall_local][:, 2] / np.maximum(bmag_h[wall_local], 1e-300)))
    _zg_h, _zg_rep = GRAD.build_host(own_h, nei_h, vol_h, Sf, ctr, bown_h, bSf_h, bCf_h, Cfi_h, plo)
    GRAD.install(GRAD.to_device(cp, _zg_h))
    m['gradient_state'] = GRAD.DEV
    m['gradient_report'] = _zg_rep
    afh = mag / dmag_h
    dg = np.zeros(nC)
    np.add.at(dg, own_h, afh)
    np.add.at(dg, nei_h, afh)
    A = AMG.ldu_to_csr(nC, own_h, nei_h, dg, -afh)
    lv = AMG.build(A, verbose=True)
    ws = STEP.StepWorkspace(cp, n_cells=nC, n_internal_faces=nIF, mean_cell_volume=float(vol_h.mean()), pressure_levels=lv, n_boundary_faces=nBF, n_wall=nW)
    import foam_fields as FF
    alpha0 = FF.gather(CASE, '0', 'alpha1', nC, 1)
    u0 = FF.gather(CASE, '0', 'U', nC, 3)
    fields = {'alpha': cp.asarray(alpha0), 'Ux': cp.asarray(np.ascontiguousarray(u0[:, 0])), 'Uy': cp.asarray(np.ascontiguousarray(u0[:, 1])), 'Uz': cp.asarray(np.ascontiguousarray(u0[:, 2])), 'phi': cp.zeros(nIF, dtype=cp.float64), 'src_U': cp.zeros(nC, dtype=cp.float64)}
    h_cell = float(vol_h.mean()) ** 0.3333333333333333
    dt_cap = float(np.sqrt(1000.0 * h_cell ** 3 / (2.0 * np.pi * 0.03)))
    dt_use = min(1e-06, 0.5 * dt_cap)
    dt_use = dt_use / 1.0
    _zng_dt0 = _zng_control_dict(CASE).get('deltaT')
    if _zng_dt0 and _zng_dt0 > 0.0:
        dt_use = _zng_dt0
    ctl = {'dt': dt_use, 'sigma': _manifest_scalar(MF, 'sigma', None)[0], 'c_alpha': _manifest_scalar(MF, 'cAlpha', None)[0], 'limiter_iterations': 3, 'rho1': 1000.0, 'rho2': 1000.0, 'mu1': 0.001, 'mu2': 0.001, 'alpha_bnd': 1.0, 'alpha_inlet': 1.0, 'wall_drag': 1.0, 'inlet_log': 1.0, 'cbc': _manifest_scalar(MF, 'cBC', None)[0], 'cbc_cap': 0.0, 'n_corr': int(_manifest_scalar(MF, 'nCorrectors', None)[0]), 'n_non_orth': 1, 'p_close': 1, 'flow_rate': MF.flow_rate(), 'flow_rate0': MF.flow_rate0(), 'u_recon': 1, 'sgpc_err': 1.0, 'u_filter': _manifest_scalar(MF, 'UBoundByUSmoothFactor', 2.0)[0], 'u_cap': 1.0, 'pressure_cycles': 12}
    import galerkin_map as GALERKIN
    gmaps = GALERKIN.build_maps(lv)
    kind, idx = AMG.csr_source_map(nC, own_h, nei_h, lv[0]['A'])
    AMG.attach_device_refresh(cp, ws.s3, lv, gmaps, nC, m['own'], m['nei'], nIF, kind, idx)
    AMG.attach_boundary(cp, ws.s3, bown=m['bown'], bmag=m['bmagSf'], bdelta=m['bdelta'], bPt=m['bPt'])
    ws.s3.coarse_jacobi = 60

    def refresh():
        ws.psolver.update(af=ws.af, rau_v=ws.rau_v)
    import ccf as CCF
    import point_layers as PT
    import amg_solver as AMG3
    pts_h, fp_h, cf_h = PT.point_layer(CASE)
    nP = int(pts_h.shape[0])
    import re as _re
    _b = open(os.path.join(CASE, 'constant', 'polyMesh', 'boundary'), 'r', errors='replace').read()
    _pt = _re.compile('(\\w+)\\s*\\{[^{}]*?type\\s+(\\w+)\\s*;[^{}]*?nFaces\\s+(\\d+)\\s*;[^{}]*?startFace\\s+(\\d+)\\s*;[^{}]*?\\}', _re.S)
    _wm = np.zeros(int(nBF), dtype=bool)
    _ca5 = _wall_contact_angles(MF)
    for _n, ty, nf, sf in _pt.findall(_b):
        if (ty == 'wall' or _n in _ca5) and int(nf) > 0:
            _lo = int(sf) - nIF
            _wm[_lo:_lo + int(nf)] = True
    wloc = SELECT.select(_wm)
    _e, fe_h, marks_h = PT.edge_layer(CASE, fp_h, nP, nIF, wloc)
    _fe4 = np.asarray(fe_h, dtype=np.int64).reshape(-1, 4)
    _g4 = nIF + np.arange(int(nBF), dtype=np.int64)
    _mk = np.zeros(marks_h.shape, dtype=np.float64)
    _mk[_fe4[_g4].ravel()] = 1.0
    marks_h = _mk
    _tf = np.ones(int(nBF), dtype=np.float64)
    _tfm = np.zeros(int(nBF), dtype=bool)
    for _zn, _zlo, _znf, _zty in _boundary_patch_ranges(CASE, nIF):
        if _zn in _TRANSITION_FACTORS:
            _tf[_zlo:_zlo + _znf] = _TRANSITION_FACTORS[_zn]
            _tfm[_zlo:_zlo + _znf] = True
    m['transition_factor'] = f(_tf)
    m['transition_mask'] = cp.asarray(_tfm)
    m['fp'] = cp.asarray(np.ascontiguousarray(fp_h, dtype=np.int64))
    m['fe'] = cp.asarray(np.ascontiguousarray(fe_h, dtype=np.int64))
    m['marks'] = f(marks_h)
    m['px'], m['py'], m['pz'] = (f(pts_h[:, 0]), f(pts_h[:, 1]), f(pts_h[:, 2]))
    m['cfx'], m['cfy'], m['cfz'] = (f(cf_h[:, 0]), f(cf_h[:, 1]), f(cf_h[:, 2]))
    m['a1s_bnd'] = cp.zeros(nBF)
    _ca7 = _wall_contact_angles(MF)
    _th7 = _wall_theta(MF)
    _zpc = np.zeros(nP, dtype=np.int64)
    _zth = np.full(nP, float(MF.theta_deg()))
    _zfp4 = np.ascontiguousarray(fp_h, dtype=np.int64)
    for _zn, _zlo, _znf, _zty in _boundary_patch_ranges(CASE, nIF):
        _zg = nIF + _zlo + np.arange(_znf, dtype=np.int64)
        _zp = _zfp4[np.repeat(4 * _zg, 4) + np.tile(np.arange(4), _znf)]
        if _zn in _ca7:
            _zpc[_zp] = 1
            _zth[_zp] = float(_th7[_zn])
        else:
            _zpc[_zp] = 2
    m['point_patch_class'] = cp.asarray(_zpc)
    m['point_patch_sin'] = f(np.sin(np.radians(_zth)))
    m['point_patch_cos'] = f(np.cos(np.radians(_zth)))
    af_pc = mag / np.maximum(dmag_h, 1e-300)
    dgp = np.zeros(nC)
    np.add.at(dgp, own_h, af_pc)
    np.add.at(dgp, nei_h, af_pc)
    lvp = AMG3.build(AMG3.ldu_to_csr(nC, own_h, nei_h, dgp, -af_pc))
    ws.ccf = CCF.CCFWorkspace(cp, nC, nIF, nP, int(fp_h.size // 4), pc_levels=AMG3.AmgWorkspace(cp, lvp), n_boundary_faces=nBF)
    ws.ccf_s3 = AMG3
    ctl.update({'pd_filter': _manifest_scalar(MF, 'pBoundBypSmoothFactor', 1.3)[0], 'filter_u_on': _manifest_scalar(MF, 'filterU', 1.0)[0], 'filter_pd_on': _manifest_scalar(MF, 'filterPd', 1.0)[0], 'fcCorrectTangent': _manifest_scalar(MF, 'fcCorrectTangent')[0], 'fcCorrectTangentRelax': _manifest_scalar(MF, 'fcCorrectTangentRelax')[0]})
    ctl.update({'pcThicknessFactor': _manifest_scalar(MF, 'pcThicknessFactor')[0], 'smoothingKernel': int(_manifest_scalar(MF, 'smoothingKernel')[0]), 'theta_deg': MF.theta_deg(), 'wallSmoothingKernel': _manifest_scalar(MF, 'wallSmoothingKernel')[0], 'smoothingRelaxFactor': _manifest_scalar(MF, 'smoothingRelaxFactor')[0], 'fcdFilter': _manifest_scalar(MF, 'fcdFilter')[0], 'pc_cycles': 16, 'nhat_cut': 0.0, 'deltaN': __import__('curvature_kernels').delta_n(float(vol_h.mean()))})
    ctl['refresh'] = refresh
    import curvature_kernels as _CURVK
    ctl['contrast'] = 0.0001 * _CURVK.delta_n(float(vol_h.mean()))
    from case_driver import run
    return run(ws=ws, mesh=m, fields=fields, control=ctl, manifest=MF, case=CASE)
if __name__ == '__main__':
    raise SystemExit(main())
