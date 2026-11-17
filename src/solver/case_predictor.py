def prepare(ws, m, F, C, cp):
    import point_layers as P
    import case_inlet as BI
    if not hasattr(ws, 'case_predT'):
        ws.case_predT = cp.zeros_like(ws.ccf.rT)
        ws.case_predbx = cp.zeros(ws.nC)
        ws.case_predby = cp.zeros(ws.nC)
        ws.case_predbz = cp.zeros(ws.nC)
        ws.case_predgx = cp.zeros(ws.nC)
        ws.case_predgy = cp.zeros(ws.nC)
        ws.case_predgz = cp.zeros(ws.nC)
        ws.case_predbf = cp.zeros(ws.nBF)
    face = (ws.p[m['nei']] - ws.p[m['own']]) / m['dmag'] * m['magSf'] - ws.ccf.sgpc
    ws.case_predbf.fill(0.0)
    out = getattr(ws, '_outlet_faces', None)
    if out is not None:
        ws.case_predbf[out] = (ws._outlet_pressure - ws.p[m['bown'][out]]) * m['bdelta'][out] * m['bmagSf'][out]
    ws.case_predT.fill(0.0)
    ws.case_predbx.fill(0.0)
    ws.case_predby.fill(0.0)
    ws.case_predbz.fill(0.0)
    gf, bf = P.grid(ws.nIF)
    gb, bb = P.grid(ws.nBF)
    gc, bc = P.grid(ws.nC)
    P.fn(cp, 'gpu_p_recon_acc2')(gf, bf, (m['own'], m['nei'], m['Sfx'], m['Sfy'], m['Sfz'], m['magSf'], face, ws.case_predT, ws.case_predbx, ws.case_predby, ws.case_predbz, ws.nIF))
    P.fn(cp, 'gpu_p_recon_acc_bnd2')(gb, bb, (m['bown'], m['bSfx'], m['bSfy'], m['bSfz'], m['bmagSf'], ws.case_predbf, ws.case_predT, ws.case_predbx, ws.case_predby, ws.case_predbz, ws.nBF))
    P.fn(cp, 'gpu_p_recon_solve')(gc, bc, (ws.case_predT, ws.case_predbx, ws.case_predby, ws.case_predbz, ws.case_predgx, ws.case_predgy, ws.case_predgz, ws.nC))
    ws.hx -= ws.rau_v * ws.case_predgx
    ws.hy -= ws.rau_v * ws.case_predgy
    ws.hz -= ws.rau_v * ws.case_predgz
    BI.update(ws, m, dict(alpha=F['alpha'], Ux=ws.hx, Uy=ws.hy, Uz=ws.hz), C, cp)
