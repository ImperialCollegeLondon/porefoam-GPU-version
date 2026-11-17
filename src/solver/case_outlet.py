def momentum(ws, m, F, C, cp):
    mask = m['case_right_io_mask']
    inflow = mask & (ws.bphi < 0)
    ws.case_io_inflow = inflow
    rho = float(C['rho1'])
    mu = float(C['mu1'])
    assert rho == float(C['rho2']) and mu == float(C['mu2'])
    coefficient = cp.where(inflow, mu * m['bmagSf'] * m['bdelta'], cp.where(mask, rho * ws.bphi, 0.0))
    ws.diag += cp.bincount(m['bown'], weights=coefficient, minlength=ws.nC)[:ws.nC]
    il = m['bUt'] == 1
    factor = cp.where(il, -rho * ws.bphi ** 2 / cp.maximum(m['bmagSf'] ** 2, 1e-300), 0.0)
    for dst, sf in [(ws.sx, m['bSfx']), (ws.sy, m['bSfy']), (ws.sz, m['bSfz'])]:
        dst += cp.bincount(m['bown'], weights=factor * sf, minlength=ws.nC)[:ws.nC]

def apply_hbya_boundary(ws, cp):
    ws.bphi[ws.case_io_inflow] = 0.0
