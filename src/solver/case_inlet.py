from pathlib import Path
import numpy as np

def build(case, m, cp, out):
    from pyporefoam import poly_mesh as PM
    import foam_fields as FF
    import case_manifest as M
    from scipy.spatial import cKDTree
    from scipy import sparse
    import cupyx.scipy.sparse as cs
    case = Path(case)
    patch = next((p for p in M.parse_boundary(str(case / 'constant/polyMesh/boundary')) if p['name'] == 'Left'))
    start, n = (int(patch['startFace']), int(patch['nFaces']))
    faces = cp.asnumpy(m['fp']).reshape(-1, 4)[start:start + n]
    points = cp.asnumpy(cp.column_stack([m['px'], m['py'], m['pz']]))
    vertices = points[faces]
    fc = cp.asnumpy(cp.column_stack([m['cfx'], m['cfy'], m['cfz']]))[start:start + n]
    tree = cKDTree(fc)
    seen = np.zeros(n, bool)
    groups = []
    maxerr = 0.0
    for pd in sorted(case.glob('processor*'), key=lambda p: int(p.name[9:])):
        pm = pd / 'constant/polyMesh'
        pb = next((p for p in M.parse_boundary(str(pm / 'boundary')) if p['name'] == 'Left'))
        nn = pb['nFaces']
        lo = pb['startFace']
        if not nn:
            groups.append(np.empty(0, dtype=np.int64))
            continue
        pts, npnt = FF._list_after(FF._payload(str(pm / 'points')), 0)
        pts = pts.reshape(npnt, 3)
        fs, nf = FF._list_after(FF._payload(str(pm / 'faces')), 0)
        fs = fs.reshape(nf, 5).astype(np.int64)
        assert np.all(fs[:, 0] == 4)
        vv = pts[fs[lo:lo + nn, 1:]]
        local_mesh = PM.PolyMesh(points=pts, faces=fs[lo:lo + nn, 1:], owner=np.zeros(nn, dtype=np.int64), neighbour=np.empty(0, dtype=np.int64))
        local_fc = local_mesh.face_centres()
        dist, idx = tree.query(local_fc)
        err = np.max(np.min(np.linalg.norm(vv[:, :, None, :] - vertices[idx, None, :, :], axis=-1), axis=2))
        maxerr = max(maxerr, float(err))
        assert np.max(dist) < 2e-12 and err < 2e-12, (pd.name, float(np.max(dist)), float(err))
        assert len(np.unique(idx)) == nn and (not seen[idx].any())
        seen[idx] = True
        groups.append(idx)
    assert seen.all()
    rows = []
    cols = []
    data = []
    for sel in groups:
        if not len(sel):
            continue
        ids = faces[sel]
        uq, iv = np.unique(ids, return_inverse=True)
        iv = iv.reshape(len(sel), 4)
        fi = np.repeat(np.arange(len(sel)), 4)
        d = np.linalg.norm(fc[sel][fi] - points[uq][iv.ravel()], axis=1)
        assert np.all(d > 0)
        w = 1 / d
        w /= np.bincount(iv.ravel(), weights=w, minlength=len(uq))[iv.ravel()]
        W = sparse.csr_matrix((w, (iv.ravel(), fi)), shape=(len(uq), len(sel)))
        A = sparse.csr_matrix((np.full(len(fi), 0.25), (fi, iv.ravel())), shape=(len(sel), len(uq)))
        S = (A @ W).tocoo()
        rows.extend(sel[S.row])
        cols.extend(sel[S.col])
        data.extend(S.data)
    S = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    err = float(np.max(np.abs(S @ np.ones(n) - 1)))
    assert err < 1e-12
    lo = start - int(len(m['own']))
    idx = cp.arange(lo, lo + n)
    assert np.array_equal(cp.asnumpy(m['bUt'] == 1), np.isin(np.arange(len(m['bown'])), np.arange(lo, lo + n)))
    m['case_inlet_S'] = cs.csr_matrix(S)
    m['case_inlet_idx'] = idx

def law(alpha, upif, pd, pc, area, S, q0, q1, pdf, pcf, xp):
    upif = 0.99 * upif + 0.01 * (S @ upif)
    X = pdf * pd + pcf * pc
    UB = xp.zeros_like(alpha)
    for phase, q in [(0, q0), (1, q1)]:
        if q <= 1e-27:
            continue
        a = xp.clip(10 * (1 - alpha if phase == 0 else alpha) - 7, 0, 1)
        sa = xp.sum(area * a)
        if float(sa) <= 1e-32:
            continue
        lp = 0.95 * xp.log(xp.maximum(upif / (q / sa) * a, 1e-24)) - 0.05
        p = 0.9 * X
        ash = S @ (a + 1e-09)
        for _ in range(2):
            p = 1.4 * (S @ (p * a + 1e-09)) / ash - 0.4 * p
        p = p + 0.1 * X
        p = a * p + (1 - a) * xp.sum(p * a) / xp.sum(a)
        p = p - xp.min(p) + 10
        p = p / (xp.max(p) + 20)
        lp = lp + pdf * (xp.clip(p, 0.2, 0.8) - p)
        u = xp.sum(a * xp.exp(lp) * area) / sa + 1e-08
        lu = xp.log(u)
        lp = lp + xp.where(lu > 0.1, -0.5 * lu - 0.09, xp.where(lu < -0.1, -0.25 * lu * (1.5 - p) + 0.09, -0.9 * lu))
        u = xp.sum(a * xp.exp(lp) * area) / sa
        lp = xp.minimum(xp.log(10.0) + xp.minimum(xp.log(u + 0.1), 0), lp)
        UB = UB - q / sa * a * xp.exp(lp)
    return UB

def update(ws, m, F, C, cp):
    idx = m['case_inlet_idx']
    own = m['bown'][idx]
    area = m['bmagSf'][idx]
    upif = -(F['Ux'][own] * m['bSfx'][idx] + F['Uy'][own] * m['bSfy'][idx] + F['Uz'][own] * m['bSfz'][idx]) / area
    ub = law(F['alpha'][own], upif, ws.p[own], ws.ccf.pc[own], area, m['case_inlet_S'], float(C['flow_rate0']), float(C['flow_rate']), float(C.get('pd_uniformization', 0.9)), float(C.get('pc_uniformization', 1.0)), cp)
    ws.bphi[idx] = ub * area
