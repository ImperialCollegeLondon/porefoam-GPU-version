from __future__ import annotations
import hashlib
import os
from typing import List
import numpy as np
NM = 1000000000.0

def _centres(path):
    from pyporefoam import poly_mesh
    m = poly_mesh.read_poly_mesh(path)
    return np.ascontiguousarray(np.asarray(m.cell_centres(), dtype=np.float64))

def _cache_file(case: str, nproc: int) -> str:
    key = []
    files = [os.path.join(case, 'constant', 'polyMesh', f) for f in ('owner', 'neighbour', 'points', 'faces')]
    for p in range(nproc):
        d = os.path.join(case, 'processor%d' % p, 'constant', 'polyMesh')
        files += [os.path.join(d, f) for f in ('owner', 'points', 'faces')]
    for f in files:
        if os.path.isfile(f):
            st = os.stat(f)
            key.append('%s:%d:%d' % (os.path.basename(f), st.st_size, int(st.st_mtime)))
    h = hashlib.sha256(('|'.join(key) + '||addrmap:v1:%d' % nproc).encode()).hexdigest()[:16]
    cdir = os.environ.get('GPU_MESH_CACHE', os.path.join(os.path.expanduser('~'), '.mesh_cache'))
    os.makedirs(cdir, exist_ok=True)
    return os.path.join(cdir, 'a_%s.npz' % h)

def _pair_leftovers(cp, cg, gi):
    out = np.empty(cp.shape[0], dtype=np.int64)
    free = np.ones(cg.shape[0], dtype=bool)
    for i in range(cp.shape[0]):
        d = ((cg - cp[i]) ** 2).sum(1)
        d[~free] = np.inf
        j = int(np.argmin(d))
        out[i] = gi[j]
        free[j] = False
    return out

def cell_map(case: str, nproc: int=None, verbose: bool=False) -> List[np.ndarray]:
    if nproc is None:
        import case_manifest as _MF
        nproc = _MF.discover(case).nproc()
    cf = _cache_file(case, nproc)
    if os.path.exists(cf):
        z = np.load(cf)
        return [z['p%d' % p] for p in range(nproc)]
    cg = _centres(case)
    nC = cg.shape[0]
    cps = [_centres(os.path.join(case, 'processor%d' % p)) for p in range(nproc)]
    if sum((c.shape[0] for c in cps)) != nC:
        raise SystemExit('cell counts disagree: global %d, processors %d' % (nC, sum((c.shape[0] for c in cps))))
    stack = np.vstack([cg] + cps)
    q = np.round(stack * 1000000000.0).astype(np.int64)
    q -= q.min(axis=0)
    n1 = int(q[:, 1].max()) + 1
    n2 = int(q[:, 2].max()) + 1
    kk = q[:, 0] * np.int64(n1 * n2) + q[:, 1] * np.int64(n2) + q[:, 2]
    kg = kk[:nC]
    order = np.argsort(kg)
    ksort = kg[order]
    dup = int(kg.size - np.unique(kg).size)
    if dup:
        raise SystemExit('global centres collide at 1 nm on %d cells' % dup)
    out = []
    off = nC
    taken = np.zeros(nC, dtype=bool)
    misses = []
    for p in range(nproc):
        n = cps[p].shape[0]
        kp = kk[off:off + n]
        off += n
        idx = np.clip(np.searchsorted(ksort, kp), 0, nC - 1)
        hit = ksort[idx] == kp
        a = np.full(n, -1, dtype=np.int64)
        a[hit] = order[idx[hit]]
        taken[a[hit]] = True
        misses.append(np.flatnonzero(~hit))
        out.append(a)
    nmiss = int(sum((m.size for m in misses)))
    if nmiss:
        if nmiss > 20000:
            raise SystemExit('%d cells unmatched on the lattice' % nmiss)
        gi = np.flatnonzero(~taken)
        if gi.size != nmiss:
            raise SystemExit('leftovers disagree: %d local, %d global' % (nmiss, gi.size))
        loc = np.vstack([cps[p][misses[p]] for p in range(nproc) if misses[p].size])
        paired = _pair_leftovers(loc, cg[gi], gi)
        used = 0
        for p in range(nproc):
            m = misses[p]
            if not m.size:
                continue
            out[p][m] = paired[used:used + m.size]
            used += m.size
    allg = np.concatenate(out)
    if not (allg.size == nC and np.unique(allg).size == nC and (allg.min() == 0) and (allg.max() == nC - 1)):
        raise SystemExit('recovered map is not a bijection onto [0,%d)' % nC)
    worst = 0.0
    for p in range(nproc):
        d = np.sqrt(((cps[p] - cg[out[p]]) ** 2).sum(1))
        worst = max(worst, float(d.max()))
    if worst > 1e-10:
        raise SystemExit('recovered map misplaces a cell by %.3e m (tolerance %.1e)' % (worst, 1e-10))
    tmp = cf[:-4] + '.%d.npz' % os.getpid()
    np.savez(tmp, **{'p%d' % p: out[p] for p in range(nproc)})
    os.replace(tmp, cf)
    return out
