from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import scipy.sparse as sp

def galerkin_map(A: sp.csr_matrix, P: sp.csr_matrix, Ac: sp.csr_matrix):
    A = A.tocsr()
    P = P.tocsr()
    Ac = Ac.tocsr()
    nnz_f = A.nnz
    nCo = Ac.shape[0]
    pos = {}
    for i in range(nCo):
        for t in range(Ac.indptr[i], Ac.indptr[i + 1]):
            pos[i, int(Ac.indices[t])] = t
    Ai, Aj = A.nonzero()
    rows_m, cols_m, vals_m = ([], [], [])
    Pip, Pix, Pdv = (P.indptr, P.indices, P.data)
    for t in range(nnz_f):
        k, l = (int(Ai[t]), int(Aj[t]))
        for a in range(Pip[k], Pip[k + 1]):
            i, pi = (int(Pix[a]), float(Pdv[a]))
            if pi == 0.0:
                continue
            for b in range(Pip[l], Pip[l + 1]):
                j, pj = (int(Pix[b]), float(Pdv[b]))
                if pj == 0.0:
                    continue
                q = pos.get((i, j))
                if q is None:
                    continue
                rows_m.append(q)
                cols_m.append(t)
                vals_m.append(pi * pj)
    M = sp.coo_matrix((vals_m, (rows_m, cols_m)), shape=(Ac.nnz, nnz_f)).tocsr()
    M.sum_duplicates()
    return M

def build_maps(levels: List[Dict[str, Any]]):
    maps = []
    for i in range(1, len(levels)):
        maps.append(galerkin_map(levels[i - 1]['A'], levels[i - 1]['P'], levels[i]['A']))
    return maps

def verify(levels, maps, rng_seed: int=3) -> Dict[str, float]:
    out = {}
    rs = np.random.default_rng(rng_seed)
    A = levels[0]['A'].tocsr().copy()
    A.data = rs.standard_normal(A.nnz)
    for i in range(1, len(levels)):
        P = levels[i - 1]['P']
        ref = (P.T @ A @ P).tocsr()
        got = levels[i]['A'].tocsr().copy()
        got.data = maps[i - 1] @ A.data
        ref.sort_indices()
        got.sort_indices()
        d = float(np.abs(ref.data - got.data).max()) if ref.nnz == got.nnz else float('inf')
        s = max(float(np.abs(ref.data).max()), 1e-300)
        out['L%d' % i] = d / s
        A = got
    return out
