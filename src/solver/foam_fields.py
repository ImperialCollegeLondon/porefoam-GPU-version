from __future__ import annotations
import os
import re
from typing import List, Tuple
import numpy as np
_HDR = re.compile('^\\s*(//|/\\*|\\||\\\\|=|\\*)', re.M)

def _payload(path: str) -> str:
    with open(path, 'r', errors='replace') as fh:
        txt = fh.read()
    i = txt.find('FoamFile')
    if i >= 0:
        d = txt.find('{', i)
        depth, j = (0, None)
        for q in range(d, len(txt)):
            if txt[q] == '{':
                depth += 1
            elif txt[q] == '}':
                depth -= 1
                if depth == 0:
                    j = q
                    break
        if j is not None:
            txt = txt[j + 1:]
    return txt

def _list_after(txt: str, start: int) -> Tuple[np.ndarray, int]:
    m = re.compile('(\\d+)\\s*\\(', re.S).search(txt, start)
    if not m:
        raise ValueError('no `N (` list found')
    n = int(m.group(1))
    o = m.end()
    depth, close = (1, None)
    for q in range(o, len(txt)):
        c = txt[q]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                close = q
                break
    if close is None:
        raise ValueError('unterminated list')
    body = txt[o:close].replace('(', ' ').replace(')', ' ')
    vals = np.fromstring(body, sep=' ')
    return (vals, n)

def read_label_list(path: str) -> np.ndarray:
    if _fmt(path) == 'binary':
        with open(path, 'rb') as fh:
            raw = fh.read()
        j = raw.find(b'//', raw.find(b'}'))
        m = re.compile(b'(\\d+)\\s*\\(').search(raw, max(j, 0))
        assert m, '%s: no `N (`' % path
        n = int(m.group(1))
        o = m.end()
        for w, dt in ((4, '<i4'), (8, '<i8')):
            if raw[o + n * w:o + n * w + 4].lstrip()[:1] == b')':
                return np.frombuffer(raw[o:o + n * w], dtype=dt).astype(np.int64)
        raise ValueError('%s: neither 32- nor 64-bit labels close the list' % path)
    vals, n = _list_after(_payload(path), 0)
    a = vals.astype(np.int64)
    assert a.size == n, '%s: header says %d, body has %d' % (path, n, a.size)
    return a

def _fmt(path: str) -> str:
    with open(path, 'rb') as fh:
        head = fh.read(2048).decode('latin-1')
    m = re.search('format\\s+(\\w+)\\s*;', head)
    return (m.group(1) if m else 'ascii').lower()

def _read_binary_field(path: str, n_cells: int, ncomp: int):
    with open(path, 'rb') as fh:
        raw = fh.read()
    i = raw.find(b'internalField')
    assert i >= 0, '%s: no internalField' % path
    if raw.find(b'uniform', i, i + 120) >= 0 and raw.find(b'nonuniform', i, i + 120) < 0:
        seg = raw[i:i + 200].decode('latin-1')
        m = re.search('uniform\\s+(\\(?[^;]+?\\)?)\\s*;', seg)
        assert m, '%s: malformed uniform entry' % path
        v = np.fromstring(m.group(1).replace('(', ' ').replace(')', ' '), sep=' ')
        out = np.zeros((n_cells, ncomp), dtype=np.float64)
        out[:, :] = v[:ncomp] if v.size >= ncomp else v[0]
        return out if ncomp > 1 else out[:, 0]
    m = re.compile(b'(\\d+)\\s*\\(').search(raw, i)
    assert m, '%s: no `N (` after internalField' % path
    n = int(m.group(1))
    if n_cells >= 0:
        assert n == n_cells, '%s: %d entries for %d' % (path, n, n_cells)
    n_cells = n
    o = m.end()
    need = n * ncomp * 8
    assert o + need <= len(raw), '%s: wants %d payload bytes, file has %d after the marker' % (path, need, len(raw) - o)
    a = np.frombuffer(raw[o:o + need], dtype='<f8').astype(np.float64)
    tail = raw[o + need:o + need + 4].lstrip()
    assert tail[:1] == b')', '%s: payload does not end at `)` -- offset or width is wrong' % path
    return a if ncomp == 1 else a.reshape(n, ncomp)

def read_internal_field(path: str, n_cells: int, ncomp: int=1):
    if _fmt(path) == 'binary':
        return _read_binary_field(path, n_cells, ncomp)
    txt = _payload(path)
    i = txt.find('internalField')
    assert i >= 0, '%s: no internalField' % path
    head = txt[i:i + 200]
    if 'uniform' in head.split('nonuniform')[0]:
        m = re.search('uniform\\s+(\\(?[^;]+?\\)?)\\s*;', head)
        assert m, '%s: malformed uniform entry' % path
        raw = m.group(1).replace('(', ' ').replace(')', ' ')
        v = np.fromstring(raw, sep=' ')
        out = np.zeros((n_cells, ncomp), dtype=np.float64)
        out[:, :] = v[:ncomp] if v.size >= ncomp else v[0]
        return out if ncomp > 1 else out[:, 0]
    vals, n = _list_after(txt, i)
    assert n == n_cells, '%s: %d entries for %d cells' % (path, n, n_cells)
    if ncomp == 1:
        assert vals.size == n, '%s: %d values for %d' % (path, vals.size, n)
        return vals
    assert vals.size == n * ncomp, '%s: %d values for %d x %d' % (path, vals.size, n, ncomp)
    return vals.reshape(n, ncomp)

def addressing(case: str, nproc: int=None) -> List[np.ndarray]:
    if nproc is None:
        import case_manifest as _MF
        nproc = _MF.discover(case).nproc()
    import mesh_addressing
    return mesh_addressing.cell_map(case, nproc)

def gather(case: str, time: str, name: str, n_global: int, ncomp: int=1, nproc: int=None):
    if nproc is None:
        import case_manifest as _MF
        nproc = _MF.discover(case).nproc()
    addr = addressing(case, nproc)
    out = np.zeros(n_global, dtype=np.float64) if ncomp == 1 else np.zeros((n_global, ncomp), dtype=np.float64)
    seen = np.zeros(n_global, dtype=bool)
    for p in range(nproc):
        fp = os.path.join(case, 'processor%d' % p, time, name)
        loc = read_internal_field(fp, int(addr[p].size), ncomp)
        out[addr[p]] = loc
        seen[addr[p]] = True
    assert seen.all(), '%s: %d cells never written' % (name, int((~seen).sum()))
    return out
