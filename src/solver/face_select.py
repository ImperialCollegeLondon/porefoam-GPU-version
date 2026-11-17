from __future__ import annotations
import numpy as np
STATS = {'select': 0, 'count': 0, 'gather': 0, 'last_len': -1, 'last_n': -1}

def _xp(a):
    if type(a).__module__.split('.')[0] == 'cupy':
        import cupy as _cp
        return _cp
    return np

def select(mask):
    xp = _xp(mask)
    idx = xp.flatnonzero(mask)
    if idx.dtype != xp.int64:
        idx = idx.astype(xp.int64)
    STATS['select'] += 1
    STATS['last_n'] = int(mask.size)
    STATS['last_len'] = int(idx.size)
    return idx

def count(mask) -> int:
    STATS['count'] += 1
    return int(_xp(mask).count_nonzero(mask))

def gather(values, idx, out=None):
    STATS['gather'] += 1
    xp = _xp(values)
    if out is None:
        return xp.take(values, idx)
    return xp.take(values, idx, out=out)
