from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

@dataclass
class FoamBinaryList:
    header: dict[str, str]
    values: np.ndarray
    primitive: str

def _primitive_dtype(primitive: str, endian: str='<') -> np.dtype:
    if primitive == 'label':
        return np.dtype(endian + 'i4')
    if primitive == 'scalar':
        return np.dtype(endian + 'f8')
    raise ValueError(f'unsupported primitive {primitive!r}')

def _parse_header(buf: bytes) -> tuple[dict[str, str], int]:
    m = re.search(b'FoamFile\\s*\\{(.*?)\\}\\s*', buf, flags=re.S)
    header: dict[str, str] = {}
    end = 0
    if m:
        end = m.end()
        text = m.group(1).decode('ascii', errors='ignore')
        for km in re.finditer('(\\w+)\\s+([^;]+);', text):
            header[km.group(1)] = km.group(2).strip()
    return (header, end)

def read_foam_binary_list(path: str | Path, *, primitive: str | None=None, endian: str='<') -> FoamBinaryList:
    raw = Path(path).read_bytes()
    header, pos = _parse_header(raw)
    rest = raw[pos:]
    m = re.search(b'(\\d+)[ \\t\\r\\n]*\\([ \\t\\r]*\\n?', rest)
    if not m:
        raise ValueError('could not find OpenFOAM binary list size')
    count = int(m.group(1))
    data_start = pos + m.end()
    cls = header.get('class', '')
    prim = primitive or ('vector' if 'vector' in cls else 'label' if 'label' in cls else 'scalar')
    if prim == 'vector':
        dtype = np.dtype(endian + 'f8')
        nbytes = count * 3 * dtype.itemsize
        data = raw[data_start:data_start + nbytes]
        if len(data) != nbytes:
            raise ValueError(f'OpenFOAM binary vector list truncated: expected {nbytes} bytes, got {len(data)}')
        vals = np.frombuffer(data[:nbytes], dtype=dtype).copy().reshape(count, 3)
    else:
        dtype = _primitive_dtype(prim, endian)
        nbytes = count * dtype.itemsize
        data = raw[data_start:data_start + nbytes]
        if len(data) != nbytes:
            raise ValueError(f'OpenFOAM binary {prim} list truncated: expected {nbytes} bytes, got {len(data)}')
        vals = np.frombuffer(data[:nbytes], dtype=dtype).copy()
    return FoamBinaryList(header=header, values=vals, primitive=prim)
