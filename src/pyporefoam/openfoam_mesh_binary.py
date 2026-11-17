from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from .poly_mesh import PolyMesh, PolyPatch, ProcessorPolyPatch
from .openfoam_native_binary import read_foam_binary_list

def read_binary_face_list(path: str | Path) -> list[tuple[int, ...]]:
    raw = Path(path).read_bytes()
    m = re.search(b'(\\d+)\\s*\\(\\s*\\n?', raw)
    if not m:
        raise ValueError('could not locate faceList count')
    n = int(m.group(1))
    pos = m.end()
    faces: list[tuple[int, ...]] = []
    i4 = np.dtype('<i4')
    native_pos = _skip_ws(raw, pos)
    if _looks_like_ascii_sized_binary_face(raw, native_pos):
        pos = native_pos
        for _ in range(n):
            pos = _skip_ws(raw, pos)
            sm = re.match(b'(\\d+)\\s*\\(', raw[pos:])
            if not sm:
                raise ValueError('could not locate OpenFOAM binary face size')
            size = int(sm.group(1))
            pos += sm.end()
            labels = np.frombuffer(raw[pos:pos + 4 * size], dtype=i4, count=size).copy()
            pos += 4 * size
            if pos < len(raw) and raw[pos:pos + 1] == b')':
                pos += 1
            faces.append(tuple((int(x) for x in labels)))
        return faces
    for _ in range(n):
        size = int(np.frombuffer(raw[pos:pos + 4], dtype=i4, count=1)[0])
        pos += 4
        labels = np.frombuffer(raw[pos:pos + 4 * size], dtype=i4, count=size).copy()
        pos += 4 * size
        faces.append(tuple((int(x) for x in labels)))
    return faces

def _skip_ws(raw: bytes, pos: int) -> int:
    while pos < len(raw) and raw[pos] in b' \t\r\n':
        pos += 1
    return pos

def _looks_like_ascii_sized_binary_face(raw: bytes, pos: int) -> bool:
    return re.match(b'\\d+\\s*\\(', raw[pos:]) is not None

def _parse_boundary(text: str) -> list[PolyPatch]:
    patches: list[PolyPatch] = []
    body = text[text.find('(') + 1:text.rfind(')')]
    pat = re.compile('(\\w+)\\s*\\{(.*?)\\}', re.S)
    for m in pat.finditer(body):
        name = m.group(1)
        block = m.group(2)
        kv = {km.group(1): km.group(2).strip() for km in re.finditer('(\\w+)\\s+([^;]+);', block)}
        typ = kv.get('type', 'patch')
        nFaces = int(kv.get('nFaces', '0'))
        startFace = int(kv.get('startFace', '0'))
        if typ == 'processor':
            patches.append(ProcessorPolyPatch(name=name, nFaces=nFaces, startFace=startFace, myProcNo=int(kv.get('myProcNo', '0')), neighbProcNo=int(kv.get('neighbProcNo', '0'))))
        else:
            patches.append(PolyPatch(name=name, type=typ, nFaces=nFaces, startFace=startFace))
    return patches

def read_binary_poly_mesh(case_dir: str | Path) -> PolyMesh:
    poly = Path(case_dir) / 'constant' / 'polyMesh'
    points = read_foam_binary_list(poly / 'points', primitive='vector').values
    faces = read_binary_face_list(poly / 'faces')
    owner = read_foam_binary_list(poly / 'owner', primitive='label').values.astype(int)
    neighbour = read_foam_binary_list(poly / 'neighbour', primitive='label').values.astype(int)
    boundary = _parse_boundary((poly / 'boundary').read_text(encoding='utf-8', errors='replace'))
    return PolyMesh(points=points, faces=faces, owner=owner, neighbour=neighbour, boundary=boundary)
