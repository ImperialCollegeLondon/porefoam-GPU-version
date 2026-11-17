from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import math
import re
import numpy as np


def _as_vector3(value: object, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).reshape(3)
    except Exception as exc:
        raise ValueError(f'{name} must be a 3-vector') from exc

def _rodrigues_rotation_tensor(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float).reshape(3)
    mag = float(np.linalg.norm(a))
    if mag < 1e-15:
        raise ValueError(f'Incorrect rotation axis: {axis!r}')
    a = a / mag
    x, y, z = a
    c = float(np.cos(float(angle)))
    s = float(np.sin(float(angle)))
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return c * np.eye(3) + s * skew + (1.0 - c) * np.outer(a, a)

def _rotation_tensor_from_normals(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    n1 = np.asarray(n1, dtype=float).reshape(3)
    n2 = np.asarray(n2, dtype=float).reshape(3)
    cross = np.cross(n1, n2)
    cross_mag_sqr = float(np.dot(cross, cross))
    dot = float(np.dot(n1, n2))
    return dot * np.eye(3) + (1.0 - dot) * np.outer(cross, cross) / (cross_mag_sqr + 1e-300) + np.outer(n2, n1) - np.outer(n1, n2)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)

def _patch_primitive_addressing(mesh: 'PolyMesh', patch: 'PolyPatch') -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]:
    point_to_local: dict[int, int] = {}
    mesh_points: list[int] = []
    local_faces: list[tuple[int, ...]] = []
    edge_to_local: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    face_edges: list[tuple[int, ...]] = []
    for facei in patch.faces:
        pts = tuple((int(p) for p in mesh.faces[int(facei)]))
        local_face: list[int] = []
        for pointi in pts:
            if pointi not in point_to_local:
                point_to_local[pointi] = len(mesh_points)
                mesh_points.append(pointi)
            local_face.append(point_to_local[pointi])
        local = tuple(local_face)
        local_faces.append(local)
        local_face_edges: list[int] = []
        for a, b in zip(local, local[1:] + local[:1]):
            key = _edge_key(a, b)
            if key not in edge_to_local:
                edge_to_local[key] = len(edges)
                edges.append(key)
            local_face_edges.append(edge_to_local[key])
        face_edges.append(tuple(local_face_edges))
    return (tuple(local_faces), tuple(mesh_points), tuple(edges), tuple(face_edges))

def _edge_pair_array(pairs: Iterable[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(tuple(((int(a), int(b)) for a, b in pairs)), dtype=int)
    if arr.size == 0:
        return np.empty((0, 2), dtype=int)
    return arr.reshape(-1, 2)

def _patch_vector_array(value: object, name: str, n_faces: int) -> np.ndarray:
    if value is None:
        raise KeyError(name)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (int(n_faces), 3):
        raise ValueError(f'{name} must have shape (nFaces, 3)')
    return arr.astype(float, copy=True)

def _label_array(value: object, name: str) -> np.ndarray:
    if value is None:
        raise KeyError(name)
    return np.asarray(value, dtype=int).reshape(-1)



@dataclass
class PolyPatch:
    name: str
    type: str = 'patch'
    nFaces: int = 0
    startFace: int = 0
    physicalType: str | None = None
    inGroups: tuple[str, ...] = ()
    attributes: dict[str, object] = field(default_factory=dict)
    _geometry_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _addressing_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _coupled_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)

    @property
    def endFace(self) -> int:
        return self.startFace + self.nFaces

    @property
    def faces(self) -> range:
        return range(self.startFace, self.endFace)


    def calcFaceTol(self, mesh: 'PolyMesh', face_centres: np.ndarray | None=None) -> np.ndarray:
        centres = mesh.face_centres()[np.asarray(list(self.faces), dtype=int)] if face_centres is None else np.asarray(face_centres, dtype=float)
        if centres.shape != (int(self.nFaces), 3):
            raise ValueError(f'polyPatch::calcFaceTol faceCentres must have shape ({self.nFaces}, 3)')
        match_tol = float(self.attributes.get('matchTolerance', getattr(self, 'matchTolerance', 0.0001)))
        tols = np.zeros(int(self.nFaces), dtype=float)
        for patch_face_i, face_i in enumerate(self.faces):
            cc = centres[int(patch_face_i)]
            pts = mesh.face_points(int(face_i))
            max_len = max((float(np.linalg.norm(point - cc)) for point in pts), default=0.0)
            tols[int(patch_face_i)] = match_tol * max_len
        return tols

    def getAnchorPoints(self, mesh: 'PolyMesh') -> np.ndarray:
        anchors = np.zeros((int(self.nFaces), 3), dtype=float)
        for patch_face_i, face_i in enumerate(self.faces):
            face = mesh.faces[int(face_i)]
            if not face:
                raise ValueError(f'polyPatch::getAnchorPoints face {int(face_i)} contains no points')
            anchors[int(patch_face_i)] = mesh.points[int(face[0])]
        return anchors


    def calcTransformTensors(self, Cf: np.ndarray, Cr: np.ndarray, nf: np.ndarray, nr: np.ndarray, smallDist: np.ndarray, absTol: float | None=None) -> None:
        cf = np.asarray(Cf, dtype=float).reshape(-1, 3)
        cr = np.asarray(Cr, dtype=float).reshape(-1, 3)
        n_face = cf.shape[0]
        if cr.shape != (n_face, 3):
            raise ValueError('coupledPolyPatch::calcTransformTensors Cr size mismatch')
        n_f = np.asarray(nf, dtype=float).reshape(n_face, 3)
        n_r = np.asarray(nr, dtype=float).reshape(n_face, 3)
        small_dist = np.asarray(smallDist, dtype=float).reshape(-1)
        if small_dist.shape != (n_face,):
            raise ValueError('coupledPolyPatch::calcTransformTensors smallDist size mismatch')
        for key in ('forwardT', 'reverseT', 'separation', 'separated', 'parallel'):
            self.attributes.pop(key, None)
            self._coupled_cache.pop(key, None)
        if n_face == 0:
            separation = np.empty((0, 3), dtype=float)
            forward = np.empty((0, 3, 3), dtype=float)
            reverse = np.empty((0, 3, 3), dtype=float)
            self._coupled_cache['separation'] = separation
            self._coupled_cache['forwardT'] = forward
            self._coupled_cache['reverseT'] = reverse
            self._coupled_cache['separated'] = False
            self._coupled_cache['parallel'] = False
            self._coupled_cache['calcTransformTensorsStatus'] = 'geometry-derived-dummy'
            return None
        tolerance = float(self.attributes.get('matchTolerance', getattr(self, 'matchTolerance', 0.0001))) if absTol is None else float(absTol)
        error = tolerance * math.sqrt(float(n_face))
        if float(np.sum(np.abs(np.einsum('ij,ij->i', n_f, n_r)))) < float(n_face) - error:
            separation = np.empty((0, 3), dtype=float)
            forward = np.asarray([_rotation_tensor_from_normals(-n_r[face_i], n_f[face_i]) for face_i in range(n_face)], dtype=float)
            reverse = np.asarray([_rotation_tensor_from_normals(n_f[face_i], -n_r[face_i]) for face_i in range(n_face)], dtype=float)
            if float(np.sum(np.abs(forward - forward[0]))) < error:
                forward = forward[:1].copy()
                reverse = reverse[:1].copy()
            self.attributes['forwardT'] = forward.copy()
            self.attributes['reverseT'] = reverse.copy()
            self.attributes['parallel'] = False
            self.attributes['separated'] = False
            self._coupled_cache['forwardT'] = forward.copy()
            self._coupled_cache['reverseT'] = reverse.copy()
            self._coupled_cache['separation'] = separation
            self._coupled_cache['parallel'] = False
            self._coupled_cache['separated'] = False
            self._coupled_cache['calcTransformTensorsStatus'] = 'geometry-derived-rotation'
            return None
        for key in ('forwardT', 'reverseT'):
            self.attributes.pop(key, None)
            self._coupled_cache.pop(key, None)
        separation = np.einsum('ij,ij->i', n_f, cr - cf)[:, None] * n_f
        same_separation = True
        for face_i in range(n_face):
            small_sqr = float(small_dist[face_i]) ** 2
            if float(np.dot(separation[face_i] - separation[0], separation[face_i] - separation[0])) > small_sqr:
                same_separation = False
                break
        if same_separation:
            if float(np.dot(separation[0], separation[0])) < float(small_dist[0]) ** 2:
                separation = np.empty((0, 3), dtype=float)
            else:
                separation = separation[:1].copy()
        separated = bool(separation.shape[0] > 0)
        if separated:
            self.attributes['separation'] = separation.copy()
        self.attributes['separated'] = separated
        self.attributes['parallel'] = True
        self._coupled_cache['separation'] = separation.copy()
        self._coupled_cache['separated'] = separated
        self._coupled_cache['parallel'] = True
        self._coupled_cache['calcTransformTensorsStatus'] = 'geometry-derived-separation'
        return None

    def calcTransforms(self) -> None:
        if self.type not in {'cyclic', 'cyclicAMI'}:
            return None
        if self.nFaces == 0:
            return None
        if self.nFaces % 2:
            raise ValueError(f'cyclicPolyPatch::calcTransforms requires an even face count for patch {self.name!r}')
        transform = str(self.attributes.get('transform', self.attributes.get('transformType', 'unknown'))).strip().lower()
        if transform in {'rotational', 'rotation'}:
            axis = _as_vector3(self.attributes.get('rotationAxis'), 'rotationAxis')
            centre = _as_vector3(self.attributes.get('rotationCentre'), 'rotationCentre')
            angle = float(self.attributes.get('rotationAngle'))
            axis_mag = float(np.linalg.norm(axis))
            if axis_mag < 1e-15:
                raise ValueError(f'Incorrect rotation axis: {axis!r}')
            axis = axis / axis_mag
            forward = _rodrigues_rotation_tensor(axis, angle)
            reverse = _rodrigues_rotation_tensor(axis, -angle)
            self.attributes['transform'] = 'rotational'
            self.attributes['rotationAxis'] = axis
            self.attributes['rotationCentre'] = centre
            self.attributes['rotationAngle'] = angle
            self.attributes['forwardT'] = forward
            self.attributes['reverseT'] = reverse
            self.attributes.pop('separated', None)
            self._coupled_cache['forwardT'] = forward
            self._coupled_cache['reverseT'] = reverse
            self._coupled_cache['calcTransformsStatus'] = 'rotational'
            return None
        if transform in {'translational', 'translation'}:
            separation_value = self.attributes.get('separationVector', self.attributes.get('separation'))
            separation = _as_vector3(separation_value, 'separationVector')
            self.attributes['transform'] = 'translational'
            self.attributes['separationVector'] = separation
            self.attributes['separation'] = separation
            self.attributes['separated'] = True
            for key in ('forwardT', 'reverseT'):
                self.attributes.pop(key, None)
                self._coupled_cache.pop(key, None)
            self._coupled_cache['separationVector'] = separation
            self._coupled_cache['separation'] = separation
            self._coupled_cache['calcTransformsStatus'] = 'translational'
            return None
        if 'forwardT' in self.attributes and 'reverseT' in self.attributes:
            self._coupled_cache['forwardT'] = np.asarray(self.attributes['forwardT'], dtype=float).reshape(3, 3)
            self._coupled_cache['reverseT'] = np.asarray(self.attributes['reverseT'], dtype=float).reshape(3, 3)
            self._coupled_cache['calcTransformsStatus'] = 'explicit'
            return None
        raise KeyError(f'cyclicPolyPatch::calcTransforms requires prescribed transforms for patch {self.name!r}')

    def _cache_computed_coupled_pairs(self, key: str, pairs: Iterable[tuple[int, int]]) -> np.ndarray:
        arr = _edge_pair_array(pairs)
        self._coupled_cache[key] = arr
        self.attributes[key] = tuple(((int(a), int(b)) for a, b in arr))
        computed = set(self._coupled_cache.get('_computedCoupledAddressingKeys', ()))
        computed.add(key)
        self._coupled_cache['_computedCoupledAddressingKeys'] = tuple(sorted(computed))
        self.clearAddressing()
        return arr.copy()

    def _require_even_cyclic_patch(self, operation: str) -> None:
        if self.type not in {'cyclic', 'cyclicAMI'}:
            raise TypeError(f'cyclicPolyPatch::{operation} requires a cyclic patch, got {self.type!r}')
        if self.nFaces % 2:
            raise ValueError(f'cyclicPolyPatch::{operation} requires an even face count for patch {self.name!r}')

    def coupledPoints(self, mesh: 'PolyMesh') -> np.ndarray:
        if 'coupledPoints' in self._coupled_cache:
            return np.asarray(self._coupled_cache['coupledPoints'], dtype=int).reshape(-1, 2).copy()
        self._require_even_cyclic_patch('coupledPoints')
        if self.nFaces == 0:
            return self._cache_computed_coupled_pairs('coupledPoints', ())
        local_faces, mesh_points, _edges, _face_edges = _patch_primitive_addressing(mesh, self)
        coupled_point = [-1] * len(mesh_points)
        half = self.nFaces // 2
        for patch_face_a in range(half):
            f_a = local_faces[patch_face_a]
            f_b = local_faces[patch_face_a + half]
            if len(f_a) != len(f_b):
                raise ValueError(f'cyclicPolyPatch::coupledPoints face vertex-count mismatch on patch {self.name!r}: {len(f_a)} != {len(f_b)}')
            for index_a, patch_point_a in enumerate(f_a):
                if coupled_point[patch_point_a] != -1:
                    continue
                index_b = (len(f_b) - index_a) % len(f_b)
                patch_point_b = f_b[index_b]
                if patch_point_a != patch_point_b:
                    coupled_point[patch_point_a] = patch_point_b
        pairs = ((i, int(point_b)) for i, point_b in enumerate(coupled_point) if int(point_b) != -1)
        return self._cache_computed_coupled_pairs('coupledPoints', pairs)

    def coupledEdges(self, mesh: 'PolyMesh') -> np.ndarray:
        if 'coupledEdges' in self._coupled_cache:
            return np.asarray(self._coupled_cache['coupledEdges'], dtype=int).reshape(-1, 2).copy()
        self._require_even_cyclic_patch('coupledEdges')
        if self.nFaces == 0:
            return self._cache_computed_coupled_pairs('coupledEdges', ())
        point_couples = self.coupledPoints(mesh)
        a_to_b = {int(a): int(b) for a, b in point_couples}
        _local_faces, _mesh_points, edges, face_edges = _patch_primitive_addressing(mesh, self)
        half = self.nFaces // 2
        edge_map: dict[tuple[int, int], int] = {}
        for patch_face_a in range(half):
            for edge_i in face_edges[patch_face_a]:
                edge = edges[int(edge_i)]
                if edge[0] in a_to_b and edge[1] in a_to_b:
                    mapped = _edge_key(a_to_b[edge[0]], a_to_b[edge[1]])
                    edge_map.setdefault(mapped, int(edge_i))
        coupled_edges: list[tuple[int, int]] = []
        for patch_face_b in range(half, self.nFaces):
            for edge_i in face_edges[patch_face_b]:
                edge = edges[int(edge_i)]
                half_a_edge_i = edge_map.pop(_edge_key(edge[0], edge[1]), None)
                if half_a_edge_i is not None and int(half_a_edge_i) != int(edge_i):
                    coupled_edges.append((int(half_a_edge_i), int(edge_i)))
        for couple_i, (edge_a, edge_b) in enumerate(coupled_edges):
            if edge_a == edge_b or edge_a < 0 or edge_b < 0:
                raise ValueError(f'cyclicPolyPatch::coupledEdges illegal couple at position {couple_i}: {(edge_a, edge_b)!r}')
        return self._cache_computed_coupled_pairs('coupledEdges', coupled_edges)



    def initGeometry(self) -> None:
        return None

    def calcGeometry(self) -> None:
        if self.type in {'cyclic', 'cyclicAMI'}:
            self.calcTransforms()
        return None


    def movePoints(self, points: np.ndarray | None=None, *, mesh: 'PolyMesh' | None=None) -> None:
        self.clearGeom()
        if self.type in {'cyclic', 'cyclicAMI'}:
            self.calcTransforms()

    def initUpdateMesh(self) -> None:
        return None

    def updateMesh(self) -> None:
        self.clearAddressing()
        if self.type in {'cyclic', 'cyclicAMI'}:
            computed_keys = tuple(self._coupled_cache.get('_computedCoupledAddressingKeys', ()))
            for key in computed_keys:
                self.attributes.pop(str(key), None)
            self._coupled_cache.pop('coupledPoints', None)
            self._coupled_cache.pop('coupledEdges', None)
            self._coupled_cache.pop('_computedCoupledAddressingKeys', None)

    def clearGeom(self) -> None:
        self._geometry_cache.clear()

    def clearAddressing(self) -> None:
        self._addressing_cache.clear()


@dataclass
class ProcessorPolyPatch(PolyPatch):
    myProcNo: int = 0
    neighbProcNo: int = 0
    matchTolerance: float = 0.0001
    transform: str = 'unknown'
    remote_cell_ids: np.ndarray | None = None
    remote_face_ids: np.ndarray | None = None
    remote_face_centres: np.ndarray | None = None
    remote_face_areas: np.ndarray | None = None
    remote_cell_centres: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.type = 'processor'
        self.attributes.setdefault('myProcNo', self.myProcNo)
        self.attributes.setdefault('neighbProcNo', self.neighbProcNo)
        self.attributes.setdefault('matchTolerance', self.matchTolerance)
        self.attributes.setdefault('transform', self.transform)




    def _local_geometry(self, mesh: 'PolyMesh') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        faces = np.asarray(list(self.faces), dtype=int)
        if faces.size != int(self.nFaces):
            raise ValueError(f'processorPolyPatch::{self.name} face range does not match nFaces')
        owners = mesh.owner[faces].astype(int)
        return (mesh.face_centres()[faces], mesh.face_area_vectors()[faces], mesh.cell_centres()[owners])

    def _remote_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        attrs = self.attributes
        remote_face_centres = self.remote_face_centres
        if remote_face_centres is None:
            remote_face_centres = attrs.get('neighbFaceCentres')
        remote_face_areas = self.remote_face_areas
        if remote_face_areas is None:
            remote_face_areas = attrs.get('neighbFaceAreas')
        remote_cell_centres = self.remote_cell_centres
        if remote_cell_centres is None:
            remote_cell_centres = attrs.get('neighbFaceCellCentres')
        return (_patch_vector_array(remote_face_centres, 'neighbFaceCentres', self.nFaces), _patch_vector_array(remote_face_areas, 'neighbFaceAreas', self.nFaces), _patch_vector_array(remote_cell_centres, 'neighbFaceCellCentres', self.nFaces))







    def initGeometry(self, mesh: 'PolyMesh' | None=None) -> None:
        if mesh is None:
            self._coupled_cache['initGeometryStatus'] = 'unsupported'
            return None
        face_centres, face_areas, face_cell_centres = self._local_geometry(mesh)
        self._coupled_cache['processorInitGeometrySendFaceCentres'] = face_centres
        self._coupled_cache['processorInitGeometrySendFaceAreas'] = face_areas
        self._coupled_cache['processorInitGeometrySendFaceCellCentres'] = face_cell_centres
        self._coupled_cache['initGeometryStatus'] = 'ok'
        return None

    def calcGeometry(self, mesh: 'PolyMesh' | None=None) -> None:
        if mesh is None:
            raise KeyError('processorPolyPatch::calcGeometry requires mesh-local face geometry evidence')
        local_face_centres, local_face_areas, _local_cell_centres = self._local_geometry(mesh)
        neighb_face_centres, neighb_face_areas, neighb_cell_centres = self._remote_geometry()
        face_normals = np.zeros((self.nFaces, 3), dtype=float)
        neighb_face_normals = np.zeros((self.nFaces, 3), dtype=float)
        match_tol = float(self.attributes.get('matchTolerance', self.matchTolerance))
        for face_i in range(int(self.nFaces)):
            mag_sf = float(np.linalg.norm(local_face_areas[face_i]))
            nbr_mag_sf = float(np.linalg.norm(neighb_face_areas[face_i]))
            av_sf = 0.5 * (mag_sf + nbr_mag_sf)
            if mag_sf < 1e-15 and nbr_mag_sf < 1e-15:
                face_normals[face_i] = np.asarray([1.0, 0.0, 0.0], dtype=float)
                neighb_face_normals[face_i] = face_normals[face_i]
            elif abs(mag_sf - nbr_mag_sf) / max(av_sf, 1e-300) > match_tol:
                raise ValueError(f'processorPolyPatch::calcGeometry face {face_i} area does not match neighbour by {100.0 * abs(mag_sf - nbr_mag_sf) / max(av_sf, 1e-300):.6g}% on patch {self.name!r}')
            else:
                face_normals[face_i] = local_face_areas[face_i] / max(mag_sf, 1e-300)
                neighb_face_normals[face_i] = neighb_face_areas[face_i] / max(nbr_mag_sf, 1e-300)
        self.remote_face_centres = neighb_face_centres.copy()
        self.remote_face_areas = neighb_face_areas.copy()
        self.remote_cell_centres = neighb_cell_centres.copy()
        self.attributes['neighbFaceCentres'] = self.remote_face_centres
        self.attributes['neighbFaceAreas'] = self.remote_face_areas
        self.attributes['neighbFaceCellCentres'] = self.remote_cell_centres
        self._coupled_cache['neighbFaceCentres'] = self.remote_face_centres.copy()
        self._coupled_cache['neighbFaceAreas'] = self.remote_face_areas.copy()
        self._coupled_cache['neighbFaceCellCentres'] = self.remote_cell_centres.copy()
        self._coupled_cache['faceCentres'] = local_face_centres.copy()
        self._coupled_cache['faceNormals'] = face_normals
        self._coupled_cache['neighbFaceNormals'] = neighb_face_normals
        self._coupled_cache['calcGeometryStatus'] = 'ok'
        self.calcTransformTensors(local_face_centres, neighb_face_centres, face_normals, neighb_face_normals, self.calcFaceTol(mesh, local_face_centres))
        return None


    def movePoints(self, points: np.ndarray | None=None, *, mesh: 'PolyMesh' | None=None) -> None:
        if mesh is None:
            self._coupled_cache['movePointsStatus'] = 'unsupported'
            return None
        self.calcGeometry(mesh)
        self._coupled_cache['movePointsStatus'] = 'ok'
        return None

    def processorUpdateMeshSendAddressing(self, mesh: 'PolyMesh') -> dict[str, np.ndarray]:
        local_faces, mesh_points, edges, face_edges = _patch_primitive_addressing(mesh, self)
        n_points = len(mesh_points)
        n_edges = len(edges)
        point_face = np.full(n_points, -1, dtype=int)
        point_index = np.full(n_points, -1, dtype=int)
        edge_face = np.full(n_edges, -1, dtype=int)
        edge_index = np.full(n_edges, -1, dtype=int)
        for face_i, face in enumerate(local_faces):
            for index_i, patch_point_i in enumerate(face):
                if point_face[int(patch_point_i)] == -1:
                    point_face[int(patch_point_i)] = int(face_i)
                    point_index[int(patch_point_i)] = int(index_i)
        for face_i, edges in enumerate(face_edges):
            for index_i, patch_edge_i in enumerate(edges):
                if edge_face[int(patch_edge_i)] == -1:
                    edge_face[int(patch_edge_i)] = int(face_i)
                    edge_index[int(patch_edge_i)] = int(index_i)
        if np.any(point_face < 0) or np.any(edge_face < 0):
            raise ValueError(f'processorPolyPatch::initUpdateMesh incomplete primitive addressing for patch {self.name!r}')
        result = {'pointFace': point_face, 'pointIndex': point_index, 'edgeFace': edge_face, 'edgeIndex': edge_index}
        self._coupled_cache['processorUpdateMeshSendPointFace'] = point_face.copy()
        self._coupled_cache['processorUpdateMeshSendPointIndex'] = point_index.copy()
        self._coupled_cache['processorUpdateMeshSendEdgeFace'] = edge_face.copy()
        self._coupled_cache['processorUpdateMeshSendEdgeIndex'] = edge_index.copy()
        self._coupled_cache['initUpdateMeshStatus'] = 'ok'
        return {key: value.copy() for key, value in result.items()}

    def initUpdateMesh(self, mesh: 'PolyMesh' | None=None) -> None:
        super().initUpdateMesh()
        self._coupled_cache.pop('neighbPoints', None)
        self._coupled_cache.pop('neighbEdges', None)
        if mesh is None:
            self._coupled_cache['initUpdateMeshStatus'] = 'unsupported'
            return None
        self.processorUpdateMeshSendAddressing(mesh)
        return None

    def updateMesh(self, mesh: 'PolyMesh' | None=None, *, neighbour_point_face: object | None=None, neighbour_point_index: object | None=None, neighbour_edge_face: object | None=None, neighbour_edge_index: object | None=None) -> None:
        super().updateMesh()
        self._coupled_cache.pop('neighbPoints', None)
        self._coupled_cache.pop('neighbEdges', None)
        if mesh is None:
            self._coupled_cache['updateMeshStatus'] = 'unsupported'
            return None
        if any((v is None for v in (neighbour_point_face, neighbour_point_index, neighbour_edge_face, neighbour_edge_index))):
            self._coupled_cache['updateMeshStatus'] = 'unsupported'
            return None
        local_faces, mesh_points, edges, face_edges = _patch_primitive_addressing(mesh, self)
        nbr_point_face = _label_array(neighbour_point_face, 'nbrPointFace')
        nbr_point_index = _label_array(neighbour_point_index, 'nbrPointIndex')
        nbr_edge_face = _label_array(neighbour_edge_face, 'nbrEdgeFace')
        nbr_edge_index = _label_array(neighbour_edge_index, 'nbrEdgeIndex')
        if nbr_point_face.shape != nbr_point_index.shape:
            raise ValueError('processorPolyPatch::updateMesh nbrPointFace/nbrPointIndex size mismatch')
        if nbr_edge_face.shape != nbr_edge_index.shape:
            raise ValueError('processorPolyPatch::updateMesh nbrEdgeFace/nbrEdgeIndex size mismatch')
        neighb_points = np.full(len(mesh_points), -1, dtype=int)
        for nbr_point_i, (face_i_raw, index_i_raw) in enumerate(zip(nbr_point_face, nbr_point_index)):
            face_i = int(face_i_raw)
            index_i = int(index_i_raw)
            if face_i < 0 or face_i >= len(local_faces):
                raise IndexError('processorPolyPatch::updateMesh nbrPointFace outside localFaces')
            face = local_faces[face_i]
            if not face:
                continue
            index = (len(face) - index_i) % len(face)
            patch_point_i = int(face[index])
            if neighb_points[patch_point_i] == -1:
                neighb_points[patch_point_i] = int(nbr_point_i)
            elif neighb_points[patch_point_i] >= 0:
                neighb_points[patch_point_i] = -2
        neighb_points[neighb_points == -2] = -1
        neighb_edges = np.full(len(edges), -1, dtype=int)
        for nbr_edge_i, (face_i_raw, index_i_raw) in enumerate(zip(nbr_edge_face, nbr_edge_index)):
            face_i = int(face_i_raw)
            index_i = int(index_i_raw)
            if face_i < 0 or face_i >= len(face_edges):
                raise IndexError('processorPolyPatch::updateMesh nbrEdgeFace outside faceEdges')
            face = face_edges[face_i]
            if not face:
                continue
            index = (len(face) - index_i - 1) % len(face)
            patch_edge_i = int(face[index])
            if neighb_edges[patch_edge_i] == -1:
                neighb_edges[patch_edge_i] = int(nbr_edge_i)
            elif neighb_edges[patch_edge_i] >= 0:
                neighb_edges[patch_edge_i] = -2
        neighb_edges[neighb_edges == -2] = -1
        self._coupled_cache['neighbPoints'] = neighb_points
        self._coupled_cache['neighbEdges'] = neighb_edges
        self._coupled_cache['updateMeshStatus'] = 'ok'
        self.clearAddressing()
        return None

    def neighbPoints(self) -> np.ndarray:
        if 'neighbPoints' not in self._coupled_cache:
            raise ValueError(f'processorPolyPatch::neighbPoints() no extended addressing for patch {self.name!r}')
        return np.asarray(self._coupled_cache['neighbPoints'], dtype=int).copy()

    def neighbEdges(self) -> np.ndarray:
        if 'neighbEdges' not in self._coupled_cache:
            raise ValueError(f'processorPolyPatch::neighbEdges() no extended addressing for patch {self.name!r}')
        return np.asarray(self._coupled_cache['neighbEdges'], dtype=int).copy()

@dataclass
class PolyMesh:
    points: np.ndarray
    faces: list[tuple[int, ...]]
    owner: np.ndarray
    neighbour: np.ndarray
    boundary: list[PolyPatch] = field(default_factory=list)
    cell_voxels: np.ndarray | None = None
    voxel_cell_ids: np.ndarray | None = None
    spacing: tuple[float, float, float] | None = None
    face_proc_addressing: np.ndarray | None = None
    solutionD: np.ndarray | None = None
    geometricD: np.ndarray | None = None
    _face_centres_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _face_area_vectors_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _mag_sf_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _cell_faces_cache: tuple[tuple[int, ...], ...] | None = field(default=None, init=False, repr=False)
    _cell_centres_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _cell_volumes_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError('points must have shape (nPoints, 3) in OpenFOAM (x,y,z) order')
        self.faces = [tuple((int(i) for i in f)) for f in self.faces]
        self.owner = np.asarray(self.owner, dtype=int)
        self.neighbour = np.asarray(self.neighbour, dtype=int)
        if len(self.owner) != len(self.faces):
            raise ValueError('owner must have one entry per face')
        if len(self.neighbour) > len(self.owner):
            raise ValueError('neighbour cannot be longer than owner')
        if self.face_proc_addressing is not None:
            self.face_proc_addressing = np.asarray(self.face_proc_addressing, dtype=int).reshape(-1)
            if self.face_proc_addressing.size != len(self.owner):
                raise ValueError('face_proc_addressing must have one entry per face')
        if self.boundary:
            first_boundary = min((p.startFace for p in self.boundary if p.nFaces > 0))
            if len(self.neighbour) != first_boundary:
                raise ValueError('neighbour length must equal first boundary face index')

    @property
    def nPoints(self) -> int:
        return int(self.points.shape[0])

    @property
    def nFaces(self) -> int:
        return len(self.faces)

    @property
    def nInternalFaces(self) -> int:
        return int(self.neighbour.shape[0])


    @property
    def nCells(self) -> int:
        owner = self.owner
        neighbour = self.neighbour
        memo = self.__dict__.get('_ncells_memo')
        if memo is not None and memo[0] is owner and (memo[1] is neighbour) and (memo[2] == owner.size) and (memo[3] == neighbour.size):
            return memo[4]
        if owner.size == 0:
            return 0
        mx = int(np.max(owner))
        if neighbour.size:
            mx = max(mx, int(np.max(neighbour)))
        value = mx + 1
        self.__dict__['_ncells_memo'] = (owner, neighbour, owner.size, neighbour.size, value)
        return value

    def patch(self, name: str) -> PolyPatch:
        for p in self.boundary:
            if p.name == name:
                return p
        raise KeyError(name)




    def face_points(self, face_i: int) -> np.ndarray:
        return self.points[np.asarray(self.faces[face_i], dtype=int)]

    def _ensure_face_geometry(self) -> None:
        if self._face_centres_cache is not None and self._face_area_vectors_cache is not None:
            return
        centres = np.zeros((self.nFaces, 3), dtype=float)
        areas = np.zeros((self.nFaces, 3), dtype=float)
        groups: dict[int, list[int]] = {}
        for facei, face in enumerate(self.faces):
            groups.setdefault(len(face), []).append(facei)
        for n_points, group in groups.items():
            if n_points == 0:
                continue
            idx = np.asarray(group, dtype=int)
            labels = np.asarray([self.faces[i] for i in group], dtype=int)
            fpts = self.points[labels]
            if n_points == 3:
                p0 = fpts[:, 0]
                p1 = fpts[:, 1]
                p2 = fpts[:, 2]
                point_sum = p0 + p1 + p2
                centres[idx] = 0.3333333333333333 * point_sum
                e1 = p1 - p0
                e2 = p2 - p0
                cross = np.empty((idx.size, 3), dtype=float)
                cross[:, 0] = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
                cross[:, 1] = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]
                cross[:, 2] = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
                areas[idx] = 0.5 * cross
                continue
            m = idx.size
            f_centre = fpts[:, 0]
            for pointi in range(1, n_points):
                f_centre = f_centre + fpts[:, pointi]
            f_centre = f_centre / float(n_points)
            sum_n = np.zeros((m, 3), dtype=float)
            sum_a = np.zeros(m, dtype=float)
            sum_ac = np.zeros((m, 3), dtype=float)
            for pointi in range(n_points):
                point = fpts[:, pointi]
                next_point = fpts[:, (pointi + 1) % n_points]
                triangle_centre_sum = point + next_point + f_centre
                e1 = next_point - point
                e2 = f_centre - point
                triangle_normal = np.empty((m, 3), dtype=float)
                triangle_normal[:, 0] = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
                triangle_normal[:, 1] = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]
                triangle_normal[:, 2] = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
                triangle_area = np.sqrt(triangle_normal[:, 0] * triangle_normal[:, 0] + triangle_normal[:, 1] * triangle_normal[:, 1] + triangle_normal[:, 2] * triangle_normal[:, 2])
                sum_n = sum_n + triangle_normal
                sum_a = sum_a + triangle_area
                sum_ac = sum_ac + triangle_area[:, None] * triangle_centre_sum
            centres[idx] = 0.3333333333333333 * sum_ac / (sum_a + 1e-300)[:, None]
            areas[idx] = 0.5 * sum_n
        self._face_centres_cache = centres
        self._face_area_vectors_cache = areas
        self._mag_sf_cache = np.sqrt(areas[:, 0] * areas[:, 0] + areas[:, 1] * areas[:, 1] + areas[:, 2] * areas[:, 2])


    def face_centres(self) -> np.ndarray:
        self._ensure_face_geometry()
        return self._face_centres_cache.copy()

    def face_area_vectors(self) -> np.ndarray:
        self._ensure_face_geometry()
        return self._face_area_vectors_cache.copy()

    def magSf(self) -> np.ndarray:
        if self._mag_sf_cache is None:
            areas = self.face_area_vectors()
            self._mag_sf_cache = np.sqrt(areas[:, 0] * areas[:, 0] + areas[:, 1] * areas[:, 1] + areas[:, 2] * areas[:, 2])
        return self._mag_sf_cache.copy()

    def clearGeom(self) -> None:
        self._face_centres_cache = None
        self._face_area_vectors_cache = None
        self._mag_sf_cache = None
        self._cell_centres_cache = None
        self._cell_volumes_cache = None
        self._reset_valid_directions()

    def clearAddressing(self) -> None:
        self._cell_faces_cache = None
        self._reset_valid_directions()

    def movePoints(self, points: np.ndarray) -> None:
        moved = np.asarray(points, dtype=float)
        if moved.shape != self.points.shape:
            raise ValueError('movePoints points must match existing point shape')
        self.points = moved.copy()
        self.clearGeom()






    def _reset_valid_directions(self) -> None:
        self.solutionD = None
        self.geometricD = None




    def cell_centres(self) -> np.ndarray:
        if self._cell_centres_cache is None:
            self._ensure_cell_geometry()
        return self._cell_centres_cache.copy()

    def cell_volumes(self) -> np.ndarray:
        if self._cell_volumes_cache is None:
            self._ensure_cell_geometry()
        return self._cell_volumes_cache.copy()

    def _ensure_cell_geometry(self) -> None:
        if self._cell_centres_cache is not None and self._cell_volumes_cache is not None:
            return
        face_centres = self.face_centres()
        n_cells = self.nCells
        n_internal = self.nInternalFaces
        n_faces = self.nFaces
        owner = self.owner
        neighbour = self.neighbour
        weighted_cell_centres = np.zeros((n_cells, 3), dtype=float)
        cell_volumes = np.zeros(n_cells, dtype=float)
        c_est = np.zeros((n_cells, 3), dtype=float)
        n_cell_faces = np.zeros(n_cells, dtype=int)
        np.add.at(c_est, owner, face_centres)
        np.add.at(n_cell_faces, owner, 1)
        if n_internal:
            np.add.at(c_est, neighbour, face_centres[:n_internal])
            np.add.at(n_cell_faces, neighbour, 1)
        has_faces = n_cell_faces > 0
        if np.any(has_faces):
            c_est[has_faces] = c_est[has_faces] / n_cell_faces[has_faces].astype(float)[:, None]
        groups: dict[int, list[int]] = {}
        face_len = np.empty(n_faces, dtype=np.int64)
        for facei, face in enumerate(self.faces):
            face_len[facei] = len(face)
            groups.setdefault(len(face), []).append(facei)
        tet_count = np.where(face_len == 3, 1, face_len)
        tet_starts = np.zeros(n_faces + 1, dtype=np.int64)
        np.cumsum(tet_count, out=tet_starts[1:])
        tet_count_nei = tet_count[:n_internal]
        tet_starts_nei = np.zeros(n_internal + 1, dtype=np.int64)
        np.cumsum(tet_count_nei, out=tet_starts_nei[1:])

        def _tet_mag_grouped(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
            b_a = b - a
            c_a = c - a
            cross0 = b_a[:, 1] * c_a[:, 2] - b_a[:, 2] * c_a[:, 1]
            cross1 = b_a[:, 2] * c_a[:, 0] - b_a[:, 0] * c_a[:, 2]
            cross2 = b_a[:, 0] * c_a[:, 1] - b_a[:, 1] * c_a[:, 0]
            d_a = d - a
            return 0.16666666666666666 * (cross0 * d_a[:, 0] + cross1 * d_a[:, 1] + cross2 * d_a[:, 2])

        def _tet_centre_grouped(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
            return 0.25 * (a + b + c + d)
        total_own = int(tet_starts[-1])
        flat_vol_own = np.zeros(total_own, dtype=float)
        flat_ctr_own = np.zeros((total_own, 3), dtype=float)
        total_nei = int(tet_starts_nei[-1])
        flat_vol_nei = np.zeros(total_nei, dtype=float)
        flat_ctr_nei = np.zeros((total_nei, 3), dtype=float)
        for n_points, group in groups.items():
            if n_points == 0:
                continue
            g = np.asarray(group, dtype=np.int64)
            labels = np.asarray([self.faces[i] for i in group], dtype=int)
            fpts = self.points[labels]
            d_own = c_est[owner[g]]
            if n_points == 3:
                a = fpts[:, 2]
                b = fpts[:, 1]
                c = fpts[:, 0]
                pos = tet_starts[g]
                flat_vol_own[pos] = _tet_mag_grouped(a, b, c, d_own)
                flat_ctr_own[pos] = _tet_centre_grouped(a, b, c, d_own)
            else:
                fc = face_centres[g]
                base = tet_starts[g]
                for pointi in range(n_points):
                    prev_pointi = (pointi - 1) % n_points
                    a = fpts[:, pointi]
                    b = fpts[:, prev_pointi]
                    pos = base + pointi
                    flat_vol_own[pos] = _tet_mag_grouped(a, b, fc, d_own)
                    flat_ctr_own[pos] = _tet_centre_grouped(a, b, fc, d_own)
            internal_mask = g < n_internal
            if not np.any(internal_mask):
                continue
            gi = g[internal_mask]
            fpts_i = fpts[internal_mask]
            d_nei = c_est[neighbour[gi]]
            if n_points == 3:
                a = fpts_i[:, 0]
                b = fpts_i[:, 1]
                c = fpts_i[:, 2]
                pos = tet_starts_nei[gi]
                flat_vol_nei[pos] = _tet_mag_grouped(a, b, c, d_nei)
                flat_ctr_nei[pos] = _tet_centre_grouped(a, b, c, d_nei)
            else:
                fc = face_centres[gi]
                base = tet_starts_nei[gi]
                for pointi in range(n_points):
                    next_pointi = (pointi + 1) % n_points
                    a = fpts_i[:, pointi]
                    b = fpts_i[:, next_pointi]
                    pos = base + pointi
                    flat_vol_nei[pos] = _tet_mag_grouped(a, b, fc, d_nei)
                    flat_ctr_nei[pos] = _tet_centre_grouped(a, b, fc, d_nei)
        rep_own = np.repeat(owner, tet_count)
        np.add.at(weighted_cell_centres, rep_own, flat_vol_own[:, None] * flat_ctr_own)
        np.add.at(cell_volumes, rep_own, flat_vol_own)
        if total_nei:
            rep_nei = np.repeat(neighbour, tet_count_nei)
            np.add.at(weighted_cell_centres, rep_nei, flat_vol_nei[:, None] * flat_ctr_nei)
            np.add.at(cell_volumes, rep_nei, flat_vol_nei)
        cell_centres = weighted_cell_centres / (cell_volumes + 1e-300)[:, None]
        self._cell_centres_cache = cell_centres
        self._cell_volumes_cache = cell_volumes.copy()









def _strip_foam_comments(text: str) -> str:
    text = re.sub('//.*', '', text)
    text = re.sub('/\\*.*?\\*/', '', text, flags=re.S)
    return text

def _body_after_header(text: str) -> str:
    text = _strip_foam_comments(text)
    m = re.search('\\n\\s*\\d+\\s*\\(', text)
    if not m:
        return text
    return text[m.start():]

def _parse_points(text: str) -> np.ndarray:
    body = _body_after_header(text)
    triples = re.findall('\\(([-+0-9eE.\\s]+)\\)', body)
    pts = []
    for t in triples:
        nums = [float(x) for x in t.split()]
        if len(nums) == 3:
            pts.append(nums)
    return np.asarray(pts, dtype=float)

def _parse_faces(text: str) -> list[tuple[int, ...]]:
    body = _body_after_header(text)
    faces: list[tuple[int, ...]] = []
    for m in re.finditer('(\\d+)?\\s*\\(([^()]*)\\)', body):
        vals_txt = m.group(2).split()
        if not vals_txt:
            continue
        vals = [int(x) for x in vals_txt]
        if m.group(1) is not None and int(m.group(1)) != len(vals):
            continue
        faces.append(tuple(vals))
    return faces

def _parse_labels(text: str) -> np.ndarray:
    body = _body_after_header(text)
    uniform = re.search('(?:^|\\n)\\s*(\\d+)\\s*\\{\\s*([-+]?\\d+)\\s*\\}\\s*(?:\\n|$)', body)
    if uniform:
        return np.full(int(uniform.group(1)), int(uniform.group(2)), dtype=int)
    m = re.search('\\d+\\s*\\((.*?)\\)', body, flags=re.S)
    if not m:
        return np.zeros(0, dtype=int)
    vals = [int(x) for x in re.findall('[-+]?\\d+', m.group(1))]
    return np.asarray(vals, dtype=int)

def _parse_boundary(text: str) -> list[PolyPatch]:
    text = _strip_foam_comments(text)
    patches: list[PolyPatch] = []
    m = re.search('\\d+\\s*\\((.*)\\)\\s*$', text, flags=re.S)
    body = m.group(1) if m else text
    for pm in re.finditer('([A-Za-z_][\\w.]*)\\s*\\{([^{}]*)\\}', body, flags=re.S):
        name = pm.group(1)
        b = pm.group(2)
        typ = _lookup_word(b, 'type', 'patch')
        nFaces = int(float(_lookup_word(b, 'nFaces', '0')))
        startFace = int(float(_lookup_word(b, 'startFace', '0')))
        if typ == 'processor':
            patches.append(ProcessorPolyPatch(name=name, type='processor', nFaces=nFaces, startFace=startFace, myProcNo=int(float(_lookup_word(b, 'myProcNo', '0'))), neighbProcNo=int(float(_lookup_word(b, 'neighbProcNo', '0'))), matchTolerance=float(_lookup_word(b, 'matchTolerance', '1e-4')), transform=_lookup_word(b, 'transform', 'unknown')))
        else:
            patches.append(PolyPatch(name=name, type=typ, nFaces=nFaces, startFace=startFace))
    patches.sort(key=lambda p: p.startFace)
    return patches

def _lookup_word(block: str, key: str, default: str) -> str:
    m = re.search(f'\\b{re.escape(key)}\\s+([^;]+);', block)
    return m.group(1).strip().split()[0] if m else default

def _read_openfoam_poly_mesh_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')

def _foam_file_declares_binary(path: Path) -> bool:
    if not path.exists():
        return False
    raw = path.read_bytes()[:4096]
    return re.search(b'\\bformat\\s+binary\\s*;', raw) is not None

def _read_optional_label_list(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    if _foam_file_declares_binary(path):
        from .openfoam_native_binary import read_foam_binary_list
        return read_foam_binary_list(path, primitive='label').values.astype(int)
    return _parse_labels(_read_openfoam_poly_mesh_text(path))

def read_poly_mesh(case_dir: str | Path) -> PolyMesh:
    root = Path(case_dir) / 'constant' / 'polyMesh'
    face_proc_addressing = _read_optional_label_list(root / 'faceProcAddressing')
    if any((_foam_file_declares_binary(root / name) for name in ('points', 'faces', 'owner', 'neighbour'))):
        from .openfoam_mesh_binary import read_binary_poly_mesh
        mesh = read_binary_poly_mesh(case_dir)
        if (root / 'boundary').exists():
            mesh.boundary = _parse_boundary(_read_openfoam_poly_mesh_text(root / 'boundary'))
        if face_proc_addressing is not None:
            if face_proc_addressing.size != mesh.nFaces:
                raise ValueError('faceProcAddressing must have one entry per face')
            mesh.face_proc_addressing = face_proc_addressing
        return mesh
    points = _parse_points(_read_openfoam_poly_mesh_text(root / 'points'))
    faces = _parse_faces(_read_openfoam_poly_mesh_text(root / 'faces'))
    owner = _parse_labels(_read_openfoam_poly_mesh_text(root / 'owner'))
    neighbour = _parse_labels(_read_openfoam_poly_mesh_text(root / 'neighbour'))
    boundary = _parse_boundary(_read_openfoam_poly_mesh_text(root / 'boundary')) if (root / 'boundary').exists() else []
    return PolyMesh(points=points, faces=faces, owner=owner, neighbour=neighbour, boundary=boundary, face_proc_addressing=face_proc_addressing)
