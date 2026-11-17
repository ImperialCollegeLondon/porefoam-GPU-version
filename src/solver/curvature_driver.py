from __future__ import annotations
import curvature_kernels as K

class CurvatureWorkspace:

    def __init__(self, cp, n_cells: int, n_internal_faces: int, mean_cell_volume: float, n_wall: int=0, n_boundary_faces: int=0):
        self.cp = cp
        self.nC = int(n_cells)
        self.nIF = int(n_internal_faces)
        self.nW = int(n_wall)
        self.nBF = int(n_boundary_faces)
        self.deltaN = K.delta_n(mean_cell_volume)

        def _f(n):
            return cp.zeros(int(n), dtype=cp.float64)
        self.gradf = _f(self.nIF)
        self.gx, self.gy, self.gz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.nx, self.ny, self.nz = (_f(self.nC), _f(self.nC), _f(self.nC))
        self.kappa = _f(self.nC)
        self.Fsig = _f(self.nIF)
        self.nHatf = _f(self.nIF)
        self.degenerate = cp.zeros(1, dtype=cp.uint64)
        self._gf = K.grid(self.nIF)
        self._gc = K.grid(self.nC)
        self._gw = K.grid(max(self.nW, 1))
        self._gb = K.grid(max(self.nBF, 1))
        K.compile_all(cp)


def curvature_step(ws: CurvatureWorkspace, *, alpha, own, nei, dmag, Sfx, Sfy, Sfz, w, vol, magSf, sigma: float, bown=None, bSfx=None, bSfy=None, bSfz=None, wall_cells=None, wnx=None, wny=None, wnz=None, costheta=None, tangent_tol: float=1e-12, contrast: float=0.0, alpha_boundary=None) -> None:
    cp = ws.cp
    gf, bf = ws._gf
    gc, bc = ws._gc
    nIF, nC = (ws.nIF, ws.nC)
    K.fn(cp, 'gpu_alpha_gradf')(gf, bf, (alpha, own, nei, dmag, ws.gradf, nIF))
    ws.gx.fill(0.0)
    ws.gy.fill(0.0)
    ws.gz.fill(0.0)
    K.fn(cp, 'gpu_grad_accumulate')(gf, bf, (alpha, own, nei, Sfx, Sfy, Sfz, w, ws.gx, ws.gy, ws.gz, nIF))
    if bown is not None and ws.nBF > 0:
        gb, bb = ws._gb
        K.fn(cp, 'gpu_grad_accumulate_bnd' + ('_field' if alpha_boundary is not None else ''))(gb, bb, (alpha_boundary if alpha_boundary is not None else alpha, bown, bSfx, bSfy, bSfz, ws.gx, ws.gy, ws.gz, ws.nBF))
    K.fn(cp, 'gpu_grad_divide_volume')(gc, bc, (ws.gx, ws.gy, ws.gz, vol, nC))
    K.fn(cp, 'gpu_nhat')(gc, bc, (ws.gx, ws.gy, ws.gz, ws.nx, ws.ny, ws.nz, cp.float64(ws.deltaN), nC))
    if wall_cells is not None and ws.nW > 0:
        gw, bw = ws._gw
        K.fn(cp, 'gpu_contact_angle')(gw, bw, (ws.nx, ws.ny, ws.nz, wall_cells, wnx, wny, wnz, costheta, ws.degenerate, cp.float64(tangent_tol), ws.gx, ws.gy, ws.gz, cp.float64(contrast), ws.nW))
    ws.kappa.fill(0.0)
    K.fn(cp, 'gpu_kappa_accumulate')(gf, bf, (ws.nx, ws.ny, ws.nz, own, nei, Sfx, Sfy, Sfz, w, ws.kappa, nIF))
    if bown is not None and ws.nBF > 0:
        gb, bb = ws._gb
        K.fn(cp, 'gpu_kappa_accumulate_bnd')(gb, bb, (ws.nx, ws.ny, ws.nz, bown, bSfx, bSfy, bSfz, ws.kappa, ws.nBF))
    K.fn(cp, 'gpu_kappa_finish')(gc, bc, (ws.kappa, vol, nC))
    K.fn(cp, 'gpu_nhat_face')(gf, bf, (ws.nx, ws.ny, ws.nz, own, nei, w, Sfx, Sfy, Sfz, ws.nHatf, nIF))
    K.fn(cp, 'gpu_surface_tension_face')(gf, bf, (ws.kappa, ws.gradf, magSf, own, nei, w, ws.Fsig, cp.float64(sigma), nIF))
