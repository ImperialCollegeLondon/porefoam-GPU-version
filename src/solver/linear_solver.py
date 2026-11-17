from __future__ import annotations
import time
from typing import Optional
import amg_solver as _AMG
_REGISTRY = []

class SolveReport:
    __slots__ = ('iters', 'residual', 'tol', 'tol_honoured', 'dispatch_s')

    def __init__(self, iters, residual=None, tol=None, tol_honoured=False, dispatch_s=0.0):
        self.iters = int(iters)
        self.residual = residual
        self.tol = tol
        self.tol_honoured = bool(tol_honoured)
        self.dispatch_s = float(dispatch_s)

    def __repr__(self):
        r = 'None' if self.residual is None else '%.6e' % self.residual
        return 'SolveReport(iters=%d, residual=%s, tol_honoured=%s, dispatch_s=%.6f)' % (self.iters, r, self.tol_honoured, self.dispatch_s)

class LinearSolver:

    def setup(self, structure) -> 'LinearSolver':
        raise NotImplementedError

    def update(self, **values) -> 'LinearSolver':
        raise NotImplementedError

    def solve(self, *, b, x, **kw) -> SolveReport:
        raise NotImplementedError

class SAAMG(LinearSolver):

    def __init__(self, cp, structure=None, *, name: str='', updatable: bool=True, cycles: Optional[int]=None):
        self._cp = cp
        self._ws = None
        self._rs = None
        self._zr_s = None
        self._zr_r = None
        self._zr_buf = None
        self._zr_n = 0
        self.name = str(name)
        self.updatable = bool(updatable)
        self.default_cycles = cycles
        self.n_solve = 0
        self.n_update = 0
        self.total_iters = 0
        self.t_solve = 0.0
        self.t_update = 0.0
        self.n_levels = 0
        self.n_rows = 0
        self.nnz = 0
        _REGISTRY.append(self)
        if structure is not None:
            self.setup(structure)

    def setup(self, structure) -> 'SAAMG':
        self._ws = structure
        L = getattr(structure, 'L', None)
        if L:
            self.n_levels = len(L)
            self.n_rows = int(L[0]['n'])
            self.nnz = int(sum((int(e['dv'].size) for e in L)))
        return self

    def _squared_residual(self, b, x, out):
        cp = self._cp
        _AMG._mv(cp, self._ws.L[0], x, self._zr_s)
        cp.subtract(b, self._zr_s, out=self._zr_r)
        cp.multiply(self._zr_r, self._zr_r, out=self._zr_s)
        cp.sum(self._zr_s, keepdims=True, out=out)


    def update(self, *, af=None, rau_v=None, fn=None) -> 'SAAMG':
        t0 = time.perf_counter()
        if fn is not None:
            fn()
        elif af is not None and self.updatable:
            _AMG.refresh_device(self._cp, self._ws, af, rau_v=rau_v)
        self.n_update += 1
        self.t_update += time.perf_counter() - t0
        return self

    def solve(self, *, b, x, cycles=None, tol=None, residual: bool=False, sync: bool=False, **kw) -> SolveReport:
        n = int(self.default_cycles if cycles is None else cycles)
        cp = self._cp
        if sync:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _rec = self._zr_buf is not None and self._zr_n < 40000
        if _rec:
            _k = self._zr_n
            self._squared_residual(b, x, self._zr_buf[_k, 0:1])
        if kw:
            _AMG.solve(cp, self._ws, b=b, x=x, cycles=n, **kw)
        else:
            _AMG.solve(cp, self._ws, b=b, x=x, cycles=n)
        if _rec:
            self._squared_residual(b, x, self._zr_buf[_k, 1:2])
            cp.multiply(b, b, out=self._zr_s)
            cp.sum(self._zr_s, keepdims=True, out=self._zr_buf[_k, 2:3])
            self._zr_n += 1
        if sync:
            cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        self.t_solve += dt
        self.n_solve += 1
        self.total_iters += n
        r = self.residual_norm(b=b, x=x) if residual else None
        return SolveReport(iters=n, residual=r, tol=tol, tol_honoured=False, dispatch_s=dt)

    def residual_norm(self, *, b, x) -> float:
        cp = self._cp
        e = self._ws.L[0]
        if self._rs is None or self._rs.size != int(e['n']):
            self._rs = cp.zeros(int(e['n']), dtype=cp.float64)
        _AMG._mv(cp, e, x, self._rs)
        return float(cp.linalg.norm(b - self._rs))

    def stats(self) -> dict:
        return {'name': self.name, 'levels': self.n_levels, 'rows': self.n_rows, 'nnz': self.nnz, 'n_solve': self.n_solve, 'n_update': self.n_update, 'total_iters': self.total_iters, 'dispatch_solve_s': self.t_solve, 'dispatch_update_s': self.t_update, 'updatable': self.updatable}

