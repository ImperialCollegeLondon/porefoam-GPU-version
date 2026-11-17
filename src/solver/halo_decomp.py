from __future__ import annotations
import numpy as np
_XPORTS = ('nccl', 'stage')
_XPORT = 'nccl'
if 'nccl' not in _XPORTS:
    raise SystemExit('transport %r is not one of %r.' % ('nccl', _XPORTS))
_GATHER = [None]

def _gather_kernel(cp):
    if _GATHER[0] is None:
        _GATHER[0] = cp.ElementwiseKernel('raw float64 src, int64 idx', 'float64 dst', 'dst = src[idx]', 'gpu_halo_gather')
    return _GATHER[0]

class Decomp(object):
    __slots__ = ('rank', 'size', 'nC', 'nIF', 'nL', 'nG', 'nLG', 'nLF', 'nLBF', 'gcell', 'gface', 'gbface', 'own_l', 'nei_l', 'bown_l', '_nb', '_csend', '_crecv', '_fsend', '_frecv', 'comm', 'cp', '_sbuf', '_hsbuf', '_hrbuf', 'stage', 'nPartFace', 'nOwnFace', 'xport', '_nccl', '_nc', '_ncdt', '_hold')

    @staticmethod
    def serial(nC, nIF, nBF=0):
        d = Decomp()
        d.rank, d.size = (0, 1)
        d.nC, d.nIF = (int(nC), int(nIF))
        d.nL, d.nG, d.nLG = (int(nC), 0, int(nC))
        d.nLF, d.nLBF = (int(nIF), int(nBF))
        d.gcell = np.arange(int(nC), dtype=np.int64)
        d.gface = np.arange(int(nIF), dtype=np.int64)
        d.gbface = np.arange(int(nBF), dtype=np.int64)
        d.own_l = d.nei_l = d.bown_l = None
        d._nb = ()
        d._csend = d._crecv = d._fsend = d._frecv = ()
        d.comm = None
        d.cp = None
        d._sbuf = d._hsbuf = d._hrbuf = None
        d.stage = False
        d.nPartFace = 0
        d.nOwnFace = int(nIF)
        d.xport = 'none'
        d._nccl = d._nc = d._ncdt = d._hold = None
        return d

    @staticmethod
    def build(part, own, nei, rank, bown=None):
        part = np.ascontiguousarray(part, dtype=np.int64)
        own = np.ascontiguousarray(own, dtype=np.int64)
        nei = np.ascontiguousarray(nei, dtype=np.int64)
        nC = int(part.size)
        nIF = int(own.size)
        size = int(part.max()) + 1
        r = int(rank)
        d = Decomp()
        d.rank, d.size = (r, size)
        d.nC, d.nIF = (nC, nIF)
        cells = np.flatnonzero(part == r).astype(np.int64)
        nL = int(cells.size)
        po, pn = (part[own], part[nei])
        mine = (po == r) | (pn == r)
        loc = mine & (po == r) & (pn == r)
        prt = mine & (po != pn)
        f_loc = np.flatnonzero(loc).astype(np.int64)
        f_prt = np.flatnonzero(prt).astype(np.int64)
        far = np.where(po[f_prt] == r, pn[f_prt], po[f_prt])
        peers = np.unique(far) if f_prt.size else np.zeros(0, np.int64)
        peers = np.sort(peers[peers != r]).astype(np.int64)
        blocks_a, blocks_b, fsend, frecv = ([], [], [], [])
        for s in peers:
            sel = f_prt[far == s]
            a = sel[po[sel] == r]
            b = sel[po[sel] == s]
            blocks_a.append(a)
            blocks_b.append(b)
        off = int(f_loc.size)
        for s, a in zip(peers, blocks_a):
            fsend.append((int(s), off, off + int(a.size)))
            off += int(a.size)
        for s, b in zip(peers, blocks_b):
            frecv.append((int(s), off, off + int(b.size)))
            off += int(b.size)
        gface = np.concatenate([f_loc] + blocks_a + blocks_b) if f_prt.size else f_loc
        d.gface = np.ascontiguousarray(gface, dtype=np.int64)
        d.nLF = int(d.gface.size)
        d.nPartFace = int(f_prt.size)
        d.nOwnFace = int(f_loc.size) + int(sum((a.size for a in blocks_a)))
        g2l = np.full(nC, -1, dtype=np.int64)
        g2l[cells] = np.arange(nL, dtype=np.int64)
        ghosts, crecv, csend = ([], [], [])
        pos = nL
        for s, a, b in zip(peers, blocks_a, blocks_b):
            fs = np.concatenate([a, b])
            need = np.unique(np.where(po[fs] == s, own[fs], nei[fs]))
            give = np.unique(np.where(po[fs] == r, own[fs], nei[fs]))
            ghosts.append(need)
            crecv.append((int(s), pos, pos + int(need.size)))
            pos += int(need.size)
            csend.append((int(s), g2l[give]))
        gh = np.concatenate(ghosts) if ghosts else np.zeros(0, np.int64)
        d.nG = int(gh.size)
        d.nL, d.nLG = (nL, nL + d.nG)
        d.gcell = np.ascontiguousarray(np.concatenate([cells, gh]), dtype=np.int64)
        g2l[gh] = nL + np.arange(d.nG, dtype=np.int64)
        d.own_l = np.ascontiguousarray(g2l[own[d.gface]], dtype=np.int64)
        d.nei_l = np.ascontiguousarray(g2l[nei[d.gface]], dtype=np.int64)
        assert d.own_l.min() >= 0 and d.nei_l.min() >= 0, 'unmapped face cell'
        if bown is not None:
            bown = np.ascontiguousarray(bown, dtype=np.int64)
            d.gbface = np.flatnonzero(part[bown] == r).astype(np.int64)
            d.bown_l = np.ascontiguousarray(g2l[bown[d.gbface]], dtype=np.int64)
        else:
            d.gbface = np.zeros(0, dtype=np.int64)
            d.bown_l = None
        d.nLBF = int(d.gbface.size)
        d._nb = tuple((int(s) for s in peers))
        d._csend, d._crecv = (tuple(csend), tuple(crecv))
        d._fsend, d._frecv = (tuple(fsend), tuple(frecv))
        d.comm = None
        d.cp = None
        d._sbuf = d._hsbuf = d._hrbuf = None
        d.stage = False
        d.xport = None
        d._nccl = d._nc = d._ncdt = d._hold = None
        return d



    def halo(self, x):
        nb = self._nb
        if not nb:
            return x
        if self.xport == 'nccl':
            return self._halo_nccl(x)
        cp = self.cp
        reqs = []
        for s, idx in self._csend:
            b = self._sbuf[s]
            _gather_kernel(cp)(x, idx, b)
        for s, idx in self._csend:
            b = self._sbuf[s]
            if self.stage:
                h = self._hsbuf[s]
                b.get(out=h)
                reqs.append(self.comm.Isend(h, dest=s, tag=11))
            else:
                reqs.append(self.comm.Isend(b, dest=s, tag=11))
        rr = []
        for s, lo, hi in self._crecv:
            if self.stage:
                rr.append((s, lo, hi, self.comm.Irecv(self._hrbuf[s], source=s, tag=11)))
            else:
                rr.append((s, lo, hi, self.comm.Irecv(x[lo:hi], source=s, tag=11)))
        for s, lo, hi, q in rr:
            q.Wait()
            if self.stage:
                h = self._hrbuf[s]
                x[lo:hi] = cp.asarray(h)
        for q in reqs:
            q.Wait()
        return x

    def _halo_nccl(self, x):
        cp = self.cp
        nccl = self._nccl
        st = cp.cuda.get_current_stream().ptr
        for s, idx in self._csend:
            _gather_kernel(cp)(x, idx, self._sbuf[s])
        nccl.groupStart()
        for s, idx in self._csend:
            b = self._sbuf[s]
            if b.size:
                self._nc.send(int(b.data.ptr), int(b.size), self._ncdt, int(s), st)
        for s, lo, hi in self._crecv:
            if hi > lo:
                v = x[lo:hi]
                self._nc.recv(int(v.data.ptr), int(hi - lo), self._ncdt, int(s), st)
        nccl.groupEnd()
        return x

    def own_faces(self, F):
        nb = self._nb
        if not nb:
            return F
        if self.xport == 'nccl':
            return self._own_faces_nccl(F)
        reqs = []
        for s, lo, hi in self._fsend:
            if hi > lo:
                if self.stage:
                    reqs.append(self.comm.Isend(self.cp.asnumpy(F[lo:hi]), dest=s, tag=22))
                else:
                    reqs.append(self.comm.Isend(F[lo:hi], dest=s, tag=22))
        rr = []
        for s, lo, hi in self._frecv:
            if hi > lo:
                if self.stage:
                    h = np.zeros(hi - lo)
                    rr.append((lo, hi, h, self.comm.Irecv(h, source=s, tag=22)))
                else:
                    rr.append((lo, hi, None, self.comm.Irecv(F[lo:hi], source=s, tag=22)))
        for lo, hi, h, q in rr:
            q.Wait()
            if h is not None:
                F[lo:hi] = self.cp.asarray(h)
        for q in reqs:
            q.Wait()
        return F

    def _own_faces_nccl(self, F):
        cp = self.cp
        nccl = self._nccl
        st = cp.cuda.get_current_stream().ptr
        nccl.groupStart()
        for s, lo, hi in self._fsend:
            if hi > lo:
                v = F[lo:hi]
                self._nc.send(int(v.data.ptr), int(hi - lo), self._ncdt, int(s), st)
        for s, lo, hi in self._frecv:
            if hi > lo:
                v = F[lo:hi]
                self._nc.recv(int(v.data.ptr), int(hi - lo), self._ncdt, int(s), st)
        nccl.groupEnd()
        return F

    def allmax_(self, x):
        if not self._nb:
            return x
        if self.xport == 'nccl':
            nccl = self._nccl
            st = self.cp.cuda.get_current_stream().ptr
            self._nc.allReduce(int(x.data.ptr), int(x.data.ptr), int(x.size), self._ncdt, nccl.NCCL_MAX, st)
            return x
        from mpi4py import MPI
        h = np.asarray(x.get(), dtype=np.float64)
        self.comm.Allreduce(MPI.IN_PLACE, h, op=MPI.MAX)
        x[...] = self.cp.asarray(h)
        return x


    def check(self, part=None, own=None):
        o = {}
        o['rank'] = self.rank
        o['size'] = self.size
        o['nL'] = self.nL
        o['nG'] = self.nG
        o['nLF'] = self.nLF
        o['nPartFace'] = self.nPartFace
        o['nOwnFace'] = self.nOwnFace
        o['peers'] = self._nb
        o['face_send'] = self._fsend
        o['face_recv'] = self._frecv
        o['xport'] = self.xport
        o['ghost_unique'] = self.nG == 0 or bool(np.unique(self.gcell[self.nL:]).size == self.nG)
        o['gface_unique'] = bool(np.unique(self.gface).size == self.nLF)
        if part is not None and own is not None:
            g = self.gface
            mineown = part[own[g]] == self.rank
            nsend = sum((hi - lo for _, lo, hi in self._fsend))
            nrecv = sum((hi - lo for _, lo, hi in self._frecv))
            o['owned_by_rule'] = int(mineown.sum())
            o['owned_by_layout'] = int(self.nOwnFace)
            o['send_plus_recv'] = int(nsend + nrecv)
        return o

def partition_from_case(case, groups, nproc=4):
    import mesh_addressing
    cm = mesh_addressing.cell_map(case, nproc=nproc)
    nC = int(sum((a.size for a in cm)))
    part = np.full(nC, -1, dtype=np.int64)
    for r, g in enumerate(groups):
        for p in g:
            part[cm[p]] = r
    if int(part.min()) < 0:
        raise SystemExit('partition_from_case: %d cells unassigned' % int((part < 0).sum()))
    return part
