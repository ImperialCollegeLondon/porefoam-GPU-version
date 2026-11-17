from __future__ import annotations
import hashlib
import os
import re

class ManifestError(AssertionError):
    pass
U_TYPE_CODE = {'fixedValue': 0, 'slowTwoPhaseFlux': 1, 'inletOutlet': 2, 'zeroGradient': 2}
P_TYPE_CODE = {'fixedFluxPressure': 0, 'zeroGradient': 0, 'fixedRelaxedMeanValue': 1, 'fixedValue': 1}
A_TYPE_CODE = {'fixedValue': 2, 'zeroGradient': 0, 'calculated': 0, 'hysteresisContactAngle': 1, 'constantAlphaContactAngle': 1, 'dynamicAlphaContactAngle': 1, 'timeVaryingAlphaContactAngle': 1}
PROC_PATCH_TYPES = ('processor', 'processorCyclic')

def _read(path):
    with open(path, 'rb') as fh:
        b = fh.read()
    return b.replace(b'\r\n', b'\n').replace(b'\r', b'\n').decode('utf-8', 'replace')

def _strip_comments(txt):
    txt = re.sub('/\\*.*?\\*/', ' ', txt, flags=re.S)
    txt = re.sub('//[^\\n]*', ' ', txt)
    return txt

def _kv(block):
    out = {}
    for m in re.finditer('^[ \\t]*([A-Za-z_]\\w*)[ \\t]+([^;{}\\n]*);', block, re.M):
        out.setdefault(m.group(1), m.group(2).strip())
    return out

def _named_blocks(txt, start_at=0):
    out = []
    pat = re.compile('\\n([ \\t]*)([A-Za-z_]\\w*)[ \\t]*\\n\\1\\{[ \\t]*\\n(.*?)\\n\\1\\}', re.S)
    for m in pat.finditer(txt, start_at):
        out.append((m.group(2), m.group(3), m.start()))
    return out

def parse_boundary(path):
    txt = _strip_comments(_read(path))
    out = []
    for name, body, _pos in _named_blocks(txt):
        d = _kv(body)
        if 'nFaces' not in d or 'startFace' not in d:
            continue
        out.append({'name': name, 'type': d.get('type', ''), 'nFaces': int(d['nFaces']), 'startFace': int(d['startFace']), 'neighbProcNo': int(d['neighbProcNo']) if 'neighbProcNo' in d else None})
    return out

def parse_field_bcs(path):
    txt = _read(path)
    i = txt.rfind('boundaryField')
    if i < 0:
        return {}
    body_txt = txt[i:]
    out = {}
    for name, body, _pos in _named_blocks(_strip_comments(body_txt)):
        d = _kv(body)
        if 'type' in d:
            out[name] = d
    for m in re.finditer('\\n([ \\t]*)([A-Za-z_]\\w*)[ \\t]*\\n\\1\\{', body_txt):
        name = m.group(2)
        if name in out:
            continue
        tail = body_txt[m.end():m.end() + 8192]
        cut = re.search('\\b(?:nonuniform|List<|uniform\\s*\\()|\\(\\s*\\n', tail)
        head = tail[:cut.start()] if cut else tail
        d = _kv(head)
        if 'type' in d:
            d['_truncated_scan'] = '1'
            out[name] = d
    return out

def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _owner_note(path):
    if not os.path.exists(path):
        return {}
    head = open(path, 'rb').read(4096).decode('utf-8', 'replace')
    m = re.search('note\\s+"([^"]*)"', head)
    if not m:
        return {}
    return {k: int(v) for k, v in re.findall('(\\w+):\\s*(\\d+)', m.group(1))}

class Manifest(object):

    def __init__(self, case):
        self.case = os.path.abspath(case)
        self.discovered = {}
        self.provenance = {}

    def _set(self, key, value, source):
        self.discovered[key] = value
        self.provenance[key] = source

    def get(self, key):
        if key in self.discovered:
            return self.discovered[key]
        raise ManifestError('manifest has no value for %r on %s' % (key, self.case))

    def source_of(self, key):
        return 'discovered'

    def nproc(self):
        return int(self.get('nproc'))

    def left_n(self):
        return int(self.get('left_n'))

    def left_a(self):
        return float(self.get('left_a'))

    def flow_rate(self):
        return float(self.get('flow_rate'))

    def flow_rate0(self):
        try:
            return float(self.get('flow_rate0'))
        except ManifestError:
            raise ManifestError("no flowRate0 for this case.  It is a REQUIRED key of slowTwoPhaseFlux, so its absence means 0/U was not parsed or the inlet is not that BC.  Add it to the case, or add a flow_rate0 to this profile -- do not default it to zero, which would select one branch by default.")

    def case_scalar(self, key):
        return self.discovered.get('case_' + key)

    def case_scalar_source(self, key):
        return self.source_of('case_' + key)

    def contact_angles(self):
        out = {}
        for src in ('contact_angle_wall', 'contact_angle_other'):
            for k, v in (self.discovered.get(src) or {}).items():
                patch = k.split('.')[0]
                out.setdefault(patch, set()).add(float(v))
        return {p: sorted(s)[0] if len(s) == 1 else sorted(s) for p, s in out.items()}

    def theta_deg(self):
        return float(self.get('theta_deg'))

    def inlet_patch(self):
        return self.get('inlet_patch')

    def u_bc(self):
        return dict(self.get('U_BC'))

    def p_bc(self):
        return dict(self.get('P_BC'))

    def a_bc(self):
        return dict(self.get('A_BC'))

    def patches(self):
        return list(self.discovered.get('patches', []))

    def wall_patch_ranges(self, n_internal_faces):
        out = []
        for p in self.patches():
            if p['type'] == 'wall' and p['nFaces'] > 0:
                out.append((p['name'], p['startFace'] - int(n_internal_faces), p['nFaces']))
        return out

    def output_times(self):
        return list(self.discovered['output_times'])



    def attach_mesh(self, nC, nIF, nBF, bmag, patch_lo, patch_nf, patch_name):
        self._set('mesh_nC', int(nC), "caller's _mesh_arrays")
        self._set('mesh_nIF', int(nIF), "caller's _mesh_arrays")
        self._set('mesh_nBF', int(nBF), "caller's _mesh_arrays")
        ip = self.inlet_patch()
        area = None
        for nm, lo, nf in zip(patch_name, patch_lo, patch_nf):
            if str(nm) == str(ip):
                area = float(bmag[int(lo):int(lo) + int(nf)].sum())
                self._set('left_n_mesh', int(nf), "nFaces of patch %r in the caller's patch list" % ip)
        if area is not None:
            self._set('left_a', area, "sum |Sf| over patch %r, from the caller's boundary face areas" % ip)
        return self.verify_mesh()

    def verify_mesh(self):
        bad = []
        for k, disc in (('n_cells', self.discovered.get('mesh_nC')), ('n_internal_faces', self.discovered.get('mesh_nIF'))):
            got = self.discovered.get(k)
            if got is not None and disc is not None and (int(got) != int(disc)):
                bad.append('%s: text says %s, mesh says %s' % (k, got, disc))
        if 'left_n_mesh' in self.discovered:
            if int(self.discovered['left_n_mesh']) != int(self.discovered.get('left_n', -1)):
                bad.append('left_n: boundary file says %s, mesh patch list says %s' % (self.discovered.get('left_n'), self.discovered['left_n_mesh']))
        if bad:
            raise ManifestError('MANIFEST MESH CHECK FAILED on %s:\n    %s' % (self.case, '\n    '.join(bad)))
        return 0


_CACHE = {}

def discover(case, force=False):
    key = os.path.abspath(case)
    if not force and key in _CACHE:
        return _CACHE[key]
    mf = _discover(key)
    _CACHE[key] = mf
    return mf

def _discover(root):
    mf = Manifest(root)
    procs = sorted([d for d in os.listdir(root) if re.fullmatch('processor\\d+', d)], key=lambda s: int(s[9:]))
    mf._set('nproc', len(procs), 'count of processor* dirs in the case root')
    if procs and [int(p[9:]) for p in procs] != list(range(len(procs))):
        pass
    p0 = os.path.join(root, procs[0]) if procs else root
    tdirs = sorted([d for d in os.listdir(p0) if d and d[0].isdigit()], key=float)
    with_a = [d for d in tdirs if os.path.exists(os.path.join(p0, d, 'alpha1'))]
    mf._set('time_dirs', tdirs, '%s/<t> directory names' % (procs[0] if procs else '.'))
    mf._set('output_times', with_a, '%s/<t> carrying alpha1, sorted by float' % (procs[0] if procs else '.'))
    mf._set('n_time_dirs', len(with_a), 'len(output_times), t=0 included')
    mf._set('n_targets', max(0, len(with_a) - 1), "number of output times minus t=0")
    tsets = {}
    for pd in procs:
        pp = os.path.join(root, pd)
        tsets[pd] = frozenset((d for d in os.listdir(pp) if d and d[0].isdigit() and os.path.exists(os.path.join(pp, d, 'alpha1'))))
    _ref = tsets.get(procs[0]) if procs else frozenset()
    _mis = {pd: sorted(_ref ^ s) for pd, s in tsets.items() if s != _ref}
    mf._set('output_times_consistent', not _mis, 'the set of alpha1-carrying time directories compared across all %d processors' % len(procs))
    if _mis:
        mf._set('output_times_disagreements', _mis, 'cross-rank time sets')
    dts = []
    for t in with_a:
        f = os.path.join(p0, t, 'uniform', 'time')
        if not os.path.exists(f):
            continue
        d = _kv(_strip_comments(_read(f)))
        try:
            dts.append((t, float(d['deltaT']), int(d.get('index', -1))))
        except (KeyError, ValueError):
            pass
    mf._set('dt_observed', [(t, v) for t, v, _ in dts], '%s/<t>/uniform/time, key deltaT' % (procs[0] if procs else '.'))
    mf._set('step_index', [(t, i) for t, _, i in dts], '%s/<t>/uniform/time, key index' % (procs[0] if procs else '.'))
    if dts:
        lo = min((v for _, v, _ in dts))
        hi = max((v for _, v, _ in dts))
        mf._set('dt_span', hi / lo if lo else float('inf'), 'max/min of dt_observed')
    _pimple = {}
    _fvs = _first_existing(os.path.join(root, 'system', 'fvSolution'), os.path.join(p0, 'system', 'fvSolution'))
    if _fvs:
        _t = _strip_comments(_read(_fvs))
        for _nm, _body, _ in _named_blocks(_t):
            if _nm in ('PIMPLE', 'PISO'):
                _pimple.update(_kv(_body))
        mf._set('pimple_keys', sorted(_pimple), 'system/fvSolution, PIMPLE/PISO sub-dictionary')
    _tp = _first_existing(os.path.join(root, 'constant', 'transportProperties'), os.path.join(p0, 'constant', 'transportProperties'))
    _tpd = _kv(_strip_comments(_read(_tp))) if _tp else {}

    def _last_number(s):
        for _tok in reversed(str(s).replace(']', ' ').split()):
            try:
                return float(_tok)
            except ValueError:
                continue
        return None
    for _k, _src, _dd, _dk, _dflt in (('cBC', 'system/fvSolution PIMPLE', _pimple, 'cBC', None), ('cAlpha', 'system/fvSolution PIMPLE', _pimple, 'cAlpha', None), ('beta', 'system/fvSolution PIMPLE', _pimple, 'beta', None), ('lambda', 'system/fvSolution PIMPLE', _pimple, 'lambda', None), ('lambdaS', 'system/fvSolution PIMPLE', _pimple, 'lambdaS', None), ('filterU', 'system/fvSolution PIMPLE', _pimple, 'filterU', 1.0), ('UBoundByUSmoothFactor', 'system/fvSolution PIMPLE', _pimple, 'UBoundByUSmoothFactor', 2.0), ('pcThicknessFactor', 'system/fvSolution PIMPLE', _pimple, 'pcThicknessFactor', None), ('fcCorrectTangent', 'system/fvSolution PIMPLE', _pimple, 'fcCorrectTangent', None), ('fcCorrectTangentRelax', 'system/fvSolution PIMPLE', _pimple, 'fcCorrectTangentRelax', None), ('fcdFilter', 'system/fvSolution PIMPLE', _pimple, 'fcdFilter', None), ('pcRefCell', 'system/fvSolution PIMPLE', _pimple, 'pcRefCell', None), ('pcRefValue', 'system/fvSolution PIMPLE', _pimple, 'pcRefValue', None), ('smoothingKernel', 'system/fvSolution PIMPLE', _pimple, 'smoothingKernel', None), ('smoothingRelaxFactor', 'system/fvSolution PIMPLE', _pimple, 'smoothingRelaxFactor', None), ('wallSmoothingKernel', 'system/fvSolution PIMPLE', _pimple, 'wallSmoothingKernel', None), ('pBoundBypSmoothFactor', 'system/fvSolution PIMPLE', _pimple, 'pBoundBypSmoothFactor', 1.3), ('filterPd', 'system/fvSolution PIMPLE', _pimple, 'filterPd', 1.0), ('nAlphaSubCycles', 'system/fvSolution PIMPLE', _pimple, 'nAlphaSubCycles', 1.0), ('nCorrectors', 'system/fvSolution PIMPLE', _pimple, 'nCorrectors', None), ('nOuterCorrectors', 'system/fvSolution PIMPLE', _pimple, 'nOuterCorrectors', None), ('nNonOrthogonalCorrectors', 'system/fvSolution PIMPLE', _pimple, 'nNonOrthogonalCorrectors', 0.0), ('sigma', 'constant/transportProperties', _tpd, 'sigma', None)):
        _raw = _dd.get(_dk)
        _v = _last_number(_raw) if _raw is not None else None
        if _v is not None:
            mf._set('case_' + _k, _v, '%s, key %s (%s)' % (_src, _dk, 'default %g' % _dflt if _dflt is not None else 'required'))
        elif _dflt is not None:
            mf._set('case_' + _k, float(_dflt), "ABSENT from %s; default used" % _src)
        else:
            pass
    cd = _first_existing(os.path.join(root, 'system', 'controlDict'), os.path.join(p0, 'system', 'controlDict'))
    if cd:
        d = _kv(_strip_comments(_read(cd)))
        claimed = {k: d[k] for k in ('adjustTimeStep', 'deltaT', 'maxCo', 'maxAlphaCo', 'endTime', 'writeControl', 'writeInterval', 'maxDeltaT') if k in d}
        mf._set('dt_claimed', claimed, 'system/controlDict')
        if claimed.get('adjustTimeStep') == 'no' and mf.discovered.get('dt_span', 1.0) > 1.000001:
            pass
    tp = _first_existing(os.path.join(root, 'constant', 'transportProperties'), os.path.join(p0, 'constant', 'transportProperties'))
    if tp:
        d = _kv(_strip_comments(_read(tp)))
        for k in ('sigma', 'theta'):
            if k in d:
                m = re.search('([-+0-9.eE]+)\\s*$', d[k])
                if m:
                    mf._set('tp_' + k, float(m.group(1)), 'constant/transportProperties, key %s' % k)
    gb = _first_existing(os.path.join(root, 'constant', 'polyMesh', 'boundary'))
    per_proc = []
    for pd in procs:
        f = os.path.join(root, pd, 'constant', 'polyMesh', 'boundary')
        if os.path.exists(f):
            per_proc.append((pd, parse_boundary(f)))
    if gb:
        patches = parse_boundary(gb)
        src = 'constant/polyMesh/boundary (global)'
    elif per_proc:
        agg, order = ({}, [])
        for _pd, pl in per_proc:
            for p in pl:
                if p['type'] in PROC_PATCH_TYPES:
                    continue
                if p['name'] not in agg:
                    agg[p['name']] = {'name': p['name'], 'type': p['type'], 'nFaces': 0, 'startFace': None, 'neighbProcNo': None}
                    order.append(p['name'])
                agg[p['name']]['nFaces'] += p['nFaces']
        patches = [agg[n] for n in order]
        src = 'SUM over processor*/constant/polyMesh/boundary (no global mesh)'
    else:
        patches, src = ([], 'NOT FOUND')
    mf._set('patches', patches, src)
    mf._set('patch_names', [p['name'] for p in patches], src)
    mf._set('wall_patches', [p['name'] for p in patches if p['type'] == 'wall' and p['nFaces'] > 0], src)
    gn = _owner_note(os.path.join(root, 'constant', 'polyMesh', 'owner'))
    if 'nCells' in gn:
        mf._set('n_cells', gn['nCells'], 'constant/polyMesh/owner header note, key nCells')
    if 'nInternalFaces' in gn:
        mf._set('n_internal_faces', gn['nInternalFaces'], 'constant/polyMesh/owner header note')
    pn = [(pd, _owner_note(os.path.join(root, pd, 'constant', 'polyMesh', 'owner'))) for pd in procs]
    if pn and all(('nCells' in n for _, n in pn)):
        mf._set('proc_cells', [n['nCells'] for _, n in pn], 'processor*/constant/polyMesh/owner header notes')
        tot = sum((n['nCells'] for _, n in pn))
        if 'n_cells' not in mf.discovered:
            mf._set('n_cells', tot, 'SUM of processor* owner notes (no global mesh present)')
        elif tot != mf.discovered['n_cells']:
            pass
    if pn and all(('nInternalFaces' in n for _, n in pn)):
        mf._set('proc_internal_faces', [n['nInternalFaces'] for _, n in pn], 'processor*/constant/polyMesh/owner header notes')
        mf._set('intra_faces', sum((n['nInternalFaces'] for _, n in pn)), 'SUM of processor* nInternalFaces')
    for fld, table, key in (('U', U_TYPE_CODE, 'U_BC'), ('pd', P_TYPE_CODE, 'P_BC'), ('alpha1', A_TYPE_CODE, 'A_BC')):
        f = _first_existing(os.path.join(p0, '0', fld), os.path.join(root, '0', fld))
        if not f:
            continue
        bcs = parse_field_bcs(f)
        codes, unknown = ({}, [])
        for nm, d in bcs.items():
            ty = d.get('type', '')
            if ty in PROC_PATCH_TYPES:
                continue
            if ty not in table:
                unknown.append((nm, ty))
                continue
            if table[ty] != 0:
                codes[nm] = table[ty]
        mf._set(key, codes, '%s/0/%s boundaryField type -> code (%s)' % (procs[0] if procs else '.', fld, ', '.join(('%s=%s' % kv for kv in sorted({nm: d.get('type', '') for nm, d in bcs.items() if d.get('type') not in PROC_PATCH_TYPES}.items())))))
        if unknown:
            pass
    fU = _first_existing(os.path.join(p0, '0', 'U'), os.path.join(root, '0', 'U'))
    if fU:
        bcs = parse_field_bcs(fU)
        for nm, d in sorted(bcs.items()):
            if d.get('type') == 'slowTwoPhaseFlux':
                mf._set('inlet_patch', nm, 'the slowTwoPhaseFlux patch in 0/U')
                for k in ('flowRate0', 'flowRate1', 'gradientFactor0', 'gradientFactor1', 'pdUniformization', 'pcUniformization'):
                    if k in d:
                        try:
                            mf._set('inlet_' + k, float(d[k]), '0/U boundaryField/%s/%s' % (nm, k))
                        except ValueError:
                            pass
                break
        ip = mf.discovered.get('inlet_patch')
        if ip:
            n = sum((p['nFaces'] for p in patches if p['name'] == ip))
            mf._set('left_n', n, 'nFaces of patch %r in %s' % (ip, src))
            if 'inlet_flowRate1' in mf.discovered:
                mf._set('flow_rate', mf.discovered['inlet_flowRate1'], '0/U boundaryField/%s/flowRate1' % ip)
            if 'inlet_flowRate0' in mf.discovered:
                mf._set('flow_rate0', mf.discovered['inlet_flowRate0'], '0/U boundaryField/%s/flowRate0' % ip)
    disagree = []
    fpc = _first_existing(os.path.join(p0, '0', 'pc'), os.path.join(root, '0', 'pc'))
    if fpc:
        _pct = {}
        for nm, d in parse_field_bcs(fpc).items():
            t = d.get('type')
            if t and (not nm.startswith('procBoundary')):
                _pct[nm] = t
        mf._set('pc_bc_types', dict(_pct), '0/pc boundaryField types')
        _bad = {n: t for n, t in _pct.items() if t != 'zeroGradient'}
        if _bad:
            raise ManifestError('0/pc is not zeroGradient on %s' % (_bad,))
    for fld in ('U', 'pd', 'alpha1'):
        ref = None
        for pd in procs:
            f = os.path.join(root, pd, '0', fld)
            if not os.path.exists(f):
                continue
            sig = {nm: d for nm, d in sorted(parse_field_bcs(f).items()) if d.get('type') not in PROC_PATCH_TYPES}
            if ref is None:
                ref = (pd, sig)
                continue
            for nm in sorted(set(ref[1]) | set(sig)):
                a, b = (ref[1].get(nm), sig.get(nm))
                if a is None or b is None:
                    disagree.append('%s: patch %r on %s but not %s' % (fld, nm, pd if b else ref[0], ref[0] if b else pd))
                    continue
                for k in sorted(set(a) | set(b)):
                    if k == '_truncated_scan':
                        continue
                    if k != 'type' and any((str(x or '').lstrip().startswith(('uniform', 'nonuniform')) for x in (a.get(k), b.get(k)))):
                        continue
                    if a.get(k) != b.get(k):
                        disagree.append('%s/%s/%s: %s=%r but %s=%r' % (fld, nm, k, ref[0], a.get(k), pd, b.get(k)))
    mf._set('bc_consistent', not disagree, 'every processor*/0/{U,pd,alpha1} boundaryField compared key by key against processor0 (%d processors)' % len(procs))
    if disagree:
        mf._set('bc_disagreements', disagree, 'cross-processor comparison')
    fa = _first_existing(os.path.join(p0, '0', 'alpha1'), os.path.join(root, '0', 'alpha1'))
    if fa:
        wall_set = set(mf.discovered.get('wall_patches', []))
        ang, ang_other = ({}, {})
        for nm, d in parse_field_bcs(fa).items():
            for k in ('theta0', 'thetaA', 'thetaR'):
                if k in d:
                    try:
                        v = float(d[k])
                    except ValueError:
                        continue
                    (ang if nm in wall_set else ang_other)['%s.%s' % (nm, k)] = v
        if ang or ang_other:
            mf._set('contact_angle_wall', ang, '0/alpha1 boundaryField on the polyMesh `wall` patches (this BC OVERRIDES transportProperties theta)')
            mf._set('contact_angle_other', ang_other, '0/alpha1 contact angles on non-wall patches')
            vals = sorted(set(ang.values()))
            if len(vals) == 1:
                mf._set('theta_deg', vals[0], 'the single thetaA/thetaR across the wall patches in 0/alpha1')
            elif len(vals) > 1:
                pass
            else:
                pass
            if ang_other:
                pass
            tt = mf.discovered.get('tp_theta')
            if tt is not None and vals and all((v != tt for v in vals)):
                pass
    h = hashlib.md5()
    npm = 0
    for pd in ['.'] + list(procs):
        pm = os.path.join(root, pd, 'constant', 'polyMesh')
        if not os.path.isdir(pm):
            continue
        for fn in sorted(os.listdir(pm)):
            fp = os.path.join(pm, fn)
            if not os.path.isfile(fp):
                continue
            npm += 1
            h.update(('%s/%s:%d|' % (pd, fn, os.path.getsize(fp))).encode())
            if fn == 'boundary':
                h.update(hashlib.md5(open(fp, 'rb').read()).hexdigest().encode())
    mf._set('mesh_digest', h.hexdigest()[:16], 'md5 over (name,size) of %d polyMesh files plus the content md5 of every `boundary`' % npm)
    return mf
