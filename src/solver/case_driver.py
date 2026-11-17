from pathlib import Path
import hashlib, json, os, re, time
import numpy as np

def transition_factors(case, mf):
    import case_manifest as M
    p = Path(case) / 'processor0/0/alpha1'
    bc = M.parse_field_bcs(str(p))
    result = {}
    for name in mf.contact_angles():
        b = bc[name]
        result[name] = float(b['transitionFactor'])
    return result

def fixed_alpha(case, alpha, bown, nif):
    import foam_fields as F
    import case_manifest as M
    case = Path(case)
    addr = F.addressing(str(case), M.discover(str(case)).nproc())
    count = ones = 0
    for i, amap in enumerate(addr):
        base = case / ('processor%d' % i)
        pm = base / 'constant/polyMesh'
        patch = next((x for x in M.parse_boundary(str(pm / 'boundary')) if x['name'] == 'Left'))
        n = patch['nFaces']
        count += n
        txt = F._payload(str(base / '0/alpha1'))
        mat = re.search('\\bLeft\\s*\\{([^{}]*)\\}', txt, re.S)
        assert mat
        body = mat[1]
        assert re.search('\\btype\\s+fixedValue\\s*;', body)
        v = re.search('\\bvalue\\s+uniform\\s+([^;]+);', body)
        if v:
            values = np.full(n, float(v[1]))
        else:
            values, nv = F._list_after(body, body.index('value'))
            assert nv == n and len(values) == n
        if not n:
            continue
        own = F.read_label_list(str(pm / 'owner'))[patch['startFace']:patch['startFace'] + n]
        assert np.array_equal(values, alpha[amap[own]]), (i, 'fixedValue owner mismatch')
        ones += int((values == 1).sum())
    patches = M.parse_boundary(str(case / 'constant/polyMesh/boundary'))
    left = next((x for x in patches if x['name'] == 'Left'))
    assert left['nFaces'] == count
    mask = np.zeros(len(bown), dtype=bool)
    lo = left['startFace'] - nif
    mask[lo:lo + count] = True
    values = alpha[bown].copy()
    assert int((values[mask] == 1).sum()) == ones
    return (values, mask, dict(n_faces=count, one_faces=ones, zero_faces=count - ones, values_sha256=hashlib.sha256(values[mask].tobytes()).hexdigest()))

def run(*, ws, mesh, fields, control, manifest, case):
    import cupy as cp
    import sys
    R = sys.modules.get('__main__')
    if not hasattr(R, '_zng_control_dict'):
        import solver_main as R
    import time_step as STEP
    m, F, C = (mesh, fields, control)
    out = Path(os.environ['GPU_CASE_OUTPUT'])
    out.mkdir(parents=True, exist_ok=False)
    alpha0 = cp.asnumpy(F['alpha'])
    vol = cp.asnumpy(m['vol'])
    bown = cp.asnumpy(m['bown'])
    assert ws.steps == 0 and np.count_nonzero(cp.asnumpy(F['Ux'])) == 0
    vals, mask, bcproof = fixed_alpha(case, alpha0, bown, ws.nIF)
    assert np.array_equal(cp.asnumpy(m['bUt']) == 1, mask)
    m['alpha_fixed'] = cp.asarray(vals)
    m['alpha_fixed_mask'] = cp.asarray(mask)
    C['alpha_subcycles'] = int(manifest.case_scalar('nAlphaSubCycles'))
    C['co_target'] = 0.5
    assert C['alpha_subcycles'] == 1
    assert getattr(ws, 'ccf', None) is not None, 'CCF setup must be enabled before loading the case'
    assert float(C['flow_rate']) == float(C['flow_rate0']) > 0
    import case_manifest as XM
    ubcs = XM.parse_field_bcs(str(Path(case) / '0/U'))
    assert ubcs['Right']['type'] == 'inletOutlet' and re.search('\\buniform\\s*\\(\\s*0\\s+0\\s+0\\s*\\)', ubcs['Right']['inletValue'])
    m['case_right_io_mask'] = m['bPt'] == 1
    C['case_right_recon'] = 1
    C['case_mixed_outlet'] = 1
    import case_inlet as BI
    C['case_full_inlet'] = 1
    assert all((x['type'] == 'zeroGradient' for x in XM.parse_field_bcs(str(Path(case) / '0/pc')).values()))
    C['case_native_predictor'] = 1
    BI.build(case, m, cp, out)
    limits = R._zng_control_dict(case)
    dt = float(limits['deltaT'])
    now = 0.0
    oldco = 0.0
    factor = 1.0
    maxsteps = int(os.environ.get('GPU_CASE_MAX_STEPS', '64'))
    end = float(os.environ.get('GPU_CASE_END', '10'))
    assert 0 < end <= 10
    schedule_path = os.environ.get('GPU_CASE_TIME_SCHEDULE')
    schedule = json.loads(Path(schedule_path).read_text()) if schedule_path else None
    schedule = [float(row['time']) for row in schedule if float(row['time']) > 0] if schedule else None
    if schedule:
        assert abs(schedule[-1] - end) < 1e-14
    save_each = os.environ.get('GPU_CASE_SAVE_EACH') == '1'
    write_steps = int(os.environ.get('GPU_CASE_WRITE_STEPS', '2000'))
    assert write_steps > 0

    def snapshot(label, t):
        _b = m['bown']
        _ab = F['alpha'][_b]
        _aup = cp.where(ws.bphi >= 0, _ab, cp.where(m['bUt'] == 1, m['alpha_fixed'], _ab))
        np.savez(out / (label + '.npz'), alpha=cp.asnumpy(F['alpha']), U=np.column_stack([cp.asnumpy(F[k]) for k in ['Ux', 'Uy', 'Uz']]), pd=cp.asnumpy(ws.p), pc=cp.asnumpy(ws.ccf.pc), bphi=cp.asnumpy(ws.bphi), phi=cp.asnumpy(F['phi']), sgpc=cp.asnumpy(ws.ccf.sgpc), sgpc_bnd=cp.asnumpy(ws.ccf.sgpc_bnd), time=np.array(t), alpha_bown=cp.asnumpy(_ab), alpha_bnd_upwind=cp.asnumpy(_aup), pd_bown=cp.asnumpy(ws.p[_b]), pc_bown=cp.asnumpy(ws.ccf.pc[_b]))
    n = 0
    porevol = float(vol.sum())
    initial = float(alpha0 @ vol)
    integrated = 0.0
    start = time.monotonic()
    np.savez(out / 'initial_fields.npz', alpha=alpha0, vol=vol, bown=bown, alpha_fixed=vals, alpha_fixed_mask=mask, inlet_face=cp.asnumpy(m['bUt'] == 1), outlet_face=cp.asnumpy(m['bPt'] == 1))
    assert float(cp.abs(F['phi']).max()) == 0 and float(cp.abs(ws.bphi).max()) == 0
    import case_checkpoint as CK
    ck_identity = CK.identity(case, Path(__file__).parent, C)
    resume = os.environ.get('GPU_CASE_RESUME')
    if resume:
        restored = CK.restore(resume, cp=cp, ws=ws, fields=F, expected_identity=ck_identity)
        now = restored['time']
        n = restored['step']
        dt = restored['dt']
        oldco = restored['oldco']
        factor = restored['factor']
        integrated = restored['integrated']
        assert restored['initial'] == initial and restored['porevol'] == porevol
    checkpoint_step = int(os.environ.get('GPU_CASE_CHECKPOINT_STEP', '0'))
    checkpoint_period = float(os.environ.get('GPU_CASE_CHECKPOINT_SECONDS', '600'))
    checkpoint_last = time.monotonic()

    def checkpoint():
        state = dict(time=now, step=n, dt=dt, oldco=oldco, factor=factor, integrated=integrated, initial=initial, porevol=porevol)
        receipt = CK.save(out / ('checkpoint_%07d.npz' % n), cp=cp, ws=ws, fields=F, meta=state, identity=ck_identity)
        (out / 'latest_checkpoint.json').write_text(json.dumps(receipt, indent=2))
        checkpoints = sorted(out.glob('checkpoint_*.npz'), key=lambda p: int(p.stem.split('_')[1]))
        for obsolete in checkpoints[:-2]:
            assert obsolete.parent.resolve() == out.resolve()
            obsolete.unlink()
            obsolete.with_suffix('.receipt.json').unlink(missing_ok=True)
        return receipt

    def _nw(t):
        k = int(np.floor(t * 10000.0 + 1e-09)) + 1
        return k if k <= 50 else (k + 4) // 5 * 5
    _keep = [t for t in (0.005, 0.05, 0.25, 0.5, 1.0, 2.0, 3.0) if t > now * 1.000000000001]
    next_w = _nw(now)
    max_wall = float(os.environ.get('GPU_CASE_MAX_WALL_SECONDS', '1e100'))
    while now < end and n < maxsteps and (time.monotonic() - start < max_wall):
        tic = time.monotonic()
        co = float((cp.abs(F['phi']) / (m['dmag'] * m['magSf'])).max()) * dt
        aco = float((cp.abs(ws.ccf.deltaS) / cp.maximum(ws.mu_f, 1e-300)).max()) * dt * C['sigma'] if n else 0.0
        f = min(limits['maxCo'] / (co + 1e-300), limits['maxAlphaCo'] / (aco + 1e-300))
        if f > 1.2 and (co < oldco * factor or f > 1.5):
            factor = min(f, 1 + 0.1 * (f - 1), 1.3)
            dt = min(factor * dt, limits['maxDeltaT'])
        elif f < 0.95:
            factor = min(f, 0.81)
            dt = min(factor * dt, limits['maxDeltaT'])
        else:
            factor = 1.0
        oldco = co
        sub = min(dt, end - now)
        if schedule:
            sub = schedule[n] - now
        assert np.isfinite(sub) and sub > 0
        for half in range(2):
            pre = F['alpha'][m['bown']].copy()
            C['dt'] = sub * 0.5
            C['relax_dels'] = 0.5 * half
            STEP.step(ws, mesh=m, fields=F, control=C)
            qa = cp.where(ws.bphi >= 0, pre, cp.where(m['bUt'] == 1, m['alpha_fixed'], pre))
            integrated -= float(cp.sum(ws.bphi * qa)) * sub * 0.5
        C['dt'] = sub
        C['relax_dels'] = 0.0
        now += sub
        n += 1
        if save_each or n % write_steps == 0:
            snapshot('step_%07d' % n, now)
        cp.cuda.get_current_stream().synchronize()
        volume = float(cp.sum(F['alpha'] * m['vol']))
        row = dict(step=n, time=now, dt=sub, Co_previous=co, alphaCo_previous=aco, alpha_min=float(F['alpha'].min()), alpha_max=float(F['alpha'].max()), saturation=volume / porevol, phase_boundary_integral=integrated, mass_residual=(volume - initial - integrated) / porevol, boundary_net_flux=float(ws.bphi.sum()), boundary_abs_flux=float(cp.abs(ws.bphi).sum()), inlet_Q=float(ws.bphi[m['bUt'] == 1].sum()), inlet_Qalpha=float((ws.bphi * m['alpha_fixed'])[m['bUt'] == 1].sum()), outlet_Q=float(ws.bphi[m['bPt'] == 1].sum()), outlet_Qalpha=float((ws.bphi * F['alpha'][m['bown']])[m['bPt'] == 1].sum()), velocity_absmax=max((float(cp.abs(F[k]).max()) for k in ['Ux', 'Uy', 'Uz'])), pd_absmax=float(cp.abs(ws.p).max()), pd_mean=float(cp.mean(ws.p)), pd_range=float(cp.max(ws.p) - cp.min(ws.p)), wall_seconds=time.monotonic() - tic)
        _a = F['alpha']
        _pcf = ws.ccf.pc
        _am = float(cp.mean(_a))
        _bm = float(cp.mean(1.0 - _a))
        row.update(alpha_D=row['alpha_max'] - row['alpha_min'], alpha_cell_mean=_am, U_magmax=float(cp.sqrt(F['Ux'] * F['Ux'] + F['Uy'] * F['Uy'] + F['Uz'] * F['Uz']).max()), U_avgmag=(float(cp.mean(F['Ux'])) ** 2 + float(cp.mean(F['Uy'])) ** 2 + float(cp.mean(F['Uz'])) ** 2) ** 0.5, pd_min=float(cp.min(ws.p)), pd_max=float(cp.max(ws.p)), Pc_monitor=float(cp.mean(_pcf * _a)) / (abs(_am) + 1e-16) - float(cp.mean(_pcf * (1.0 - _a))) / (abs(_bm) + 1e-16), Co_end=float((cp.abs(F['phi']) / (m['dmag'] * m['magSf'])).max()) * sub)
        if not all((np.isfinite(v) for v in row.values())):
            raise AssertionError(row)
        if now * 10000.0 >= next_w - 1e-09:
            snapshot('w_%05d' % next_w, now)
            next_w = _nw(now)
        if row['alpha_cell_mean'] >= 0.999 or row['alpha_cell_mean'] <= 0.001:
            raise RuntimeError('average(alpha1) limit reached')
        if _keep and now >= _keep[0] * 0.999999999999:
            while _keep and now >= _keep[0] * 0.999999999999:
                _keep.pop(0)
            CK.save(out / ('keep_%07d.npz' % n), cp=cp, ws=ws, fields=F, meta=dict(time=now, step=n, dt=dt, oldco=oldco, factor=factor, integrated=integrated, initial=initial, porevol=porevol), identity=ck_identity)
        if checkpoint_step and n == checkpoint_step or time.monotonic() - checkpoint_last >= checkpoint_period:
            checkpoint()
            checkpoint_last = time.monotonic()
    if os.environ.get('GPU_CASE_CHECKPOINT_FINAL') == '1':
        checkpoint()
    np.savez(out / 'final_fields.npz', **{k: cp.asnumpy(v) for k, v in F.items() if isinstance(v, cp.ndarray)}, pd=cp.asnumpy(ws.p), pc=cp.asnumpy(ws.ccf.pc), bphi=cp.asnumpy(ws.bphi), time=np.array(now))
    return 0
