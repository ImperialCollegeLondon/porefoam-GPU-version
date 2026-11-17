from pathlib import Path
import hashlib, json, os, datetime
import numpy as np

def identity(case, source, control):
    src = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(source).glob('*.py'))}
    c = {k: v for k, v in control.items() if isinstance(v, (int, float, str, bool)) and k not in ['dt', 'relax_dels']}
    return dict(source=src, control=c, case=str(Path(case).resolve()))

def save(path, *, cp, ws, fields, meta, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    entries = []
    arrays = {}
    seen = set()
    skipped = []

    def visit(obj, route):
        if isinstance(obj, (cp.ndarray, np.ndarray)):
            key = 'a%05d' % len(arrays)
            arrays[key] = cp.asnumpy(obj) if isinstance(obj, cp.ndarray) else obj.copy()
            entries.append(dict(route=route, kind='cuda' if isinstance(obj, cp.ndarray) else 'numpy', key=key))
            return
        if obj is None or isinstance(obj, (bool, int, float, str)):
            entries.append(dict(route=route, kind='scalar', value=obj))
            return
        if isinstance(obj, np.generic):
            entries.append(dict(route=route, kind='scalar', value=obj.item()))
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, (list, tuple)):
            entries.append(dict(route=route, kind='container', type=type(obj).__name__, size=len(obj)))
            for i, v in enumerate(obj):
                visit(v, route + [i])
        elif isinstance(obj, dict):
            entries.append(dict(route=route, kind='container', type='dict', keys=list(obj)))
            for k, v in obj.items():
                visit(v, route + [k])
        elif type(obj).__module__.startswith('gpu_') and hasattr(obj, '__dict__'):
            for k, v in vars(obj).items():
                visit(v, route + [k])
        else:
            skipped.append(dict(route=route, type=type(obj).__module__ + '.' + type(obj).__name__))
    cp.cuda.get_current_stream().synchronize()
    visit(ws, ['ws'])
    visit(fields, ['fields'])
    manifest = dict(schema=1, identity=identity, meta=meta, entries=entries, skipped_rebuilt_objects=skipped, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    arrays['__manifest__'] = np.frombuffer(json.dumps(manifest, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8)
    with tmp.open('xb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    receipt = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size, n_arrays=len(arrays) - 1, meta=meta, skipped_rebuilt_objects=skipped)
    path.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt

def restore(path, *, cp, ws, fields, expected_identity):
    path = Path(path)
    receipt = json.loads(path.with_suffix('.receipt.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256'], 'checkpoint hash mismatch'
    root = {'ws': ws, 'fields': fields}

    def get(obj, k):
        return obj[k] if isinstance(obj, (dict, list, tuple)) else getattr(obj, k)

    def put(obj, k, v):
        if isinstance(obj, (dict, list)):
            obj[k] = v
        elif isinstance(obj, tuple):
            raise TypeError('tuple replacement requires parent reconstruction')
        else:
            setattr(obj, k, v)

    def assign(route, v):
        obj = root
        for k in route[:-1]:
            obj = get(obj, k)
        if isinstance(obj, tuple):
            old = list(obj)
            old[route[-1]] = v
            assign(route[:-1], tuple(old))
        else:
            put(obj, route[-1], v)

    def value(route):
        obj = root
        for k in route:
            obj = get(obj, k)
        return obj
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive['__manifest__'].tobytes())
        assert manifest['schema'] == 1
        assert manifest['identity'] == expected_identity, 'source/input/control/environment identity mismatch'
        for e in manifest['entries']:
            route = e['route']
            kind = e['kind']
            try:
                old = value(route)
            except (AttributeError, KeyError, IndexError):
                old = None
            if kind == 'container':
                typ = e['type']
                cls = {'dict': dict, 'tuple': tuple, 'list': list}[typ]
                if not isinstance(old, cls):
                    assign(route, {} if typ == 'dict' else cls([None] * e['size']))
                elif typ != 'dict':
                    assert len(old) == e['size'], route
            elif kind == 'scalar':
                assign(route, e['value'])
            else:
                a = archive[e['key']]
                cls = cp.ndarray if kind == 'cuda' else np.ndarray
                if isinstance(old, cls):
                    assert old.shape == a.shape and old.dtype == a.dtype, route
                    old[...] = cp.asarray(a) if kind == 'cuda' else a
                else:
                    assign(route, cp.asarray(a) if kind == 'cuda' else a.copy())
    cp.cuda.get_current_stream().synchronize()
    return manifest['meta']
