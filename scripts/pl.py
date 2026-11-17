import glob
import json
import os
import sys
best = None
n = 0
for pattern in sys.argv[1:]:
    for j in glob.glob(pattern):
        n += 1
        try:
            r = json.load(open(j))
        except Exception:
            continue
        p = r.get('path')
        meta = r.get('meta') or {}
        st = meta.get('step')
        if not p or st is None or (not os.path.isfile(p)):
            continue
        if best is None or st > best[0]:
            best = (st, p, meta.get('time'))
if best is None:
    sys.exit(9)
sys.stdout.write(best[1] + '''
''')
