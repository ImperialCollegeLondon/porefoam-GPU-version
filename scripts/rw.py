import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC = os.path.join(ROOT, 'src')
TREE = os.path.join(SRC, 'solver')
sys.path.insert(0, SRC)
import pyporefoam  # noqa: E402
assert os.path.realpath(pyporefoam.__file__).startswith(SRC + os.sep), pyporefoam.__file__
sys.path.insert(0, TREE)
script = os.path.join(TREE, 'solver_main.py')
sys.argv = [script]
runpy.run_path(script, run_name='__main__')
