import os as _os
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
_os.environ.setdefault('MKL_NUM_THREADS', '1')
_os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
