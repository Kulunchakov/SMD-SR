import numpy as np
import pickle, sys

import models.mirror_descent as mirror_descent
import structures.objectives as objectives
import models.restarter      as restarter
import structures.oracles    as oracles
import sklearn.linear_model  as lm
import draw.main as draw
import exp_setup as es
from   other.utils import *

def warn(*args, **kwargs): pass
import warnings; warnings.warn = warn

# parse_arguments
p, args = dict(), my_map(lambda x: float(x) if '.' in x else int(x), sys.argv[1:])
p['dim'], p['s'], b, p['seed'], p['mu'], p['sigma'], dl, dr, p['noise'] = args
p['L'], budget, do_lass, do_restart = 1.0, b, dl, dr

if p['noise'] == 2: p['dim'] = 65536

# define everything
types = ["normal", "t", "hadamard"]
np.random.seed(p['seed'])

optimum     = objectives.sparse_vector(p)
prox        = objectives.Proximal_Sparse(p)
objective   = objectives.QuadraticSimple(p, optimum, is_diag=1)
oracle      = oracles.Oracle_LS_Gaussian(objective, p['sigma'], types[p['noise']])

# launch a simulation
args, st = (p['seed'], oracle, prox, budget), ct()
if not do_lass:
    if do_restart:
        es.launch_restarter(*args)
    else:
        es.launch_mirror_descent(*args)
else:
    if do_restart:
        es.launch_restarter(*args, noisy=1)
    else:
        es.launch_lasso(*args)
    
elapsed(st)

