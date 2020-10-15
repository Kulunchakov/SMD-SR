import numpy as np
import pickle, sys, os
import pywt, pywt.data, pickle
import matplotlib.image as mpimg

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

filename, wav_type = "{}.png".format(sys.argv[3]), sys.argv[2]
original   = rgb2gray(mpimg.imread(filename))
coeffs     = pywt.wavedec2(original, wav_type)
vec, sizes = convert_pyr_2_vec(coeffs)

p = {'mu': 0.1, 'L': 1.0, 's_level': 0.001}
p['noise'], p['sigma'] = "normal", float(sys.argv[4])
p['dim'], _ = find_size_of_vec_from_pyr(coeffs)
p['s']  = int(p['dim'] * p['s_level'])
do_restart, p['seed'] = int(sys.argv[6]), int(sys.argv[7])
budget, beta, m_0 = int(sys.argv[5]), -abs(float(sys.argv[1])), 20000 # 1000*1000

optimum = sparsify(vec, int(p['s_level'] * p['dim']))
coeffs  = convert_vec_2_pyr(optimum, sizes)
image   = pywt.waverec2(coeffs, wav_type)
xz      = np.zeros(p['dim'])

np.random.seed(p['seed'])

prox      = objectives.Proximal_Sparse(p)
objective = objectives.QuadraticImage(p, image, optimum)
oracle    = oracles.Oracle_Image(objective, p['sigma'], p['noise'], wav_type)
oracle.p["image"] = sys.argv[3]

print("sparsity:", p['s'])
print("optimum sparsity:", np.linalg.norm(objective.x_optimum, ord=0))
print("initial error: ".ljust(22) + "{:.5E}".format(objective(xz)))

args, st = (p['seed'], oracle, prox, budget), ct()

if do_restart:
    br   = es.launch_restarter(*args, beta=beta, m_0=m_0)
    name = "sole_restart_" + br.name_construct_2d()
    data_md = extract_history(br.history_full, "f(xk)")
    draw.plot_result(data_md, title="Mirror Descent convergence", name=name)
    print("restart. xk: ".ljust(22) + \
          "{:.5E}".format(br.history_full[-1]['f(hat_xk)']))
else:
    md = es.launch_mirror_descent(*args, beta=beta)
    name = "sole_md_" + md.name_construct_2d()
    data_rs = extract_history(md.history, "f(xk)")
    draw.plot_result(data_rs, title="Restart convergence", name=name)

    print("mirror desc. xk: ".ljust(22) + "{:.5E}".format(md.history[-1]['f(xk)']))

elapsed(st)
