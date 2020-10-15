# python analysis.py 0.0075 bior4.4 lenin0 0.0 10 0 0

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

def define_all(p):
    types = ["normal", "t", "hadamard"]
    np.random.seed(p['seed'])
    global optimum, prox, objective, oracle
    optimum     = objectives.sparse_vector(p)
    prox        = objectives.Proximal_Sparse(p)
    objective   = objectives.QuadraticSimple(p, optimum, is_diag=1)
    oracle      = oracles.Oracle_LS_Gaussian(objective, p['sigma'], types[p['noise']])

# parse_arguments()
# np.random.seed(p['seed'])

# st=ct(); define_all(p)
# if not do_lass:
#     es.launch_exp(p['seed'], oracle, prox, budget)
# else:
#     es.launch_noisy_exp(p['seed'], oracle, prox, budget)
# elapsed(st)

filename, wav_type = "{}.png".format(sys.argv[3]), sys.argv[2]
original   = rgb2gray(mpimg.imread(filename))
coeffs     = pywt.wavedec2(original, wav_type)
vec, sizes = convert_pyr_2_vec(coeffs)

p = {'mu': 0.1, 'L': 1.0, 's_level': 0.004}
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

print("dimensionality:", p['dim'])
print("sparsity:", p['s'])
in_er = objective(xz)
print("initial error: ".ljust(26) + "{:.5E}".format(in_er))

args, st = (p['seed'], oracle, prox, budget), ct()

for beta in [0.05, 0.01, 0.5, 0.6, 0.9, 1.5, 2.0, 2.5]:
    md = es.launch_mirror_descent(*args, beta=beta)
    print("beta: ", beta, end=";\t")
    print("error".ljust(10) + "{:.5E}".format(md.history[-1]['f(xk)']-in_er))

elapsed(st)


# for i in range(1, 2):
#     example = "{}_xk_(\d+)_0.1_1.0_65536_50_50_0.1_2_0.5_0_".format(i)
#     l = pieces = example.split("_")
#     args = my_map(lambda x: float(x) if '.' in x else int(x), [l[5], l[6], l[3], l[8], l[9]])
#     p['dim'], p['s'], p['mu'], p['sigma'], p['noise'] = args


#     pieces[2] = r"(\d+)"
#     pattern_x = "_".join(pieces)
#     pattern_y = pattern_x.replace("xk", "yk")

#     files = gf("/home/akulunch/Papers/code/implementation/Sparse_Python/numerics")
#     data_x, data_y = dict(), dict()
#     for f in files:
#         a = re.search(pattern_x, efn(f))
#         if a: data_x[int(a.group(1))] = np.loadtxt(f)
#         a = re.search(pattern_y, efn(f))
#         if a: data_y[int(a.group(1))] = np.loadtxt(f)

#     p['seed'] = int(example.split("_")[0])
#     define_all(p)

#     n_stages = len(lkeys(data_x))
#     x_arr = np.zeros(2 * n_stages)
#     for i in range(n_stages):
#         x_arr[2 * i] = objective(data_x[i+1])
#         x_arr[2 * i + 1] = objective(data_y[i+1])
#     n_arr = np.zeros(2 * n_stages)
#     for i in range(n_stages):
#         n_arr[2 * i] = np.linalg.norm(data_x[i+1])
#         n_arr[2 * i + 1] = np.linalg.norm(data_y[i+1])
#     mn = x_arr.mean() / n_arr.mean()
#     n_arr *= mn

#     plt.figure(figsize=(10,6))
#     plt.plot(x_arr, c='r', lw=1.3)
#     plt.plot(n_arr, c='g', lw=1.3)
#     # plt.scatter(2*np.arange(n_stages), x_arr[::2], c='r')
#     # plt.scatter(2*np.arange(n_stages)+1, x_arr[1::2], c='g')

#     plt.savefig("figures/{}_".format(p['seed']) + "_".join(example.split("_")[3:])+".png")
#     # plt.show()