from other.utils import *
import numpy as np
import sys, os 

p = {'L': 1.0, 'dim': 500000, 's': 500}
d_0 = p['dim']
print('number of files:', 
        len(os.listdir("exps/")))
budget = 200000 // 50
s_0 = p['s']
do_median = int(sys.argv[1])

for noise in [0]:
    p['s'], p['noise'], p['dim'] = s_0, noise, d_0
    betas = [0.1, 0.01, 0.5]
    beta  = betas[p['noise']]

    if p['noise'] == 2: p['dim'] = 65536
    
    for mu in [1.0, 0.1]:
        for sigma in [0.001, 0.1]:
            st=ct()
            # print("p['mu'], p['sigma'] = {}, {}".format(mu, sigma))
            p['mu'], p['sigma'] = mu, sigma
            store_aggregated_results(p, budget, 0, beta, do_median)
            store_aggregated_results(p, budget, 1, beta, do_median)
            store_aggregated_results(p, budget, 2, beta, do_median)
            elapsed(st)

    # betas = [0.2, 0.005, 2.5]
    # beta  = betas[p['noise']]

    # budgets   = [10000, 10000, 10000]
    # budget_0  = budgets[p['noise']] // 50

    # p['dim'] = 65536 if p['noise'] == 2 else 50000
    # for mu in [1.0, 0.1]:
    #     for sigma in [10.0, 5.0]:
    #         p['mu'], p['sigma'] = mu, sigma
    #         # p['sigma'] = 5.0 if p['noise'] == 1 else 10.0
    #         # p['sigma'] = 5.0 if noise == 1 else 10.0
    #         print("p['mu'], p['sigma'] = {}, {}".format(mu, p['sigma']))

    #         store_aggregated_results(p, budget_0, 0, beta, do_median)
    #         store_aggregated_results(p, budget_0, 7, "", do_median)
    print("\n\n\n")