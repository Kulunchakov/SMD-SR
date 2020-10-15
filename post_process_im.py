from other.utils import *
import numpy as np
import sys, os 

def store_aggregated_results_local(p, wav_type, mode, beta, filename=None):
    name  = filename+"_" if filename is not None else ""
    name += str(beta)[:5] + "_"
    if mode==0:
        name += str(p['m_0']) + "_"
    name += str(p['dim']) + "_" + wav_type + "_" + str(p['sigma']) + "_" + str(mode) + "_"

    files = [f for f in os.listdir("exps") if starts(efn(f), name)]
    if len(files)==0: 
        print("name:", name); print("no files!"); return
        
    files_0 = files[:]
    budget = np.min([len(np.loadtxt("exps/" + f)) for f in files_0])
    budget = int(budget)
    files = []
    for f in files_0:
        aa = np.loadtxt("exps/" + f)
        if len(aa) >= budget and aa[-1] <= aa[0]:
            files.append(f)
    if mode==0:
        # print(files)
        print(len(files), "files collected.")
    arr = np.zeros((len(files), budget))
    to_delete = []
    for (i,f) in enum(files): 
        arr[i,:] = np.loadtxt("exps/" + f)[:budget]
        to_delete.append("exps/" + f)
    name = "agg_" + name + str(len(files))
    m, q_025, q_075 = np.median(arr, axis=0), \
                      np.quantile(arr, 0.25, axis=0), np.quantile(arr, 0.75, axis=0)
    np.savetxt("av_exps/" + name + "_med.txt", m)
    np.savetxt("av_exps/" + name + "_q_025.txt", q_025)
    np.savetxt("av_exps/" + name + "_q_075.txt", q_075)

# p = {'dim': 20320, 'sigma': 0.0, 'm_0': 10000}
p = {'dim': 13231, 'sigma': 0.0, 'm_0': 40000}

wav_type = "bior4.4"
# beta = 0.0075

#     store_aggregated_results_local(p, wav_type, 0, beta)
# for beta in [0.001, 0.002, 0.0075, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]:
for (name, beta, dim) in zip(["lenin2", "lenin3"], [0.7, 2.0], [661556, 341982]):
    for m_0 in [20000]:
        p["m_0"], p["dim"] = m_0, dim
        for sigma in [0.0]:
            p["sigma"] = sigma
            store_aggregated_results_local(p, wav_type, 0, beta, name)
            store_aggregated_results_local(p, wav_type, 1, beta, name)
            store_aggregated_results_local(p, wav_type, 2, beta, name)

print("\n\n\n")
