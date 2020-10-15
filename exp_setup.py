import numpy as np

import models.acc_mirror_descent as acc_md
import models.mirror_descent as mirror_descent
import structures.objectives as objectives
import models.restarter as restarter
import structures.oracles as oracles
import draw.main as draw
import matrix.main as mx
from other.utils import *
import sklearn.linear_model as lm

import pickle

def launch_lasso_exp_simple(ind, X, y, notches, objective, verbose=0):
    np.random.seed(ind); p = objective.p

    base_value = objective(np.zeros(p['dim']))
    output = np.ones_like(notches, dtype=np.float64)
    for (i, notch) in enum(notches):
        print("notch:", notch)
        if verbose:
            print("accessible budget of oracle regressors:", notch, end='\t')
        alpha = 2.0 * np.sqrt(2.0) * p['sigma'] * np.sqrt(np.log(p['dim']) / notch)
        lasso = lm.Lasso(alpha=alpha, max_iter=notch, tol=1e-12, selection="random")
        lasso.fit(X[:notch,:], y[:notch])
        if verbose:
            print(lasso.n_iter_, end= '\n')

        output[i] = objective(lasso.coef_)
        output[i] = base_value if output[i] >= base_value else output[i]
    # output *= output[0] 
    # output[0] = output[-1] 
    return output

def launch_lasso_exp(ind, oracle, budget, notches, verbose=True):
    np.random.seed(ind); oracle.reset(); p = oracle.p
    X, y = lasso_data_gen(ind, oracle, budget)
    
    data = launch_lasso_exp_simple(ind, X, y, notches, oracle.objective)
    data_0 = data[:]; data = [data_0[0] for _ in range(notches[0])]
    for i in range(1, len(notches)):
        data.extend([data_0[i] for _ in range(notches[i]-notches[i-1])])
    data = np.array(data[:budget])
    # filename construction
    name = str(p['mu']) + "_" + str(p['L'])
    name += "_" + str(p['dim']) + "_" + str(p['s'])
    name += "_" + str(p['s']) + "_" + str(p['sigma']) + "_" + str(p['noise'])
    if verbose:
        print("lasso xk: ".ljust(12) + "{:.3E}".format(data[-1]))
    np.savetxt("exps/" + name + "_7_" + str(ind)  + ".txt", data[::50])
    # update_mean_std("agg_" + name + "_7_", data)
    return data

def launch_lasso(ind, oracle, prox, budget):
    notches = list(np.hstack((np.arange(10, 1500, 200), 
                         np.arange(1600, 3000, 400), 
                         np.arange(3200, budget, 2000)))) + [budget]
    data = launch_lasso_exp(ind, oracle, budget, notches)

def launch_restarter(ind, oracle, prox, budget, beta=None,
                                        verbose=True, m_0=None, noisy=0):
    np.random.seed(ind); oracle.reset()
    xz = np.zeros(oracle['dim'])
    workhorse = mirror_descent.Averaged
    if beta is None:
        betas = [-0.1, -0.01, -0.5] if not noisy else [-0.2, -0.005, -2.5]
        beta  = betas[oracle.p['noise']]

    the_model = restarter.Restarter(workhorse, oracle, prox, xz, beta=beta, verbose=verbose,
                                    cp_thresh=0.5, min_n_pstage=10)
    if m_0 is not None:
        the_model.m_0 = m_0
    the_model.make_pass(budget)
    the_model.store_result(ind)
    if verbose:
        print_error("restart hat_xk: ", the_model.history_full[-1]['f(xk)'])
    return the_model

def launch_mirror_descent(ind, oracle, prox, budget, beta=None, verbose=True):
    np.random.seed(ind); oracle.reset()
    xz = np.zeros(oracle['dim'])
    if beta is None:
        betas = [-0.1, -0.01, -0.5]
        beta  = betas[oracle.p['noise']]
    md_model = mirror_descent.Averaged(oracle, prox, xz, beta=beta, trace=1)
    md_model.make_pass(budget)
    md_model.store_result(ind)
    if verbose:
        print_error("mirror desc. xk: ", md_model.history[-1]['f(xk)'])
    return md_model

def launch_exp(ind, oracle, prox, budget, verbose=True):
    np.random.seed(ind); oracle.reset()
    xz = np.zeros(oracle['dim'])
    workhorse = mirror_descent.Averaged
    betas = [-0.1, -0.0075, -0.5]
    beta  = betas[oracle.p['noise']]
    the_model = restarter.Restarter(workhorse, oracle, prox, xz, beta=beta, verbose=0,
                                    cp_thresh=0.07, min_n_pstage=4)

    the_model.make_pass(budget)
    the_model.store_result(ind)
    if verbose:
        print("restart hat_xk: ".ljust(22) + "{:.3E}".format(\
                                            the_model.history_full[-1]['f(xk)']))

    betas = [-0.1, -0.0075, -0.5]
    beta  = betas[oracle.p['noise']]
    np.random.seed(ind); oracle.reset()
    md_model = mirror_descent.Averaged(oracle, prox, xz, beta=beta, trace=1)
    md_model.make_pass(budget)
    md_model.store_result(ind)
    if verbose:
        print("mirror desc. xk: ".ljust(12) + "{:.3E}".format(\
                                            md_model.history[-1]['f(xk)']))

    # plot_restart_vs_lasso_basic(oracle.p, 
    #                         extract_history(the_model.history_full, 'f(hat_xk)')[:budget],
    #                         extract_history(md_model.history, 'f(xk)')[:budget])    

def make_experiment(ind, oracle, prox, budget, verbose=True):
    np.random.seed(ind); oracle.reset(); xz = np.zeros(oracle['dim'])
    prox_eu = objectives.Proximal_Euclidean(); beta = -0.1 if oracle.p['noise'] < 2 else -0.5
    acc_restart = restarter.Restarter(acc_md.Averaged, oracle, prox, xz, beta=beta, verbose=0,
                                                       cp_thresh=0.07, min_n_pstage=4)
    acc_restart.make_pass(budget=budget)
    print("restart accelerated hat_xk: ".ljust(22) + "{:.3E}".format(\
                                        acc_restart.history_full[-1]['f(hat_xk)']))

    np.random.seed(ind); oracle.reset()
    acc_mid = acc_md.Averaged(oracle, prox, xz, beta=beta, trace=1); acc_mid.make_pass(budget=budget)
    print("accelerated mirror desc. xk: ".ljust(22) + "{:.3E}".format(acc_mid.history[-1]['f(xk)']))

    np.random.seed(ind); oracle.reset();
    base_restart = restarter.Restarter(mirror_descent.Averaged, oracle, prox, xz, beta=beta, verbose=0,
                                                                cp_thresh=0.07, min_n_pstage=4)
    base_restart.make_pass(budget=budget)
    print("base restart hat_xk: ".ljust(15) + "{:.3E}".format(base_restart.history_full[-1]['f(hat_xk)']))

    np.random.seed(ind); oracle.reset()
    md = mirror_descent.Averaged(oracle, prox, xz, beta=beta, trace=1); md.make_pass(budget=budget)
    print("mirror desc. xk: ".ljust(12) + "{:.3E}".format(md.history[-1]['f(xk)']))
    p = oracle.p
    name = str(p['mu']) + "_" + str(p['L']) + "_" + str(p['dim']) + "_" + str(p['s']) + "_"
    name += str(p['sigma']) + "_" + str(p['seed'])
    compare([acc_restart.history_full, base_restart.history_full, acc_mid.history, md.history], 
            ['f(hat_xk)', 'f(hat_xk)', 'f(xk)', 'f(xk)'], colors=['b', 'g', 'm', 'r'],
            legends=['ACC restart', 'BASE restart', 'ACC md', 'MD'], loglog=1, name=name)

def launch_exp_acc(ind, oracle, prox, budget, verbose=True):
    np.random.seed(ind); oracle.reset()
    xz = np.zeros(oracle['dim'])
    workhorse = acc_md.Averaged
    prox_eu = objectives.Proximal_Euclidean()
    beta = -0.1 if oracle.p['noise'] < 2 else -0.5
    the_model = restarter.Restarter(workhorse, oracle, prox, xz, beta=beta, verbose=0,
                                    cp_thresh=0.07, min_n_pstage=4)

    the_model.make_pass(budget=budget)
    the_model.store_result(ind)
    if verbose:
        print("restart accelerated hat_xk: ".ljust(22) + "{:.3E}".format(\
                                            the_model.history_full[-1]['f(hat_xk)']))

    beta = -0.1 if oracle.p['noise'] < 2 else -0.5
    np.random.seed(ind); oracle.reset()
    md_model = acc_md.Averaged(oracle, prox, xz, beta=beta, trace=1)
    md_model.make_pass(budget=budget)
    md_model.store_result(ind)
    if verbose:
        print("accelerated mirror desc. xk: ".ljust(22) + "{:.3E}".format(\
                                            md_model.history[-1]['f(xk)']))

    np.random.seed(ind); oracle.reset()
    xz = np.zeros(oracle['dim'])
    workhorse = mirror_descent.Averaged
    prox_eu = objectives.Proximal_Euclidean()
    beta = -0.1 if oracle.p['noise'] < 2 else -0.5
    the_model_0 = restarter.Restarter(workhorse, oracle, prox, xz, beta=beta, verbose=0,
                                    cp_thresh=0.07, min_n_pstage=4)
    the_model_0.make_pass(budget=budget)
    the_model_0.store_result(ind)

    if verbose:
        print("base restart hat_xk: ".ljust(15) + "{:.3E}".format(\
                                            the_model_0.history_full[-1]['f(hat_xk)']))

    beta = -0.1 if oracle.p['noise'] < 2 else -0.5
    np.random.seed(ind); oracle.reset()
    md_model_0 = mirror_descent.Averaged(oracle, prox, xz, beta=beta, trace=1)
    md_model_0.make_pass(budget=budget)
    md_model_0.store_result(ind)
    if verbose:
        print("mirror desc. xk: ".ljust(12) + "{:.3E}".format(\
                                            md_model_0.history[-1]['f(xk)']))

    plot_restart_vs_lasso_basic(oracle.p, 
                        extract_history(the_model.history_full, 'f(hat_xk)')[:budget],
                        extract_history(the_model_0.history_full, 'f(hat_xk)')[:budget])
                        # extract_history(md_model_0.history, 'f(xk)')[:budget],  
                        # extract_history(md_model.history, 'f(xk)')[:budget])

def create_setups(p):
    base_p = p.copy()
    np.random.seed(0)
    setups = dict()
    for mu, sigma in [[0.01, 0.001], [1.0, 0.001], [1.0, 1.0], [1.0, 0.1], [0.1, 0.001], \
                        [0.1, 0.1], [0.2, 0.1], [0.1, 1.0], [0.1, 2.0]]:
        p['mu'], p['sigma'] = mu, sigma
        key = tuple([mu, sigma])
        setups[key] = dict()
        setups[key]['optimum'] = objectives.sparse_vector(p)
        setups[key]['prox']    = objectives.Proximal_Sparse(p)
        setups[key]['prox_eu'] = objectives.Proximal_Euclidean()
        setups[key]['objective'] = objectives.QuadraticSimple(p, \
                                        setups[key]['optimum'], is_diag=1)
        setups[key]['oracle'] = oracles.Oracle_LS_Gaussian(\
                                        setups[key]['objective'], p['sigma'])
    p = base_p.copy()
    return setups