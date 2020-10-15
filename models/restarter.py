from structures.structures import OptimizationMethod as OptimizationMethod
from structures.structures import Subscriptable as Subscriptable
from structures.structures import Assignable as Assignable
from structures.structures import NamedClass as NamedClass
# from models.acc_mirror_descent import AccMD_Structure as AccMD_Structure
from models.mirror_descent import MD_Structure as MD_Structure
from models.mirror_descent import Averaged as Averaged
from models.mirror_descent import *
from models.averaged_mirror_descent import *
from other.statistics import *
# from post_processing.evolution import *
from other.utils import *
import numpy as np
import math, os, time

log = np.log
norm = np.linalg.norm
arr = np.array

class Restarter(MD_Structure):
    def __init__(self, method, oracle, prox, xz, trace=1, verbose=False, do_setup=1, 
                beta=0.0, do_batch=1, cp_thresh=0.5, min_n_pstage=0):
        # ida : index_start_average
        assert beta;
        self.method, self.prox, self.oracle = method, prox, oracle
        self.p, self.n_stages = oracle.p.copy(), 0
        self.p['done_switch'] = False

        self.iter, self.niters = 0, []
        self.beta_0 = beta(0) if not isinstance(beta, float) else beta
        self.beta, self.do_batch = beta, do_batch
        self.trace, self.verbose = trace, verbose
        self.min_n_pstage = min_n_pstage

        if do_setup: self.setup()
        self.initialize(xz)
        self.cp_thresh = cp_thresh
        # self.oracle.reset()
        self.show("m_0 = {} iterations\nK_bar = {} stages".\
                                        format(self.m_0, math.ceil(self.K_bar)))
        self.evolution = []
    def setup(self):
        mu, n, s = self['mu'], self['dim'], self['s']
        nu = 2 * self['L'] * (log(self['dim']) + 1.0)
        self.m_0 = int_floor(0.5 * (nu * self['s']) * (log(n) + 1))
        self.K_bar = 30000

    def initialize(self, xz):
        self.init_x, self.x, self.y = xz.copy(), xz.copy(), xz.copy()
        self.history_full  = []; self.history_stage = []
        self.n_pstage = self.n_astage = self.switched = 0;

        self.oracle.reset()
        self.last_oracle_ncalls = -1
        self.update_history()
        self.cusum_res, self.cp, self.started = [], [], time.time()

    def update_history_full(self):
        if not self.switched and self.n_pstage < 40:
            pack = self.curr_model.history[:]
            for p in pack:
                p['f(hat_xk)'] = p['f(xk)']
        else:
            pack = self.curr_model.history
        self.history_full.extend(pack)
        for a in pack:
            self.evolution.extend([a['f(hat_xk)'] for _ in range(a['copy'])])
        self.show("self.evolution length = {}".format(len(self.evolution)))

    def update_history(self):
        objective = self.oracle.objective
        pack = {'copy': self.oracle.calls - self.last_oracle_ncalls}
        pack['xk'], pack['yk'] = self.x, self.y
        pack['f(xk)'] = objective(self.x); pack['f(yk)'] = objective(self.y)

        self.last_oracle_ncalls = self.oracle.calls
        if hasattr(self, "last_notched_time"):
            self.time_notch = time.time() - self.last_notched_time
            self.last_notched_time = time.time(); pack['time'] = self.time_notch
        self.history_stage.append(pack)

    def preliminary_stage(self):
        self.n_pstage += 1
        self.show('Do prel. stage: {}/{}/{}'.format(self.m_0, self.n_pstage, self.K_bar))
        print_error("current error: ", self.oracle.objective(self.x))
        self.curr_model = self.method(self.oracle, self.prox, self.y, self.beta, 
                                    self.trace, 0, oracle_reset=0)
        self.curr_model.make_pass(num=min([self.budget, self.m_0]))

        if self.n_pstage < 40:
            self.x = self.curr_model.solution_xk
        else:
            self.x = self.curr_model.solution_hat_xk
        self.y = sparsify(self.x, self['s'])
        self.update_history()
        self.update_history_full()

    def asymptotic_stage(self):
        self.n_astage += 1
        if not self.do_batch:
            self.beta, self.m_0 = 2.0 * self.beta, 2.0 * self.m_0
        else:
            self.oracle.batch_size = 2 ** self.n_astage
            # self.beta = 2.0 * self.beta

        self.show('Do asymp. stage: {}/{}/{}'.format(self.m_0, self.n_astage, self.K_bar))
        self.curr_model = self.method(self.oracle, self.prox, self.y, self.beta, 
                                    self.trace, 0, oracle_reset=0)
        self.curr_model.make_pass(num=int(self.m_0))

        self.x = self.curr_model.solution_hat_xk
        self.y = sparsify(self.x, self['s'])
        self.update_history()
        self.update_history_full()

    def make_pass(self, budget=None):
        self.last_notched_time = self.start_time = time.time()
        self.budget = budget
        while True:
            if not self.switched:
                self.preliminary_stage()
            else:
                self.asymptotic_stage()

            self.load_dynamics()
            if self.check_finish(): return
            self.check_switch()

    def check_switch(self):
        if not self.switched and self.cusum_detect_switch():
            if self.n_pstage >= self.min_n_pstage:
                self.show(f"Switch at {self.n_pstage} stage")
                self.switched = 1
                if self['noise']==1: self.beta *= 3

    def check_finish(self):
        # if self.oracle.calls >= self.budget: 
        if len(self.evolution) >= self.budget: 
            self.oracle.batch_size = 1
            return True
    def cusum_detect_switch(self):
        # num=50; 
        num=150; 
        for start in range(0, len(self.curr_model.history) - num, num):
            pack = self.curr_model.history[start:start+num]
            x = np.arange(num); y = np.log([i['f(xk)'] for i in pack])
            x_bar = x.mean(); y_bar = y.mean(); d_x = x - x_bar; d_y = y - y_bar
            beta = np.dot(d_x, d_y) / np.dot(d_x, d_x)
            alpha = y_bar - beta * x_bar
            y_pred = array([alpha+beta*v for v in x])
            self.cusum_res.append(np.dot(y_pred-y, y_pred-y))
            if start==0: 
                self.cp.append(self.cusum_res[-1]); mean = self.cusum_res[-1]
            else: 
                self.cp.append( max([0, self.cusum_res[-1] - (mean + 0.5) + self.cp[-1]]) )
                mean = (mean*(len(self.cusum_res)-1) + self.cusum_res[-1]) / \
                                                                len(self.cusum_res)
            # return self.cp[-1] > 0.05
            # print("cusum test value:", self.cp[-1])
            return self.cp[-1] > self.cp_thresh


    def load_dynamics(self):
        if hasattr(self.oracle, "wav_type"):
            name = self.name_construct_2d()
        else:
            name = self.name_construct_1d()
        self.store_evolution(name)
        ind = self.n_pstage + self.n_astage
        if ind % 10 == 0:
            fnm = str(self['seed']) + "_xk_" + str(ind) + "_" + name + "_0.txt"
            np.savetxt("numerics/" + fnm, self.x)
            fnm = str(self['seed']) + "_yk_" + str(ind) + "_" + name + "_0.txt"
            np.savetxt("numerics/" + fnm, self.y)

        with open("dynamics/rs_job_info_" + name + " (stage).txt", 'w+') as f:
            n_stages = self.budget//self.m_0
            f.write("{}/{} stages\n".format(self.n_pstage, n_stages))
            passed  = time.time() - self.started
            remains = (n_stages - self.n_pstage) * passed / self.n_pstage
            f.write("{} seconds passed \n".format(int(passed)))
            f.write("{} seconds remains\n".format(int(remains)))

    def store_result(self, ind=0):
        if hasattr(self.oracle, "wav_type"):
            name = self.name_construct_2d()
        else:
            name = self.name_construct_1d()
        self.store_evolution(name)

    def name_construct_2d(self):
        p = self.oracle.p
        name  = p["image"] + "_" + str(abs(self.beta_0))[:5] + "_" + str(self.m_0) 
        name += "_" + str(p['dim']) + "_"
        name += self.oracle.wav_type + "_{}".format(p['sigma'])
        return name
    def name_construct_1d(self):
        name = str(self['mu']) + "_" + str(self['L'])
        name += "_" + str(self['dim']) + "_" + str(self['s']) + "_" + str(self['s'])
        name += "_" + str(self['sigma']) + "_" + str(self['noise']) + "_"
        name += str(abs(self.beta_0))[:5]
        return name
    def store_evolution(self, name, root="exps/"):
        # evolution = extract_history(self.history_full, 'f(xk)')[:self.budget]
        np.savetxt(root + name + "_0_" + str(self['seed']) + ".txt", self.evolution[::200])


    def __getitem__(self, index):
        if isinstance(index, int):
            return self.history[:, index:index+1] if index >= 0 else self.history[:, index:]
        elif isinstance(index, str):
            if index in self.p: return self.p[index]
            elif index in self.__dict__: return self.__dict__[index]
            else: raise KeyError

    def show(self, text):
        if self.verbose:
            print(text)