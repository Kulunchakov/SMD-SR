from structures.structures import OptimizationMethod as OptimizationMethod
from structures.structures import Subscriptable as Subscriptable
from structures.structures import Assignable as Assignable
from structures.structures import NamedClass as NamedClass
# from models.acc_mirror_descent import AccMD_Structure as AccMD_Structure
from models.acc_mirror_descent import MD_Structure as MD_Structure
from models.acc_mirror_descent import Averaged as Averaged
from models.acc_mirror_descent import *
from models.averaged_mirror_descent import *
from other.statistics import *
# from post_processing.evolution import *
from other.utils import *
import numpy as np
import math, os

log = np.log
norm = np.linalg.norm
arr = np.array

class Restarter(MD_Structure):
    def __init__(self, method, oracle, prox, xz, trace=1, verbose=False, do_setup=1, 
                beta=0.0, do_batch=1, isa=0, cp_thresh=0.5, min_n_pstage=0):
        # ida : index_start_average
        assert beta;
        self.method, self.prox = method, prox
        self.oracle = oracle
        self.p, self.n_stages = oracle.p.copy(), 0
        self.p['done_switch'] = False

        self.iter, self.niters = 0, []
        self.beta_0 = beta(0) if not isinstance(beta, float) else beta
        self.beta, self.do_batch = beta, do_batch
        self.trace, self.verbose = trace, verbose
        self.isa = isa; self.min_n_pstage = min_n_pstage
        # self.simple_prox = Proximal_Simple()
        # self.prox = Proximal_Simple()
        if do_setup: self.setup()
        self.initialize(xz)
        self.cp_thresh = cp_thresh
        # self.oracle.reset()
        if self.verbose:
            print("m_0 = {} iterations\nK_bar = {} stages".\
                                        format(self.m_0, math.ceil(self.K_bar)))

    def setup(self):
        mu, n, s = self['mu'], self['dim'], self['s']
        upsilon = norm(self.oracle.objective.A, ord=np.inf)
        nu = 2 * self['L'] * (log(self['dim']) + 1.0)
        self.m_0 = int_floor(0.1 * (nu * self['s']) * (log(n) + 1))
        # nu = 2 * upsilon * (log(self['dim']) + 1.0)
        # self.m_0 = int_floor((upsilon * self['s'] / mu) * (log(n) + 1))
        # self.m_0 = int_floor((self['L'] * self['s'] / mu)  * (log(n) + 1) )
        # w = mu/(self['L']*np.log(self['dim']))
        # self.m_0 = 40 * int_floor(np.log(1.0/ self['s']) / np.log(1 - w))
        # self.m_0 = 2 * int_floor(s * np.log(self['dim']) / np.sqrt(self['mu']))
        self.K_bar = 30000

    def initialize(self, xz):
        self.init_x = xz.copy()
        self.x, self.y = xz.copy(), xz.copy()
        self.history_full  = []; self.history_stage = []
        self.n_pstage = self.n_astage = 0;

        self.oracle.reset()
        self.last_oracle_ncalls = -1
        self.update_history()
        self.cusum_res = []; self.cp = []; self.mean = None
        # self.update_history_full()

    def update_history_full(self):
        if self.n_astage == 0:
            pack = self.curr_model.history
            for p in pack:
                p['f(hat_xk)'] = p['f(xk)']
            self.history_full.extend(pack)
        else:
            pack = self.curr_model.history
            self.history_full.extend(pack)

            
    
    def update_history(self):
        objective = self.oracle.objective
        pack = {'copy': self.oracle.calls - self.last_oracle_ncalls}
        pack['xk'] = self.x; pack['yk'] = self.y;
        pack['f(xk)'] = objective(self.x);
        pack['f(yk)'] = objective(self.y);
        self.last_oracle_ncalls = self.oracle.calls
        if hasattr(self, "last_notched_time"):
            self.time_notch = time.time() - self.last_notched_time
            self.last_notched_time = time.time()
            pack['time'] = self.time_notch
        self.history_stage.append(pack)

    def preliminary_stage(self):
        print("preliminary_stage:", self.n_pstage, " --- ", self.oracle.batch_size)
        # if self.n_pstage==0: np.random.seed(0)

        self.n_pstage += 1
        self.show('Do prel. stage: {}/{}/{}'.format(self.m_0, self.n_pstage, self.K_bar))
        # if self.n_pstage < 15:
        #     sc, lim = -0.05, 180
        #     isa = self.isa
        # else:
        #     sc, lim = -0.1, 70
        #     isa = 0

        # beta = lambda x: sc if x<=lim else sc * (x - lim)
        # self.beta = -0.2 if self.n_pstage <= 14 else -1.0 * np.sqrt(self.n_pstage-14)
        self.curr_model = self.method(self.oracle, self.prox, self.y, self.beta, 
                                    self.trace, 0, oracle_reset=0, isa=self.isa)
        self.curr_model.make_pass(num=self.m_0)

        self.x = self.curr_model.solution_xk
        self.y = sparsify(self.x, self['s'])
        self.update_history()
        self.update_history_full()

    def asymptotic_stage(self):
        print("asymptotic_stage:", self.n_astage, " --- ", self.oracle.batch_size)
        self.n_astage += 1
        if not self.do_batch:
            self.beta *= 2.0 
            self.m_0 *= 2.0
        else:
            self.oracle.batch_size = 2**self.n_astage

        self.show('Do asymp. stage: {}/{}/{}'.format(self.m_0, \
                                                        self.n_astage, self.K_bar))
        self.curr_model = self.method(self.oracle, self.prox, self.y, self.beta, 
                                    self.trace, 0, oracle_reset=0, isa=self.isa)
        self.curr_model.make_pass(num=int(self.m_0))

        self.x = self.curr_model.solution_xk
        self.y = sparsify(self.x, self['s'])
        self.update_history()
        self.update_history_full()

    def make_pass_new(self, num=None, budget=None):
        assert num is None or budget is None
        
        self.last_notched_time = self.start_time = time.time()
        if num is not None:
            self.budget = num
            for i in range(min([num, self.K_bar])):
                self.preliminary_stage()
            for i in range(num - self.K_bar):
                self.asymptotic_stage()
        if budget is not None:
            self.budget = budget
            for i in range(self.K_bar):
                self.preliminary_stage()
                if self.oracle.calls >= budget:
                    self.oracle.batch_size = 1
                    return
            while True:
                self.asymptotic_stage()
                if self.oracle.calls >= budget:
                    self.oracle.batch_size = 1
                    return

    def make_pass(self, budget=None, do_prelim=1):
        self.last_notched_time = self.start_time = time.time()
        
        if budget is not None:
            self.budget = budget
            while True:
                if not do_prelim: break
                self.preliminary_stage()
                if self.oracle.calls >= budget: 
                    self.oracle.batch_size = 1
                    # print("Max stat of CuSum: ", max(self.cp))
                    return
                if self.cusum_detect_switch():
                    if self.n_pstage >= self.min_n_pstage:
                        if self.verbose: print(f"Switch at {self.n_pstage} stage")
                        self.beta *= 1.0 
                        break
            while True:
                self.asymptotic_stage()
                if self.oracle.calls >= budget: 
                    self.oracle.batch_size = 1
                    return
    def cusum_detect_switch(self):
        num=50; 
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
            print("cusum test value:", self.cp[-1])
            return self.cp[-1] > self.cp_thresh


    def store_result(self, ind):
        root = "/home/akulunch/Papers/code/implementation/Sparse_Python/exps/"
        name = str(self['mu']) + "_" + str(self['L'])
        name += "_" + str(self['dim']) + "_" + str(self['s'])
        name += "_" + str(self['s']) + "_" + str(self['sigma']) + "_" + str(self['noise']) + "_"
        name += str(abs(self.beta_0))[:5]

        evolution = extract_history(self.history_full, 'f(hat_xk)')[:self.budget]

        np.savetxt(root + name + "_0_" + str(ind)  + ".txt", evolution)
        # update_mean_std("agg_" + name + "_0_", evolution)
        

    # def make_pass(self, budget=None):
    #     self.last_notched_time = self.start_time = time.time()
        
    #     if budget is not None:
    #         self.budget = budget
    #         for i in range(self.K_bar):
    #             self.preliminary_stage()
    #             if self.oracle.calls >= budget: 
    #                 self.oracle.batch_size = 1
    #                 return
    #         while True:
    #             self.asymptotic_stage()
    #             if self.oracle.calls >= budget: 
    #                 self.oracle.batch_size = 1
    #                 return

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.history[:, index:index + 1] if index >= 0 else self.history[:, index:]
        elif isinstance(index, str):
            if index in self.p: return self.p[index]
            elif index in self.__dict__: return self.__dict__[index]
            else: raise KeyError

    def show(self, text):
        if self.verbose:
            print(text)

# class Proximal_Simple():
#     def __init__(self):
#         pass
#     def __call__(self, zeta, x, x_0, beta):
#         return x - zeta / beta

# elif "AccMD_Structure" in str(self.method):
#     eta_0 = min([0.25/self.nu, self['s']*10 / \
#                         (np.sqrt(self.varsigma)*(self.m_0**1.5))])
#     beta = 1.0 / eta_0


# v2 = objective(sparsify(md_model.history[-1]['xk'], 30))
# v3 = objective(sparsify(md_model.history[-1]['hat_xk'], 30))