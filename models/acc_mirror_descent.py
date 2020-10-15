from structures.structures import OptimizationMethod as OptimizationMethod
from structures.structures import Subscriptable as Subscriptable
from structures.structures import Assignable as Assignable
from structures.structures import NamedClass as NamedClass
from structures.stepper import Stepper as Stepper
import structures.objectives as objectives
from numpy.random import choice
from other.utils import log_progress, timer_decorator, update_mean_std, extract_history
from utils import store_string
import numpy as np
import time, os

class MD_Structure(NamedClass, OptimizationMethod):
    def __init__(self, oracle, prox_operator, xz, beta, 
                trace, verbose, oracle_reset=True, isa=-1):
        # ida : index_start_average
        # np.random.seed(0)
        self.oracle = oracle; 
        self.beta_0 = beta(0) if not isinstance(beta, float) else beta
        self.beta_func = (lambda x: beta) if isinstance(beta, float) else beta

        self.prox_operator = prox_operator
        self.p = oracle.p.copy()
        self.trace, self.verbose = trace, verbose
        self.cur_step, self.sum_steps = None, 0 # for averaging
        self.isa = isa
        self.initialize(xz, oracle_reset)
        self.beta_f = lambda x: 0.5 * (x + 1)
        self.eta_f = lambda x: 0.5 * (x + 1)

    def initialize(self, xz, oracle_reset=True):
        self.x, self.iter, self.history = xz.copy(), 0, []
        self.xag = xz.copy()
        self.init_x, self.hat_x = xz.copy(), np.zeros_like(xz)
        if oracle_reset:
            self.oracle.reset()
        self.last_oracle_ncalls = self.oracle.calls - 1
        self.update_history()

    @timer_decorator
    def step(self):
        self.iter += 1
        i = self.iter
        self.xmd = convex(1.0 / self.beta_f(i), self.x, self.xag)

        if self.beta_func(0) < 0:  gradient, phi = self.oracle(self.xmd, 1)
        else:  gradient = self.oracle(self.xmd)

        if self.beta_func(0) < 0: 
            step = abs(self.beta_func(self.iter)) * np.linalg.norm(phi, ord=np.inf)**2
        else: step = self.beta_func(self.iter)

        # step = 20.0 / np.log(1.0 + self.eta_f(i))
        step = 4.0 / np.log(1.0 + self.eta_f(i))

        # step *= self.oracle.batch_size
        self.cur_step =  step;
        self.sum_steps += step * (self.iter >= self.isa)
        
        # gradient = self.oracle(self.xmd)
        self.x = self.prox_operator(gradient, self.x, self.init_x, step)
        self.xag = convex(1.0/self.beta_f(i), self.x, self.xag)

        # scale_0 = self.prox_l_one.scale
        # self.prox_l_one.scale /= step
        # self.x = self.prox_l_one(self.x - (gradient / step))
        # self.x = self.prox_operator(gradient, self.x, self.init_x, step)
        # self.x -= gradient / step
        # self.prox_l_one.scale = scale_0

        if self.iter >= self.isa:
            self.hat_x += self.cur_step * self.x
        if self.trace: self.update_history()

    @timer_decorator
    def update_history(self):
        objective = self.oracle.objective
        pack = {'copy': self.oracle.batch_size}
        pack['call'] = self.oracle.calls
        pack['f(xk)'] = objective(self.xag); 

        if self.iter < self.isa:
            pack['f(hat_xk)'] = pack['f(xk)']
        elif self.sum_steps:
            pack['f(hat_xk)'] = objective(self.hat_x / self.sum_steps)
        else: 
            pack['f(hat_xk)'] = pack['f(xk)']

        self.history.append(pack)
        self.last_oracle_ncalls = self.oracle.calls
        if hasattr(self, "last_notched_time"):
            self.time_notch = time.time() - self.last_notched_time
            self.last_notched_time = time.time()
            pack['time'] = self.time_notch

    @timer_decorator
    def make_pass(self, num=None, budget=None):
        x = num if num is not None else budget
        alpha = 2.0 * np.sqrt(2.0) * self['sigma'] * np.sqrt(np.log(self['dim']) / x)

        self.prox_l_one = objectives.Proximal_LOne(alpha)
        self.last_notched_time = self.start_time = time.time()
        if num is not None:
            self.budget = num
            for _ in range(num):
                self.step()
            self.final_processing()
        elif budget is not None:
            self.budget = budget
            while True:
                self.step()
                if self.oracle.calls >= budget: self.final_processing(); return
    def final_processing(self):
        self.history[-1]['xk'] = self.xag;
        self.history[-1]['hat_xk'] = self.hat_x / self.sum_steps;
        self.solution_xk = self.history[-1]['xk'];
        self.solution_hat_xk = self.history[-1]['hat_xk'];

    def set_beta(self, beta):
        self.beta_func = lambda x: beta


    def store_result(self, ind):
        root = "/home/akulunch/Papers/code/implementation/Sparse_Python/exps/"
        name = str(self['mu']) + "_" + str(self['L'])
        name += "_" + str(self['dim']) + "_" + str(self['s'])
        name += "_" + str(self['s']) + "_" + str(self['sigma']) + "_" + str(self['noise']) + "_" 
        name += str(abs(self.beta_0))[:5]

        if os.path.exists("D:\\Data\\" + name + "_1_" + str(ind)  + ".txt"):
            return

        evolution = extract_history(self.history, 'f(hat_xk)')[:self.budget]
        # np.savetxt("D:\\Data\\" + name + "_1_" + str(ind)  + ".txt", evolution)
        np.savetxt(root + name + "_1_" + str(ind)  + ".txt", evolution)
        # update_mean_std("agg_" + name + "_1_", evolution)

        evolution = extract_history(self.history, 'f(xk)')[:self.budget]
        # np.savetxt("D:\\Data\\" + name + "_2_" + str(ind)  + ".txt", evolution)
        np.savetxt(root + name + "_2_" + str(ind)  + ".txt", evolution)
        # update_mean_std("agg_" + name + "_2_", evolution)


class Averaged(Subscriptable,Assignable,MD_Structure):
    def __init__(self, oracle, prox, xz, beta=None, trace=1,verbose=0,oracle_reset=1,isa=-1):
        MD_Structure.__init__(self, oracle, prox, xz, beta, trace, verbose, oracle_reset, isa)

def convex(w, a, b):
    return w*a + (1.0-w)*b