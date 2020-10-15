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
    def __init__(self, oracle, prox, xz, beta, trace, verbose, oracle_reset=True):
        # ida : index_start_average
        # np.random.seed(0)
        self.oracle = oracle; 
        self.beta_0 = beta(0) if not isinstance(beta, float) else beta
        self.beta_func = (lambda x: beta) if isinstance(beta, float) else beta

        self.prox = prox
        self.p = oracle.p.copy()
        self.trace, self.verbose = trace, verbose
        self.sum_steps = 0 # for averaging
        self.from_restart = not oracle_reset
        self.initialize(xz, oracle_reset)

    def initialize(self, xz, oracle_reset=True):
        self.x, self.iter, self.history = xz.copy(), 0, []
        self.init_x, self.hat_x = xz.copy(), np.zeros_like(xz)

        if oracle_reset: self.oracle.reset()
        self.last_oracle_ncalls = self.oracle.calls - 1
        self.update_history()
        self.started = time.time()

    @timer_decorator
    def step(self):
        self.iter += 1
        if self.beta_func(0) < 0:  
            gradient, phi = self.oracle(self.x, 1)
            step = abs(self.beta_func(self.iter)) * np.linalg.norm(phi, ord=np.inf)**2
        else:  
            gradient = self.oracle(self.x)
            step = self.beta_func(self.iter)

        self.sum_steps += step
        self.x = self.prox(gradient, self.x, self.init_x, step)

        self.hat_x += step * self.x
        if self.trace: self.update_history()

    @timer_decorator
    def update_history(self):
        objective = self.oracle.objective
        pack = {'copy': self.oracle.batch_size}
        pack['call'] = self.oracle.calls
        pack['f(xk)'] = objective(self.x); 

        if self.sum_steps:
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
    def make_pass(self, num=None):
        self.last_notched_time = self.start_time = time.time()
        self.budget = num
        for _ in range(num):
            self.step()
            self.load_dynamics()
        self.final_processing()

    def load_dynamics(self):
        p = self.oracle.p
        if self.iter % 10000 == 0:
            if hasattr(self.oracle, "wav_type"):
                name = self.name_construct_2d()
            else:
                name = self.name_construct_1d()
            if not self.from_restart:
                self.store_evolution(name)

            if self.from_restart:
                f = open("dynamics/rs_job_info_" + name + ".txt", 'w+')
            else:
                f = open("dynamics/md_job_info_" + name + ".txt", 'w+')

            f.write("{}/{} iterations\n".format(self.iter, self.budget))
            passed  = time.time() - self.started
            remains = (self.budget - self.iter) * passed / self.iter
            f.write("{} seconds passed \n".format(int(passed)))
            f.write("{} seconds remains\n".format(int(remains)))
            f.close()
            
    def final_processing(self):
        self.history[-1]['xk'] = self.x;
        self.history[-1]['hat_xk'] = self.hat_x / self.sum_steps;
        self.solution_xk = self.history[-1]['xk'];
        self.solution_hat_xk = self.history[-1]['hat_xk'];

    def set_beta(self, beta):
        self.beta_func = lambda x: beta

    def store_result(self, ind=0):
        if hasattr(self.oracle, "wav_type"):
            name = self.name_construct_2d()
        else:
            name = self.name_construct_1d()
        self.store_evolution(name)

    def name_construct_2d(self):
        p = self.oracle.p
        name  = p["image"] + "_" + str(abs(self.beta_0))[:5] + "_" + str(p['dim']) + "_"
        name += self.oracle.wav_type + "_{}".format(p['sigma'])
        return name
    def name_construct_1d(self):
        name = str(self['mu']) + "_" + str(self['L'])
        name += "_" + str(self['dim']) + "_" + str(self['s']) + "_" + str(self['s'])
        name += "_" + str(self['sigma']) + "_" + str(self['noise']) + "_"
        name += str(abs(self.beta_0))[:5]
        return name
    def store_evolution(self, name, root="exps/"):
        evolution = extract_history(self.history, 'f(hat_xk)')[:self.budget]
        np.savetxt(root + name + "_1_" + str(self['seed'])  + ".txt", evolution[::200])

        evolution = extract_history(self.history, 'f(xk)')[:self.budget]
        np.savetxt(root + name + "_2_" + str(self['seed'])  + ".txt", evolution[::200])

            
class Averaged(Subscriptable,Assignable,MD_Structure):
    def __init__(self, oracle, prox, xz, beta=None, trace=1, verbose=0, oracle_reset=1):
        MD_Structure.__init__(self, oracle, prox, xz, beta, trace, verbose, oracle_reset)