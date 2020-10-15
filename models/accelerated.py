from structures.structures import OptimizationMethod as OptimizationMethod
from structures.structures import Subscriptable as Subscriptable
from structures.structures import NamedClass as NamedClass
from structures.stepper import Stepper as Stepper
from numpy.random import choice, binomial
from numpy import array, ones, array_equal
from other.utils import log_progress, timer_decorator
from collections import defaultdict


class Accelerated(NamedClass, Subscriptable, OptimizationMethod):
    def __init__(self, objective, oracle, step_function=None, prox_regime="I", gamma_func=None, 
                        prob_dist=None, trace=1, verbose=0, name=''):
        self.objective, self.oracle = objective, oracle
        self.oracle.set_beta(beta)

        self.step_function = Stepper(step_function)
        self.n = self.objective.n

        self.is_sampling_uniform = prob_dist is None # TO FIX
        self.prob_dist = ones(objective.n) / objective.n if prob_dist is None else prob_dist


        gamma_func = (lambda x: objective['mu']) if gamma_func is None else gamma_func
        self.gamma_k = Stepper(gamma_func)
        self.prox_regime = prox_regime
        self.trace, self.verbose, self.p = trace, verbose, dict()
        self.create_name(name)


    def initialize(self, xz):
        self.solution, self.iter, self.history = xz.copy(), 0, []
        
        self.oracle.reset()
        self.last_oracle_ncalls = -1
        self.update_history()
        self.x_bar = xz.copy() # for the second prox

    @timer_decorator
    def modify_grad(self, grad, ind):
        anchor_grad = self.get_anchor_grad(ind)
        if self.is_sampling_uniform:
            return grad - anchor_grad + self.z_bar
        else:
            return ((grad - anchor_grad) / (self.prob_dist[ind] * self.n)) + self.z_bar

    @timer_decorator
    def common_update(self):
        self.iter += 1

        random_ind = choice(self.n, p=self.prob_dist)
        grad = self.oracle(self.solution, random_ind)
        modified_grad = self.modify_grad(grad, random_ind) # g_k
        if self.prox_regime == 'I':
            self.objective.prox_scale_update(self.step_function[self.iter])
            self.solution = self.objective.prox(self.solution - self.step_function[self.iter] * modified_grad)
        elif self.prox_regime == 'II':
            eta_k = self.step_function[self.iter]
            weight = self.objective['mu'] * eta_k

            self.x_bar = (1 - weight) * self.x_bar + weight * self.solution - eta_k * modified_grad
            self.objective.prox_scale_update( 1.0 / self.gamma_k[self.iter] )
            self.solution = self.objective.prox(self.x_bar)
    
    @timer_decorator
    def step(self):
        self.common_update()
        # update anchor gradient and z_bar
        self.post_step_update()
        self.update_history()
        # if self.prox_regime == "II": self.side_processing()


    @timer_decorator
    def update_history(self):
        pack = defaultdict(int)
        pack['copy'] = self.oracle.calls - self.last_oracle_ncalls
        self.last_oracle_ncalls = self.oracle.calls
        pack['xk'] = self.solution
        self.history.append(pack)

    def create_name(self, text):
        self.name  = "{} with proximal operator of the type: {}\n".format(text, self.prox_regime)
        self.name += "sampling is " + ("NOT " if not array_equal(self.prob_dist, ones(self.n) / self.n) else '') + "uniform\n"
        self.name += "proximal operator is {}".format("erased" if self.objective.prox_operator is None else self.objective.prox_operator)
        
