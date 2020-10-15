from structures.structures import OptimizationMethod as OptimizationMethod
from structures.structures import Subscriptable as Subscriptable
from structures.structures import NamedClass as NamedClass
from structures.stepper import Stepper as Stepper
from numpy.random import choice, binomial
from numpy import array, ones, array_equal
from other.utils import log_progress, timer_decorator
from collections import defaultdict


class Incremental(NamedClass, Subscriptable, OptimizationMethod):
    def __init__(self, objective, oracle, step_function=None, prox_regime="I", gamma_func=None, 
                        prob_dist=None, beta=0, trace=1, verbose=0, name=''):
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
            if self.oracle.beta:
                return ((grad - anchor_grad) / (self.prob_dist[ind] * self.n)) + self.z_bar + self.oracle.beta * self.solution
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
        
        return grad, random_ind

    @timer_decorator
    def step(self):
        new_grad, random_ind = self.common_update()
        # update anchor gradient and z_bar
        self.post_step_update(new_grad, random_ind)
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
        

class SAGA(Incremental):
    @timer_decorator
    def get_anchor_grad(self, index=None):
        return self.anchor_grads[index,:]

    @timer_decorator
    def post_step_update(self, new_grad, random_ind):
        # random_ind = choice(self.n)
        # new_grad = self.oracle(self.solution, random_ind)
        self.z_bar = self.z_bar + ((new_grad - self.anchor_grads[random_ind,:]) / self.n)
        self.anchor_grads[random_ind,:] = new_grad

    @timer_decorator
    def initialize_anchor_grads(self):
        self.anchor_grads = array([list(self.oracle(self.solution, i)) for i in range(self.n)])

    @timer_decorator
    def make_pass(self, n_steps=10):
        self.print_intro()
        self.initialize_anchor_grads()
        self.z_bar = self.anchor_grads.mean(axis=0)
        it = log_progress(range(n_steps), name='Iterations') if self.verbose else range(n_steps)
        for i in it:
            self.step()    


class randomSVRG(Incremental):
    @timer_decorator
    def update_z_bar(self):
        self.z_bar = self.oracle(self.anchor_x)
        
    @timer_decorator
    def get_anchor_grad(self, index=None):
        return self.oracle(self.anchor_x, index)

    @timer_decorator
    def post_step_update(self, new_grad, random_ind):
        if binomial(1, 1.0 / self.n):
            self.anchor_x = self.solution
            self.update_z_bar()
            
    @timer_decorator
    def make_pass(self, n_steps=10):
        self.print_intro()
        # local initialization
        self.anchor_x = self.solution
        self.update_z_bar()

        it = log_progress(range(n_steps), name='Iterations') if self.verbose else range(n_steps)
        for _ in it:
            self.step()


# elif self.averaging == "I":
#     self.objective.prox_scale_update(self.step_function[self.iter])
#     self.solution = 
# elif self.averaging == "II":
#     eta_k = self.step_function[self.iter]
#     mu = self.objective['mu']
#     self.x_bar = (1 - mu * eta_k) * self.x_bar + mu * eta_k * self.solution - eta_k * modified_grad
#     self.objective.prox_scale_update( 1.0 / self.gamma_k[self.iter] )
#     self.solution = self.objective.prox(self.x_bar)