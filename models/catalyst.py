from structures.structures import Subscriptable as Subscriptable
from structures.structures import Assignable as Assignable
from structures.nesterov_sequence import *
from models.mirror_descent import *
from other.statistics import *
from other.utils import *

from numpy import inf, concatenate, repeat


class Standalone(Subscriptable):
    def __init__(self, objective, nesterov_seq, kpa, oracle, trace=1, verbose=0):
        self.objective = objective 
        self.nesterov_seq = nesterov_seq
        self.oracle = oracle
        self.p = dict() # easyli accessed meta_parameters
        self.transfer_params(objective)
        self.p['kpa'] = kpa
        self.trace, self.verbose = trace, verbose

        
    def initialize(self, xz, budget=None):
        # xz = x_0 : initial estimate
        self.current_estimate = self.current_extrapolator = xz
        self.current_index = 0
        self.budget = budget if budget is not None else inf
        self.history = []
        self.oracle.reset()
        self.last_oracle_ncalls = -1
        if self.trace:
            self.update_history(xz, xz)

    def step(self):
        # ykm = y_{k-1} : extrapolator
        # alk = alpha_{k} : extrapolator
        # alkm = alpha_{k-1} : extrapolator
        self.current_index += 1
        ykm = self.current_extrapolator
        k = self.current_index
        xk = self.objective.prox(ykm - ( self.oracle(ykm) / (self['kpa'] + self['mu']) ))
        
        # metaparameters update
        alk  = self.nesterov_seq[self.current_index]
        alkm = self.nesterov_seq[self.current_index-1]
        etak = (self['kpa'] + self['mu']) * (1-alk) * (alkm**2) / (self['kpa'] * alk)
        
        # in the standalone approach the estimates xk and 
        # the optimum of the surrogate function are the same
        xkm = self.history[-1]['xk']
        assert 0 < alkm < 1
        vk = xk + ( 1. / alkm - 1 ) * (xk - xkm)
        yk = vk*etak + (1-etak)*xk
        
        self.current_estimate = xk 
        self.current_extrapolator = yk 
        if self.trace:
            self.update_history(xk, yk)
        

    def update_history(self, xk, yk):
        # TODO
        # for i in range(self.oracle.calls - self.last_oracle_ncalls):
        pack = {'copy': self.oracle.calls - self.last_oracle_ncalls}
        pack['xk'] = xk
        # pack['xk'], pack['yk'] = xk, yk
        # if hasattr(self.objective, 'optimum'): 
        #     pack['exact_ek'] = pack['ek'] = self.objective(xk) - self.objective.optimum
        # else:
        #     pack['value'] = self.objective(xk) 
        self.history.append(pack)
        self.last_oracle_ncalls = self.oracle.calls

    def make_pass(self, n_passes=inf):
        assert n_passes != inf or self.budget != inf
        stage_index = 0
        while (len(self.history) < self.budget) and (stage_index < n_passes):
            stage_index += 1
            if self.verbose:
                print('Do Catalyst iteration: {}/{}'.format(stage_index, n_passes))
                print('Current history: {}/{}'.format(len(self.history), self.budget))
            self.step()
        self.iter, self.history = self.budget, self.history[:self.budget]


    def print_current_state(self, dim_threshold=1000):
        if max(self.current_estimate.shape) < dim_threshold:
            print('Current solution:', self.current_estimate)
        else:
            print('Solution is too large, try another threshold')

        print("{} iterations passed", len(self.history))
        print("Precision:", self.history[-1]['exact_ek'])
    

    def transfer_params(self, dictionary):
        for key in dictionary.p:
            self.p[key] = dictionary[key]
     
    
# class Old_Catalyst(MirrorDescent):
#     def __init__(self, oracle, initial, sigma, omega, solver, budget, verbose=False, do_ladder=True):
#         self.oracle, self.solver, self.budget = oracle, solver, budget if budget is not None else inf
#         self.verbose = verbose

#         self.stage = {'inner_iter': 0, 'n_stage': 0, 'ek': -1,
#                       'xk': initial, 'yk': initial.copy(), 'batch': 1}
#         self.init_parametrize(omega, sigma)
#         self.storage, self.history = [self.stage], initial.reshape((-1, 1))

#         self.ns = NesterovSequence(np.sqrt(self['mu'] / (self['mu'] + self['kappa'])))

#     def init_parametrize(self, omega, sigma):
#         self.p = dict()
#         self.p.update(self.oracle.p)
#         self.p['kappa'] = self['L'] - self['mu']
#         self.p['omega'], self.p['sigma'] = omega, sigma
#         self.p['tk'] = 2*((self['L'] + self['kappa']) / (self['kappa'] + self['mu'])) * np.log(self['L'] / self['mu'])
#         self.p['tk'] = max(int_ceil(self.p['tk']), 1)
#         self.show("Number of inner iterations: {}".format(self.p['tk']))
#         self.p['q'] = self.p['mu'] / (self.p['mu'] + self.p['kappa'])
#         self.p['rho'] = 0.9 * np.sqrt(self.p['q'])

#         # remember about logarithm
#         self['tk'] = int_ceil((self.p['kappa'] + self.p['L']) / (self.p['kappa'] + self.p['mu']))
#         self.stage['ek'] = self.calculate_precision()

#     def calculate_precision(self):
#         # we replace the precise calculation of the initial gap f(xk) - f* with its upper bound
#         return (1. / 9) * self.p['L'] * self.p['omega'] ** 2 * alg.bin_power(1 - self.p['rho'],
#                                                                              self.stage['n_stage'])

#     def calculate_batch(self):
#         # return min(20, int_ceil(self['sigma'] ** 2 / ((self['mu'] + self['kappa']) * self.stage['ek'] * self['tk'])))
#         return int_ceil(self['sigma'] ** 2 / ((self['mu'] + self['kappa']) * self.stage['ek'] * self['tk']))

#     def form_new_stage(self):
#         self.stage['n_stage'] += 1
#         self.stage['ek'] *= (1 - self.p['rho'])
#         self.stage['batch'] = self.calculate_batch()
#         if self.stage['batch'] > 1:
#             self.show("I'm setting batch size to be {}".format(self.stage['batch']))
#             self.oracle.set_batch(self.stage['batch'])

#     def warm_start(self):
#         if len(self.storage) == 1:
#             return self.storage[-1]['xk']
#         p = self.storage[-1]
#         return p['xk'] + (1 - self['q']) * (p['yk']-self.storage[-2]['yk'])

#     def do_stage(self):
#         if self.history.shape[1] > self.budget: return

#         self.form_new_stage()
#         self.oracle.change_smoothing(regularization=self['kappa'], prox_center=self.stage['yk'])

#         # def __init__(self, oracle, initial, sigma, omega, budget=None, verbose=False):
#         # def __init__(self, oracle, initial, sigma, omega, budget=None, verbose=False)
#         model = self.solver(self.oracle, self.warm_start(), self['sigma'], self['omega'], do_setup=False,
#                             budget=inf, verbose=self.verbose)
#         model.make_pass(num=1) # I use the fact that 1 stage will definetely cover self['tk'] iterations

#         self.stage['xk'] = model[-1].reshape((-1, 1))
#         self.ns.gen_new()
#         self.stage['yk'] = self.stage['xk'] + self.ns.get_beta()*(self.stage['xk'] - self.storage[-1]['xk'])
#         self.stage['inner_iter'] = model.iter
#         self.storage.append(self.stage)

#         self.history = np.concatenate((self.history, np.repeat(model.history[:,1:], self.stage['batch'], axis=1)), axis=1)
#         self.iter = max(self.history.shape)

#     def print_current_state(self):
#         print('Batch after switch is:', self.stage['batch'])
#         print("Current history lasts:", self.history.shape[1])
#         print("Precision:", self.stage['ek'])
#         print("Iterations:", self['tk'])

#     def make_pass(self, n_passes=inf):
#         assert n_passes != inf or self.budget != inf
#         stage_index = 0
#         while (self.history.shape[1] < self.budget) and (stage_index < n_passes):
#             stage_index += 1
#             if self.verbose:
#                 print('Do Catalyst iteration: {}/{}'.format(stage_index, n_passes))
#                 print('Current history: {}/{}'.format(self.history.shape[1], self.budget))
#             try:
#                 self.do_stage()
#             except:
#                 self.oracle.remove_smoothing(); raise
#         self.oracle.remove_smoothing()
#         self.iter, self.history = self.budget, self.history[:, :self.budget]

#     def __setitem__(self, key, value):
#         if key in self.p:
#             self.p[key] = value
#         elif key in self.__dict__:
#             self.__dict__[key] = value
#         else:
#             raise KeyError
