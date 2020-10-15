from structures.nesterov_sequence import *
from models.mirror_descent import *
from other.statistics import *
from other.utils import *

import algos.main as alg
from numpy import inf, concatenate, repeat
norm = np.linalg.norm


class CatalystBase(MirrorDescent):
    def __init__(self, oracle, initial, omega, solver, budget=None, verbose=False, do_ladder=True):
        self.oracle, self.solver, self.budget = oracle, solver, budget if budget is not None else inf

        self.stage = {'inner_iter': 0, 'n_stage': 0,
                      'kappa': oracle['L'] - oracle['mu'],
                      't_k': 2, 'e_k': -1,
                      'x_k': initial, 'y_k': initial.copy(),
                      'batch': 1}
        self.init_parametrize(omega)
        self.storage = [self.stage]
        self.history = initial.reshape((-1, 1))

        self.ns = NesterovSequence(np.sqrt(self['mu'] / (self['mu'] + self.stage['kappa'])))
        self.verbose = verbose

    def init_parametrize(self, omega):
        self.p = dict()
        self.p.update(self.oracle.p)
        self.p['omega'] = omega
        self.p['rho'] = 0.9 * np.sqrt(self.p['mu'] / (self.p['mu'] + self.stage['kappa']))
        self.p['sigma'] = self.oracle['sigma']
        self.stage['t_k'] = int_ceil((self.stage['kappa'] + self.p['L']) / (self.stage['kappa'] + self.p['mu']))
        self.stage['e_k'] = self.calculate_precision()

    def calculate_precision(self):
        return (1. / 9) * self.p['L'] * self.p['omega'] ** 2 * alg.bin_power(1 - self.p['rho'],
                                                                             self.stage['n_stage'])

    def calculate_batch(self):
        return min(20, int_ceil(
            self['sigma'] ** 2 / ((self['mu'] + self.stage['kappa']) * self.stage['e_k'] * self.stage['t_k'])))

    def form_new_stage(self):
        new_stage = self.stage.copy()
        new_stage['n_stage'] += 1
        new_stage['e_k'] *= (1 - self.p['rho'])
        self.stage = new_stage

    def do_stage(self):
        self.form_new_stage()
        self.oracle.change_smoothing(regularization=self.stage['kappa'], prox_center=self.stage['y_k'])
        model = self.solver(self.oracle, self.stage['y_k'], self['omega'], verbose=self.verbose)
        model.make_pass(self.stage['t_k'])

        self.stage['x_k'] = model[-1].reshape((-1, 1))
        self.ns.gen_new()
        self.stage['y_k'] = self.stage['x_k'] + self.ns.get_beta() * (self.stage['x_k'] - self.storage[-1]['x_k'])
        self.stage['inner_iter'] = model.iter
        self.storage.append(self.stage)

        self.history = concatenate((self.history, repeat(model.history[:,1:], self.stage['batch'], axis=1)), axis=1)
        self.iter = max(self.history.shape)

    def print_current_state(self):
        print('Batch after switch is:', self.stage['batch'])
        print("Current history lasts:", self.history.shape[1])
        print("Precision:", self.stage['e_k'])
        print("Iterations:", self.stage['t_k'])

    def make_pass(self, n_passes=inf):
        assert n_passes != inf or self.budget != inf
        stage_index = 0
        while (self.history.shape[1] < self.budget) and (stage_index < n_passes):
            stage_index += 1
            if self.verbose:
                print('Do Catalyst iteration: {}/{}'.format(stage_index, n_passes))
                print('Current history: {}/{}'.format(self.history.shape[1], self.budget))
            try:
                self.do_stage()
            except:
                self.oracle.remove_smoothing(); raise
        self.oracle.remove_smoothing()
        self.iter, self.history = self.budget, self.history[:, :self.budget]