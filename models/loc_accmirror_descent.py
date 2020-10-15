from models.mirror_descent import *
from models.acc_mirror_descent import *
from models.averaged_mirror_descent import *
from other.statistics import *
from post_processing.evolution import *
from other.utils import *
import numpy as np
norm = np.linalg.norm


class LOC_AccMirrorDescent(MirrorDescent):
    def __init__(self, oracle, initial, sigma, omega, budget=None, verbose=False, do_setup=False):
        self.oracle, self.solution = oracle, initial.copy()
        self.p, self.n_stages = dict(), 0
        self.init_parametrize(sigma, omega)

        self.history, self.iter, self.niters = initial, 0, []
        self.verbose, self.budget = verbose, budget
        if do_setup: self.setup()

    def init_parametrize(self, sigma, omega):
        self.p = self.oracle.p
        self.p['sigma'], self.p['omega'] = sigma, omega
        self.p['done_switch'] = False

    def setup(self):
        self.show("Calculate the number of iterations to be passed")
        bias_part = sqrt(32) * sqrt(self['L'] / self['mu'])

        switch_point = np.log((.25 * self['L'] * self['mu'] * sq(self['omega'])) / sq(self['sigma'])) / np.log(4)
        switch_point = int_floor(max(switch_point, 0))
        if (switch_point + 1) * bias_part >= self.budget:
            self.show("There'll be no switch")
            self.stages_to_perform = int_floor(self.budget / bias_part)
            self.budget = int_ceil(self.stages_to_perform * bias_part)
        else:
            rest = self.budget - int_ceil(switch_point * bias_part)
            factor = sq(self['mu']*self['omega']/self['sigma'])
            noise_stages_r = (3/32) * factor * rest + (4 ** switch_point)
            noise_stages = int_floor(np.log(noise_stages_r) / np.log(4))
            self.stages_to_perform = switch_point + noise_stages
            noise_iterates = 32 * (1/factor) * (4 ** self.stages_to_perform - 4 ** switch_point) / 3
            self.budget = int_ceil(switch_point * bias_part) + int_ceil(noise_iterates)
        self.show("At the setup we cshoose: {} stages ({} iterates)".format(self.stages_to_perform, self.budget))

    def stage(self, i, num):
        n_iterations = self.calculate_iterations()
        if self.history.shape[1] + n_iterations > self.budget: return

        self.n_stages += 1
        self.show('Do localization iteration ' + mycol.bd + '{}/{}'.format(i, num) + mycol.e)
        self.show("Loc-MD iteration. Pass of gradient descent with {}{}{} iterations".format(
            mycol.bd, self.calculate_iterations(), mycol.e))

        sub_model = ACC_MirrorDescent(self.oracle, self.solution, self['sigma'], self['omega'],
                                      n_iterations)
        # sub_model = AveragedMirrorDescent(self.oracle, lambda x: init_md_gamma, self.solution, verbose=self.verbose)
        sub_model.make_pass(n_iterations)

        self.solution = sub_model[-1] #get_averaged_estimates(sub_model, start=n_iterations//2)[:, -1:]
        self.history = np.concatenate((self.history, sub_model.history[:,1:]), axis=1)
        self.p['omega'] /= 2

        self.niters.append(n_iterations)
        self.iter = self.history.shape[1]

    def make_pass(self, num=None):
        assert num is not None or self.budget is not None
        if num is not None:
            for i in range(num):
                self.stage(i, num)
        elif self.budget is not None:
            while self.iter < self.budget:
                self.show('Do localization iteration {}{}/{}{}'.format(mycol.bd, self.iter, self.budget, mycol.e))
                self.stage()

    def calculate_iterations(self):
        bias_part = sqrt(8 * 4 * self['L'] / self['mu'])
        noise_part = 1024 * sq(self['sigma'] / (self['mu'] * self['omega']))
        if noise_part > bias_part and not self.p['done_switch']:
            # if self.verbose:
            print('{}Switch is done at {}-th stage{} ({} iterations made)'.format(mycol.bd, len(self.niters),
                                                                                  mycol.e, self.iter))
            self.p['done_switch'] = True
        return int_ceil(max(bias_part, noise_part))

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.history[:, index:index + 1] if index >= 0 else self.history[:, index:]
        elif isinstance(index, str):
            if index in self.p: return self.p[index]
            elif index in self.__dict__: return self.__dict__[index]
            else: raise KeyError

    def show(self, text):
        if self.verbose: print(text)
