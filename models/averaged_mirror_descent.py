from structures.stepper import Stepper
from other.utils import *
import numpy as np


class AveragedMirrorDescent:
    def __init__(self, oracle, step_function, init_vec, verbose = False):
        self.oracle, self.stepper = oracle, Stepper(step_function)
        self.solution = init_vec.copy()
        self.history = self.solution
        self.verbose = verbose
        self.iter = 0
        self.p = dict()

    def iterate(self):
        self.iter += 1
        self.solution = self.solution - self.stepper[self.iter] * self.oracle(self.solution)

    def make_pass(self, n_iterations):
        #for i in log_progress(range(n_iterations)):
        for i in range(n_iterations):
            self.iterate()
            self.history = np.hstack((self.history, self.solution))

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.history[:, index:index + 1] if index >= 0 else self.history[:, index:]
        elif isinstance(index, str):
            if index in self.p:
                return self.p[index]
            elif index in self.__dict__:
                return self.__dict__[index]
            else:
                raise KeyError

