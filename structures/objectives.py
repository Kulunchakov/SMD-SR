from structures.structures import Subscriptable as Subscriptable
from structures.structures import NamedClass as NamedClass
from numpy.random import choice
from numpy import dot
import scipy, os
import matrix.main as mx
import numpy as np
norm = np.linalg.norm
log = np.log

class Proximal_Euclidean(NamedClass):
    def __init__(self):
        pass

    def __call__(self, zeta, x, x_0, beta):
        return x - zeta / beta
        # return arg


class Proximal_LOne(NamedClass):
    def __init__(self, scale, name=''):
        # proximal operator defined as argmin( scale * f(x) + 0.5 * ||x - y|| )
        self.scale = scale
        self.name = name

    def update(self, new_scale):
        self.scale = new_scale

    def __call__(self, arg):
        return (arg - np.sign(arg) * self.scale) * (np.abs(arg) >= self.scale)

class Proximal_Sparse(NamedClass):
    def __init__(self, params):
        # proximal operator defined for||x - y||_p^2 penalization
        self.n = params["dim"]
        self.p = 1.0 + (1.0 / log(self.n))
        self.q = log(self.n) + 1.0
        self.vartheta_const = np.e * log(self.n) * \
                        np.power(self.n, (self.p - 1)*(2 - self.p)/self.p)

    def grad(self, arg, p):
        temp =  np.power(np.abs(arg), p - 1)
        lp_norm = np.linalg.norm(arg, ord=p)
        if lp_norm > 0.1**35:
            return temp * np.sign(arg) * ( lp_norm ** (2 - p) )
        else:
            return temp * 0
    def __call__(self, zeta, x, x_0, beta):
        Q = self.grad(x - x_0, self.p) - (zeta / (self.vartheta_const * beta))
        return x_0 + self.grad(Q, self.q)

class Function(NamedClass):
    def __init__(self, input_params, name=''):
        self.p = input_params.copy()
        self.name = name

    def __call__(self, arg, index=None):
        return self.value(arg) if index is None else self.value_term(arg, index)

class QuadraticImage(Function, Subscriptable):
    def __init__(self, input_params, data, optimum, name=''):
        Function.__init__(self, input_params, name)
        self.data = data
        # self.A = generate_diag_matrix(input_params) # becomes a vector

        self.x_optimum = optimum
        self.optimum = self(self.x_optimum)

    def grad(self, arg, phi):
        return phi * np.dot(phi, arg - self.x_optimum)
    
    def value(self, arg):
        return np.linalg.norm(arg - self.x_optimum) ** 2

class QuadraticSimple(Function, Subscriptable):
    def __init__(self, input_params, optimum, name='', is_diag=1):
        Function.__init__(self, input_params, name)
        self.is_diag = is_diag
        if is_diag:
            self.A = generate_diag_matrix(input_params) # becomes a vector
        else:
            mat_name = f"matrices\\{self['dim']}.txt"
            if os.path.exists(mat_name):
                self.A = symmetrize(np.loadtxt(mat_name))
            else:
                self.A = symmetrize(generate_matrix(self))
        self.x_optimum = optimum
        self.optimum = self(self.x_optimum)

    def grad(self, arg, phi):
        return phi * np.dot(phi, arg-self.x_optimum)
    
    def full_gradient(self, arg):
        if self.is_diag:
            return self.A * (arg-self.x_optimum)
        else:
            return self.A.dot(arg-self.x_optimum) 

    def value(self, arg):
        if self.is_diag:
            output = 0.5 * np.dot(arg-self.x_optimum, self.A * (arg-self.x_optimum))
        else:
            output = 0.5 * np.dot(arg-self.x_optimum, self.A.dot(arg-self.x_optimum))
        try: return output[0, 0]
        except: return output


def symmetrize(matrix):
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    return 0.5 * (matrix + matrix.T)
def generate_diag_matrix(self):
    # eigenvalues = [self['mu']] + list(np.random.uniform(self['mu'], self['L'], \
    #                 size=self['dim'] - 2)) + [self['L']]
    # return np.array(eigenvalues)
    return np.linspace(self['mu'], self['L'], self['dim'])
    # base = np.log(1.00045)
    # return self['mu'] + np.logspace(np.log(self['mu'])/base, np.log(self['L'])/base, num=self['dim'], base=1.002)
    # return self['L'] * np.ones(self['dim'])
def generate_matrix(self):
    # U = mx.random_orthonormal_matrix(dim=self['dim'])
    mat = np.random.randn(d,d)
    U, _, _ = np.linalg.svd(mat)

    eigenvalues = [self['mu']] + list(np.random.uniform(self['mu'], self['L'], size=self['dim'] - 2)) + [self['L']]
    return U.dot(np.diag(eigenvalues)).dot(U.T)
def sparse_vector(p, shuffle=0):
    import math
    s, d = p['s'], p['dim']
    i = max(math.floor(d / s) - 1, 0)
    values = np.random.normal(size=s)
    result = [values[0]]
    for _ in range(s-2): result.extend([0] * i + [values[_+1]])
    if s > 1: result.extend([0] * (d - 1 - len(result)) + [values[-1]])
    else: result.extend([0] * (d - len(result)))
    return np.array(result)
