from structures.structures import Subscriptable as Subscriptable
from scipy.stats import t

import matplotlib.image as mpimg
import other.utils as utils
import numpy as np
import PIL, pywt
   
class Oracle(Subscriptable):
    def __init__(self, objective, sigma=0, type_="normal"):
        self.objective = objective
        self.update_parameters()
        self.sigma, self.batch_size = sigma, 1
        self.calls = 0
        self.type = type_

    def update_parameters(self):
        self.p = dict()
        self.p.update(self.objective.p)
    def generate_regressor(self): pass
    def generate_noise(self): pass

    def grad(self, arg, return_regressors=0):
        if self.batch_size > 1:
            output = np.zeros_like(arg)
            if return_regressors: output_regr = np.zeros_like(arg)

            for _ in range(self.batch_size):
                self.calls = self.calls + 1
                phi = self.generate_regressor()
                xi  = self.generate_noise()
                output = output + self.objective.grad(arg, phi) + xi * phi
                if return_regressors: output_regr = phi

            if return_regressors:
                return output / self.batch_size, output_regr # output_regr / self.batch_size
            else:
                return output / self.batch_size
        else:
            phi = self.generate_regressor()
            xi  = self.generate_noise()
            self.calls = self.calls + 1
            if return_regressors:
                output = self.objective.grad(arg, phi) + xi * phi, phi
                return output
            else:
                output = self.objective.grad(arg, phi) + xi * phi
                return output

    def __call__(self, arg, return_regressors=0):
        return self.grad(arg, return_regressors)

    def set_beta(self, beta):
        self.beta = beta

    def set_batch(self, batch_size):
        self.batch_size = batch_size
        if hasattr(self, "size_storage") and self.size_storage < batch_size:
            self.size_storage = batch_size + 1

    def reset(self):
        self.calls = 0
        if 'dataset' in self.objective:
            self.objective.dataset.reset_accesses()
        if hasattr(self, "flush_index"):
            self.flush_index = 0
            self.storage = np.zeros((0,0))

class Oracle_Image(Oracle):
    def __init__(self, objective, sigma=0, type_="normal", wav_type='haar'):
        Oracle.__init__(self, objective, sigma, type_)
        # try different wavelet
        self.wav_type = wav_type

        self.data = self.objective.data
        self.shape = self.data.shape
        self.size  = self.data.size

    def generate_regressor(self):
        gen = np.random.randint
        if len(self.shape) == 2: # then we deal with an image
            row, col = gen(self.shape[0]), gen(self.shape[1])
            sub_image           = np.zeros_like(self.data); 
            sub_image[row, col] = 1.0
            pyr = pywt.wavedec2(sub_image, self.wav_type)
        else: # then we deal with a sound
            ind = gen(self.size)
            sub_sound      = np.zeros_like(self.data); 
            sub_sound[ind] = 1.0 
            pyr = pywt.wavedec(sub_sound, self.wav_type)

        return utils.convert_pyr_2_vec(pyr)[0]
    
    def generate_noise(self):
        if self.type == "normal":
            return self.sigma * np.random.randn(1)
        elif self.type == "t":
            return self.sigma * t.rvs(4, size=1) / np.sqrt(2)
        elif self.type == "hadamard":
            return self.sigma * np.random.randn(1)

class Oracle_LS_Gaussian(Oracle):
    def __init__(self, objective, sigma=0, type_="normal"):
        Oracle.__init__(self, objective, sigma, type_)
        self.is_diag = self.objective.is_diag
        self.size_storage = 300
        # becomes a vector
        if self.is_diag:
            self.regressor_cov = np.tile(np.sqrt(self.objective.A),(self.size_storage,)) 
        else:
            self.regressor_mean = np.zeros(self['dim'])
            self.regressor_cov = self.objective.A

        self.fill_storage()
        self.flush_index = 0

    def fill_storage(self):
        self.flush_index, K = self.calls, self.size_storage
        if self.is_diag:
            if self.type == "normal":
                self.storage = self.regressor_cov * np.random.randn(self['dim'] * K)
            elif self.type == "t":
                self.storage = self.regressor_cov * t.rvs(4, size=self['dim'] * K) / np.sqrt(2)
            elif self.type == "hadamard":
                self.storage = np.zeros((0,0)); return
            self.storage = self.storage.reshape(K, self['dim'])
        else:
            self.storage = np.random.multivariate_normal(self.regressor_mean, 
                                    self.regressor_cov, size=K)

    def generate_regressor(self):
        if self.type == "hadamard":
            return np.sqrt(self.objective.A) * get_hadamard_vec(self['dim'])
        if (self.calls - self.flush_index) >= len(self.storage):
            self.fill_storage()
        return self.storage[self.calls - self.flush_index]

    # def grad(self, arg, return_regressors=0):
    #     mb = self.batch_size
    #     if mb > 1:
    #         if (mb + self.calls - self.flush_index) >= self.storage.shape[0]:
    #             self.fill_storage()
    #         index = self.calls - self.flush_index

    #         self.calls += mb
    #         phis = self.storage[index:index+mb,:]
    #         xis  = self.generate_noise(size=mb)

    #         output = (phis.T * (xis+phis.dot(arg-self.objective.x_optimum))).mean(axis=1)
    #         return (output, np.mean(phis, axis=0)) if return_regressors else output
    #     else:
    #         phi = self.generate_regressor()
    #         xi  = self.generate_noise()
    #         self.calls = self.calls + 1
    #         output = self.objective.grad(arg, phi) + xi * phi
    #         return (output, phi) if return_regressors else output

    def generate_noise(self, size=1):
        if self.type == "normal":
            return self.sigma * np.random.randn(size)
        elif self.type == "t":
            return self.sigma * t.rvs(4, size=size) / np.sqrt(2)
        elif self.type == "hadamard":
            return self.sigma * np.random.randn(size)


class Quadratic(Oracle):
    def __init__(self, objective, sigma=0):
        Oracle.__init__(self, objective, sigma=sigma)

class Universal(Oracle):
    def __init__(self, objective, sigma=0):
        Oracle.__init__(self, objective, sigma=sigma)

# TODO delete excessive

class LeastSquares(Oracle):
    def __init__(self, objective, sigma=0):
        Oracle.__init__(self, objective, sigma=sigma)


def get_hadamard_vec(n):
    q = int(np.log(n)/np.log(2))
    bits = 2*np.random.binomial(1, 0.5, q) - 1
    regr, pos = np.ones(n), 1
    for i, b in enumerate(bits):
        regr[pos:pos+2**i] = b * regr[:2**i]
        pos *= 2
    return regr

def multivariate_student_t(X, mu, Sigma, df):    
    #multivariate student T distribution

    [n,d] = X.shape
    Xm = X-mu
    V = df * Sigma
    V_inv = np.linalg.inv(V)
    (sign, logdet) = slogdet(np.pi * V)

    logz = -gamma(df/2.0 + d/2.0) + gamma(df/2.0) + 0.5*logdet
    logp = -0.5*(df+d)*np.log(1+ np.sum(np.dot(Xm,V_inv)*Xm,axis=1))

    logp = logp - logz            

    return logp


# def pyr_2_vec_haar(pyr):
#     temp = np.cumsum([np.sum(j.size for j in i) for i in pyr])
#     output = np.zeros(temp[-1])
#     counter = 0
#     for level in pyr:
#         if counter==0:
#             output[counter] = level[0][0]; counter += 1; continue
#         for ent in level:
#             for row in range(ent.shape[0]):
#                 for col in range(ent.shape[1]):
#                     output[counter] = ent[row][col]; counter += 1
#     return output


# def vec_2_pyr_haar(vec):
#     # n_levels = int(np.log(vec.size) / np.log(2))
#     output = []
#     n_level = 0; cur_pos = 0
#     while cur_pos < vec.size:
#         if cur_pos == 0:
#             output.append(np.array([[vec[0]]])); cur_pos+=1; continue
#         to_append = []
#         for _ in range(3):
#             cur_ent = np.array(vec[cur_pos:cur_pos + 4**n_level]).reshape((2**n_level, 2**n_level))
#             cur_pos += 4**n_level
#             to_append.append(cur_ent)
#         output.append(tuple(to_append))
#         n_level += 1
#     return output