
class Oracle(Subscriptable):
    def __init__(self, objective, sigma=0, beta=0):
        self.objective = objective
        self.p = dict(); self.update_parameters()
        self.sigma, self.beta, self.batch_size = sigma, 0, 1
        self.calls = 0

    def update_parameters(self):
        self.p.update(self.objective.get_params())

    def betanizing(self, grad, arg):
        if self.beta == 0: return grad
        return grad - self.beta * arg

    def grad(self, arg, index=None, separate=0):
        self.calls = self.calls + (self.objective.n if (index is None and hasattr(self.objective, "n")) else 1)
        dim = max(arg.shape)
        if self.sigma != 0:
            return self.objective.grad(arg, index) + self.noise(dim) if not separate else (self.objective.grad(arg, index), self.noise(dim))
        else:
            return self.objective.grad(arg, index)

    def __call__(self, arg, index=None):
        if self.batch_size > 1:
            return np.concatenate([self.grad(arg, index) for _ in range(self.batch_size)], axis=1).mean(axis=1).reshape((1,))
        else:
            return self.grad(arg, index)

    def set_beta(self, beta):
        self.beta = beta

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def noise(self, dim):
        return self.sigma * np.random.randn(dim, )

    def reset(self):
        self.calls = 0
        if 'dataset' in self.objective:
            self.objective.dataset.reset_accesses()


# to check gradient
# eps = 0.0001
# check_grad  = []
# for i in range(len(dataset[0][0])):
#     direction = np.zeros_like(dataset[0][0])
#     direction[i] = 1
#     term = (objective.value_term(dataset[0][0] + eps*direction, 0) - objective.value_term(dataset[0][0] + 0, 0)) / eps
#     check_grad.append(term)
# np.array(check_grad[:10])

# class Old_ObjectiveLS:
#     def __init__(self, input_params, regularization=0, prox_center=None):
#         self.p = input_params.copy(); self.init_parametrize(input_params)
#         self.A = self.symmetrize(self.generate_matrix())
#         self.b = np.random.rand(input_params['dim'], 1)

#         self.change_smoothing(regularization)
#         self.change_prox_center(prox_center)

#     def init_parametrize(self, input_params):
#         self.p['mu_0'] = input_params['mu']
#         self.p['L_0'] = input_params['L']

#     def remove_smoothing(self):
#         self.do_smooth = False
#         self.regularization = 0
#         self.prox_center = None
#         self.p['mu'] = self.p['mu_0']
#         self.p['L'] = self.p['L_0']

#     def change_smoothing(self, regularization=0):
#         if regularization == 0:
#             self.change_prox_center(None)
#         self.do_smooth = regularization > 0
#         self.regularization = regularization
#         self.p['mu'] = self.p['mu_0'] + regularization
#         self.p['L'] = self.p['L_0'] + regularization

#     def change_prox_center(self, prox_center):
#         self.prox_center = prox_center

#     def symmetrize(self, matrix):
#         assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
#         return 0.5 * (matrix + matrix.T)

#     def gradient(self, arg):
#         assert len(arg.shape) > 1 and arg.shape[1] == 1
#         if self.do_smooth:
#             return self.A.dot(arg) + self.b + self.regularization * (arg - self.prox_center)
#         else:
#             return self.A.dot(arg) + self.b

#     def generate_matrix(self):
#         U = mx.random_orthonormal_matrix(dim=self['dim'])
#         eigenvalues = [self['mu']] + list(np.random.uniform(self['mu'], self['L'], size=self['dim'] - 2)) + [self['L']]
#         return U.dot(np.diag(eigenvalues)).dot(U.T)

#     def value(self, arg):
#         temp = 0.5 * np.dot(arg.T, self.A.dot(arg)) + self.b.T.dot(arg)
#         output = temp + 0.5 * self.regularization * norm(arg - self.prox_center) ** 2 if self.do_smooth else temp

#         try: return output[0, 0]
#         except: return output

#     def __call__(self, arg):
#         return self.value(arg)

#     def eval_2d(self, x, y):
#         assert x.shape == y.shape
#         output = np.zeros_like(x)
#         for (i, j) in np.ndindex(x.shape):
#             output[i, j] = self(np.array([x[i, j], y[i, j]]))
#         return output

#     def find_optimum(self):
#         return np.linalg.solve(self.A, -self.b)

#     def closeness_of_solution(self, solution):
#         return norm(solution - self.optimum()) / norm(solution)

#     def get_params(self):
#         return self.p

#     def __getitem__(self, item):
#         if item in self.p:
#             return self.p[item]
#         elif item in self.__dict__:
#             return self.__dict__[item]
#         else:
#             raise KeyError
