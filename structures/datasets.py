from numpy import random, insert, diag, sqrt, squeeze, array, asarray
from structures.structures import NamedClass as NamedClass
from sklearn import preprocessing, datasets
from pandas import read_csv, DataFrame
import matrix.main as mx
from utils import *


class Basic(NamedClass):
    def shape(self):
        return self.data.shape
    def size(self):
        return self.data.size
    def len(self):
        return len(self)
    def reset_accesses(self):
        self.accesses = 0
    def dropout(self, vector):
        return array([(0 if random.binomial(1, self.dropout_delta) else i/(1-self.dropout_delta)) for i in vector])
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        self.accesses += 1 
        X = self.data[index,:]
        X = squeeze(asarray(X.todense())) if self.is_sparse else X
        X = X if self.dropout_delta == 0 else self.dropout(X)
        y = self.responses[index] if self.label_encoding is None else self.label_encoding[self.responses[index]]
        return X, y

    def prep(self, x, arg):
        return insert(x, 0, arg)


class From_LIBSVM(Basic):
    def __init__(self, p, filename, normalize=True, free_term=1, name=''):
        self.data, self.responses = datasets.load_svmlight_file(filename)
        self.shape, self.name = self.data.shape, name
        if normalize:
            self.data = preprocessing.normalize(self.data)
        self.dropout_delta, self.is_sparse, self.accesses = p['dropout'], 1, 0
        self.label_encoding = None


class From_CSV(Basic):
    def __init__(self, p, filename, label_ind=1, feature_inds=None, label_encoding=None, 
                normalize=True, free_term=1, name=''):
        self.data = read_csv(filename)
        self.feature_inds = my_range(label_ind+1, self.data.shape[1]) if feature_inds is None else feature_inds
        self.label_encoding, self.label_ind = label_encoding, label_ind
        if free_term: self.data['free_term'] = 1
        
        if normalize:
            self.normalize_data()
        
        self.responses, self.data = self.data.as_matrix()[:,label_ind], self.data.as_matrix()[:, self.feature_inds]
        self.dropout_delta, self.is_sparse, self.accesses = p['dropout'], 0, 0
        self.name, self.shape = name, self.data.shape

    def normalize_data(self):
        print("> normalize data")
        names = list(self.data)
        clmn_to_normalize = [names[i] for i in self.feature_inds] 
        x = self.data[clmn_to_normalize].values # returns a numpy array
        x_scaled = preprocessing.normalize(x.T).T
        df_temp = DataFrame(x_scaled)
        self.data[clmn_to_normalize] = df_temp


class Synthetic(Basic):
    def __init__(self, optimum, filename, normalize=False, name=''):
        Basic.__init__(self, filename, 0, None, None, normalize, 0, name)
        self.optimum = optimum


def generate_synthetic_data(filename, p, binary_label=False):
    # Generate a dataset [y, Q], where 
    # y is a response variable, based on a hidden linear dependence
    # Q is a (p['n'] x p['dim']) matrix with min/max eigenvalues as p['mu'], p['L'] respectively
    # 
    # Other parameters:
    # p: dictionary of meta-parameters for data generation
    # p['n']: number of samples to generate
    def generate_matrix(dim, mu, L):
        U = mx.random_orthonormal_matrix(dim)
        eigenvalues = [mu] + list(random.uniform(mu, L, size=dim - 2)) + [L]
        return U.dot(diag(eigenvalues)).dot(U.T)
    def noise(dim, sigma):
        return sigma * random.randn(dim, )

    file = open(filename, 'w+')
    # construct a random semi_orthonormal_matrix Q, such that the min/max eigenvalues of Q.T.dot(Q) are mu / L
    Q = mx.random_semi_orthonormal_matrix((p['n'], p['dim']))
    M = generate_matrix(p['dim'], sqrt(p['mu']), sqrt(p['L']))
    Q = Q.dot(M)

    beta = 10 * (random.rand(p['dim'], ) - 0.5)
    response = Q.dot(beta) + noise(p['n'], p['sigma'])
    if binary_label:
        response = binarize_responses(response)
    prec = 10
    # file.write("{},{}\n".format("label", ','.join([str(1+a) for a in range(p['dim'])])))
    # for i in range(Q.shape[0]):
    #     file.write("{},{}\n".format(str(response[i])[:prec], ','.join([str(a)[:prec] for a in Q[i,:]])))
    file.write("{},{}\n".format("label", ','.join([str(1+a) for a in range(p['dim'])])))
    for i in range(Q.shape[0]):
        file.write("{},{}\n".format(str(response[i]), ','.join([str(a) for a in Q[i,:]])))

    file.close()
    return Q, beta, response


def binarize_responses(responses, threshold=0):
    return array([(i>threshold) for i in responses])



# TEST GRADIENTS
# point = x0
# fs_ls.grad_term(x0,1)[:5]

# eps = 0.000001
# grad = []
# for i in log_progress(range(len(point))):
#     z = np.zeros_like(point)
#     z[i] = 1
#     grad.append((fs_ls(point + eps*z, 1) - fs_ls(point, 1)) / eps) 
    
# np.array(grad)[:5]