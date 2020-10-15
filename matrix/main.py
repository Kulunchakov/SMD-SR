import numpy as np    


def random_arg(dim):
    return np.random.rand((dim, 1))


def eigvals(matrix):
    return np.linalg.eigvals(matrix)


def random_semi_orthonormal_matrix(shape):
    assert isinstance(shape, tuple)
    mean_value = 35
    A = np.random.exponential(mean_value, size=shape) - mean_value
    Q, _ = np.linalg.qr(A)
    return Q


def random_orthonormal_matrix(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H

def check_L(func, L, ntries = 1000):
    norm = np.linalg.norm
    grad = func.gradient
    for i in range(ntries):
        x, y = np.random.rand(func.dim, 1), np.random.rand(func.dim, 1)
        if norm(grad(x) - grad(y)) > L*norm(x - y):
            print(x, y)
            return x, y
    print('Successful')
    return None, None

def check_mu(func, mu, ntries = 1000):
    norm = np.linalg.norm
    grad = func.gradient
    for i in range(ntries):
        x, y = np.random.rand(func.dim, 1), np.random.rand(func.dim, 1)
        if norm(grad(x) - grad(y)) < mu*norm(x - y):
            print(x, y)
            return x, y
    print('Successful')
    return None, None