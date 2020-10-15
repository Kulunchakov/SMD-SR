from numpy import sqrt, ones


def get_acc_md_gamma_from_class(entry, n_iterations):
    return min(.49 / entry['L'], sqrt(6)*(entry['omega'] / entry['sigma']) * sqrt(1 / ((n_iterations+2)**3)))


def get_md_gamma_from_class(entry, n_iterations):
    return min(.49 / entry['L'], (entry['omega'] / entry['sigma']) * sqrt(.5 / n_iterations))


def get_md_gamma(params):
    return min(.49/params['L'], (params['omega']/params['sigma'])*sqrt(.5/params['N']))


def init_approximation(oracle):
    return ones((oracle.p['dim'], 1))