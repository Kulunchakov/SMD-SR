from numpy import roots

class Sequence:
    def __init__(self, alpha_m1, q):
        # alpha_m1 = alpha_{-1}
        # q = mu / (mu + kpa)
        self.q = q
        self.arr = [self.solve_quadratic_eq(alpha_m1)]

    def solve_quadratic_eq(self, alpha=None):
        def extract_right_root(candidates):
            return [c for c in candidates if 0 < c < 1][0]
        alpha = self[-1] if alpha is None else alpha
        candidates = roots([1, alpha**2-self.q, -alpha**2])
        return extract_right_root(candidates)
    
    def gen_new(self):
        self.arr.append(self.solve_quadratic_eq())

    def size(self):
        return len(self.arr)
    
    def __getitem__(self, index):
        try:
            return self.arr[index]
        except IndexError:
            assert index >= 0
            for i in range(index + 1 - self.size()):
                self.gen_new()
            return self.arr[index]