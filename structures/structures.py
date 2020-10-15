from other.utils import *
import time

class Subscriptable():
    def __getitem__(self, index):
        if isinstance(index, int) and 'history' in self:
            return self.history[:, index:index + 1] if index >= 0 else self.history[:, index:]
        elif isinstance(index, int) and 'dataset' in self:
            return self.dataset[index]
        elif isinstance(index, str):
            if index in self.p:
                return self.p[index]
            elif index in self.__dict__:
                return self.__dict__[index]
            else:
                print(index)
                raise KeyError

    def __contains__(self, index):
        if isinstance(index, int):
            return len(self.history) > index
        elif isinstance(index, str):
            return index in self.p or index in self.__dict__
        else:
            raise KeyError


class OptimizationMethod:
    def print_intro(self):
        if not self.verbose: return
        print(bold('{}'.format(self.name)))
    def print_outro(self):
        if not self.verbose: return
        print("{:.2f} seconds elapsed;\t{} calls to oracle;\t".format(self.history[-1]['make_pass'], self.oracle.calls))
        print("Approximately {:.3f} seconds for one call".format(self.history[-1]['make_pass'] / self.oracle.calls))
        if 'dataset' in self.objective:
            print("{} accesses to {}".format(self.objective.dataset.accesses, self.objective.dataset))
        print('\n\n')



class NamedClass:
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.__str__()


class Assignable():
    def __setitem__(self, key, value):
        if key in self.p:
            self.p[key] = value
        elif key in self.__dict__:
            self.__dict__[key] = value
        else:
            raise KeyError
