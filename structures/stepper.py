class Stepper():
    def __init__(self, func):
        self.func = func

    def __getitem__(self, index):
        return self.func(index)