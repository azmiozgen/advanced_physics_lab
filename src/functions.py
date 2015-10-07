import numpy as np

class Sigmoid(object):

    def activate(self, z):
        return 1. / (1. + np.exp(-z))

    def diff(self, z):
        return self.activate(z) * (1. - self.activate(z))

class Tanh(object):

    def activate(self, z):
        return np.tanh(z)

    def diff(self, z):
        return -(self.activate(z) ** 2) + 1

class Linear(object):

    def activate(self, z):
        return z

    def diff(self, z):
        return 1

class ReLU(object):

    def activate(self, z):
        return max(0, z)

    def diff(self, z):
        if z > 0:
            return 1
        else:
            return 0

class QuadraticCost(object):

    def cost(self, y, out):
        return 0.5 * ((y - out) ** 2.)

    def diff(self, y, out):
        return (out - y)

class CrossEntropyCost(object):

    def cost(self, y, out):
        return -np.nan_to_num((y * np.log(out) + (1. - y) * np.log(1. - out)))

    def diff(self, y, out):
        return (y - out) / (out * (out - 1.))
