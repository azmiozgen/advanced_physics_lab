import matplotlib.pylab as plt
import numpy as np

class Sigmoid(object):

    def activate(self, z):
        return 1. / (1. + np.exp(-z))

    def diff(self, z):
        return self.activate(z) * (1. - self.activate(z))

class QuadraticCost(object):

    def cost(self, y, out):
        return 0.5 * ((y - out) ** 2.)

    def diff(self, y, out):
        return (out - y)

class CrossEntropyCost(object):

    def cost(self, y, out):
        return -(y * np.log(out) + (1. - y) * np.log(1. - out))

    def diff(self, y, out):
        return (y - out) / (out * (out - 1.))

def singleNeuronModel(params, x=1.0, y=0.0, costFunction=QuadraticCost(), \
                             eta=0.15, activationFunction=Sigmoid(), epoch=300):
    '''
    Models a single neuron for given params (weight and bias).
    i.e. params=[[1.2, 1.453], [6.35442, 3]]
    Input x = 1.0 and expected output y = 0.0.
    Learning rate eta = 0.15.
    Epoch (total training number) is 300.
    '''
    costs = []
    for index, (weight, bias) in enumerate(params):
        costs.append([])
        for i in range(epoch):
            z = x * weight + bias
            out = activationFunction.activate(z)
            weight -= eta * costFunction.diff(y, out) * \
                      activationFunction.diff(z) * out
            bias -= eta * costFunction.diff(y, out) * activationFunction.diff(z)
            costs[index].append(costFunction.cost(y, out))

        # Last out value with updated weights and biases.
        z = x * weight + bias
        out = activationFunction.activate(z)
        costs[index].append(costFunction.cost(y, out))

    # Print last weight, bias and output and drawing.
    for index, cost in enumerate(costs):
        plt.plot(range(epoch + 1), cost, label="initial w, b = {}".format(params[index]))
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
