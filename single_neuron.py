import matplotlib.pylab as plt
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidPrime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def quadraticCost(y, out):
    '''
    '''
    return 0.5 * ((y - out) ** 2)

def quadraticCostPrime(y, out):
    '''
    '''
    return (out - y)

def singleNeuronModel(weight, bias, x=1.0, y=0.0, costFunction=quadraticCost,
                      eta=0.15, activationFunction = sigmoid, epoch=300):
    '''
    '''
    allCosts = []
    for i in range(epoch):
        z = x * weight + bias
        out = sigmoid(z)
        weight = weight - eta * quadraticCostPrime(y, out) * sigmoidPrime(z) * out
        bias = bias - eta * quadraticCostPrime(y, out) * sigmoidPrime(z)
        allCosts.append(quadraticCost(y, out))

    print "weight: {}, bias: {}, output: {}".format(weight, bias, allCosts[-1])
    plt.plot(range(epoch), allCosts)
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.show()
