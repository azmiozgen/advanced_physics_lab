import matplotlib.pylab as plt
import numpy as np

def compareTrainCost(*args):
    '''
    Models a single neuron for given parameters of args.
    arg[0] = (x, y)          ## input and expected output
    arg[1] = (weight, bias)  ## initial weight and bias
    arg[2] = costFunction
    arg[3] = activationFunction
    arg[4] = eta             ## learning rate
    arg[5] = llambda         ## regularization constant
    arg[6] = epoch
    '''

    costs = []
    for index, arg in enumerate(args):
        (x, y) = arg[0]          ## input and expected output
        (weight, bias) = arg[1]  ## initial weight and bias
        costFunction = arg[2]
        activationFunction = arg[3]
        eta = arg[4]             ## learning rate
        llambda = arg[5]
        epoch = arg[6]
        costs.append([])
        for i in range(epoch):
            z = x * weight + bias
            out = activationFunction.activate(z)
            weight =  weight - \
                      eta * llambda * weight - \
                      eta * costFunction.diff(y, out) * activationFunction.diff(z) * out
            bias = bias - \
                   eta * costFunction.diff(y, out) * activationFunction.diff(z)
            costs[index].append(costFunction.cost(y, out))

        ## Last out value with updated weights and biases.
        z = x * weight + bias
        out = activationFunction.activate(z)
        costs[index].append(costFunction.cost(y, out))

    ## Draw train cost vs. epoch
    for index, cost in enumerate(costs):
        plt.plot(range(epoch + 1), cost, label="model {}".format(index + 1))
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
