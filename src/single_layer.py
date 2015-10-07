import matplotlib.pylab as plt
import numpy as np

def default_weight_initializer(self):
    '''
    Initialize each weight using a Gaussian distribution with mean 0
    and standard deviation 1 over the square root of the number of
    weights connecting to the same neuron.  Initialize the biases
    using a Gaussian distribution with mean 0 and standard
    deviation 1.
    '''

    ## Weights for hidden and output layers
    self.weights = [np.random.randn(y, x) / np.sqrt(x)
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]

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
        (x, y) = arg[0]
        layerSize = arg[1]          ## Neuron numbers in the layer
        bias = np.random.randn(layerSize, 1)      ## Random initialization of biases
        weight = np.random.randn(layerSize, len(x))## Random initialization of weights
        costFunction = arg[2]
        activationFunction = arg[3]
        eta = arg[4]             ## learning rate
        llambda = arg[5]
        epoch = arg[6]
        costs.append([])
        for i in range(epoch):
            z = np.dot(weight, x) + bias
            out = np.vectorize(activationFunction.activate)(z)

            ## Update weights and biases
            weight =  weight - \
                      eta * llambda * weight / len(x) - \
                      eta * np.vectorize(costFunction.diff)(y, out) * np.vectorize(activationFunction.diff)(z) * out / len(x)
            bias = bias - \
                   eta * np.vectorize(costFunction.diff)(y, out) * np.vectorize(activationFunction.diff)(z) / len(x)

            costs[index].append(costFunction.cost(y, out))

        ## Last out value with updated weights and biases.
        z = np.dot(x * weight) + bias
        out = np.vectorize(activationFunction.activate)(z)
        costs[index].append(costFunction.cost(y, out))

    ## Draw train cost vs. epoch
    for index, cost in enumerate(costs):
        plt.plot(range(epoch + 1), cost, label="model {}".format(index + 1))
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
