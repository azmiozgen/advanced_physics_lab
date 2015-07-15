Improving the way neural networks learn (Nielsen, Chapter 3)
====================================================================

In this chapter it's explained a **suite of techniques** which can be used to **improve the implementation of backpropagation,** and so **improve the way our networks learn.**

The techniques we'll develop in this chapter include: a **better choice of cost function, known as the *cross-entropy cost function*;** four ***"regularization" methods*** **(L1 and L2 regularization, dropout, and artificial expansion of the training data),** which make our networks better at **generalizing beyond the training data;** a **better method for initializing the weights** in the network; and **a set of heuristics to help choose good hyper-parameters for the network.** 

## The cross-entropy cost function ##

> Soon after beginning to learn the piano I gave my first performance before an audience. I was nervous, and began playing the piece an octave too low. I got confused, and couldn't continue until someone pointed out my error. I was very embarassed. Yet while unpleasant, **we also learn quickly when we're decisively wrong.** You can bet that the next time I played before an audience I played in the correct octave! By contrast, **we learn more slowly when our errors are less well-defined.**

Ideally, we hope and expect that our neural networks will learn fast from their errors. Is this what happens in practice? To answer this question, let's look at a **toy example.** The example involves **a neuron with just one input:**

![simple_neuron](http://neuralnetworksanddeeplearning.com/images/tikz28.png)

We'll train this neuron taking the **input 1 to the output 0.** Of course, this is such a **trivial task that we could easily figure out an appropriate weight and bias by hand,** without using a learning algorithm. However, it turns out to be **illuminating** to use gradient descent to attempt to **learn a weight and bias.** So let's take a look at how the neuron learns.

To make things definite, we'll pick the **initial weight to be 0.6** and the **initial bias to be 0.9** generically. The **initial output** from the neuron is **0.82,** so quite a bit of learning will be needed before our neuron gets near the **desired output, 0.0.** The learning rate is $ \eta = 0.15. $ The cost is the **quadratic cost function,** $ C $. Code and the result shown below.


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
	
	def singleNeuronModel(weight, bias, x=1.0, y=0.0, costFunction=QuadraticCost(),
	                      eta=0.15, activationFunction=Sigmoid(), epoch=300):
	    '''
	    Models a single neuron for given weight and bias.
	    Input x = 1.0 and expected output y = 0.0.
	    Learning rate eta = 0.15.
	    Epoch (total training number) is 300.
	    '''
	    allCosts = []
	    for i in range(epoch):
	        z = x * weight + bias
	        out = activationFunction.activate(z)
	        weight -= eta * costFunction.diff(y, out) * activationFunction.diff(z) * out
	        bias -= eta * costFunction.diff(y, out) * activationFunction.diff(z)
	        allCosts.append(costFunction.cost(y, out))
	
	    # Last out value with updated weights and biases.
	    z = x * weight + bias
	    out = activationFunction.activate(z)
	    allCosts.append(costFunction.cost(y, out))
	
	    # Print last weight, bias and output and drawing.
	    print "weight: {}, bias: {}, output: {}".format(weight, bias, allCosts[-1])
	    plt.plot(range(epoch + 1), allCosts)
	    plt.xlabel('epochs')
	    plt.ylabel('cost')
	    plt.show()

-----

	singleNeuronModel(weight=0.6, bias=0.9)

![rapid_learning](https://lh3.googleusercontent.com/-XMhTxcBFyJc/VZULqrdycxI/AAAAAAAAAU4/tRj1DIVkbTQ/s0/single_neuron%2528w%253D0.6%252Cb%253D0.9%2529.png "single_neuron&#40;w=0.6,b=0.9&#41;")

As you can see, the **neuron rapidly learns** a weight and bias that drives down the cost, and gives an output 0.008. Suppose, however, that we instead choose both the **starting weight and the starting bias to be 2.0.** In this case the **initial output is 0.98, which is very badly wrong.** Let's look at how the neuron learns to output 0 in this case. 

![slow_learning](https://lh3.googleusercontent.com/-yWskvCDvRP8/VZULicYhmpI/AAAAAAAAAUs/LN7yVyhkN4Q/s0/single_neuron%2528w%253D2.0%252Cb%253D2.0%2529.png "single_neuron&#40;w=2.0,b=2.0&#41;")

Although this example uses the same learning rate $ (\eta = 0.15) $, we can see that learning starts out much more slowly. Indeed, **for the first 150 or so learning epochs, the weights and biases don't change much at all.** 

	singleNeuronModel(weight=2.0, bias=2.0)

This behaviour is strange when contrasted to human learning. As I said at the beginning of this section, we often learn fastest when we're badly wrong about something. But we've just seen that our artificial neuron has a lot of difficulty learning when it's badly wrong - far more difficulty than when it's just a little wrong. What's more, it turns out that this behaviour occurs not just in this toy model, but in more general networks. Why is learning so slow? And can we find a way of avoiding this slowdown?

To understand the origin of the problem, let's compute the partial derivatives. Recall that we're using the quadratic cost function,

\begin{equation}
	C = \frac {(y − a)^2}{2}
	\hspace{2.5 cm} (1)
\end{equation}

where $ a $ is the neuron's output when the training input $ x = 1 $ is used, and $ y = 0 $ is the corresponding desired output. Recall that $ a = \sigma(z) $, where $ z = wx + b $. Using the chain rule to differentiate with respect to the weight and bias we get

\begin{equation}
	\frac {\partial C}{\partial w} = \frac {\partial C}{\partial a}
	\frac {\partial a}{\partial \sigma} \frac {\partial \sigma}{\partial w} =
	(a - y) \sigma^\prime(z) x = 
	a \sigma^\prime(z) \hspace{2.5 cm} (2)\\
	\frac {\partial C}{\partial b} = \frac {\partial C}{\partial a}
	\frac {\partial a}{\partial \sigma} \frac {\partial \sigma}{\partial b} =
	(a - y) \sigma^\prime(z) = 
	a \sigma^\prime(z) \hspace{2.5 cm} (3)
\end{equation}

where substituted $ x = 1 $ and $ y = 0 $. To understand the behaviour of these expressions, let's look more closely at the $ \sigma^\prime (z) $ term on the right-hand side. Recall the shape of the $ \sigma $ function:

![sigmoid_function](https://lh3.googleusercontent.com/ixnXyVlSgBDv2QDdRHSGCjTI7rTV72ILnSoAjWjklEk=s0 "sigmoid_function")

We can see from this graph that when the **neuron's output is close to 1, the curve gets very flat,** and so $ \sigma^\prime(z) $ gets very small. Therefore $ \partial C / \partial w $ and $ \partial C / \partial b $ **get very small.** This is the origin of the ***learning slowdown***. 

## Introducing the cross-entropy cost function ##

How can we address the learning slowdown? It turns out that we can solve the problem by **replacing the quadratic cost with a different cost function, known as the** ***cross-entropy.*** To understand the cross-entropy, let's move a little away from our super-simple toy model. We'll suppose instead that we're trying to train **a neuron with several input variables,** $ x_1, x_2, … $ **corresponding weights** $ w_1, w_2, … $ and a bias, $ b $:

![neuron_with_multiple_input](http://neuralnetworksanddeeplearning.com/images/tikz29.png)


The output from the neuron is, of course, $ a = \sigma(z) $, where $ z = \sum_j w_j x_j + b $ is the weighted sum of the inputs. We define the ***cross-entropy cost function*** for this neuron by

\begin{equation}
	C = -\frac{1}{n} \sum_x [y \ln a + (1 − y) \ln (1 − a)]
	\hspace{2.5 cm}(4)
\end{equation}

where $ n $ is the **total number of items of training data,** the sum is over all training inputs, $ x $, and $ y $ is the corresponding desired output. The quantity $ −[y \ln a + (1 − y) \ln (1 − a)] $ is sometimes known as the [***binary entropy***](https://en.wikipedia.org/wiki/Binary_entropy_function).

**Two properties** in particular make it reasonable to **interpret the cross-entropy as a cost function.** First, it's **non-negative,** that is, $ C > 0 $. $ y $ takes only $ 0 $ or $ 1 $ and $ a $ takes values between $ 0 $ and $ 1 $. Then $ \ln(1 - a) < 0 $ and $ \ln(a) < 0 $. The minus sign out of front provides $ C > 0 $.

Second, **if the neuron's actual output is close to the desired output, i.e.,** $ y = y(x) $ **for all training inputs** $ x $ **, then the cross-entropy will be close to zero.** 

	>>> import sympy as sp
	>>> a, y = sp.symbols("a y")
	>>> expr = - (y * sp.ln(a) + (1 - y) * sp.ln(1 - a))
	>>> expr.subs({y:0, a:0.00001})
		1.00000500002878e-5
	>>> expr.subs({y:1, a:0.99999})
		1.00000500005137e-5

To see this, let's compute the partial derivative of the cross-entropy cost with respect to the weights. We substitute $ a = \sigma(z) $ into (4), and apply the **chain rule twice,** obtaining:

\begin{equation}
	\begin{split}
		\frac{\partial C}{\partial w_j} & = −\frac{1}{n}\sum_x \left ( 
		\frac{y}{\sigma(z)} − \frac{(1 − y)}{1−\sigma(z)} \right) 
		\frac{\partial \sigma}{\partial w_j} \hspace{2.5 cm}(5) \\
		& = −\frac{1}{n}\sum_x \left ( 
		\frac{y}{\sigma(z)} − \frac{(1 − y)}{1−\sigma(z)} \right) 
		\sigma^\prime(z) x_j \hspace{2.5 cm}(6)\\
		& = \frac{1}{n} \sum_x  \frac{\sigma ^ \prime(z) x_j}
		{\sigma(z)[1 - \sigma(z)]}(\sigma(z) - y) \hspace{2.5 cm}(7)
	 \end{split}
\end{equation}

We know that $ \sigma^\prime(z) = \sigma(z)[1−\sigma(z)] $. Therefore (7) becomes,

\begin{equation}
	\frac{\partial C}{\partial w_j} = \frac{1}{n} \sum_x x_j [\sigma(z) − y]
	\hspace{2.5 cm}(8)
\end{equation}

This is a beautiful expression. It tells us that the **larger the error, the faster the neuron will learn.** In a similar way, it can be easily verified that

\begin{equation}
	\frac{\partial C}{\partial b} = \frac{1}{n} \sum_x [\sigma(z) − y]
	\hspace{2.5 cm}(9)
\end{equation}

Let's return to the toy example we played with earlier, and explore what happens when we **use the cross-entropy instead of the quadratic cost.** To re-orient ourselves, we'll begin with the case where the quadratic cost did just fine, with **starting weight 0.6 and starting bias 0.9.** 

	singleNeuronModel(weight=0.6, bias=0.9, costFunction=CrossEntropyCost())

![cross_entropy_cost](https://lh3.googleusercontent.com/-xP058zOE2KI/VZaBS3nLgfI/AAAAAAAAAVY/OcJVL7bxpMg/s0/single_neuron%2528w%253D0.6%252Cb%253D0.9%2529_cross_entropy_cost.png "single_neuron&#40;w=0.6,b=0.9&#41;_cross_entropy_cost")

Now with **starting weight 2.0 and starting bias 2.0.**

	singleNeuronModel(weight=2.0, bias=2.0, costFunction=CrossEntropyCost())

![cross_entropy_cost_further_weights](https://lh3.googleusercontent.com/-a-04S0ASN1g/VZaBbz1zOUI/AAAAAAAAAVk/cVsHwrnmW6A/s0/single_neuron%2528w%253D2.0%252Cb%253D2.0%2529_cross_entropy_cost.png "single_neuron&#40;w=2.0,b=2.0&#41;_cross_entropy_cost")


**Success!** This time the **neuron learned quickly,** just as we hoped. 

We've been studying the cross-entropy for a single neuron. However, it's easy to **generalize the cross-entropy to many-neuron multi-layer networks.** In particular, suppose $ y = y_1, y_2, … $ are the desired values at the output neurons, i.e., the neurons in the final layer, while $ a^L_1, a^L_2, … $are the actual output values. Then we define the cross-entropy by

\begin{equation}
		C = -\frac{1}{n} \sum_x \sum_j \left[ y_j \ln a^L_j + 
		(1 - y_j) \ln (1 - a^L_j) \right ] \hspace{2.5 cm}(10) 
\end{equation}

When should we use the cross-entropy instead of the quadratic cost? In fact, the **cross-entropy is nearly always the better choice, provided the output neurons are sigmoid neurons.** To see why, consider that when we're setting up the network **we usually initialize the weights and biases using some sort of randomization.** It may happen that those initial choices result in the network being **decisively wrong for some training input** - that is, an output neuron will have saturated near 1, when it should be 0, or vice versa. If we're **using the quadratic cost that will slow down learning.** 

## Softmax ##

The idea of ***softmax*** is to **define a new type of output layer** for our neural networks. It begins in the same way as with a sigmoid layer, by forming the weighted inputs as $ z^L_j = \sum_k w^L_{jk} a^{L − 1} _k + b^L_j $. However, in a softmax layer we apply the so-called **softmax function** to the $ z^L_j $. According to this function, the activation $ a^L_j $ of the $ j^{th} $ output neuron is

\begin{equation}
		a^L_j = \frac{e^{z^L_j}}{\sum_k e^{z^L_k}} 
		\hspace{2.5 cm}(11) 
\end{equation}

where **in the denominator we sum over all the output neurons.**

To better understand Equation (11), suppose we have a network with four output neurons, and four corresponding weighted inputs, which we'll denote $ z^L_1, z^L_2, z^L_3 $ and $ z^L_4 $. As you **increase** $ z^L_4 $, you'll see an **increase in the corresponding output activation,** $ a^L_4 $, and a **decrease in the other output activations.** Similarly, if you **decrease** $ z^L_4 $ **then** $ a^L_4 $ **will decrease, and all the other output activations will increase.** In fact, if you look closely, you'll see that in both cases the total change in the other activations exactly **compensates for the change in** $ a^L_4 $. The reason is that the **output activations are guaranteed to always sum up to** $ 1 $, 

\begin{equation}
		\sum_j a^L_j = \frac{\sum_j e^{z^L_j}}{\sum_k e^{z^L_k}} = 1
		\hspace{2.5 cm}(12) 
\end{equation}

And, of course, **similar statements hold for all the other activations.**

Equation (11) implies that the **output from the softmax layer is a set of positive numbers which sum up to 1.** In other words, the output from the softmax layer can be thought of as a **probability distribution.**

In many problems it's convenient to be able to **interpret the output activation** $ a^L_j $ **as the network's estimate of the probability that the correct output is** $ j $. So, for instance, in the MNIST classification problem, we can interpret $ a^L_j $ as the network's estimated probability that the correct digit classification is $ j $.

**The learning slowdown problem:**  How a softmax layer lets us address the learning slowdown problem? To understand that, let's define the ***log-likelihood cost function.*** We'll use $ x $ to denote a training input to the network, and $ y $ to denote the corresponding desired output. Then the log-likelihood cost associated to this training input is

\begin{equation}
		C \equiv −\ln a^L_y \hspace{2.5 cm}(13)
\end{equation}

So, for instance, if we're training with MNIST images, and input an image of a 7, then the log-likelihood cost is $ −\ln a^L_7 $. To see that this makes intuitive sense, consider the case when the network is doing a good job, that is, it is confident the input is a $ 7 $. In that case it will estimate a value for the corresponding probability $ a^L_7 $ which is close to $ 1 $, and so the cost $ −\ln a^L_7 $ will be small. By contrast, when the network isn't doing such a good job, the probability $ a^L_7 $ will be smaller, and the cost $ −\ln a^L_7 $ will be larger. Unlike quadratic cost function and cross-entropy function, no need to sum all the cost in the output neurons, only **corresponding neuron must be considered.** Because the cost of output neurons are related each other to make $ \sum_j a^L_j = 1 $.

What about the learning slowdown problem? With a little algebra you can show that. 

\begin{equation}
	\begin{split}
		\frac{\partial C}{\partial b^L_j} & = a^L_j - y_j 
		\hspace{2.5 cm}(14) \\
		\frac{\partial C}{\partial w^L_{jk}} & = a^{L - 1}_k(a^L_j - y_j)
		\hspace{2.5 cm}(15)
	\end{split}
\end{equation}

Here we used $ y $ to denote the desired output from the network - e.g., output a "$ 7 $" if an image of a $ 7 $ was input. But in the equations here $ y $ to denote the vector of output activations which corresponds to $ 7 $, that is, a vector which is all $ 0 $ s, except for a $ 1 $ in the $ 7^{th} $ location.

Just as in the earlier analysis, these expressions **ensure that we will not encounter a learning slowdown.** In fact, it's useful to think of a softmax output layer with log-likelihood cost as being quite similar to a sigmoid output layer with cross-entropy cost.

## Overfitting ##

***Overfitting*** is a **major problem** in neural networks. This is especially true in modern networks, which often have **very large numbers of weights and biases.** 

Let's sharpen this problem up by constructing a situation where our network does a bad job generalizing to new situations. We'll use our 30 hidden neuron network, with its 23,860 parameters. But **we won't train the network using all 50,000 MNIST training images.** Instead, we'll use **just the first 1,000 training images.** Using that **restricted set** will make the problem with generalization much more evident. We'll train in a similar way to before, using the **cross-entropy cost function,** with a learning rate of $ \eta = 0.5 $ and a **mini-batch size of 10.** However, we'll train for **400 epochs**, a somewhat larger number than before, because **we're not using as many training examples.** Get [network2.py](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py) from GitHub to look at the way the cost function changes:

	>>> import mnist_loader 
	>>> training_data, validation_data, test_data =     mnist_loader.load_data_wrapper()
	>>> import network2 
	>>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost) 
	>>> net.large_weight_initializer()
	>>> net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)

Using the results we can plot the way the cost changes as the network learns. This and the next four graphs were generated by the program [overfitting.py](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/fig/overfitting.py):

![overfitting_cost](http://neuralnetworksanddeeplearning.com/images/overfitting1.png)

This looks encouraging, showing a **smooth decrease in the cost**, just as we expect. Let's now look at how the classification accuracy on the test data changes over time:

![overfitting_accuracy](http://neuralnetworksanddeeplearning.com/images/overfitting2.png)

If we just look at that cost, it appears that our model is still getting "better". But the test accuracy results show the **improvement is an illusion.** What our network learns after epoch 280 **no longer generalizes to the test data.** And so it's not useful learning. We say the network is **overfitting or overtraining beyond epoch 280.**

Another sign of overfitting may be seen in the classification accuracy on the training data:

![overfitting_accuracy_training](http://neuralnetworksanddeeplearning.com/images/overfitting4.png)

The accuracy rises all the way up to 100 percent. That is, **our network correctly classifies all 1,000 training images!** Meanwhile, our test accuracy tops out at just 82.27 percent. So our network really is learning about peculiarities of the training set, not just recognizing digits in general. It's almost as though our network is **merely memorizing the training set**, without understanding digits well enough to generalize to the test set.

The obvious way to detect overfitting is to use the approach above, keeping track of accuracy on the test data as our network trains. **If we see that the accuracy on the test data is no longer improving, then we should stop training.** Of course, strictly speaking, this is not necessarily a sign of overfitting. It might be that accuracy on the test data and the training data both stop improving at the same time. Still, adopting this strategy will prevent overfitting.

In fact, we'll use a variation on this strategy. Recall that when we load in the MNIST data we load in three data sets:

	>>> import mnist_loader 
	>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	
Up to now we've been using the 'training_data' and 'test_data', and ignoring the 'validation_data'. The 'validation_data' contains 10,000 images of digits, images which are different from the 50,000 images in the MNIST training set, and the 10,000 images in the MNIST test set. 

Instead of using the 'test_data' to prevent overfitting, we will use the 'validation_data'. To do this, we'll use much the same strategy as was described above for the 'test_data'. That is, we'll compute the classification accuracy on the 'validation_data' at the end of each epoch. **Once the classification accuracy on the 'validation_data' has saturated, we stop training.** This strategy is called ***early stopping***. Of course, in practice we won't immediately know when the accuracy has saturated. Instead, we **continue training until we're confident that the accuracy has saturated.** It requires **some judgement** to determine when to stop. 

You can think of the validation data as a type of training data that **helps us learn good hyper-parameters.** This approach to finding good hyper-parameters is sometimes known as the hold out method, since the **'validation_data' is kept apart from the 'training_data'.**

What happens when we use the full training set of 50,000 images?  Here's a graph showing the results for the classification accuracy on both the training data and the test data. 

![comparison_accuracy_test_training](http://neuralnetworksanddeeplearning.com/images/overfitting_full.png)

As you can see, the accuracy on the test and training data remain **much closer together** than when we were using 1,000 training examples. Overfitting is still going on, but it's been greatly reduced. Our network is **generalizing much better** from the training data to the test data. In general, **one of the best ways of reducing overfitting is to increase the size of the training data.** Unfortunately, **training data can be expensive or difficult to acquire, so this is not always a practical option.**

## Regularization ##

Fortunately, there are other techniques which can reduce overfitting, even when we have a fixed network and fixed training data. These are known as ***regularization*** techniques. One of the most commonly used regularization techniques are ***weight decay*** or ***L2 regularization***. The idea of L2 regularization is to **add an extra term to the cost function**, a term called the ***regularization term***. Here's the regularized cross-entropy:

\begin{equation}
		C = -\frac{1}{n} \sum_{xj} \left [
		y_j \ln a^L_j + (1 − y_j) \ln (1 − a^L_j) \right ] + 
		\frac{\lambda}{2n} \sum_w w^2 
		\hspace{1.0 cm}(16)
\end{equation}

The first term is just the usual expression for the cross-entropy. But we've added a **second term**, namely the **sum of the squares of all the weights in the network.** This is scaled by a factor $ \lambda / 2n $, where $ \lambda > 0 $ is known as the ***regularization parameter***, and $ n $ is, as usual, the size of our training set. 

Of course, it's possible to regularize other cost functions, such as the quadratic cost. This can be done in a similar way:

\begin{equation}
		C = -\frac{1}{2n} \sum_x || y - a^L ||^2 + 
		\frac{\lambda}{2n} \sum_w w^2 
		\hspace{1.0 cm}(17)
\end{equation}

Intuitively, the effect of regularization is to make it so the **network prefers to learn small weights**, all other things being equal. Large weights will only be allowed if they **considerably improve the first part of the cost function.** Regularization can be viewed as a way of compromising between **finding small weights and minimizing the original cost function.** When $ \lambda $ **is small we prefer to minimize the original cost function, but when** $ \lambda $ **is large we prefer small weights.**

To construct such an example, we first need to figure out how to apply our stochastic gradient descent learning algorithm in a regularized neural network. In particular, we need to know how to compute the partial derivatives $ \partial C / \partial w $ and $ \partial C / \partial b $ for all the weights and biases in the network. 

\begin{equation}
	\begin{split}
		\frac{\partial C}{\partial w} & = \frac{\partial C_0}{\partial w} + 
		\frac{\lambda}{n} w \hspace{2.5 cm}(18) \\
		\frac{\partial C}{\partial b} & = \frac{\partial C_0}{\partial b}
		\hspace{2.5 cm}(19)
	\end{split}
\end{equation}

The $ \partial C_0 / \partial w $ and $ \partial C_0 / \partial b $ terms can be computed using backpropagation. Then **add** $ \frac{\lambda}{n} w $ **to the partial derivative of all the weight terms.** The partial derivatives with respect to the biases are unchanged, and so the **gradient descent learning rule for the biases doesn't change** from the usual rule:

\begin{equation}
	b \to b − \eta \frac{\partial C_0}{\partial b}\hspace{2.5 cm}(20)
\end{equation}

The learning rule for the weights becomes:

\begin{equation}
	\begin{split}
		w & \to w − \eta \frac{\partial C_0}{\partial w} - 
		\frac{\eta \lambda}{n} w \hspace{2.5 cm}(21)\\
		&= \left (1 - \frac{\eta \lambda}{n} \right ) w - \eta 
		\frac{\partial C_0}{\partial w}
		\hspace{2.5 cm}(22)
	\end{split}
\end{equation}

This is exactly the same as the usual gradient descent learning rule, except we first **rescale the weight** $ w $ **by a factor** $ 1 − \frac{\eta \lambda}{n} $. This rescaling is sometimes referred to as ***weight decay***, **since it makes the weights smaller.** At first glance it looks as though this means the weights are being driven **unstoppably toward zero. But that's not right, since the other term may lead the weights to increase.**

For stochastic gradient descent Equation (20) becomes

\begin{equation}
	b \to b - \frac{\eta}{m} \sum_x \frac{\partial C_x}{\partial b}
	\hspace{2.5 cm}(23)
\end{equation}

Equation (21) becomes

\begin{equation}
	w \to \left (1 - \frac{\eta \lambda}{n} \right ) w - 
	\frac{\eta}{m} \sum_x \frac{\partial C_x}{\partial w}
	\hspace{2.5 cm}(24)
\end{equation}

where the sum is over training examples $ x $ in the mini-batch, and $ C_x $ is the (unregularized) cost for each training example. And 

Let's see how regularization changes the performance of our neural network. We'll use a network with 30 hidden neurons, a mini-batch size of 10, a learning rate of 0.5, and the cross-entropy cost function. However, this time we'll use a regularization parameter of $ \lambda = 0.1 $. 
 
	>>> import mnist_loader 
	>>> training_data, validation_data, test_data = 
	mnist_loader.load_data_wrapper() 
	>>> import network2 
	>>> net = network2.Network([784, 30, 10],
	cost=network2.CrossEntropyCost)
	>>> net.large_weight_initializer()
	>>> net.SGD(training_data[:1000], 400, 10, 0.5, 
	evaluation_data=test_data, lmbda = 0.1, 
	monitor_evaluation_cost=True, 
	monitor_evaluation_accuracy=True, 
	monitor_training_cost=True, 
	monitor_training_accuracy=True)
	
The cost on the training data decreases over the whole time, much as it did in the earlier, unregularized case.

![cost_on_the_entire_data](http://neuralnetworksanddeeplearning.com/images/regularized1.png)

But this time the accuracy on the 'test_data' continues to increase for the entire 400 epochs:

![accuracy_on_the_test_data](http://neuralnetworksanddeeplearning.com/images/regularized2.png)

Clearly, the use of **regularization has suppressed overfitting.** What's more, the accuracy is considerably higher, with a peak classification accuracy of $ 87.1 $ percent, compared to the peak of $ 82.27 $ percent obtained in the unregularized case. It seems that, **empirically, regularization is causing our network to generalize better, and considerably reducing the effects of overfitting.**

Let' train our network with full 50,000 images. The hyper-parameters the same as before - 30 epochs, learning rate 0.5, mini-batch size of 10. However, we need to modify the regularization parameter. The reason is because the size $ n $ of the training set has changed from $ n = 1,000 $ to $ n = 50,000 $. If we continued to use $ \lambda = 0.1 $ that would mean **much less weight decay**, and thus **much less of a regularization effect.** We compensate by changing to $ \lambda = 5.0 $.
 
	>>> net.large_weight_initializer()
	>>> net.SGD(training_data, 30, 10, 0.5, 
	evaluation_data=test_data, lmbda = 5.0, 
	monitor_evaluation_accuracy=True, 
	monitor_training_accuracy=True)

![accuracy_on_test_and_training](http://neuralnetworksanddeeplearning.com/images/regularized_full.png)

There's lots of good news here. First, our classification accuracy on the test data is up, from $ 95.49 $ percent when running unregularized, to $ 96.49 $ percent. That's a big improvement. Second, we can see that the **gap between results on the training and test data is much narrower than before**, running at under a percent. That's still a significant gap, but we've obviously made substantial progress reducing overfitting.

Finally, let's see what test classification accuracy we get when we use 100 hidden neurons and a regularization parameter of $ \lambda = 5.0 $. 
 
	>>> net = network2.Network([784, 100, 10], 
	cost=network2.CrossEntropyCost)
	>>> net.large_weight_initializer()
	>>> net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, 
	evaluation_data=validation_data, 
	monitor_evaluation_accuracy=True)
	
The final result is a classification accuracy of $ 97.92 $ percent on the validation data. 

Empirically, when doing multiple runs of our MNIST networks, but with different (random) weight initializations, I've found that the **unregularized runs will occasionally get "stuck"**, apparently **caught in local minima** of the cost function. By contrast, the **regularized runs have provided much more easily replicable results.**

Heuristically, **if the cost function is unregularized, then the length of the weight vector is likely to grow**, all other things being equal. Over time **this can lead to the weight vector being very large indeed** and can cause the **weight vector to get stuck pointing in more or less the same direction, since changes due to gradient descent only make tiny changes to the direction.**

## Other techniques for regularization ##

There are three other approaches to reducing overfitting: ***L1 regularization***, ***dropout***, and ***artificially expanding the training set size***. 

### L1 regularization ### 

In this approach we modify the unregularized cost function by adding the sum of the absolute values of the weights:

\begin{equation}
	C = C_0 + \frac{\lambda}{n} \sum_w |w|
	\hspace{2.5 cm}(25)
\end{equation}

Intuitively, this is similar to L2 regularization, **penalizing large weights**, and tending to **make the network prefer small weights.** 

To do that, we'll look at the partial derivatives of the cost function. Differentiating (25) we obtain:

\begin{equation}
	\newcommand{\sgn}{\mathop{\mathrm{sgn}}}
	\frac{\partial C}{\partial w} = \frac{\partial C_0}{\partial w} + 
	\frac{\lambda}{n} \sgn(w)
	\hspace{2.5 cm}(25)
\end{equation}

where $ \newcommand{\sgn}{\mathop{\mathrm{sgn}}} \sgn(w) $ is the sign of $ w $, that is, +1 if $ w $ is positive, and −1 if $ w $ is negative. The resulting update rule for an L1 regularized network is

\begin{equation}
	\newcommand{\sgn}{\mathop{\mathrm{sgn}}}
	w\to w − \frac{\eta \lambda}{n} \sgn(w) -  
	\eta \frac{\partial C_0}{\partial w}
	\hspace{2.5 cm}(26)
\end{equation}

In both expressions the effect of regularization is to **shrink the weights.** This accords with our intuition that both kinds of regularization penalize large weights. But the way the weights shrink is different. **In L1 regularization, the weights shrink by a constant amount toward 0. In L2 regularization, the weights shrink by an amount which is proportional to w.** And so when a particular weight has a large magnitude, $ |w| $, **L1 regularization shrinks the weight much less than L2 regularization does.** By contrast, **when** $ |w| $ **is small, L1 regularization shrinks the weight much more than L2 regularization.** The net result is that L1 regularization tends to concentrate the weight of the network in a **relatively small number of high-importance connections,** while the other weights are driven toward zero.

Partial derivative $ \partial C / \partial w $ isn't defined when $ w = 0 $. The reason is that the function $ |w| $ has a **sharp "corner"** at $ w = 0 $, and so **isn't differentiable** at that point. That's okay, though. We'll use Equations (25) and (26) with the convention that $ 	\newcommand{\sgn}{\mathop{\mathrm{sgn}}} \sgn(0) = 0 $. 

### Dropout ###

Dropout is a **radically different technique** for regularization. In dropout we **modify the network itself.** Suppose we're trying to train a network:

![dropout_sample_net](http://neuralnetworksanddeeplearning.com/images/tikz30.png)


With dropout, we start by **randomly (and temporarily) deleting half the hidden neurons** in the network, **while leaving the input and output neurons untouched.** After doing this, we'll end up with a network along the following lines. 

![dropout_deleted_neurons](http://neuralnetworksanddeeplearning.com/images/tikz31.png)

After doing this over a mini-batch of examples, we **update the appropriate weights and biases.** We then **repeat the process**, first restoring the dropout neurons, then choosing a new random subset of hidden neurons to delete, estimating the gradient for a different mini-batch, and updating the weights and biases in the network.

By repeating this process over and over, our network will **learn a set of weights and biases.** Of course, those weights and biases will have been learnt under conditions in which half the hidden neurons were dropped out. When we actually run the full network that means that **twice as many hidden neurons** will be active. To **compensate for that, we halve the weights outgoing from the hidden neurons.**

Heuristically, when we dropout different sets of neurons, **it's rather like we're training different neural networks.** And so the dropout procedure is like **averaging the effects of a very large number of different networks.** The **different networks will overfit in different ways, and so, hopefully, the net effect of dropout will be to reduce overfitting.**

In other words, if we think of our network as a model which is making predictions, then we can think of dropout as a way of making sure that the **model is robust to the loss of any individual piece of evidence.** 

### Artificially expanding the training data ###
Obtaining more training data is a great idea. Unfortunately, it can be expensive, and so is not always possible in practice. However, there's another idea which can work nearly as well, and that's to artificially expand the training data. Suppose, for example, that we take an MNIST training image of a five,

![normal_five](http://neuralnetworksanddeeplearning.com/images/more_data_5.png)

and rotate it by a small amount, let's say 15 degrees:

![rotated_five](http://neuralnetworksanddeeplearning.com/images/more_data_rotated_5.png)

It's still recognizably the same digit. And yet at the pixel level it's quite different to any image currently in the MNIST training data. It's conceivable that **adding this image to the training data might help our network learn more about how to classify digits.** Also we can expand our training data by making **many small rotations of all the MNIST training images**, and then using the expanded training data to improve our network's performance.

### An aside on big data and what it means to compare classification accuracies ###

Let's look again at how our neural network's accuracy varies with training set size:

![accuracy_vs._training_set_size](http://neuralnetworksanddeeplearning.com/images/more_data_log.png)

Suppose that instead of using a neural network we use some other machine learning technique to classify digits. For instance, let's try using the ***support vector machines (SVM).*** Here's how SVM performance varies as a function of training set size and the neural net results as well, to make comparison easy. You can see that the **neural network outperforms SVM.**

![neural_net_vs._SVM](http://neuralnetworksanddeeplearning.com/images/more_data_comparison.png)

However it' s not the general case. Some algorithm may fits some training data than other algorithm and other training data. So the message to take away, especially in practical applications, is that **what we want is both better algorithms and better training data.** It's fine to look for better algorithms, but make sure you're not focusing on better algorithms to the exclusion of easy wins getting more or better training data.

## Weight initialization ##

When we create our neural networks, we have to make choices for the initial weights and biases. The prescription is to choose both the weights and biases using **independent Gaussian random variables, normalized to have mean** $ 0 $ **and standard deviation** $ 1 $. 

It turns out that we can do quite a bit better than initializing with normalized Gaussians. To see why, let's consider the weighted sum $ z = \sum_j w_j x_j + b $ of inputs to our hidden neuron. With selection of weights to be normalized Gaussians, $ z $ has a very broad Gaussian distribution, **not sharply peaked at all.** Because, let' s say if there is $ 500 $ weight and 1 bias term, total $ 501 $ parameters we have. It means $  \sqrt{501} = 22.4 $ standard deviation. This makes the hidden neurons will have **saturated**, provided that $ \sigma(z) $ very close to either $ 1 $ or  $ 0 $.

![gaussian_mean0_variance22.4](https://lh3.googleusercontent.com/-dyDRsew7UAE/VaVznnqH36I/AAAAAAAAAWc/yC865e0mAwM/s0/gaussian_mean0_variance22.4.png "gaussian_mean0_variance22.4")

Is there some way we can choose better initializations for the weights and biases, so that we don't get this kind of saturation, and so **avoid a learning slowdown**? Suppose we have a neuron with $ n_{in} $ input weights. Then we shall initialize those weights as Gaussian random variables with mean $ 0 $ and standard deviation $ \sqrt \frac{1}{n_{in}} $. That is, we'll **squash the Gaussians down, making it less likely that our neuron will saturate.** We'll continue to choose the bias as a Gaussian with mean $ 0 $ and standard deviation $ 1 $. Suppose, as we did earlier, that $ 500 $ of the inputs are $ 0 $ and $ 500 $ are $ 1 $. Then $ z $ has a Gaussian distribution with mean $ 0 $ and standard deviation $ \sqrt{3/2} = 1.22 $ This is **much more sharply** peaked than before.

![gaussian_sharpy](https://lh3.googleusercontent.com/-O7BhA-gnnVo/VaV1r2ynk4I/AAAAAAAAAWs/VMpO3W7EGEk/s0/gaussian_mean0_variance1.22.png "gaussian_mean0_variance1.22")

Such a neuron is much less likely to saturate, and correspondingly much less likely to have problems with a learning slowdown.

	>>> import mnist_loader
	>>> training_data, validation_data, test_data = 
	mnist_loader.load_data_wrapper()
	>>> import network2
	>>> net = network2.Network([784, 30, 10], 
	cost=network2.CrossEntropyCost)
	>>> net.large_weight_initializer()
	>>> net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, 
	evaluation_data=validation_data, 
	monitor_evaluation_accuracy=True)
	
We can also train using the new approach to weight initialization. This is actually even easier, since network2' s default way of initializing the weights is using this new approach. That means we can omit the net.large_weight_initializer() call above:

	>>> net = network2.Network([784, 30, 10], 
	cost=network2.CrossEntropyCost)
	>>> net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, 
	evaluation_data=validation_data, 
	monitor_evaluation_accuracy=True)

![weight_initializing_comparison](http://neuralnetworksanddeeplearning.com/images/weight_initialization_30.png)

In both cases, we end up with a classification accuracy somewhat over 96 percent. The final classification accuracy is almost exactly the same in the two cases. But the **new initialization technique brings us there much, much faster.** 

## How to choose a neural network's hyper-parameters? ##

### Broad strategy ###

Suppose, for example, that you're attacking MNIST for the first time. You start out enthusiastic, but are a little discouraged when your first network fails completely, as in the example above. The way to go is to strip the problem down. **Get rid of all the training and validation images except images which are 0s or 1s.** Then try to train a network to distinguish 0s from 1s. Not only is that an inherently easier problem than distinguishing all ten digits, it also reduces the amount of training data by 80 percent, **speeding up** training by a factor of 5. That enables much more **rapid experimentation**, and so gives you more rapid insight into how to build a good network.

You can further speed up experimentation by stripping your network down to the simplest network likely to do meaningful learning. If you believe a **[784, 10]** network can likely do better-than-chance classification of MNIST digits, then begin your experimentation with such a network. It'll **be much faster than training a [784, 30, 10]** network, and you can build back up to the latter.

You can get another speed up in experimentation by **increasing the frequency of monitoring.** Of course, a minute isn't really very long to wait for all training, but if you want to trial dozens of hyper-parameter choices it's annoying. We can **get feedback more quickly** by monitoring the validation accuracy more often, **say, after every 1,000 training images.** Furthermore, instead of using the full 10,000 image validation set to monitor performance, we can get a much faster estimate using just **100 validation images.** 

And so we can continue, **individually adjusting each hyper-parameter, gradually improving performance.** Once we've explored to find an improved value for $ \eta $, then we move on to find a good value for $ \lambda $. Then experiment with a more complex architecture, say a network with 10 hidden neurons. Then adjust the values for $ \eta $ and $ \lambda $ again. Then increase to 20 hidden neurons. And then adjust other hyper-parameters some more. 

### Learning rate ###

Suppose we run three MNIST networks with three different learning rates, $ \eta = 0.025 $, $ \eta = 0.25 $ and $ \eta = 2.5 $, respectively. Here's a graph showing the behaviour of the training cost as we train. 

![learning_rates_comparison](http://neuralnetworksanddeeplearning.com/images/multiple_eta.png)

With $ \eta = 0.025 $ the cost **decreases smoothly** until the final epoch. With $ \eta = 0.25 $ the cost initially decreases, but after about 20 epochs it is **near saturation**, and thereafter most of the changes are merely small and apparently random oscillations. Finally, with $ \eta = 2.5 $ the cost makes **large oscillations** right from the start.

Of course, choosing $ \eta $ so small creates another problem, namely, that it **slows down stochastic gradient descent.** An even better approach would be to start with $ \eta = 0.25 $, train for 20 epochs, and then switch to $ \eta = 0.025 $. We'll discuss such variable learning rate schedules later. 

With this picture in mind, we can set $ \eta $ as follows. First, we **estimate the threshold value** for $ \eta $ at which the cost on the training data immediately begins decreasing, instead of oscillating or increasing. This estimate doesn't need to be too accurate. You can estimate the order of magnitude by **starting with** $ \eta = 0.01 $. If the cost decreases during the first few epochs, then you should successively try $ \eta = 0.1, 1.0, … $ **until you find a value for** $ \eta $ **where the cost oscillates or increases during the first few epochs.** Alternately, **if the cost oscillates or increases during the first few epochs when** $ \eta = 0.01 $, **then try** $ \eta = 0.001, 0.0001, … $ **until you find a value for** $ \eta $ **where the cost decreases during the first few epochs.** 

### Use early stopping to determine the number of training epochs ###

As we discussed earlier in the chapter, ***early stopping*** means that at the end of each epoch we should compute the classification accuracy on the validation data. When that stops improving, **terminate.** This makes setting the number of epochs very simple. Furthermore, early stopping also **automatically prevents us from overfitting.** 

### Learning rate schedule ###

We've been holding the learning rate $ \eta $ constant. However, it's often advantageous to vary the learning rate. Early on during the learning process it's likely that the weights are badly wrong. And so it's best to **use a large learning rate that causes the weights to change quickly.** 

How should we set our learning rate schedule? Many approaches are possible. The idea is to **hold the learning rate constant until the validation accuracy starts to get worse.** Then decrease the learning rate by some amount, say a factor of two or ten. We repeat this many times, until, say, the learning rate is a factor of 1,024 (or 1,000) times lower than the initial value. Then we **terminate.**

### The regularization parameter ###

The suggestion is starting initially with no regularization $ \lambda = 0.0 $, and determining a value for $ \eta $, as above. Using that choice of $ \eta $, we can then use the validation data to select a good value for $ \lambda $. Start by trialling let' s say $ \lambda = 1.0 $ and then increase or decrease by factors of 10, as needed to improve performance on the validation data. 

### Mini-batch size ###

Let' s first suppose that we're doing ***online learning***, i.e., that we're **using a mini-batch size of 1.**

The obvious worry about online learning is that using mini-batches which contain just a single training example **will cause significant errors in our estimate of the gradient.** In fact, though, the errors turn out to not be such a problem. The reason is that the **individual gradient estimates don't need to be super-accurate.** All we need is an estimate accurate enough that our cost function tends to keep decreasing. 

>It's as though you are trying to get to the North Magnetic Pole, but have a wonky compass that's 10-20 degrees off each time you look at it. Provided you stop to check the compass frequently, and the compass gets the direction right on average, you'll end up at the North Magnetic Pole just fine.

**Summing up:** Keep in mind that the heuristics described here are rules of thumb, not rules cast in stone. You should be on the lookout for signs that things aren't working, and be willing to experiment. 

One thing that becomes clear as you read these articles and, especially, as you engage in your own experiments, is that **hyper-parameter optimization is not a problem that is ever completely solved.** There's always another trick you can try to improve performance. 

## Other techniques ##

### Variations on stochastic gradient descent ###

#### Hessian technique ####

To begin our discussion it helps to put neural networks aside for a bit. Instead, we're just going to consider the abstract problem of minimizing a cost function $ C $ which is a function of many variables, $ w = w1, w2, …, $ so $ C = C(w) $. By Taylor's theorem, the cost function can be approximated near a point $ w $ by

\begin{equation}
	\begin{split}
		C(w + \Delta w) =  C&(w)  + \sum_j \frac{\partial C}{\partial w_j} 
		\Delta w_j \\ & + \frac{1}{2}\sum_{jk} \Delta w_j 
		\frac{\partial^2 C}{\partial w_j \partial w_k} \Delta w_k + ...
		\hspace{2.5 cm} (27)
	\end{split}
\end{equation}

We can rewrite this more compactly as

\begin{equation}
	C(w + \Delta w) = C(w) + \nabla C \cdot \Delta w + 
	\frac{1}{2} {\Delta w}^T H \Delta w + …
	\hspace{1.5 cm} (28)
\end{equation}

where $ \nabla C $ is the usual gradient vector, and $ H $ is a matrix known as the ***Hessian matrix***, whose $ jk^{th} $ entry is $ \partial^2 C / \partial w_j \partial w_k $. Suppose we approximate $ C $ by discarding the higher-order terms represented by $ … $ above,

\begin{equation}
	C(w + \Delta w) \approx C(w) + \nabla C \cdot \Delta w + 
	\frac{1}{2} {\Delta w}^T H \Delta w
	\hspace{2.5 cm} (29)
\end{equation}

Using calculus we can show that the expression on the right-hand side can be minimized by choosing

\begin{equation}
	\Delta w = −H^{−1} \nabla C
	\hspace{2.5 cm} (30)
\end{equation}

Provided (29) is a good approximate expression for the cost function, then we'd expect that moving from the point $ w $ to $ w + \Delta w = w − H^{−1} \nabla C $ should significantly decrease the cost function. That suggests a possible algorithm for minimizing the cost:

* Choose a starting point, $ w $.
* Update $ w $ to a new point $ w′ = w − H^{−1} \nabla C $, where the Hessian $ H $ and $ \nabla C $ are computed at $ w $.
* Update $ w′ $ to a new point $ w′′ = w′ − H′^{−1} \nabla′ C $, where the Hessian $ H′ $ and $ \nabla′ C $ are computed at $ w′ $.
* …

In practice, (29) is only an approximation, and it's better to take smaller steps. We do this by repeatedly changing $ w $ by an amount $ Δ\Delta w = −\eta H^{−1} \nabla C $, where $ \eta $ is known as the learning rate.

This approach to minimizing a cost function is known as the ***Hessian technique or Hessian optimization.*** There are theoretical and empirical results showing that Hessian methods **converge on a minimum in fewer steps than standard gradient descent.** In particular, by incorporating information about second-order changes in the cost function it's possible for the Hessian approach to avoid many pathologies that can occur in gradient descent. Furthermore, there are versions of the backpropagation algorithm which can be used to compute the Hessian.

Unfortunately, while it has many desirable properties, it has one very undesirable property: the **Hessian matrix is very big.** Suppose you have a neural network with $ 10^7 $ weights and biases. Then the corresponding Hessian matrix will contain $ 10^7 \times 10^7 = 10^{14} $ entries. That's a lot of entries to compute, especially when you're going to need to invert the matrix as well! That makes Hessian optimization difficult to apply in practice. 

#### Momentum-based gradient descent ####

Unlike Hessian technique momentum-based gradient descent **avoids large matrices of second derivatives.** The momentum technique modifies gradient descent in two ways that make it more similar to the physical picture. First, it introduces a notion of **"velocity"** for the parameters we're trying to optimize. The gradient acts to change the velocity, **not (directly) the "position"**, in much the same way as physical forces change the velocity, and only **indirectly affect position.** Second, the **momentum method introduces a kind of friction term,** which tends to **gradually reduce the velocity.**

We introduce velocity variables $ v = v_1, v_2, … $ one for each corresponding $ w_j $ variable. Then we replace the gradient descent update rule $ w \to w′ = w − \eta \nabla C $ by

\begin{equation}
	\begin{split}
		v & \to \mu v - \eta \nabla C \hspace{2.5 cm} (31) \\  
		w & \to  w + v' \hspace{2.5 cm} (32) 
	\end{split}
\end{equation}

In these equations, $ \mu $ is a hyper-parameter which **controls the amount of damping or friction in the system.** To understand the meaning of the equations it's helpful to first consider the case where $ \mu = 1 $, which corresponds to **no friction.** When that's the case, inspection of the equations shows that the **"force"** $ \nabla C $ is now modifying the velocity, $ v $, and the velocity is controlling the rate of change of $ w $. Intuitively, we **build up the velocity by repeatedly adding gradient terms to it.** That means that if the gradient is in (roughly) the same direction through several rounds of learning, we can build up quite a bit of steam moving in that direction. Think, for example, of what happens if we're moving straight down a slope. With **each step the velocity gets larger down the slope, so we move more and more quickly to the bottom of the valley.** This can enable the **momentum technique to work much faster than standard gradient descent.** Of course, a problem is that **once we reach the bottom of the valley we will overshoot.** Or, if the gradient should change rapidly, then we could find ourselves moving in the wrong direction. That's the reason for the $ \mu $ hyper-parameter in (31). To be a little more precise, you should think of $ 1 − \mu $ as the amount of friction in the system. When $ \mu = 1 $, as we've seen, there is no friction, and the velocity is completely driven by the gradient $ \nabla C $. By contrast, when $ \mu = 0 $ there's a lot of friction, the velocity can't build up, and Equations (31) and (31) reduce to the usual equation for gradient descent, $ w\to w′ = w − \eta \nabla C $

It' s avoided naming the hyper-parameter $ \mu $ up to now. The reason is that the standard name for $ \mu $ is badly chosen: it's called the ***momentum co-efficient***. This is potentially confusing, since $ \mu $ is not at all the same as the notion of momentum from physics. Rather, it is much more **closely related to friction.** 

A nice thing about the momentum technique is that it takes **almost no work to modify an implementation of gradient descent** to incorporate momentum. We can still use backpropagation to compute the gradients, just as before, and use ideas such as sampling stochastically chosen mini-batches. In this way, we can get some of the advantages of the Hessian technique, using information about how the gradient is changing. In practice, the momentum technique is commonly used, and **often speeds up learning.**


## Other models of artificial neuron ##

Up to now we've built our neural networks using sigmoid neurons. In practice, networks built using other model neurons sometimes outperform sigmoid networks. 

Perhaps the simplest variation is the $ \tanh $ (pronounced "tanch") neuron, which replaces the sigmoid function by the ***hyperbolic tangent function***. The output of a $ \tanh $ neuron with input $ x $, weight vector $ w $, and bias $ b $ is given by

\begin{equation}
	\tanh(w \cdot x + b) \hspace{2.5 cm} (33) 
\end{equation}

Recall that the $ \tanh $ function is defined by

\begin{equation}
	\tanh(z) \equiv \frac{e^z − e^{−z}}{e^z + e^{−z}} \hspace{2.5 cm} (34) 
\end{equation}

With a little algebra it can easily be verified that

\begin{equation}
	\sigma(z) = \frac{1 + \tanh(z / 2)}{2} \hspace{2.5 cm} (35) 
\end{equation}

that is, $ \tanh $ is just a **rescaled version of the sigmoid function.** We can also see graphically that the $ \tanh $ function has the same shape as the sigmoid function,

![tanh](https://lh3.googleusercontent.com/-WdNc0bJLb2I/VaWzLYtYaQI/AAAAAAAAAXE/U_PDsxTK3-c/s0/tanh_function.png "tanh_function")

One difference between tanh neurons and sigmoid neurons is that the **output from tanh neurons ranges from** $ -1 $ to $ 1 $, not $ 0 $ to $ 1 $. This means that if you're going to build a network based on tanh neurons you may need to **normalize your outputs.**

If some of the input activations have different signs, replace the sigmoid by an activation function, such as $ \tanh $, which allows both positive and negative activations. Indeed, because $ \tanh $ is **symmetric about zero**, $ \tanh(−z) = −tanh(z) $. 

Another variation on the sigmoid neuron is the ***rectified linear neuron or rectified linear unit***. The output of a rectified linear unit with input $ x $, weight vector $ w $, and bias $ b $ is given by

\begin{equation}
	\max(0, w \cdot x + b) \hspace{2.5 cm} (36) 
\end{equation}

Graphically, the rectifying function $ \max(0, z) $ looks like this:

![rectifying_function](https://lh3.googleusercontent.com/-4GxP-qqhtBo/VaW3--JDLZI/AAAAAAAAAXU/EgWAXlQqNEw/s0/rectifying_function.png "rectifying_function")

Like the sigmoid and tanh neurons, rectified linear units can be used to **compute any function**, and they can be trained using ideas such as backpropagation and stochastic gradient descent.

