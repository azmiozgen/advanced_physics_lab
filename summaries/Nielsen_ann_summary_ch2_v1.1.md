How the backpropagation algorithm works (Nielsen, Chapter 2)
====================================================================

The backpropagation algorithm was originally **introduced in the 1970s,** but its importance **wasn't fully appreciated until a famous 1986 paper by David Rumelhart, Geoffrey Hinton, and Ronald Williams.** That paper describes several neural networks where backpropagation works far faster than earlier approaches to learning, **making it possible to use neural nets to solve problems which had previously been insoluble.** Today, the backpropagation algorithm is the **workhorse of learning** in neural networks.

**This chapter is more mathematically involved** than the rest of the book. The reason, of course, is understanding. At the heart of backpropagation is an expression for the partial derivative `$ \frac {\partial C}{\partial w} $` of the cost function `$ C $` with respect to any weight `$ w $` (or bias `$ b $`) in the network. The **expression tells us how quickly the cost changes when we change the weights and biases.** 

## Warm up: a fast matrix-based approach to computing the output from a neural network ##

We'll use `$ w^l_{jk} $` to denote the weight for the connection from the `$ k^{th} $` neuron in the `$ (l − 1)^{th} $` layer to the `$ j^{th} $` neuron in the `$ l^{th} $` layer. So, for example, the diagram below shows the weight on a connection from the **fourth neuron in the second layer to the second neuron in the third layer** of a network:

![weight_notation](http://neuralnetworksanddeeplearning.com/images/tikz16.png)

We use a similar notation for the network's biases and activations. Explicitly, we use `$ b^l_j $` for the bias of the `$ j^{th} $` neuron in the `$ l^{th} $` layer. And we use `$ a^l_j $` for the activation of the `$ j^{th} $` neuron in the `$ l^{th} $` layer. The following diagram shows examples of these notations in use:

![bias_and_activation_notation](http://neuralnetworksanddeeplearning.com/images/tikz17.png)

The activation `$ a^l_j $` of the `$ j^{th} $` neuron in the `$ l^{th} $` layer is related to the activations in the `$ (l-1)^{th} $` layer by the equation

`
\begin{equation}
	a^l_j = σ(\sum_k w^l_{jk} a^{l−1}_k + b^l_j) \hspace{5 cm}(1)
\end{equation}
`

where the sum is over all neurons `$ k $` in the `$ (l-1)^{th} $` layer. Equation (1) can be rewritten in the beautiful and compact **vectorized form**

`
\begin{equation}
	a^l = σ(w^l \cdot a^{l−1}+b^l) \hspace{5 cm}(2)
\end{equation}
`

This expression gives us a much more global way of thinking about how the activations in one layer relate to activations in the previous layer: we just apply the weight matrix to the activations, then add the bias vector, and finally apply the `$ σ $` function. The expression is also useful in practice, because **most matrix libraries provide fast ways of implementing matrix multiplication, vector addition, and vectorization.** 

## The two assumptions we need about the cost function

The goal of backpropagation is to compute the partial derivatives `$ \partial C / \partial w $` and `$ \partial C / \partial b $` of the cost function `$ C $` with respect to any weight `$ w $` or bias `$ b $` in the network. For backpropagation to work we need to make **two main assumptions about the form of the cost function.** We'll use the quadratic cost function from last chapter. In the notation of the last section, the quadratic cost has the form

`
\begin{equation}
	C = \frac{1}{2n} \sum_x || y(x) − a^L(x)||^2 \hspace{5 cm}(3)
\end{equation}
`

where `$ n $` is the total number of training examples; the sum is over individual training examples, `$ x $`; `$ y = y(x) $` is the corresponding desired output; `$ L $` denotes the number of layers in the network; and `$ a^L = a^L(x) $` is the vector of activations output from the network when `$ x $` is input.

The first assumption we need is that the **cost function can be written as an average** `$ C = \frac {1}{n} \sum_x C_x $` over cost functions `$ C_x $` for individual training examples, `$ x $`. This is the case for the quadratic cost function, where the cost for a single training example is `$ C_x = \frac {1}{2} ||y − a^L||^2 $`. 

The second assumption we make about the cost is that it can be written as a function of the outputs from the neural network:

![output_cost](http://neuralnetworksanddeeplearning.com/images/tikz18.png)

For example, the quadratic cost function satisfies this requirement, since the quadratic cost for a single training example `$ x $` may be written as

`
\begin{equation}
	C = \frac{1}{2}||y − a^L||^2 = \frac{1}{2} \sum_j (y_j − 
	a^L_j)^2 \hspace{5 cm}(4)
\end{equation}
`

and thus is a **function of the output activations.** Remember, though, that the input training example `$ x $` is fixed, and so the output `$ y $` is also a **fixed parameter**. In particular, it's not something we can modify by changing the weights and biases in any way, i.e., it's **not something which the neural network learns.** And so it makes sense to regard `$ C $` as a **function of the output activations** `$ a^L $` **alone**, with `$ y $` merely a parameter that helps define that function.

## The Hadamard product, `$ s \odot t $`

Suppose `$ s $` and `$ t $` are two vectors of the same dimension. Then we use `$ s \odot t $` to denote the ***elementwise product*** of the two vectors. Thus the components of `$ s \odot t $` are just `$ (s \odot t)_j = s_j t_j $`. As an example,

`
\begin{equation}
	\left[ \begin{array}{c} 1 \\ 2 \end{array} \right] \odot 
	\left[ \begin{array}{c} 3 \\ 4 \end{array} \right] = \left[ 
	\begin{array}{c} 1 \times 3 \\ 2 \times 4 \end{array} \right] = 
	\left[ \begin{array}{c} 3 \\ 8 \end{array} \right] \hspace{5 cm}(5)
\end{equation}
`

This kind of elementwise multiplication is sometimes called the ***Hadamard product*** or ***Schur product***. Good matrix libraries usually provide fast implementations of the Hadamard product, and that comes in handy when implementing backpropagation.

## The four fundamental equations behind backpropagation ##

We first introduce an intermediate quantity, `$ \delta_j^l $`, which we call the error in the `$ j^{th} $` neuron in the `$ l^{th} $` layer. Backpropagation will give us a procedure to compute the error `$ \delta_j^l $`, and then will relate it to `$ \partial C / \partial w^l_{jk} $` and `$ \partial C / \partial b^l_j $`.

We define the error `$ \delta^l_j $`  of neuron `$ j $` in layer `$ l $` by

`
\begin{equation}
	\delta^l_j \equiv \frac {\partial C}{\partial z^l_j} \hspace{5 cm}(6)
\end{equation}
`

**Backpropagation is based around four fundamental equations.** Together, those equations give us a way of computing both the error `$ \delta^l $` and the gradient of the cost function. 

**An equation for the error in the output layer,** `$ \delta^L $`: The components of `$ \delta^L $` are given by

`
\begin{equation}
	\delta^L_j = \frac {\partial C}{\partial a^L_j} \sigma^\prime 
	(z^L_j) \hspace{5 cm}(7)
\end{equation}
`

This is a very natural expression. The first term on the right, `$ \partial C / \partial a^L_j $`, just measures how fast the cost is changing as a function of the `$ j^{th} $` output activation. If, for example, `$ C $` doesn't depend much on a particular output neuron, `$ j $`, then `$ \delta^L_j $` will be small, which is what we'd expect. The second term on the right, `$ \sigma^\prime(z^L_j) $`, measures how fast the activation function `$ \sigma $` is changing at `$ z^L_j $`. If we're using the quadratic cost function then `$ C = \frac {1}{2} \sum_j (y_j − a_j)^2 $`, and so `$ \partial C / \partial a^L_j = (a_j − y_j) $`, which obviously is **easily computable.**

It's easy to rewrite the Equation (7) in a matrix-based form, as

`
\begin{equation}
	\delta^L = (a^L - y) \odot \sigma^\prime (z^L) \hspace{5 cm}(8)
\end{equation}
`

**An equation for the error** `$ \delta^l $` **in terms of the error in the next layer,** `$ \delta_l + 1 $`: In particular

`
\begin{equation}
	\delta^l = ((w^{l + 1})^T \delta^{l + 1}) \odot \sigma^\prime (z^l) 
	\hspace{5 cm}(9)
\end{equation}
`

We can think intuitively of this as **moving the error backward through the network,** giving us some sort of measure of the error at the output of the `$ l^{th} $` layer. 

By combining (8) with (9) we can compute the error `$ \delta^l $` for any layer in the network. We start by using (8) to compute `$ \delta^L $`, then apply Equation (9) to compute `$ \delta^{L − 1} $`, then `$ \delta^{L − 2} $`, and so on, **all the way back through the network.**

**An equation for the rate of change of the cost with respect to any bias in the network:** In particular:

`
\begin{equation}
	\frac {\partial C}{\partial b^l_j} = \delta^l_j
	\hspace{5 cm}(10)
\end{equation}
`

or in shorthand as

`
\begin{equation}
	\frac {\partial C}{\partial b} = \delta
	\hspace{5 cm}(11)
\end{equation}
`

where it is understood that `$ \delta $` is being **evaluated at the same neuron as the bias** `$ b $`.

**An equation for the rate of change of the cost with respect to any weight in the network:** In particular:

`
\begin{equation}
	\frac {\partial C}{\partial w^l_{jk}} = a^{l − 1}_k\delta^l_j
	\hspace{5 cm}(12)
\end{equation}
`

This tells us how to compute the partial derivatives `$ \partial C / \partial w^l_{jk} $` in terms of the quantities `$ \delta^l $` and `$ a^{l − 1} $`, which we already know how to compute. The equation can be rewritten in a **less index-heavy notation** as

`
\begin{equation}
	\frac {\partial C}{\partial w} = a_{in} \delta_{out}
	\hspace{5 cm}(13)
\end{equation}
`

where it's understood that `$ a_{in} $` is the activation of the neuron input to the weight `$ w $`, and `$ \delta_{out} $` is the error of the neuron output from the weight `$ w $`. 

A nice consequence of Equation (13) is that **when the activation** `$ a_{in} $` **is small,** `$ a_{in} \approx 0 $`, **the gradient term** `$ \partial C / \partial w $` **will also tend to be small.** In this case, we'll say the **weight learns slowly,** meaning that it's not changing much during gradient descent. In other words, one consequence of (13) is that **weights output from low-activation neurons learn slowly.** 

Recall from the graph of the sigmoid function in the last chapter that the `$ \sigma $` function becomes **very flat** when `$ \sigma (z^L_j) $` is approximately 0 or 1. When this occurs we will have `$ \sigma^\prime (z^L_j) \approx 0 $`. And so the lesson is that **a weight in the final layer will learn slowly if the output neuron is either low activation** `$ (\approx 0) $` **or high activation** `$ (\approx 1) $`.

Summing up, we've learnt that **a weight will learn slowly if either the input neuron is low-activation, or if the output neuron has saturated, i.e., is either high- or low-activation.**

The ***four fundamental equations*** turn out to **hold for any activation function, not just the standard sigmoid function** (that's because, the **proofs don't use any special properties of σ**). And so we can use these equations to **design activation functions** which have particular desired learning properties. As an example to give you the idea, **suppose we were to choose a (non-sigmoid) activation function** `$ \sigma $` **so that** `$ \sigma^\prime $` **is always positive, and never gets close to zero.** That would **prevent the slow-down of learning that occurs when ordinary sigmoid neurons saturate.** 

![four_fundamental_eqs_of_backprop](http://neuralnetworksanddeeplearning.com/images/tikz21.png)

The proof of four fundamental equations of backpropagation is just coming from systematically applying the **chain rule** to the cost function so that we may be able to get **gradients** of it.

## The backpropagation algorithm ##

The backpropagation equations provide us with a way of computing the gradient of the cost function. Let's explicitly write this out in the form of an algorithm:

1. **Input** `$ x $`: Set the corresponding activation `$ a^1 $` for the input layer.

2. **Feedforward**: For each `$ l = 2, 3, …, L $` compute `$ z^l = w^l a^{l − 1} + b^l $` and `$ a^l = \sigma (z^l) $`.

3. **Output error** `$ \delta^L $`: Compute the vector `$ \delta^L = \nabla_a C \odot \sigma^\prime (z^L) $`.

4. **Backpropagate the error**: For each `$ l = L − 1, L − 2, …, 2 $` compute `$ \delta^l = ((w^{l + 1})^T \delta^{l + 1})\odot \sigma^\prime (z^l) $`.

5. **Output**: The gradient of the cost function is given by `$ \frac{\partial C}{\partial w^l_{jk}} = a^{l − 1}_k \delta^l_j $` and `$ \frac{\partial C}{\partial b^l_j} = \delta^l_j $`.

Examining the algorithm you can see **why it's called backpropagation.** We compute the **error vectors** `$ \delta^l $` **backward, starting from the final layer.** The backward movement is a consequence of the fact that the **cost is a function of outputs** from the network. To understand how the cost varies with earlier weights and biases we need to **repeatedly apply the chain rule,** working backward through the layers to obtain usable expressions.

In practice, it's common to **combine backpropagation with a learning algorithm such as stochastic gradient descent,** in which we compute the gradient for many training examples. In particular, given a mini-batch of `$ m $` training examples, the following algorithm applies a gradient descent learning step based on that mini-batch:

1. **Input a set of training examples**

2. **For each training example** `$ x $`: Set the corresponding input activation `$ a^{x, 1} $`, and perform the following steps:

- **Feedforward**: For each `$ l = 2, 3, …, L $` compute `$ z^{x, l} = w^l a^{x,l − 1} + b^l $` and `$ a^{x, l} = \sigma (z^{x, l}) $` .

- **Output error** `$ \delta^{x, L} $`: Compute the vector `$ \delta^{x, L} = \nabla_a C_x \odot \sigma^\prime(z^{x, L}) $`.

- **Backpropagate the error**: For each `$ l = L − 1, L − 2, …, 2 $` compute `$ \delta^{x, l} = ((w^{l + 1})^T \delta^{x, l + 1}) \odot \sigma^\prime (z^{x, l}) $`.

3. **Gradient descent**: For each `$ l = L, L − 1, …, 2 $` update the weights according to the rule `$ w^l \to w^l − \frac{\eta}{m}\sum_x\delta^{x, l} (a^{x, l − 1})^T $`, and the biases according to the rule `$ b^l \to b^l − \frac{\eta}{m} \sum_x \delta^{x, l} $`.

Of course, to implement stochastic gradient descent in practice you also need an **outer loop** generating mini-batches of training examples, and an **outer loop stepping through multiple epochs of training.** 

## In what sense is backpropagation a fast algorithm? ##

`
\begin{equation}
	\frac {\partial C}{\partial w_j} \approx 
	\frac {C(w + \epsilon e_j) - C(w)}{\epsilon}
	\hspace{5 cm}(14)
\end{equation}
`

where `$ \epsilon > 0 $` is a small positive number, and `$ e_j $` is the unit vector in the `$ j^{th} $` direction. In other words, we can estimate `$ \partial C / \partial w_j $` by computing the cost `$ C $` for two slightly different values of `$ w_j $`, and then applying Equation (14). The same idea will let us compute the partial derivatives `$ \partial C / \partial b $` with respect to the biases.

This approach looks very **promising.** It's simple conceptually, and extremely easy to implement, using just a few lines of code. Certainly, it **looks much more promising than the idea of using the chain rule to compute the gradient!**

Unfortunately, while this approach appears promising, when you implement the code it turns out to be **extremely slow.** To understand why, imagine we have a **million weights** in our network. Then for each distinct weight `$ w_j $` we need to compute `$ C(w + \epsilon e_j) $` in order to compute `$ \partial C / \partial w_j $`. That means that **to compute the gradient we need to compute the cost function a million different times.** 

What's clever about backpropagation is that it enables us to simultaneously compute all the partial derivatives `$ \partial C / \partial w_j $` using **just one forward pass** through the network, followed by **one backward pass** through the network. Roughly speaking, the **computational cost of the backward pass is about the same as the forward pass.** And so the **total cost of backpropagation is roughly the same as making just two forward passes** through the network.

## Backpropagation: the big picture ##

To improve our intuition about what the algorithm is really doing, let's imagine that we've **made a small change** `$ \Delta w^l_{jk} $` **to some weight** in the network, `$ w^l_{jk} $`:

![delta_change_in_weight](http://neuralnetworksanddeeplearning.com/images/tikz22.png)

That **change in weight will cause a change in the output activation from the corresponding neuron:**

![delta_change_in_neuron](http://neuralnetworksanddeeplearning.com/images/tikz23.png)

That, in turn, will **cause a change in all the activations in the next layer:**

![change_in_the_next_layer](http://neuralnetworksanddeeplearning.com/images/tikz24.png)

Those changes will in turn cause changes in the next layer, and then the next, and so on **all the way through to causing a change in the final layer, and then in the cost function:**

![delta_change_in_cost_function](http://neuralnetworksanddeeplearning.com/images/tikz25.png)

The **change** `$ \Delta C $` **in the cost is related to the change** `$ \Delta w^l_{jk} $` **in the weight** by the equation

`
\begin{equation}
	\Delta C \approx 
	\frac {\partial C}{\partial w^l_{jk}} \Delta w^l_{jk} 
	\hspace{5 cm}(15)
\end{equation}
`

Being careful to express everything along the way in terms of easily computable quantities, then we should be able to compute `$ \partial C / \partial w^l_{jk} $`.

Let's try to carry this out. The **change** `$ \Delta w^l_{jk} $` **causes a small change** `$ \Delta a^l_j $` **in the activation of the** `$ j^{th} $` **neuron in the** `$ l^{th} $` **layer.** This change is given by

`
\begin{equation}
	\Delta a^l_j \approx 
	\frac {\partial a^l_j}{\partial w^l_{jk}} \Delta w^l_{jk} 
	\hspace{5 cm}(16)
\end{equation}
`

The **change in activation** `$ \Delta a^l_j $` **will cause changes in all the activations in the next layer, i.e., the** `$ (l + 1)^{th} $` **layer.** We'll concentrate on the way just a single one of those activations is affected, say `$ a^{l + 1}_q $`,

![affected_activation](http://neuralnetworksanddeeplearning.com/images/tikz26.png)

In fact, it'll cause the following change:

`
\begin{equation}
	\Delta a^{l + 1}_q \approx 
	\frac {\partial a^{l + 1}_q}{\partial a^l_j} 
	\Delta a^l_j 
	\hspace{5 cm}(17)
\end{equation}
`

Substituting in the expression from Equation (16), we get:

`
\begin{equation}
	\Delta a^{l + 1}_q \approx 
	\frac {\partial a^{l + 1}_q}{\partial a^l_j} 
	\frac {\partial a^l_j}{\partial w^l_{jk}} 
	\Delta w^l_{jk} 
	\hspace{5 cm}(18)
\end{equation}
`

Of course, the change `$ \Delta a^{l + 1}_q $` will, in turn, cause changes in the activations in the next layer. In fact, we can imagine a path all the way through the network from `$ w^l_{jk} $` to `$ C $`, with **each change in activation causing a change in the next activation, and, finally, a change in the cost at the output.** If the path goes through activations `$ a^l_j, a^{l + 1}_q, …,a^{L − 1}_n, a^L_m $` then the resulting expression is

`
\begin{equation}
	\Delta C \approx 
	\frac {\partial C}{\partial a^l_m} 
	\frac {\partial a^L_m}{\partial a^{L - 1}_n} 
	\frac {\partial a^{L - 1}_n}{\partial a^{L - 2}_p} ...
	\frac {\partial a^{l + 1}_q}{\partial a^l_j}
	\frac {\partial a^l_j}{\partial w^l_{jk}}
	\Delta w^l_{jk} 
	\hspace{5 cm}(19)
\end{equation}
`

Of course, there's **many paths** by which a change in `$ w^l_{jk} $` **can propagate to affect the cost,** and **we've been considering just a single path.** To compute the **total change in** `$ C $` it is plausible that we should **sum over all the possible paths between the weight and the final cost,** i.e.,

`
\begin{equation}
	\Delta C \approx 
	\sum_{mnp...q}
	\frac {\partial C}{\partial a^l_m} 
	\frac {\partial a^L_m}{\partial a^{L - 1}_n} 
	\frac {\partial a^{L - 1}_n}{\partial a^{L - 2}_p} ...
	\frac {\partial a^{l + 1}_q}{\partial a^l_j}
	\frac {\partial a^l_j}{\partial w^l_{jk}}
	\Delta w^l_{jk} 
	\hspace{5 cm}(20)
\end{equation}
`

Comparing with (15) we see that

`
\begin{equation}
	\frac {\partial C}{\partial w^l_{jk}} \approx 
	\sum_{mnp...q}
	\frac {\partial C}{\partial a^l_m} 
	\frac {\partial a^L_m}{\partial a^{L - 1}_n} 
	\frac {\partial a^{L - 1}_n}{\partial a^{L - 2}_p} ...
	\frac {\partial a^{l + 1}_q}{\partial a^l_j}
	\frac {\partial a^l_j}{\partial w^l_{jk}}
	\Delta w^l_{jk} 
	\hspace{5 cm}(21)
\end{equation}
`

What the equation tells us is that **every edge between two neurons in the network is associated with a rate factor** which is just the **partial derivative of one neuron's activation with respect to the other neuron's activation.** 

![sum_of_rate_factors](http://neuralnetworksanddeeplearning.com/images/tikz27.png)

What I've been providing up to now is a **heuristic argument, a way of thinking about what's going on when you perturb a weight in a network.** Let me sketch out a line of thinking you could use to further develop this argument. And so you can think of the backpropagation algorithm as **providing a way of computing the sum over the rate factor for all these paths.** Or, to put it slightly differently, the backpropagation algorithm is a **clever way of keeping track of small perturbations to the weights (and biases) as they propagate through the network, reach the output, and then affect the cost.**
