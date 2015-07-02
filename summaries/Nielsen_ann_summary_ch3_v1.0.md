Improving the way neural networks learn (Nielsen, Chapter 3)
====================================================================

In this chapter it's explained a **suite of techniques** which can be used to **improve the implementation of backpropagation,** and so **improve the way our networks learn.**

The techniques we'll develop in this chapter include: a **better choice of cost function, known as the *cross-entropy cost function*;** four ***"regularization" methods*** **(L1 and L2 regularization, dropout, and artificial expansion of the training data),** which make our networks better at **generalizing beyond the training data;** a **better method for initializing the weights** in the network; and **a set of heuristics to help choose good hyper-parameters for the network.** 

## The cross-entropy cost function ##

> Soon after beginning to learn the piano I gave my first performance before an audience. I was nervous, and began playing the piece an octave too low. I got confused, and couldn't continue until someone pointed out my error. I was very embarassed. Yet while unpleasant, **we also learn quickly when we're decisively wrong.** You can bet that the next time I played before an audience I played in the correct octave! By contrast, **we learn more slowly when our errors are less well-defined.**

Ideally, we hope and expect that our neural networks will learn fast from their errors. Is this what happens in practice? To answer this question, let's look at a toy example. The example involves a neuron with just one input:


We'll train this neuron to do something ridiculously easy: take the input 1 to the output 0. Of course, this is such a trivial task that we could easily figure out an appropriate weight and bias by hand, without using a learning algorithm. However, it turns out to be illuminating to use gradient descent to attempt to learn a weight and bias. So let's take a look at how the neuron learns.

To make things definite, I'll pick the initial weight to be 0.6 and the initial bias to be 0.9. These are generic choices used as a place to begin learning, I wasn't picking them to be special in any way. The initial output from the neuron is 0.82, so quite a bit of learning will be needed before our neuron gets near the desired output, 0.0. Click on "Run" in the bottom right corner below to see how the neuron learns an output much closer to 0.0. Note that this isn't a pre-recorded animation, your browser is actually computing the gradient, then using the gradient to update the weight and bias, and displaying the result. The learning rate is η=0.15, which turns out to be slow enough that we can follow what's happening, but fast enough that we can get substantial learning in just a few seconds. The cost is the quadratic cost function, C, introduced back in Chapter 1. I'll remind you of the exact form of the cost function shortly, so there's no need to go and dig up the definition. Note that you can run the animation multiple times by clicking on "Run" again.

As you can see, the neuron rapidly learns a weight and bias that drives down the cost, and gives an output from the neuron of about 0.09. That's not quite the desired output, 0.0, but it is pretty good. Suppose, however, that we instead choose both the starting weight and the starting bias to be 2.0. In this case the initial output is 0.98, which is very badly wrong. Let's look at how the neuron learns to output 0 in this case. Click on "Run" again:


Although this example uses the same learning rate (η=0.15), we can see that learning starts out much more slowly. Indeed, for the first 150 or so learning epochs, the weights and biases don't change much at all. Then the learning kicks in and, much as in our first example, the neuron's output rapidly moves closer to 0.0.

This behaviour is strange when contrasted to human learning. As I said at the beginning of this section, we often learn fastest when we're badly wrong about something. But we've just seen that our artificial neuron has a lot of difficulty learning when it's badly wrong - far more difficulty than when it's just a little wrong. What's more, it turns out that this behaviour occurs not just in this toy model, but in more general networks. Why is learning so slow? And can we find a way of avoiding this slowdown?

To understand the origin of the problem, consider that our neuron learns by changing the weight and bias at a rate determined by the partial derivatives of the cost function, ∂C/∂w and ∂C/∂b. So saying "learning is slow" is really the same as saying that those partial derivatives are small. The challenge is to understand why they are small. To understand that, let's compute the partial derivatives. Recall that we're using the quadratic cost function, which, from Equation (6), is given by
C=(y−a)22,(54)
where a is the neuron's output when the training input x=1 is used, and y=0 is the corresponding desired output. To write this more explicitly in terms of the weight and bias, recall that a=σ(z), where z=wx+b. Using the chain rule to differentiate with respect to the weight and bias we get
∂C∂w∂C∂b==(a−y)σ′(z)x=aσ′(z)(a−y)σ′(z)=aσ′(z),(55)(56)
where I have substituted x=1 and y=0. To understand the behaviour of these expressions, let's look more closely at the σ′(z) term on the right-hand side. Recall the shape of the σ function:

-4
-3
-2
-1
0
1
2
3
4
0.0
0.2
0.4
0.6
0.8
1.0
z
sigmoid function
We can see from this graph that when the neuron's output is close to 1, the curve gets very flat, and so σ′(z) gets very small. Equations (55) and (56) then tell us that ∂C/∂w and ∂C/∂b get very small. This is the origin of the learning slowdown. What's more, as we shall see a little later, the learning slowdown occurs for essentially the same reason in more general neural networks, not just the toy example we've been playing with.

Introducing the cross-entropy cost function

