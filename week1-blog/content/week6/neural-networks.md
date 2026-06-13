# Neural Networks

> Week 6: Neural Networks I · MIT 6.036 courseware archive

## Notes – Chapter 8: Neural Networks

Notes – Chapter 8: Neural Networks
You can sequence through the Neural Networks lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 8: Neural Networks
notes as a PDF file.

## Lecture: Neural networks - basic element

Lecture: Neural networks - basic element
Lecture: Neural networks - basic element
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Introduction to neural networks

Introduction to neural networks
Unless you live under a rock with no internet access, you've been hearing a lot about “neural networks." Now that we have several useful machine-learning concepts (hypothesis classes, classification, regression, gradient descent, regularization, etc.) we are completely well equipped to understand neural networks in detail.
This, in some sense, the “third wave" of neural nets. The basic idea is founded on the 1943 model of neurons of McCulloch and Pitts and learning ideas of Hebb. There was a great deal of excitement, but not a lot of practical success: there were good training methods (e.g., perceptron) for linear functions, and interesting examples of non-linear functions, but no good way to train non-linear functions from data. Interest died out for a while, but was re-kindled in the 1980s
As with many good ideas in science, the basic idea for how to train non-linear neural networks with gradient descent, was independently developed by more than one researcher.
when several people
came up with a way to train neural networks with “back-propagation," which is a particular style of implementing gradient descent, which we will study here. By the mid-90s, the enthusiasm waned again, because although we could train non-linear networks, the training tended to be slow and was plagued by a problem of getting stuck in local optima. Support vector machines (
svm
s) (regularization of high-dimensional hypotheses by seeking to maximize the margin) and kernel methods (an efficient and beautiful way of using feature transformations to non-linearly transform data into a higher-dimensional space) provided reliable learning methods with guaranteed convergence and no local optima.
However, during the
svm
enthusiasm, several groups kept working on neural networks, and their work, in combination with an increase in available data and computation, has made them rise again. They have become much more reliable and capable, and are now the method of choice in many applications. There are many,
The number increases daily, as may be seen on
arxiv.org
.
many
variations of neural networks, which we can't even begin to survey. We will study the core “feed-forward" networks with “back-propagation" training, and then, in later chapters, address some of the major advances beyond this core.
We can view neural networks from several different perspectives:
We will mostly take view 1, with the understanding that the techniques we develop will enable the applications in view 3. View 2 was a major motivation for the early development of neural networks, but the techniques we will
Some prominent researchers are, in fact, working hard to find analogues of these methods in the brain
study do not
seem to actually account for the biological learning processes in brains.
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:19 PM (revision f808f068e)

## Basic element

Basic element
The basic element of a neural network is a “neuron," pictured schematically below. We will also sometimes refer to a neuron as a “unit" or “node."
It is a non-linear function of an input vector [mathjaxinline]x \in \mathbb {R}^ m[/mathjaxinline]
Sorry for changing our notation here. We were using [mathjaxinline]d[/mathjaxinline] as the dimension of the input, but we are trying to be consistent here with many other accounts of neural networks. It is impossible to be consistent with all of them though—there are many different ways of telling this story.
note
to a single output value [mathjaxinline]a \in \mathbb {R}[/mathjaxinline]. It is parameterized by a vector of
weights
[mathjaxinline](w_1, \ldots , w_ m) \in \mathbb {R}^ m[/mathjaxinline] and an
offset
or
threshold
[mathjaxinline]w_0 \in \mathbb {R}[/mathjaxinline].
This should remind you of our [mathjaxinline]\theta[/mathjaxinline] and [mathjaxinline]\theta _0[/mathjaxinline] for linear models.
note
In order for the neuron to be non-linear, we also specify an
activation function
[mathjaxinline]f : \mathbb {R}\rightarrow \mathbb {R}[/mathjaxinline], which can be the identity ([mathjaxinline]f(x) = x[/mathjaxinline]), but can also be any other function, though we will only be able to work with it if it is differentiable.
The function represented by the neuron is expressed as:
[mathjax]a = f(z) = f\left(\sum _{j=1}^ m x_ jw_ j + w_0\right) = f(w^ Tx + w_0)\; \; .[/mathjax]
Before thinking about a whole network, we can consider how to train a single unit. Given a loss function [mathjaxinline]L(\text {\it guess}, \text {\it actual)}[/mathjaxinline] and a dataset [mathjaxinline]\{ (x^{(1)}, y^{(1)}), \ldots , (x^{(n)},y^{(n)})\}[/mathjaxinline], we can do (stochastic) gradient descent, adjusting the weights [mathjaxinline]w, w_0[/mathjaxinline] to minimize
[mathjax]J(w, w_0) = \sum _{i} L\left(NN(x^{(i)}; w, w_0), y^{(i)}\right)\; \; .[/mathjax]
where [mathjaxinline]NN[/mathjaxinline] is the output of our neural net for a given input.
We have already studied two special cases of the neuron: linear classifiers with hinge loss and regressors with quadratic loss! Both of these have activation functions [mathjaxinline]f(x) = x[/mathjaxinline].
Study Question:
Just for a single neuron, imagine for some reason, that we decide to use activation function [mathjaxinline]f(z) = e^ z[/mathjaxinline] and loss function [mathjaxinline]L(g, a) = (g - a)^2[/mathjaxinline]. Derive a gradient descent update for [mathjaxinline]w[/mathjaxinline] and [mathjaxinline]w_0[/mathjaxinline].
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:57 PM (revision 4f166135)

## Lecture: Neural networks - layer definition

Lecture: Neural networks - layer definition
Lecture: Neural networks - layer definition
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - many layers

Lecture: Neural networks - many layers
Lecture: Neural networks - many layers
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Networks

Networks
Now, we'll put multiple neurons together into a
network
. A neural network in general takes in an input [mathjaxinline]x \in \mathbb {R}^ m[/mathjaxinline] and generates an output [mathjaxinline]a \in \mathbb {R}^ n[/mathjaxinline]. It is constructed out of multiple neurons; the inputs of each neuron might be elements of [mathjaxinline]x[/mathjaxinline] and/or outputs of other neurons. The outputs are generated by [mathjaxinline]n[/mathjaxinline]
output units
.
In this chapter, we will only consider
feed-forward
networks. In a feed-forward network, you can think of the network as defining a function-call graph that is
acyclic
: that is, the input to a neuron can never depend on that neuron's output. Data flows, one way, from the inputs to the outputs, and the function computed by the network is just a composition of the functions computed by the individual neurons.
Although the graph structure of a neural network can really be anything (as long as it satisfies the feed-forward constraint), for simplicity in software and analysis, we usually organize them into
layers
. A layer is a group of neurons that are essentially “in parallel": their inputs are outputs of neurons in the previous layer, and their outputs are the input to the neurons in the next layer. We'll start by describing a single layer, and then go on to the case of multiple layers.
Single layer
A
layer
is a set of units that, as we have just described, are not connected to each other. The layer is called
fully connected
if, as in the diagram below, the inputs to each unit in the layer are the same (i.e. [mathjaxinline]x_1, x_2, \ldots x_ m[/mathjaxinline] in this case). A layer has input [mathjaxinline]x \in \mathbb {R}^ m[/mathjaxinline] and output (also known as
activation
) [mathjaxinline]a \in \mathbb {R}^ n[/mathjaxinline].
Since each unit has a vector of weights and a single offset, we can think of the weights of the whole layer as a matrix, [mathjaxinline]W[/mathjaxinline], and the collection of all the offsets as a vector [mathjaxinline]W_0[/mathjaxinline]. If we have [mathjaxinline]m[/mathjaxinline] inputs, [mathjaxinline]n[/mathjaxinline] units, and [mathjaxinline]n[/mathjaxinline] outputs, then
[mathjaxinline]W[/mathjaxinline] is an [mathjaxinline]m\times n[/mathjaxinline] matrix,
[mathjaxinline]W_0[/mathjaxinline] is an [mathjaxinline]n \times 1[/mathjaxinline] column vector,
[mathjaxinline]X[/mathjaxinline], the input, is an [mathjaxinline]m \times 1[/mathjaxinline] column vector,
[mathjaxinline]Z = W^ T X + W_0[/mathjaxinline], the
pre-activation
, is an [mathjaxinline]n \times 1[/mathjaxinline] column vector,
[mathjaxinline]A[/mathjaxinline], the
activation
, is an [mathjaxinline]n \times 1[/mathjaxinline] column vector,
and the output vector is
[mathjax]A = f(Z) = f(W^ TX + W_0)\; \; .[/mathjax]
The activation function [mathjaxinline]f[/mathjaxinline] is applied element-wise to the pre-activation values [mathjaxinline]Z[/mathjaxinline].
What can we do with a single layer? We have already seen single-layer networks, in the form of linear separators and linear regressors. All we can do with a single layer is make a linear hypothesis (with some possible linear transformation on the output). The whole reason for moving to neural networks is to move in the direction of
non-linear
hypotheses. To do this, we will have to consider multiple layers.
Many layers
A single neural network generally combines multiple layers, most typically by feeding the outputs of one layer into the inputs of another layer.
We have to start by establishing some nomenclature. We will use [mathjaxinline]l[/mathjaxinline] to name a layer, and let [mathjaxinline]m^ l[/mathjaxinline] be the number of inputs to the layer and [mathjaxinline]n^ l[/mathjaxinline] be the number of outputs from the layer. Then, [mathjaxinline]W^ l[/mathjaxinline] and [mathjaxinline]W^ l_0[/mathjaxinline] are of shape [mathjaxinline]m^ l \times n^ l[/mathjaxinline] and [mathjaxinline]n^ l \times 1[/mathjaxinline], respectively. Let [mathjaxinline]f^ l[/mathjaxinline] be the activation function of layer [mathjaxinline]l[/mathjaxinline].
It is technically possible to have different activation functions within the same layer, but, again, for convenience in specification and implementation, we generally have the same activation function within a layer.
note
Then, the pre-activation outputs are the [mathjaxinline]n^ l \times 1[/mathjaxinline] vector
[mathjax]Z^ l = {W^ l}^ TA^{l-1} + W^ l_0[/mathjax]
and the activation outputs are simply the [mathjaxinline]n^ l \times 1[/mathjaxinline] vector
[mathjax]A^ l = f^ l(Z^ l)\; \; .[/mathjax]
Here's a diagram of a many-layered network, with two blocks for each layer, one representing the linear part of the operation and one representing the non-linear activation function. We will use this structural decomposition to organize our algorithmic thinking and implementation.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:57 PM (revision 4f166135)

## Lecture: Neural networks - activation functions

Lecture: Neural networks - activation functions
Lecture: Neural networks - activation functions
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Choices of activation function

Choices of activation function
There are many possible choices for the activation function. We will start by thinking about whether it's really necessary to have an [mathjaxinline]f[/mathjaxinline] at all.
What happens if we let [mathjaxinline]f[/mathjaxinline] be the identity? Then, in a network with [mathjaxinline]L[/mathjaxinline] layers (we'll leave out [mathjaxinline]W_0[/mathjaxinline] for simplicity, but keeping it wouldn't change the form of this argument),
[mathjax]A^ L = {W^ L}^ T A^{L-1} = {W^ L}^ T {W^{L-1}}^ T \cdots {W^1}^ T X\; \; .[/mathjax]
So, multiplying out the weight matrices, we find that
[mathjax]A^ L = W^\text {total}X\; \; ,[/mathjax]
which is a
linear
function of [mathjaxinline]X[/mathjaxinline]! Having all those layers did not change the representational capacity of the network: the non-linearity of the activation function is crucial.
Study Question:
Convince yourself that any function representable by any number of linear layers (where [mathjaxinline]f[/mathjaxinline] is the identity function) can be represented by a single layer.
Now that we are convinced we need a non-linear activation, let's examine a few common choices.
The original idea for neural networks involved using the
step
function as an activation, but because the derivative is discontinuous, we won't be able to use gradient-descent methods to tune the weights in a network with step functions, so we won't consider them further. They have been replaced, in a sense, by the sigmoid, relu, and tanh activation functions.
Study Question:
Consider sigmoid, relu, and tanh activations. Which one is most like a step function? Is there an additional parameter you could add to a sigmoid that would make it be more like a step function?
Study Question:
What is the derivative of the relu function? Are there some values of the input for which the derivative vanishes?
ReLUs are especially common in internal (“hidden") layers, and sigmoid activations are common for the output for binary classification and softmax for multi-class classification (see section
for an explanation).
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:57 PM (revision 4f166135)

## Lecture: Neural networks - training and back-propagation

Lecture: Neural networks - training and back-propagation
Lecture: Neural networks - training and back-propagation
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Error back-propagation

Error back-propagation
We will train neural networks using gradient descent methods. It's possible to use
batch
gradient descent, in which we sum up the gradient over all the points (as in section
of chapter
) or stochastic gradient descent (
sgd
), in which we take a small step with respect to the gradient considering a single point at a time (as in section
of chapter
).
Our notation is going to get pretty hairy pretty quickly. To keep it as simple as we can, we'll focus on computing the contribution of one data point [mathjaxinline]x^{(i)}[/mathjaxinline] to the gradient of the loss with respect to the weights, for
sgd
; you can simply sum up these gradients over all the data points if you wish to do batch descent.
So, to do
sgd
for a training example [mathjaxinline](x, y)[/mathjaxinline], we need to compute [mathjaxinline]\nabla _ W \text {Loss}(NN(x;W),y)[/mathjaxinline], where [mathjaxinline]W[/mathjaxinline] represents all weights [mathjaxinline]W^ l, W_0^ l[/mathjaxinline] in all the layers [mathjaxinline]l = (1, \ldots , L)[/mathjaxinline]. This seems terrifying, but is actually quite easy to do using
Remember the chain rule! If [mathjaxinline]a = f(b)[/mathjaxinline] and [mathjaxinline]b = g(c)[/mathjaxinline] (so that
[mathjaxinline]a = f(g(c))[/mathjaxinline]), then
[mathjaxinline]\frac{d a}{d c} = \frac{d a}{d b} \cdot \frac{d b}{d c} = f'(b) g'(c) = f'(g(c)) g'(c)[/mathjaxinline].
the chain rule.
Remember that we are always computing the gradient of the loss function
with respect to the weights
for a particular value of [mathjaxinline](x, y)[/mathjaxinline]. That tells us how much we want to change the weights, in order to reduce the loss incurred on this particular training example.
First, let's see how the loss depends on the weights in the final layer, [mathjaxinline]W^ L[/mathjaxinline]. Remembering that our output is [mathjaxinline]A^ L[/mathjaxinline], and using the shorthand [mathjaxinline]\text {loss}[/mathjaxinline] to stand for [mathjaxinline]\text {Loss}((NN(x;W),y)[/mathjaxinline] which is equal to [mathjaxinline]\text {Loss}(A^ L, y)[/mathjaxinline], and finally that [mathjaxinline]A^ L = f^ L(Z^ L)[/mathjaxinline] and [mathjaxinline]Z^ L = {W^ L}^ T A^{L-1}[/mathjaxinline], we can use the chain rule:
[mathjax]\frac{\partial \text {loss}}{\partial W^ L} = \underbrace{ \frac{\partial \text {loss}}{\partial A^ L}}_{\text {depends on loss function}} \cdot \underbrace{\frac{\partial A^ L}{\partial Z^ L}}_{f^{L'}} \cdot \underbrace{\frac{\partial Z^ L}{\partial W^ L}}_{\text {$A^{L-1}$}} \; \; .[/mathjax]
It might reasonably bother you that [mathjaxinline]\partial {Z^ L}/\partial {W^ L} = A^{L-1}[/mathjaxinline]. We're somehow thinking about the derivative of a vector with respect to a matrix, which seems like it might need to be a three-dimensional thing. But note that [mathjaxinline]\partial {Z^ L}/\partial {W^ L}[/mathjaxinline] is really [mathjaxinline]\partial {{W^ L}^ T A^{L-1}}/\partial {W^ L}[/mathjaxinline] and it seems okay in at least an informal sense that it's [mathjaxinline]A^{L-1}[/mathjaxinline].
note
To actually get the dimensions to match, we need to write this a bit more carefully, and note that it is true for any [mathjaxinline]l[/mathjaxinline], including [mathjaxinline]l = L[/mathjaxinline]:
[mathjax]\underbrace{\frac{\partial \text {loss}}{\partial W^ l}}_{m^ l \times n^ l} = \underbrace{A^{l-1}}_{m^ l \times 1} \;  \underbrace{\left(\frac{\partial \text {loss}}{\partial Z^ l}\right)^ T}_{1 \times n^ l}[/mathjax]
(1.1)
Yay! So, in order to find the gradient of the loss with respect to the weights in the other layers of the network, we just need to be able to find [mathjaxinline]\partial \text {loss}/\partial {Z^ l}[/mathjaxinline].
If we repeatedly apply the chain rule, we get this expression for the gradient of the loss with respect to the pre-activation in the first layer:
[mathjax]\frac{\partial \text {loss}}{\partial Z^1} = \underbrace{\underbrace{ \frac{\partial \text {loss}}{\partial A^ L} \cdot \frac{\partial A^ L}{\partial Z^ L} \cdot \frac{\partial Z^ L}{\partial A^{L-1}} \cdot \frac{\partial A^{L-1}}{\partial Z^{L-1}} \cdot \cdots \cdot \frac{\partial A^2}{\partial Z^2}}_{\partial \text {loss} / \partial Z^2} \cdot \frac{\partial Z^2}{\partial A^1}} _{\partial \text {loss} / \partial A^1} \cdot \frac{\partial A^1}{\partial Z^1} \; \; .[/mathjax]
(1.2)
This derivation was informal, to show you the general structure of the computation. In fact, to get the dimensions to all work out, we just have to write it backwards! Let's first understand more about these quantities:
[mathjaxinline]\partial \text {loss}/\partial A^ L[/mathjaxinline] is [mathjaxinline]n^ L \times 1[/mathjaxinline] and depends on the particular loss function you are using.
[mathjaxinline]\partial Z^ l / \partial A^{l-1}[/mathjaxinline] is [mathjaxinline]m^ l \times n^ l[/mathjaxinline] and is just [mathjaxinline]W^ l[/mathjaxinline] (you can verify this by computing a single entry [mathjaxinline]\partial Z^ l_ i / \partial A^{l-1}_ j[/mathjaxinline]).
[mathjaxinline]\partial A^ l/\partial Z^ l[/mathjaxinline] is [mathjaxinline]n^ l \times n^ l[/mathjaxinline]. It's a little tricky to think about. Each element [mathjaxinline]a_ i^ l = f^ l(z_ i^ l)[/mathjaxinline]. This means that [mathjaxinline]\partial a_ i^ l / \partial z_ j^ l = 0[/mathjaxinline] whenever [mathjaxinline]i \not= j[/mathjaxinline]. So, the off-diagonal elements of [mathjaxinline]\partial A^ l/\partial Z^ l[/mathjaxinline] are all 0, and the diagonal elements are [mathjaxinline]\partial a_ i^ l / \partial z_ j^ l = {f^ l}'(z_ j^ l)[/mathjaxinline].
Now, we can rewrite equation
so that the quantities match up as
[mathjax]\frac{\partial \text {loss}}{\partial Z^ l} = \frac{\partial A^ l}{\partial Z^ l} \cdot W^{l+1} \cdot \frac{\partial A^{l+1}}{\partial Z^{l+1}} \cdot \ldots W^{L-1} \cdot \frac{\partial A^{L-1}}{\partial Z^{L-1}} \cdot W^{L} \cdot \frac{\partial A^{L}}{\partial Z^{L}} \cdot \frac{\partial \text {loss}}{A^ L}\; \; .[/mathjax]
(1.3)
Using equation
to compute [mathjaxinline]\partial \text {loss}/\partial {Z^ l}[/mathjaxinline] combined with equation
, lets us find the gradient of the loss with respect to any of the weight matrices.
Study Question:
Apply the same reasoning to find the gradients of [mathjaxinline]\text {loss}[/mathjaxinline] with respect to [mathjaxinline]W_0^ l[/mathjaxinline].
This general process is called
error back-propagation
. The idea is that we first do a
forward pass
to compute all the [mathjaxinline]a[/mathjaxinline] and [mathjaxinline]z[/mathjaxinline] values at all the layers, and finally the actual loss on this example. Then, we can work backward and compute the gradient of the loss with respect to the weights in each layer, starting at layer [mathjaxinline]L[/mathjaxinline] and going back to layer 1.
I like to think of this as “blame propagation". You can think of [mathjaxinline]\text {loss}[/mathjaxinline] as how mad we are about the prediction that the network just made. Then [mathjaxinline]\partial \text {loss}/ \partial A^ L[/mathjaxinline] is how much we blame [mathjaxinline]A^ L[/mathjaxinline] for the loss. The last module has to take in [mathjaxinline]\partial \text {loss}/ \partial A^ L[/mathjaxinline] and compute [mathjaxinline]\partial \text {loss}/ \partial Z^ L[/mathjaxinline], which is how much we blame [mathjaxinline]Z^ L[/mathjaxinline] for the loss. The next module (working backwards) takes in [mathjaxinline]\partial \text {loss}/ \partial Z^ L[/mathjaxinline] and computes [mathjaxinline]\partial \text {loss}/ \partial A^{L-1}[/mathjaxinline]. So every module is accepting its blame for the loss, computing how much of it to allocate to each of its inputs, and passing the blame back to them.
note
If we view our neural network as a sequential composition of modules (in our work so far, it has been an alternation between a linear transformation with a weight matrix, and a component-wise application of a non-linear activation function), then we can define a simple API for a module that will let us compute the forward and backward passes, as well as do the necessary weight updates for gradient descent. Each module has to provide the following “methods." We are already using letters [mathjaxinline]a, x, y, z[/mathjaxinline] with particular meanings, so here we will use [mathjaxinline]u[/mathjaxinline] as the vector input to the module and [mathjaxinline]v[/mathjaxinline] as the vector output:
forward: [mathjaxinline]u \rightarrow v[/mathjaxinline]
backward: [mathjaxinline]u, v, \partial L / \partial v \rightarrow \partial L / \partial u[/mathjaxinline]
weight grad: [mathjaxinline]u, \partial L / \partial v \rightarrow \partial L / \partial W[/mathjaxinline] only needed for modules that have weights [mathjaxinline]W[/mathjaxinline]
In homework we will ask you to implement these modules for neural network components, and then use them to construct a network and train it as described in the next section.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:57 PM (revision 4f166135)

## Lecture: Neural networks - backprop with the chain rule

Lecture: Neural networks - backprop with the chain rule
Lecture: Neural networks - backprop with the chain rule
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - weight initialization

Lecture: Neural networks - weight initialization
Lecture: Neural networks - weight initialization
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Training

Training
Here we go! Here's how to do stochastic gradient descent training on a feed-forward neural network. After this pseudo-code, we motivate the choice of initialization in lines 2 and 3. The actual computation of the gradient values (e.g. [mathjaxinline]\partial \text {loss}/ \partial A^ L[/mathjaxinline]) is not directly defined in this code, because we want to make the structure of the computation clear.
Study Question:
What is [mathjaxinline]\partial Z^ l / \partial W^ l[/mathjaxinline]?
Study Question:
Which terms in the code below depend on [mathjaxinline]f^ L[/mathjaxinline]?
Initializing [mathjaxinline]W[/mathjaxinline] is important; if you do it badly there is a good chance the neural network training won't work well. First, it is important to initialize the weights to random values. We want different parts of the network to tend to “address" different aspects of the problem; if they all start at the same weights, the symmetry will often keep the values from moving in useful directions. Second, many of our activation functions have (near) zero slope when the pre-activation [mathjaxinline]z[/mathjaxinline] values have large magnitude, so we generally want to keep the initial weights small so we will be in a situation where the gradients are non-zero, so that gradient descent will have some useful signal about which way to go.
One good general-purpose strategy is to choose each weight at random from a Gaussian (normal) distribution with mean 0 and standard deviation [mathjaxinline](1/m)[/mathjaxinline] where [mathjaxinline]m[/mathjaxinline] is the number of inputs to the unit.
Study Question:
If the input [mathjaxinline]x[/mathjaxinline] to this unit is a vector of 1's, what would the expected pre-activation [mathjaxinline]z[/mathjaxinline] value be with these initial weights?
We write this choice (where [mathjaxinline]\sim[/mathjaxinline] means “is drawn randomly from the distribution")
[mathjax]W^ l_{ij} \sim \text {Gaussian}\left(0, \frac{1}{m^ l}\right)\; \; .[/mathjax]
It will often turn out (especially for fancier activations and loss functions) that computing
[mathjax]\frac{\partial \text {loss}}{\partial Z^ L}[/mathjax]
is easier than computing
[mathjax]\frac{\partial \text {loss}}{\partial A^ L}\; \; \text { and }\; \; \frac{\partial A^ L}{\partial Z^ L}\; \; .[/mathjax]
So, we may instead ask for an implementation of a loss function to provide a backward method that computes [mathjaxinline]\partial \text {loss}/\partial Z^ L[/mathjaxinline] directly.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:57 PM (revision 4f166135)

## Lecture: Neural networks - output layer activation functions

Lecture: Neural networks - output layer activation functions
Lecture: Neural networks - output layer activation functions
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Loss functions and activation functions

Loss functions and activation functions
Different loss functions make different assumptions about the range of inputs they will get as input and, as we have seen, different activation functions will produce output values in different ranges. When you are designing a neural network, it's important to make these things fit together well. In particular, we will think about matching loss functions with the activation function in the last layer, [mathjaxinline]f^ L[/mathjaxinline]. Here is a table of loss functions and activations that make sense for them:
Loss
[mathjaxinline]f^ L[/mathjaxinline]
squared
linear
hinge
linear
[mathjaxinline]\text {NLL}[/mathjaxinline]
sigmoid
[mathjaxinline]\text {NLLM}[/mathjaxinline]
softmax
But what is NLL?
Two-class classification and log likelihood
For classification, the natural loss function is 0-1 loss, but we have already discussed the fact that it's very inconvenient for gradient-based learning because its derivative is discontinuous. Hinge loss gives us a way, for binary classification problems, to make a smoother objective. An alternative loss function that has a nice probabilistic interpretation, is in popular use, and extends nicely to multi-class classification, is called
negative log likelihood
(
nll
). We will discuss it first in the two-class case, and then generalize to multiple classes.
Let's assume that the activation function on the output layer is a sigmoid and that there is a single unit in the output layer, so the output of the whole neural network is a scalar, [mathjaxinline]a^ L[/mathjaxinline]. Because [mathjaxinline]f^ L[/mathjaxinline] is a sigmoid, we know [mathjaxinline]a^ L \in [0, 1][/mathjaxinline], and we can interpret it as the probability that the input [mathjaxinline]x[/mathjaxinline] is a positive example. Let us further assume that the labels in the training data are [mathjaxinline]y \in \{ 0, 1\}[/mathjaxinline], so they can also be interpreted as probabilities.
We might want to pick the parameters of our network to maximize the probability that the network assigns the correct labels to all the points. That would be
[mathjax]\prod _{i = 1}^ n \begin{cases}  a^{(i)} &  \text {if $y^{(i)} = 1$} \\ 1 - a^{(i)} &  \text {otherwise} \end{cases}\; \; ,[/mathjax]
under the assumption that our predictions are independent. This can be cleverly rewritten as
[mathjax]\prod _{i = 1}^ n {a^{(i)}}^{y^{(i)}}(1 - a^{(i)})^{1 - y^{(i)}}\; \; .[/mathjax]
Study Question:
Be sure you can see why these two expressions are the same.
Now, because products are kind of hard to deal with, and because the log function is monotonic, the [mathjaxinline]W[/mathjaxinline] that maximizes the log of this quantity will be the same as the [mathjaxinline]W[/mathjaxinline] that maximizes the original, so we can try to maximize
[mathjax]\sum _{i = 1}^ n {y^{(i)}}\log {a^{(i)}} + (1 - y^{(i)})\log (1 - a^{(i)})\; \; ,[/mathjax]
which we can write in terms of a loss function
[mathjax]\sum _{i = 1}^ n \mathcal{L}_\text {nll}(a^{(i)}, y^{(i)})[/mathjax]
where [mathjaxinline]\mathcal{L}_\text {nll}[/mathjaxinline] is the
negative log likelihood
loss function:
[mathjax]\mathcal{L}_\text {nll}(\text {guess},\text {actual}) = -\left(\text {actual}\cdot \log (\text {guess}) + (1 - \text {actual})\cdot \log (1 - \text {guess})\right) \; \; .[/mathjax]
This loss function is also sometimes referred to as the
log loss
or
You can use any base for the logarithm and it won't make any real difference. If we ask you for numbers, use log base [mathjaxinline]e[/mathjaxinline].
cross entropy
.
Multi-class classification and log likelihood
We can extend this idea directly to multi-class classification with [mathjaxinline]K[/mathjaxinline] classes, where the training label is represented with the one-hot vector [mathjaxinline]y=\begin{bmatrix}  y_1, \ldots , y_ K \end{bmatrix}^ T[/mathjaxinline], where [mathjaxinline]y_ k=1[/mathjaxinline] if the example is of class [mathjaxinline]k[/mathjaxinline]. Assume that our network uses
softmax
as the activation function in the last layer, so that the output is [mathjaxinline]a=\begin{bmatrix}  a_1, \ldots , a_ K \end{bmatrix}^ T[/mathjaxinline], which represents a probability distribution over the [mathjaxinline]K[/mathjaxinline] possible classes. Then, the probability that our network predicts the correct class for this example is [mathjaxinline]\prod _{k=1}^ K a_ k^{y_ k}[/mathjaxinline] and the log of the probability that it is correct is [mathjaxinline]\sum _{k=1}^ K y_ k \log a_ k[/mathjaxinline], so
[mathjax]\mathcal{L}_\text {nllm}(\text {guess},\text {actual}) = - \sum _{k=1}^ K \text {actual}_ k \cdot \log (\text {guess}_ k) \; \; .[/mathjax]
We'll call this
nllm
for
negative log likelihood multiclass.
Study Question:
Show that [mathjaxinline]L_\text {nllm}[/mathjaxinline] for [mathjaxinline]K = 2[/mathjaxinline] is the same as [mathjaxinline]L_\text {nll}[/mathjaxinline].
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:57 PM (revision 4f166135)

## Video transcripts

### Lecture: Neural networks - basic element


LESLIE KAELBLING: So let's start with the most basic thing,
the most basic thing--
People, if you're feeling romantic,
you can call it a neuron.
Or it also gets called the node sometimes, or sometimes a unit.
So these are all names for one computational item.
It often gets written like this.
There's kind of a thing with a summation.
There's some stuff that comes in, dot, dot, dot.
I'll elaborate on all this in a minute.
There's some function f and there's
something that comes out.
OK.
And what comes in is a vector of inputs.
So x1 through xm.

So this is input.
And I have to apologize for the machine learning community.
We have not got our story straight
about like what the names of all the dimensions of things are.
So basically, when we're doing the neural network stuff,
we're generally speaking going to use
m for the input dimension of something and n
for the output dimension of something.
And I'm sorry because we had been
using d and some other things.

But for the duration of neural networks,
we're going to try to stick to m for input
dimensions of something, n for output dimensions of something.
So the input to this neuron node unit
is a vector of values x1 through xm.

And its output is something.
Right now, it only has a one dimensional output.
For right now, we're going to call it A.
OK.
So the input is x, which is going to be in r to the m.
The output is a, which is say a real number.
OK.
So what is this picture about?
So the next thing to know is that there are some weights.
I'm going to draw them here associated with that little--
So x1, x2 weight 1, weight 2, weight m.
Here for this guy.
And there's usually one more weight, weight zero.
So those are going to be the parameters.
Those are like the thetas, bright?
We have been doing linear classifiers
and linear aggressors, and we had these theta parameters.
Now they're going to be called w for weights,
and we're going to have a whole lot more of them.
OK.
What comes out of here is something called z.
And then there's f and then there's a.
OK.
A is called a for activation.
It's the output of an individual neuron,
but we'll call it the activation.
That's a kind of again a term that
came from the history of thinking
of this is like a neuron.
And this f is something called an activation function.

And it will, generally speaking, be nonlinear.
So we'll talk about a bunch of potential different choices
for f.
All right.
So this is a picture and a bunch of terminology.
And the question is--
This is really in the end just a function.
And what's the output of this thing?
So the output.
So a is f of z, which is f of the sum from j
is one that m of xj weight j plus w0.

OK.
That looks familiar, right?
So all this is--
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: Oh, good question.
Good question.
It's outside.

Yeah.
So we only add in W0 once.
So you can think of the weights are a vector
and the X's are a vector.
That's the null product, right?
So this is the same basic linear machine
that we've been looking at since the first lecture,
but with this extra added f.
OK.
So this is just old hat, but new picture.
Old hat, new picture.
So that is neural network unit.
And generally speaking, f will be fixed.
And our job will be to pick w, so that this thing does
what we want it to.
The difference here now is going to be not just
that we don't just have one of them,
but that we're going to have a whole big collection of them
put together.
OK.

We start with a unit, and then we go to a layer.
So a neural network is going to be a big collection of neurons.
And let me actually talk about the class
that we're going to talk about-- feed forward networks.

And in a feed forward network, the idea
is that we might now have a whole bunch of these units.
And the inputs, we might have several inputs
that go to this one, and they go to this one.
And this guy goes here, and this thing goes here,
and this goes here, and so on.
We have two outputs.
So you can wire these up.
So you can take these units, and you can think of it
as a circuit almost, right?
So you can take the inputs that come from your data,
your original X's.
They can come into some units.
Those outputs of some units can go to other ones, and so on.
And the feed forward just means that nobody
can depend on himself, right.
So the data has to flow in some sense in one direction,
from the inputs to the outputs.
You can't have any connections that
go like oh, my output actually comes back and goes in here.
So that's not OK.
Later on, we'll talk about recurrent networks.
And recurrent neural networks, we
are allowed to have these connections.
But in feed forward networks, inputs go here,
and they go straight through to the output.
So we'll restrict our attention for right now to feed forward
networks.
OK.
But they can be wired up in some sense in any way you want.
And all the math and all the stuff we're going to talk about
would go through for an arbitrary arrangement of a feed
forward network.


### Lecture: Neural networks - basic element


LESLIE KAELBLING: Again for simplicity and because there
is no particular reason to make kind of a crazy structure,
we're going to think of networks that are
organized in terms of layers.
So what's a layer?
Again, you guys have all the conceptual structure
you need already to do neural networks.
So really, this is a lot of indices and terminology
and stuff so that you can keep the basic idea straight
when you apply it to a bigger problem.
But really, there's nothing conceptually new here.
OK?
So layers.
So this is kind of a terminology-heavy lecture,
unfortunately.
So there will be many different kinds of layers
that we'll talk about, but today, we're
going to talk about a particular kind of layer which is
called a fully connected layer.

And what that means is basically that I have a set now of units.
So let me draw them like this.

OK.
And now I will have m.
So this is going to be activation 1
through activation n.
Activation 2.
And I'm going to have some inputs,
and every one of my inputs is going
to go to every one of my units, like that.
That's what makes it fully connected.
And I have x1, x2, xm.

So I might have a different number of inputs to this layer
than I have outputs.
The number of outputs is the same as the number of units,
right?
So number of outputs is number of units is n.

OK?
And generally speaking, again, we will have the same f.
We talked about the fact that we could have different f's, and
we'll talk a little bit about different f's, but we're
going to have the same f in all the nonlinearity.
So all the activation functions are going to be the same.
So we'll have the same f.

And what we'll do is actually try to--
I mean, the way you compute output, too, let's say,
is just using that formula up there.
But we can make it a lot tidier if we think
of it in terms of matrices.
So let's think about the weights in here.
So each one of these units has weights on all the inputs
and it's own little w0, so that's a lot of weights
now, right?
It's n weight vectors.
So we'll let w be an m by n matrix of weights.

And actually, I'm going to write out the whole set of things
we have.
And w0 is going to be a vector of offsets, so it's an n--

yeah, sorry.
n by 1 vector of the w0's.

OK.
So now what's the output of this thing?
So the complete output-- so A, the activation of the whole
thing--
I want to write it and I'll explain some more over there--
is f of z.

And that's going to be f of w transpose x plus w0.

OK.
So let's just kind of be sure that this is all good.
So x is our input vector here.
So x is going to be m by 1.

w is m by n, but if we transpose it,
we get something that's n by m.
And n by m times m by 1 gives us n by 1, right?
So this whole thing is n by 1.
And w0 is n by 1.
So this whole stuff inside the F is n by 1.
OK.
So that seems like an OK thing I did so far, right?
Is this cool?
The stuff that's inside the parenthesis.
Yeah?
OK.
So now, what about the f?
So that's a little funny.
Up there, f was some kind of a scalar function.
f, generally speaking, takes a real number.
So what's with f?
So f, usually, we think of it as going from r to r.
Like, one f you can think about, which we do use occasionally,
is this one.
So that's an f that we could use.
So what does it mean to apply f to an n by 1 vector?
So when I write f of a vector, and whenever
we apply f, really, we're going to apply f element-wise.
So the f, we're going to apply element-wise,
which means that when we write f of a vector--
in this case, like, z1 through zn,
that's just going to be the vector, f of z1, f of zn.

And this is going to be the thing we call A.
Does that seem good?
OK.
So this is everything there is to know about a neural network
layer.

So it's a transformation of a vector in m-dimensional space
into a vector in n-dimensional space.
And it's described using a weight
matrix, a vector of offsets, and an activation function.
So if you describe to me the w and the w0 and the f,
that's all we need to know.
And it's a transformation from this kind of vector


### Lecture: Neural networks - basic element


LESLIE KAELBLING: So far, so good.
Still not, maybe, super exciting,
because these are all just individual little linear units
with some nonlinear thing added on at the end.
So now, we get to the exciting part and my favorite figure.
So I'm going to draw this incredibly complicated figure.
It's in the notes.
I should say, I posted the notes this morning,
and then when I was preparing for lecture,
I realized that I could make them better in several ways,
so I will do that probably tonight
and post a better version.
So we're let's see.
We're going to eventually talk about putting
many layers together to make a neural network,
and then eventually, we're going to have to take gradients.
And that can be like a nightmare of indices.
So we have a choice between a nightmare of indices and vector
calculus.
And we're going to go for vector calculus.
You don't mess up the i's and the j's and the k's as badly.
And there's a nice, really beautiful modular view
that you can take.
So we're going to take this kind of modular view.
And you guys will implement a neural network training
algorithm using this kind of modular story.
So I'm going to try to stick with the modular story here.
OK.
So let's think about modules.
So I'm going to draw this picture.
It's going to have a lot of boxes,
but they are going to be excellent boxes.
OK, we'll start with the boxes.
OK, so there's one layer, two layers.

Then we've got some dot, dot, dots going on,
and now we've got two more boxes.

And the loss.
OK, good boxes.
Nice.
So I'm going to let each one of these modules--
so the boxes are going to be things
I'm going to call modules, and it'll take two to make a layer.

So now, we're going to have many layers,
and I'm going to use the superscript here
to indicate which layer we're talking about.

All right.

So in comes our x.
All right?
So we're going to think about, how does this thing
compute something?
So in comes our x.
We can also think of it as the 0th activation.
Activation 0.
The dimension of x, we will recall, is m by 1.
So I'm going to write down here the dimensions.
This is an m by 1.
But we don't just have one m.
We're going to have lots of m's and n's, one for each layer.
So we'll call this m1 by 1.
That's the dimension of x.
OK.
What's going to come out of here is something we'll call z1,
all right?
So that's going to be the vector that we get by taking the x,
multiplying by the w, adding in the w0.
We get out the z's.
That's a vector.
What's the size of z1?

n by 1, good.
In particular, n1 by 1.

We run that now, element-wise, through the f's.
And we're going to say we have to have
the same f in every layer, but we
could have different f's in different layers.
So we're going to run it through the f.
And what's going to come out of here
is A1, which is also n1 by 1.

Now we can multiply it by these weights.
Add this offset.
We'll get z2, A2, et cetera, et cetera, et cetera.
What will come into here is AL minus 1.
Here comes zL.
Here comes AL.
AL is like our answer.

OK.
Does this make sense to you as a kind of a big, hairy function?
It's just a function.
Yeah?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: They're not all the same.
They needn't be.
So let me write some more.
So let me write down this.
So n1 is going to be the same as m2,
and then this is going to be n2.
And this is n2, which is also m3, and so on.
So in fact, there is an art--
not yet a science, but an art.
In designing a neural network for your problem,
you get to pick how many layers, how big should they be,
and it matters in ways that we are not yet able, very well,
to explain.
So right.
So no, that's a good point.
So these things, they don't all have
to be the same size, right?
Now the size of the first layer is usually
dictated to you by the input that you're
trying to take in, right?
So how big is your initial feature vector.
And this final L--
so this is nL by 1.
Ah, so big L here.
Big L is the number of layers.

And one group of these guys is what we would call a layer.
The reason I'm dividing it into separate boxes
is that the software is gorgeous if you actually just divide
into these subsections.

So right.
So we're going to take something of size m1
by 1 into something of size nL by 1.
And if you're just trying to predict a single value,
it's actually a regression problem, right?
You're just trying to map some vector
into a single real number.
That's totally cool.
That would just mean that you would only have
one unit in the last layer.
So you could say, big, big, big, big, big, and then down
into one value.
That's totally fine.
Or you could predict a vector of values
and that would be OK, too.

OK.
So this is a class of hypotheses.
So let's just be clear about what the parameters are.
Right?
So the parameters, generally speaking,
we fix the f's so there's some simple form.
The parameters are the w's, right?
So for each layer, we have a matrix of weights
and a vector of offsets.

And so, now, when we think of this as a hypothesis class,
we say, OK, it's a function that maps x's into A's.
This first x, or A0, into A big L.
But to pick a particular hypothesis in that class,
we have to come up with values for all of these weight


### Lecture: Neural networks - basic element


LESLIE KAELBLING: All right, so let's talk about activation
functions for a minute.
I wrote one activation function up there,
which is f of x equals x.

That might not be the best choice of activation function.
And let's think about why.
Why might that--
So let's think about what this thing computes if the f's
are just the identity.
Right?
If the f's are just the identity function, f of x equals x,
then I could just delete all the f boxes here.
Right?
And if I delete all the f boxes, what am I going to get?
I'm going to say that AL--
let's see, so how do I compute an A?
I come over here and I say, well, A is f of z,
but f is nothing, so how do I get the z?
Well that's going to be weights transpose x plus w0.
Just for the purposes of argument and to make what I
write on the board simple, I'm going to leave off the w0's.
I shouldn't do that, really, but I'm just going to do it.
It won't really hurt.
OK.
So wL transpose--
OK, so AL is wL transpose times whatever
came in to that module.
So wL transpose times A of L minus 1.

OK?
What's A of L minus 1?
wL minus 1 transpose times A of L minus 2.
All right.
You're going to get a pattern here, right?
So this is going to turn into wL transpose wL--
never mind that guy.
w2 transpose w1 transpose x.

Right?
Because to get the output, we just
multiply by these weights the thing that came in.
And to get this, we multiply by these weights
the thing that came in.
We multiply these weights the thing that came in.
So what we have here is a product of a bunch of weight
matrices and an x.

What do you know about the product
of a bunch of weight matrices?
Yeah?
It's a matrix.
Yay.
So if we multiply a bunch of matrices-- we all
know how to do matrix multiplication-- there are
some matrix in the world, we'll call it
w total, which is just the product of all the weight
matrices.
And so what we have is that A is a linear function of x.
And if we wanted to make A be a linear function of x,
we could have done that like three weeks ago
and we wouldn't need all this stuff.
OK.
So important fact, if the f's are linear--
I mean, the same argument.
It works fine with the w0's in there,
it just is more tedious on the board.
It's still linear.
And even if the f isn't identity,
if it's some other linear transformation,
it's linear transformations all the way down,
and they all just turn into one linear transformation
and you don't need a neural network.
OK.
So only interesting-- only useful if f is nonlinear.

So that's sort of an important message.

OK.
In the notes, there's a description
of a bunch of activation functions
that we use now and then.
I'll just talk about a couple of them and you can read it.
Well, I'll talk about two.
You can read about the other ones.

So there's two activation functions that are popular.
The one that everybody loved and is still good for some things
is called the sigmoid.
It has a nice name.
Sigmoid means s-shaped.

It looks like a sigma.

And it is--

OK, everyone has intuition about that function, I know.
It looks like this.
So if this is z and this is going
to be A-- it's an activation function, so it takes in a z
and gives us an A. If this is 1 and this is 0,
it looks like that.
Oh, that's bad drawing.
This is 0.5.
So it crosses the y-axis at 0.5.
OK.
So what's most interesting about sigmoid?
So people, when they actually first invented neural networks,
really, wanted to think about having
some kind of a nonlinearity, but they wanted this one.

They wanted a step function, which
is kind of like what we have for perceptron, right?
Like, either the answer is 0 or it's 1.
For perceptron, we had, like-- oh, either it was minus 1
or it was 1.
But it was, like, either this or that, and that was all.
And, though, the only algorithm we
have for training perceptrons is the perceptron algorithm,
which is like a miracle but it only trains perceptrons.
We can't apply gradient descent to the step function
because the gradients are not helpful, right?
It's always 0.
So you could see the sigmoid function
as kind of like a smoother version of a step function.
But it's got derivatives.
And it means that if you're, like, over here,
you can tell how you would like to change things
if you want to change what category you're
assigning to something.
So you can think of it as a soft step function
or something like that.
So sigmoid is useful in that way.
Another thing that in fact is actually
now everybody's favorite activation function,
except for some cases, ReLU(z).
So this is ReLU(z), Rectified Linear Unit,
and it looks like this.

So again, this is z and this is A. It looks like--

so you can't really see that because it's--
OK, so ReLU(z).

So it's the max of 0 and z.
So it's linear.
Plain old linear if you're bigger than z and 0 otherwise.

And it is really interesting from a mathematical perspective
what it's like when you put a bunch of these guys together.
It is nonlinear, right?
So it's linear in this part of the regime,
but over here, it's off.
And if you take a neural network and use ReLU(z)'s for the f's,
what you get as output is something
that's piecewise linear.
You get a function that's, like, got pieces that are linear.
So it's not curvy.
It can't be curvy, right?
Because it's only ever 0 or a linear function.
But whether you're in the 0 regime or the linear regime
kind of argues whether this particular unit is contributing
or not to the output.
So some of the units now won't contribute to the output
if their input puts them over here.
So ReLU(z)'s are interesting and people
are using-- they're easier to study,
theoretically, in a bunch of ways,
and they also are kind of practically useful.
So by default, this is the nonlinearity that people tend
to use inside as the f's.
But we'll talk about sigmoid because it's
good for a different purpose.


### Lecture: Neural networks - basic element


LESLIE KAELBLING: So now we come to the tricky business, which
is, how do we train this thing?
Assume that we have a data set.
We have some data set.
Usual story.
So we're going to do supervised learning.

For right now, we'll think of it as doing regression, but maybe
with multiple outputs, right?
So you could do regression where you're
trying to predict a whole vector of outputs.
Or we could think of it as a regression
where there's only one unit in the last hidden layer.
It doesn't really matter.
So we're going to think, though, about supervised learning.
So we have some data set, and our data set still
looks like x1, y--
that's a 1--
1.

x-- oh gosh, so now I'm really stuck because we were using n.
I'm going to make it n star.
I don't know.
We were using n for the number of data points we have,
and now we've totally co-opted n for the number of outputs,
which is sort of frustrating, but that's what we had to do.
OK, so we have a data set and we're
going to supervise learning.
We're going to fix a structure of a feed
forward neural networks.
We're going to pick a number of layers
with a number of units in each layer,
the activation function to use in each layer,
and a loss function.
And then we're going to try to adjust the weights so
that we minimize the error on the loss function.
We won't talk about regularization today.
We're going to come back to that next week.
So let's assume that we have a loss function.
So that we have a loss function.
Some loss function, loss.
And what we want to do is minimize--
so we want to pick weights.
Sometimes when I just write "w", what I mean is all the weights.
All the matrices, all the vectors, all the weights.
We want to pick all the weights to minimize the sum
i equals 1 to that n star of loss of--
let me write this, neural network of xi, w, yi.

OK, so this is the thing we've been doing all along,
but let me just say now, right?
So when I write neural network of this, what I mean
is the prediction that you make-- the a that
comes out of that function.
So if we put in xi--
training example i-- and we have these weights fixed--
all the weights, fixed to whatever this w is--
then an a will come out.
Right?
So we can call this ai.
We could call it ai.
OK, good.
We could call this ai. ai comes out.
We compare it to yi.
The loss function tells us how mad we are.
And we want to pick our weights so that it will
minimize this on the data set.
OK, so let me just write a little note.
We'll say no regularization for now.

Not because we don't have to think about it,
but because we have enough to think about today.
So you should always wonder whether it's
good to just be minimizing loss on the training data.

OK, so that's our job.
Is that job good with everybody?
Make sense?

OK, so how are we going to do that?
We are going to do that by gradient descent.

It would be totally possible, though somewhat inconvenient,
to just take the gradient descent code that you've
been working on this week, good old gd, which takes f and df
and does what it does, and solve this problem.

There's nothing wrong with that.
It just could be a little tedious
to think of stacking up all those weight matrices and all
those vectors as a big, gigantic parameter director,
and then figuring out what the gradient is, right?
So you could totally do it.
It's totally fine.
Mathematically awesome but kind of awkward.
So what we're going to do is look
at a kind of very beautiful way for computing the gradients.
You never thought you'd hear of a beautiful way for computing
gradients, but that's what we're going to do.
So we need to compute--
so we need the gradient with respect
to the weights of the loss of neural network x.
Nah, I'm just going to write x here.
For some x and w and some y.

So if we can compute this, we can do gradient descent, right?
That's what we need.

And let me see.
I think in the last lecture, we didn't exactly
end up discussing stochastic gradient,
but it's described in the notes, so you should go and read
about that.
But basically, as long as we can compute
the gradient with respect to one point in our data set,
we could compute this whole batch gradient
if that's what we wanted.
Or we could do stochastic gradient descent, which
is where we randomly pick out an x, y pair from my data set,
make a little step based on that guy,
pick out another one at random, make a little step, and so on.
So if we can compute this, we can do stochastic gradient,
or we could do gradient with respect
to the whole sum over the data because the gradient is just
the sum of the gradients.
OK, so how are we going to compute this?

And so there is a super nice algorithm called back
propagation.

Or error back propagation if you feel a bit more careful.
Or people just call it back prop.
But error back propagation, really, it
is just computing the gradient.

But it has a nice structure.
So another name for error back propagation is the chain rule.


### Lecture: Neural networks - basic element


LESLIE KAELBLING: So calculus, imagine
that a is some function f of b, and b is some function g of c,
so that a is f of g of c.
Imagine that.
And imagine that we want to know the derivative of a
with respect to c, then that's the derivative of a
with respect to b times derivative of b with respect
to c.
And you can write that as f prime of b g prime of c.

You must all remember that.
But still, it's good to be reminded.
And what I'm going to do today is
sketch out the structure of the algorithm and the way
that things propagate and so on.
When you learned this-- probably most of you
learned this where everything was a scalar.
And now we're going to do it with matrices and vectors.
And it's going to make you slightly nervous.
And we're not going to get the transposes right.
I'm just going to try to get the structure right.
And in the homework, you can go through it in super detail
and get all the orders of the matrix multiplies
and the transposes right.
So I'm not going to get that right right now because it
will distract us from the bigger picture, I think.

So let's go.
Let's go.

How are we getting to this?
I'm always at a loss for the-- ha ha,
at a loss for the best way to do this.
I'll do a little piece of it.
And then I'll do a bigger thing.
And then we'll see how it relates
to that picture over there.
So let's just do one step.
So imagine that I want to compute--

eventually-- so remember that in gradient descent, what
we need to know is for each parameter d loss d
parameter, the partial derivative of the total loss,
like how mad we are about this output that we just generated,
how much did this parameter affect
the badness of our answer?
That's what we want to know.
And then we're going to take a step in the opposite direction
in the weight space to try to make it not be so bad.
So we want to make this guy contribute less to the loss
than he does right now.
But to figure that out, we have to figure out
how much does he contribute to the loss and in one direction.
So we're going to have to compute
d loss dw for all our w's, for all the weight matrices,
and all the weight vectors, and everything in the whole model.
But let's just think about the last one
because it's a good place to start.
So there we are.
Here's our model over there.
Weight l is the last weight matrix.
And we can ask how much--
how much did it contribute to the loss?
So let's use the chain rule.
So, d loss/dw, well we can see that dw depends on--
we can see that--
that the way the loss depends on w is that--

let's write down an intermediate result. So I'll write down--

I don't know.
We'll just do it this way, d loss/dA^L times dA^L--
we'll just do it in the steps-- dw^L.
So chain rule time.
So the loss, the actual loss that comes out of the--
that comes out of that last box, depends
on that last activation.
So d loss, d activation, that's just the derivative
of our plain old loss function.
And then deactivation dw, that's how do those last activations
depend on the w's.
Now, we don't know that directly quite yet.
So this we can deal with.
This is just-- this is just loss function prime really.
It's just, how does loss function
depend on its first argument?

And if this is a scalar--
let's just imagine for right now that there's only
one output unit.
a is a scalar.
The loss function is something that takes two scalars, maybe
a squared error for regression.
We already know how to take the derivative of a squared error,
so that's just the derivative of the loss function.
So that's not so bad.
Now, dA/dw, we don't know that quite yet.
But we can see that A gets computed via Z.
So we can turn this guy into da/dz and dZ/dw.

Chain rule again.
It's only the chain rule.
It's only the chain rule.
That's all we're ever going to do now
for the next couple of weeks or maybe the rest of the semester
is the chain rule.
But let's see how this goes.
So dA^L/dZ^L, and dZ^L/dw^L, let's think about what each
of these things is.

So dA^L/dZ^L, how do we get A from Z?
We apply f element-wise.
So in general, if A^L is a vector and Z^L is a vector--
maybe if we just have one output unit, these would be scalars.
But if we have a vector of outputs,
which would also be OK, then this is a vector
and this is a vector.
Generally speaking, d vector d vector is a matrix.
So this is going to be an N^L by N^L matrix.

And what goes in each place in this matrix
is a partial derivative.
It's the partial derivative of a_i with respect to z_j.
Those are the elements in this matrix.
So-- well, let's see.
So because this is an element-wise--
we just take the a's in each--
we take the Z, the vector of z's, and we
apply f to each one.
So the a, the third a, depends only on the third z.
It doesn't depend on anybody else.
So this matrix is going to be 0 everywhere but the diagonal
because a_2 doesn't depend on z_4.
a_2 only depends on z_2.
And how it is a_2 depend on z_2?

f prime.
So this is going to be a matrix with 0's everywhere,
and the f prime of z for each z on the diagonal.

Does that make sense?
You can ponder this more if you want to.
And then again, I'm doing the cartoon version of the calculus
just so that we can get the feeling.
So we'll stick with the cartoon version here just a little bit.
So we have a loss.
That's, again, how mad are we?
Generally speaking, this will be N^L by 1, the loss.
This will be some kind of an N^L by N^L thing.
What's this, dZ/dw?
If these were not matrices and stuff, what would this be?
How do we get Z?
How does Z depend on w?
What is Z?

It's w times A^L minus 1.
So we know that Z, Z here, is W transpose A^L minus 1.
And so if you squint for a minute, and you say, hm, well,
how does that depend on the w's?
You might say, ah, hm, looks like A^L minus 1 to me.
And that's right.
So this is A^L minus 1.

Any questions?
I have a little bit more intuitive story
to tell you about this.
Maybe I will just to get a feeling.

The feeling for how much I want--
this is, again, how much do I want
to change a particular element in the weight matrix?
So, one way to think about how much I
want to change a particular element in the weight matrix
is that it's going to depend on an entry in this vector.
This vector is the "how mad are we" vector.
I wrote a little story in the notes.
People talk about back propagation.
I like to think about it as blame propagation.
So we just got a loss, so if somebody hit us in the head.
And we are trying to figure out who to blame.
This is all about who to blame.
Which weights are the ones who caused this trouble?
And how can we fix it?
So we just got hit on the head, and we
want to know how much are these weights to blame.
So first we figure out, well, if I had a bunch of activation,
if my A is the whole vector, I might say,
well, some of these vector entries
are bigger trouble makers and other ones.
So that's d loss/dA^L. So that's of the outputs,
who caused us the most trouble?
This is how much are the inputs on just to this layer?
A^L is like-- so maybe if most of these guys are 0,
they won't matter.
If they're small values, they won't matter.
If they're big values, they will matter.
And if we actually make a--

let's do this.

Actually, I wrote a picture of my notes that I'm fond of.

In some sense, you could say that for any element in w--
so let's say this is w^L now.
And we're trying to think about d loss/dw^L,
so there's some element in dw^L. And for that element,
there are two things that contribute to d loss/dw^L So we
are actually interested in d loss/dw^L.
So what contributes to that?
In one direction, what contributes is d loss dA^:L.
Again, this is how much did the individual outputs make us mad?
And then another thing that contributes here is just A^L
minus 1.
This is coming in because this weight is the weight that turns
A^L minus 1's into L's.
The weight matrix takes the previous layer
and turns it into the new layer.
So this weight was causing a lot of trouble
if it produced an output that we hated, and it
had a lot of input.
That means it had a big influence.
And it means that if we were to decrease its influence,
we would do a better job on the L.
So I encourage you--
I have to do this myself.
So when I am working-- when I'm trying to prepare this lecture,
and I am lost in a sea of N's and L's, and M's, and A's,
and Z's and whatever, the only way
I can convince myself that I'm not messing up
is by telling myself stories like this, drawing a picture,
thinking about, OK, which values contribute to this?
Is it sensible?
Is that the right sign?
Partly I want to teach you guys machine learning.
But I also want to teach you to be a person who ask
these questions of yourself.
You'll also notice maybe in the notes
that I'm sticking in all these study questions.
They're the questions that I think
about when I'm trying to be sure that I am being sensible
when I write this stuff.
And so they are the kind of questions
that you should train yourself to ask and answer yourself.
So that's a little meta conversation
about learning things.
So anyway, this morning, I drew this picture for myself.
And it made me happy.
I don't know if it makes you happy or not.

So where are we?
So chain rule.
So chain rule-- so let me just now write a big old chain rule
thing here on the board.
So this was for computing the derivative
of the loss with respect to the weights in the last layer.
That seems not so hard.
That's just like only having one layer.
It's not so different than just doing regression or something
like that with a little nonlinearity at the end.
But we have to get it all the way through.
We have to compute the gradient with respect
to all the weights in the whole network.
But luckily, the chain rule will do it for us.
And so we'll end up with a thing that looks like.
So d loss-- let's think about d loss/dw^1.
These are the weights in the first layer.
So you say, oh my goodness, how is it
that changing one little weight in the very first layer
changes my answer and not just changes my answer,
but changes my loss?
If we're going to do gradient descent,
we have to figure that out.

Here we go.
I'm just going to write it down.
It's beautiful.

You can guess.

That's the answer.

It's not surprising, that if the way we get the output
is just this big, old long composition of functions,
then we can compute the derivative
by decomposing, going backwards.
So that's why it's called back prop.
And the way to think about it is that--
and in fact, when we do this in the computer--
when we're training, we'll do first what's called the forward
pass, which you might not be surprised to find out is taking
the x, multiplying by the w's--
f, w, f, w, f, w, f--
until you get this output here.
So that's the forward pass.
You can compute the loss.
And then in the backward pass, you just
have to actually pass values backwards.
Let's see, here's my figure.
This is a figure I'm going to make better.

So what goes back here is d loss dA^L. We tell this guy, hey,
this is how the loss depends on all the outputs that you
generated.
So you please do something about that, if you can.
Now, f does it have any parameters to change.
So f can't adjust parameters itself.
But what it can do is pass the blame.
So it's going to pass back the blame.
It's blame is going to be d loss--
always d loss-- dZ^L. So it says, yo, you guy,
you generated dZ's for me, and they caused this trouble
downstream.
This is the way in which the dZ's
you generated caused trouble.
Please make it better.
At this point, this module can compute d loss d weights.
So at this point, this guy can update by computing d loss d
weights^L and d loss/ d weights^0_L.
He has enough information to compute his gradient update.
So he says, oh, OK, you're right.
I was a contributor to this badness that we just generated.
And so I will update my weights.
But it was not all my fault because this guy gave me
really bad inputs.
And I'm going to tell him exactly how his inputs were
bad.
So I'm going say, d loss dA^: minus 1.
So now I tell the layer before me, look, you generated--
you generated inputs that cause this trouble downstream.
So please make it better.
And we do this all the way back.
So this guy is going to get as input d loss/dA^2.

And then here is going to be d loss/dZ^2.

When this guy gets d loss/dZ^2, he can update,
and so we pass it all the way back to here,
and we can update these ones.

Does that-- any questions about this at least
at the high level?

So it's really-- it's really none
of the pieces are complicated.
And if you just look at a little piece at a time, you can--
the only thing that remains to figure out
is exactly which transposes you need where
and a few things like that, what order
to multiply the matrices so everything matches up and works
out.
And it's a very simple computational setup.


### Lecture: Neural networks - basic element


LESLIE KAELBLING: What we know now is that for any given
training example, we can propagate it forward
to compute the output, compute the loss that we have
for that output, go back, compute the gradients,
and update the weights.
So we know how to do one weight update based on an input point.
So there's a couple of other important things
to think about.
And one is initialization.

So far the algorithms we've looked at
have not been very sensitive to the initialization.
We know the perceptron.
You could actually initialize it to anything if you wanted to.
But we could just set it to zero, and that worked great.
In linear regression the function we're optimizing
is convex, which means also that it's not
sensitive to the initialization.
Neural networks-- if you think again of the function
that we're optimizing-- let's think about,
what does that even mean, what function we're optimizing?
So there is it conceptually.
There's a space of weights.

And there's a space of the sum of the losses on our data set,
if those are the weights.
And this is some--
I don't know-- crazy function.
Oh, it is a function though, so it doesn't go back on itself.
So the weights-- first of all, it's
an enormous dimensional space.
And it is not convex, no matter what your loss function is.
Well, it's convex if your loss function is something simple,
like quadratic, and your activation functions
are linear.
But then we already decided that that
wasn't an interesting case.
So not convex, it matters where you start.
So we saw that in the lab, that if you
do graded descent, depending where
you start you will end up in some kind of a local optimum.
So it matters where you start.

And there's two ways in which it's
easy to think about how it matters where you start.
One thing is that you want your weights
to be initialized randomly.

So the idea here is that we're learning this very complicated
function, which is a function of functions
of functions of functions.
So it's this big, nested thing.
And intuitively you could imagine
that if you're computing a bunch of intermediate results
and you're going to use them to do a later computation,
it is uninteresting if all those intermediate results end up
being the same.
So you want to start all the little pieces of network
out with different initial values
so that they can come to be something different.
You want all those units not all just
to come to be the same unit.
That would not be helpful.
So it's important to initialize the weights at random.
That's part number one.
And then the other thing is not too big.

And there's a more detailed discussion
in the notes of what not too big means,
but let's just discuss it by looking particularly
at this guy.

But actually-- yeah, mostly this one.

For this activation function, if the input has big magnitude--
so if the weights have big magnitude, generally speaking,
you might expect the input to the unit
to have a big magnitude.
And if the input to the sigmoid has big magnitude,
the derivative is what?
Zero, roughly, very small.
If the derivatives are very small,
then you will make minuscule progress toward your goal.
So it's nice to be in this part of the regime of your sigmoid,
if you're using sigmoids.
It's nice to have-- the weights have a small magnitude.
For ReLU, you want to at least be sure
that not everybody is over here.
If you initialize all your weights to zero
and you're using ReLU, then everything will be zero,
and the derivatives will be zero, and it will be zero,
and nothing will happen.
So that's not a good thing.
So there's a discussion in the notes in a little more detail
about the initialization.
But it's really critical to do it randomly.
And generally speaking, you want to keep the weights small.
And the way you think about small is you
think about how many inputs you have to a unit
and what their magnitudes might be.
So it's something you have to think about when
you design your network.

Just a little thing about the non-linearity
and a thing about the last time neural networks were cool.
Last time neural networks were cool,
they were plagued, plagued by the problem of local optimum.
All the time we would make a network,
we would to try to chain it.
It would get stuck in a local optimum
and not find this tasty answer over here.
And that problem has gotten better.
And that's a bizarre thing to say,
because it's not as if we're better
at doing global optimization of really complicated functions.
So what's going on?
Well, so what's going on is a matter
of hot debate in the theory of neural networks, actually.
So it's a current research question.
But I think there's an intuitive understanding, which
is the following.
We now make the networks-- tend to make
these networks really big.
The hidden layers can have hundreds, or thousands,
or 10,000 units, really big.
And generally speaking, they're way bigger than--
in some sense, it feels like they need to be--
you might say, oh, here's a function
I think I need to compute.
I think it's not too complicated.
Maybe I should just make my network about that function
that I think I need to compute.
If you do that, you are likely to get stuck in local optimum.
If, on the other hand, you make your neural network too big,
maybe even a fair amount too big,
bigger than you think you need in order
to compute the answer you need to compute,
one way to think about it is that it has lots
of chances to find the answer.
You could think of it as this is not just one network.
It's a whole bunch of networks all going in parallel.
And only one little piece of it has
to get a hold of the right part of the space, and it can win.

So interestingly, this problem of local optima
is not nearly as bad as it used to be because we're
making the networks bigger.
Now, making networks bigger is its own potential concern,
and we might in particular worry about regularization.
And so that's something that we'll
come back to and talk about.
So local optima are still a problem, but maybe not as much
of a problem as they used to be.


### Lecture: Neural networks - basic element


LESLIE KAELBLING: OK, so when you design a neural network,
you pick the activation functions.
Right?
So you pick the f's.
And when you have a problem that you're trying to solve,
you pick a loss function, right?
And we talked already a little bit about how the loss function
should be in some way matched to the problem
that you're trying to solve, because that's
where you're saying, look, you generated this answer
and I wanted that one.
How unhappy am I?
Right?
That's what the loss function is about.
And that's kind of a domain-dependent thing.
There are some standard loss functions,
but you have to think about your problem on what loss
function makes sense.
But it happens that certain activation functions
in the last layer--
so, generally, we'll pick the last-layer activation because
of the type of our y of the final output
that we're trying to generate and/or because of a loss
function that we have picked.
That they kind of go together.
And then we will generally use some other activation function,
usually ReLU or sigmoid, in the rest of the network
because there's no particular reason
to make one layer different than another one.
Except the last layer is special because it
has to produce outputs of the kind that, like, our y's are.
Right?
Because we're going to compare what comes out
of the last-layer activation with the y's
in the training set.
So they have to be matched in some kind of nice way.
So for instance, if we're just doing plain-old regression,
it might make sense to use squared loss.
We talked about that last week.
And if we're doing regression and our outputs
can be real value, any real value, sigmoid
is a terrible activation function
to use in the last layer because it can only
produce values between 0 and 1.
ReLU is not a good choice either because it can only
produce positive numbers.
So generally speaking, if we're doing regression,
we would like our last-layer activation function to actually
just be the identity, right?
Now we proved that you shouldn't use this everywhere,
but if what you want to come out is just real numbers,
then you'd like the last-layer activation function
to be the identity.
OK.
What I want to do is introduce another idea,
which is a way of doing classification.
Actually, let me-- well, we've already
talked about classification with hinge loss.

If we do classification with hinge loss,
again, we need the output to be a general purpose real number.
Right?
It needs to go positive and negative.
So if using hinge loss, you should also
use f of x equals x in the last layer.
But what I just want to do is introduce you
to one more idea of a loss function which
is useful for classification, which is called negative log
likelihood.

So it's good for classification.
And it's useful when our last-layer activation
is sigmoid.
They go together.

So one thing about a sigmoidal activation function
is that it produces values between 0 and 1,
and one way to interpret values between 0 and 1
is as probabilities.
So if we could interpret as a probability.

All right.
Chain rule can go.

So let me talk about this loss function very briefly.
It has two good properties.
One is that the probabilistic interpretation is nice.
The other one, which you can read about in the notes,
is that it extends easily to multi-class.

So we haven't really talked much about
multi-class classification, but sometimes you
have discrete outwards but you have more than two.
Right?
You might want to say, oh, there's
four different types of people, or something,
and I would like to have four different possible outputs.
Or there's 10,000 different kind of categories
my images could be in, so I would need to somehow pick 1
out of 10,000 as my output.
So, in the notes, there's a discussion
of softmax as an activation function.
And this negative log likelihood extends nicely to that case.
So you'll have to read about that
and do some homework problems.
But let me just illustrate the negative log likelihood thing.
So imagine that you had a guess which
is in 0 to 1, and your actual label--

so this is going to be y, right?
The y for your training example.
And this is unfortunately called a. aL, the output.
So we had a particular y, which is the output that we wanted.
We put an x into our network and we got an a that came out.
And let's imagine that it's just one-dimensional
and that our f was a sigmoid, so that our fL, our very last f,
was a sigmoid.
So that our guess--
our thing that's coming out of the network-- is in the range 0
to 1.
And let's imagine that our actual values--
before, for classification, we've
been using plus 1 minus 1, but for this purpose--
and you have to keep this straight in your head
when you're setting up a problem.
For this purpose, it's better if your labels are 0, 1.
In fact, it's required.
OK.
So you'd say our labels are 0 and 1.
Either I am not in the class or I am in the class.
And the output is going to be something between 0 and 1.
And you can now interpret all of these things
as probability that this particular x
is in the positive class.

OK.
So our loss function.
What should the loss function be?
So imagine that we predict some value, p.
You could think of it as p.
As a probability.
If we predict value p and it's true,
then we want the p to be as high as possible.
If it should be true, then we want our predicted p
to be close to 1.
And if it should be 0, then we want our predicted p
to be what?
So you can think of our loss function--
I'm going to write it in two different ways.

One way to write it is that our loss should be the negative--
because loss is unhappiness--
of the guess if actual is 1, and 1 minus guess otherwise.

Right?
So if we're supposed to predict 1, then our loss--
unfortunately, the sign is not--
this is how happy we are, right?
We want the guess to be high if the actual's 1.
So if we negate this, then this gives us
a loss instead of a happiness.
And the cool thing is that you can write this in another way.
Right?
So another way to write this--
it's sort of the same thing exactly--
is guess to the actual--
guess raised to the actual power--
times 1 minus guess to the 1 minus the actual power.
OK.
Why is this the same as that?
This is a cheap trick.
Actual's either 0 or 1.
So one of these guys is going to be 1
and the other one is going to be 0.
The one that's 0 causes us to multiply by 1.
So this is just another way of writing that
but it kind of looks nicer.
And I might take like one more minute, and then I'm out.
OK, so that would be one loss function that we could use.
It's still a little bit unbeautiful for some reasons,
again, that I describe in more detail in the notes.
What we're going to do is instead now
take the log of this thing.

Log is a monotonic transformation,
so if you optimize the log of a function,
you'll get the same optimizer as if you
optimize the function itself.
And if we take the log of this, we
get actual times log of guess plus 1 minus actual times
log of 1 minus guess.

And again, this actual is either 1 or 0,
so we'll either get one of these terms.
If we wanted the answer to be 1, we'll get log of guess,
and if we wanted the answer to be 0,
we'll get log of 1 minus guess.
And we're going to negate the whole thing.

So if you try to minimize that, if that's your loss function,
then you have-- this is negative log likelihood.
And that's a good thing to use for classification,
but you have to take care that you're y's are in 0, 1
and you've used a sigmoid as your output f.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/9c36c444e5df10eef7ce4d052e4a2ed1/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Neural_Networks.pdf
