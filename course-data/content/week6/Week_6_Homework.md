This homework builds on the material in the notes on
neural networks up through and including section 6 on loss functions.

In particular, in this homework we consider neural networks with
multiple layers. Each layer has multiple inputs and outputs, and can
be broken down into two parts:

* A linear module that implements a linear transformation: $z_j = (\sum^{m}_{i=1} x_i w_{i,j}) + {w_0}_j$ specified by a weight matrix $W$ and a bias vector $W_0$. The output is $[z_1, \ldots, z_n]^T$.

* An activation module that applies an activation function to the outputs of the linear module for some activation function $f$, such as Tanh or ReLU in the hidden layers or Softmax (see below) at the output layer. We write the output as: : $[f(z_1), \ldots, f(z_m)]^T$, although technically, for some activation functions such as softmax, each output will depend on all the $z_i$, not just one.

We will use the following notation for quantities in a network:

* Inputs to the network are $x_1, \ldots, x_d$.

* Number of layers is $L$

* There are $m^l$ inputs to layer $l$

* There are $n^l = m^{l+1}$ outputs from layer $l$

* The weight matrix for layer $l$ is $W^l$, an $m^l \times n^l$ matrix, and the bias vector (offset) is $W_0^l$, an $n^l \times 1$ vector

* The outputs of the linear module for layer $l$ are known as pre-activation values and denoted $z^l$

* The activation function at layer $l$ is $f^l(\cdot)$

* Layer $l$ activations are $a^l = [f^l(z^l_1), \ldots, f^l(z^l_{n^l})]^T$

* The output of the network is the values $a^L = [f^L(z^L_1), \ldots, f^L(z^L_{n^L})]^T$

* Loss function $Loss(a, y)$ measures the loss of output values $a$ when the target is $y$

1) Loss functions and output activations: classification

When doing classification, it's natural to think of the output values as being discrete: +1 and -1. But it is generally difficult to use optimization-based methods without somehow thinking of the outputs as being continuous (even though you will have to discretize when it's time to make a prediction).

1.1) Hinge loss, linear activation

When we looked at the SVM objective for classification, we did this:

* Defined the output space to be $\mathbb{R}$

* Developed the hinge loss function

$$
Loss(a, y) = L_h(y a) = \begin{cases} 0 & \text{if $y a$ > 1}\\ 1 - ya & \text{otherwise}\end{cases}
$$

where $a$ is the continuous output (we're using $a$ here to be consistent with the neural network terminology of activation) and $y$ is the desired/target output

* Tried to find parameters $\theta$ of our model to minimize loss summed over the training data

Consider a single "neuron" with a linear activation function; that is, where $a_1^L = \sum_k w^L_{k,1} x_k + {w_{0,1}^L}$. In this case, we have $L = 1$ and $f^L(z) = z$.

1.1.A)
Write a short program to compute the gradient of the loss function with respect to the weight vector (not the bias): $\nabla_{w^L} Loss(a_1^L, y)$ when $Loss(a, y) = L_h(y a)$.

* x is a column vector

* y is a number, a label

* a is a number, an activation

It should return a column vector.

123def hinge_loss_grad(x, y, a): passXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Run Code
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1.2) Log loss, sigmoidal activation

Another way to make the output for a classifier continuous is to make it be in the range $(0, 1)$, which admits the interpretation of being the predicted probability that the example is positive. A convenient way to make the activation of a unit be in the range $(0, 1)$ is to use a sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

The figure below shows a sigmoid activation function on the left, with the rectified linear (ReLU) activation function on the right for comparison.

1.2.A)
What is an expression for the derivative of the sigmoid with respect
to $z$, expressed as a function of $z$, its input?

Enter a Python expression (use ** for exponentiation) involving e and z:

Check Syntax
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1.2.B)
What is an expression for the derivative of the sigmoid with respect
to $z$, but this time expressed as a function of $o = \sigma(z)$, its
output?

Hint: Think about the expression $1 - \frac{1}{1 + e^{-z}}$. (Here is a review of computing derivatives.)

Enter a Python expression (use ** for exponentiation) involving only o (note: eand z are not allowed, and remember $o = \sigma(z)$):

Check Syntax
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

In this model, we will consider positive points to have label +1, and negative points to have label 0.

We need a loss function that works well when we are predicting
probabilities. A good choice is to ask what probability is assigned
to the correct label. We will interpret the value outputted by our
classifier as the probability that the example is positive. So, if
the output value is $a$ and the true label is $+1$, then the
probability assigned to the true label is $a$; on the other hand, if
the true label is $0$, then the probability assigned to the true label
is $1 - a$. Because we actually will be interested in the probability
of the predictions on the whole data set, we'd want to choose weights
to maximize

$$
\prod_t P(a^{(t)}, y^{(t)})
$$

where $P(a^{(t)}, y^{(t)})$ is the probability that the network predicts the
correct label for data point $(t)$.

Using a notational trick (which turns an if expression into a
product) that might seem unmotivated now, but will be useful later, we
can write the probability $P(a, y)$ as

$$
P(a, y) = a^y (1 - a)^{(1 - y)}.
$$

1.2.C)
What is the value of $P(a, y)$ when $y=0$?

Enter an expression for $P(a, y)$ when $y=0$ in terms of a:

Check Syntax
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1.2.D)
What is the value of $P(a, y)$ when $y=1$?

Enter an expression for $P(a, y)$ when $y=1$ in terms of a:

Check Syntax
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1.2.E)
Find a simplified expression for $\log P(a, y)$ that does not use exponentiation. Note that we
refer to the natural logarithm $\ln$ as $\log$ throughout this assignment, consistent
with the lecture notes.

Enter an expression in terms of y and a; you can use log(.) to indicate natural log:

Check Syntax
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

In fact, because log is a monotonic function, the same weights that
maximize the product of the probabilities will minimize the
negative log likelihood ("likelihood" is the same as
probability; we just use that name here because the phrase is an idiom
in machine learning, abbreviated NLL):

$$
Loss(a, y) = NLL(a, y) = -y \log a - (1 - y) \log (1 - a).
$$

Our objective function (over our $n$ data points) will then be

$$
\sum NLL(a^{(t)}, y^{(t)}) = -\sum_{t = 1}^n \left[ y^{(t)} \log a^{(t)} + (1 - y^{(t)}) \log (1 - a^{(t)}) \right] .
$$

Remember that $a^{(t)}$ is our model's output for training example $t$, and $y^{(t)}$ is the true label (+1 or 0).

Now, we can think about a single unit with a sigmoidal activation function, trained to minimize NLL.
So, $a_1^L = \sigma(\sum_k w^L_{k,1} x_k + {w_{0,1}^L})$. In this case, we have $L = 1$.

1.2.F)
Write a formula for the gradient of the NLL with respect to the first
weight, $\nabla_{w^L_{1,1}} NLL(a_1^L, y)$, for a single training
example. Hint: consider using the chain rule; the final answer (expression) is very short.

Write an expression in terms of x_1, a_1, and y:

Check Syntax
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1.2.G)
Write a formula for the gradient of the NLL with respect to the full
weight vector, $\nabla_{W^L} NLL(a_1^L, y)$, for a single training
example.

Enter an expression in terms of x, a_1, and y:

Check Syntax
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2) Multiclass classification

What if we needed to classify homework problems into three categories: enlightening, boring, impossible? We can do this by using a "one-hot" encoding on the output, and using three output units with what is called a "softmax" (SM) activation module. It's not a typical activation module, since it takes in all $n_L$ pre-activation values $z^L_j$ in $\mathbb{R}$ and returns $n_L$ output values $a^L_j \in [0, 1]$ such that $\sum_j a^L_j = 1$. This can be interpreted as representing a probability distribution over the possible categories.

The individual entries are computed as

$$
a_j = \frac{e^{z_j}}{\displaystyle\sum_{k=1}^{n^L} e^{z_k}}
$$

We'll describe the relationship of the vector $a$ on the vector $z$ as

$$
a = {\rm SM}(z)
$$

The network below shows a one-layer network with a linear module followed by a softmax activation module.

2.A)
What probability distribution over the categories is represented by $z^L = [-1, 0, 1]^T$?

Enter a distribution (a list of three numbers adding up to 1) for the three categories. Your answers should be numeric (please enter numbers, and do not use the symbol $e$):

Submit
View Answer Ask for Help</button[Truncated]