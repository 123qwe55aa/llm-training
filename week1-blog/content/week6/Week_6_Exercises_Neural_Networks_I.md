For these exercises, you should review the notes on Neural Networks.
1) Prediction

Consider the following data set:

X = np.array([[0, 1, 2],
[0, 1, 2]])
Y = np.array([[0, 1, 0]])

The columns of X and Y are the data points and corresponding labels.

We will be looking at the behavior of the following simple two-layer network:

Assume that within each layer, each unit has the step activation function $f(z)$ given by

$$
f(z) = \begin{cases}1 & \text{if $z > 0$} \\ 0 & \text{otherwise} \end{cases}.
$$

Let the weights in the first layer (layer 1) be:

* $w^1_{0,1} = -0.5$, $w^1_{1, 1} = 1$, $w^1_{2, 1} = 0$

* $w^1_{0,2} = 1.5$, $w^1_{1, 2} = -1$, $w^1_{2, 2} = 0$

1A)
Enter a matrix $Z$ where each column represents the outputs of the
hidden units ($f^1(z_1^1)$ and $f^1(z_2^1)$) for each of the input vectors in X.

Enter a Python list of lists [[a,b,c],[d,e,f]], each list is a row of the matrix.

View Answer

1B)
Pick weights for the second layer $w^2_{0,1}, w^2_{1,1}, w^2_{2,1}$ so that the desired outputs are predicted correctly.

Enter a Python list of 3 numbers [$w^2_{0,1}, w^2_{1,1}, w^2_{2,1}$]

View Answer

2) Training

Now, we will consider the
classification of a different set of X and Y data,
with a different single layer network having the
following structure and activation function:

$$
z = w^1_{1,1} x_1 + w^1_{2,1} x_2 + w^1_{0,1}
$$

In this network we have $f^1(z) = z$, so our output $a_1^1 = z_1^1.$

Assume the initial weights are $w^1_{0,1} = 1, w^1_{1,1} = 1, w^1_{2,1} = 1$, and the step size is 0.5 (not usually a good idea, but okay for now).

The current training example is $x^{(i)} = [1, 2]^T$, $y^{(i)} = -1.$

2A)
What is the output value $a^1_1$, given current input $x^{(i)}$ and the current weights?

Enter a number

View Answer

2B)
What will the values of weights $w^1_{0,1}, w^1_{1,1}, w^1_{2,1}$ be after one step of stochastic gradient descent at the given training example $x^{(i)} = [1, 2]^T$, $y^{(i)} = -1$ using
our definition of hinge loss $L_h(v) = {\it max}(0, 1-v)$?

Enter a Python list of 3 numbers [$w^1_{0,1}$, $w^1_{1,1}$, $w^1_{2,1}$] (to 3 decimal places)

View Answer

2C)
What would the output value $a^1_1$ be, for this same input $x$, with these new weights?

Enter a number

View Answer

2D)
What would happen to the $v_i$ if we did another SGD update, for that same point, with step size 0.5, as before?

Enter a Python list of 3 numbers [$w^1_{0,1}$, $w^1_{1,1}$, $w^1_{2,1}$]

View Answer

2E)
Now what would the output be?

Enter a number

View Answer

2F)
What if we do one more update, for that same point?

Enter a Python list of 3 numbers [$w^1_{0,1}$, $w^1_{1,1}$, $w^1_{2,1}$]

View Answer