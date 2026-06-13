For this homework, you will need to understand the notes on Recurrent Neural Networks.

1) Where is the gradient?

In backpropagation through time, we have a nice recursive equation for $\delta_i^{s_{t-1}}$, which is the degree to which unit $i$ of the state at time $t-1$ was responsible for the losses from time $t$ through $n$ (the end of the sequence):

$$
\delta_i^{s_{t-1}} = \sum_{j = 1}^m \frac{\partial s_{t;j}}{\partial s_{t-1; i}} \left[\frac{\partial}{\partial s_{t;j}} {\rm Loss}(y_t, p_t) + \delta_j^{s_t}\right]
$$

For concreteness, consider a sequence of length $n = 5$ and assume that $s$ has a single dimension, so we can omit that subscript.

1A)
We are interested in the impact of the state at time $1$ on the error at time $5$. Assume that the losses for all times $t < 5$ are 0.

Which of the following expressions are correct formulas for that quantity?

$$
\delta^{s_1} = \frac{\partial s_2}{\partial s_1} \delta^{s_2}
$$

$$
\delta^{s_1} = \frac{\partial s_2}{\partial s_1} \frac{\partial s_3}{\partial s_2} \delta^{s_3}
$$

$$
\delta^{s_1} = \left(\sum_{t=2}^5 \frac{\partial s_t}{\partial s_{t-1}}\right) \left(\frac{\partial}{\partial s_5} {\rm Loss}(y_5, p_5)\right)
$$

$$
\delta^{s_1} = \left(\prod_{t=2}^5 \frac{\partial s_t}{\partial s_{t-1}}\right) \left(\frac{\partial}{\partial s_5} {\rm Loss}(y_5, p_5)\right)
$$

1B)
If the network were actually 20 layers deep, $\frac{\partial s_t}{\partial s_{t-1}} = 0.1$ for all $t$, and $\delta^{s_{20}} = 1$, what is $\delta^{s_1}$?

Enter a number:

1C)
Is this gradient exploding or vanishing?

Pick one:

--
Exploding
Vanishing

2) RNN

Consider a simple recurrent neural network model given by

$$
s_{t} = \text{sign}_0(W^{ss} s_{t-1} + W^{sx} x_t), \;\; t=1,2,\ldots
$$

where we have omitted any offset parameters (set biases to zero) and the non-linear activation function, applied element-wise, is a sign function that also returns zero: $\text{sign}_0(z) = 1$ if $z>0$, $\text{sign}_0(0) = 0$, and $-1$ otherwise. In our problem here, $s\in \mathbb{R}^2$ and $x\in \mathbb{R}$. The parameters are set (not learned) as follows:

$$
W^{ss} = \left[ \begin{array}{cc} -1 & 0\\ 0 & -1 \end{array} \right],\;\;\; W^{sx} = \left[ \begin{array}{c} 1 \\ 1 \end{array} \right], \;\;\; s_0 = \left[ \begin{array}{c} 0 \\ 0 \end{array} \right]
$$

We will use the RNN model to translate variable length input sequences to corresponding state vectors. In other words, a sequence $x_1,x_2$ will be represented by the resulting $s_2$; similarly, $x_1,x_2,x_3$ will be represented by $s_3$, and so on.

Once each sequence has an associated state vector, we can use them, e.g., to train a linear classifier. To this end, our three training examples are binary sequences:

$$
x^{(1)} = (x^{(1)}_1,x^{(1)}_2) = (1,0)
$$

$$
x^{(2)} = (x^{(2)}_1,x^{(2)}_2,x^{(2)}_3) = (0,0,1)
$$

$$
x^{(3)} = (x^{(3)}_1) = (0)
$$

Note that the input sequences do not have to have the same length! (This is an advantage of RNNs).

2A)
What is the sequence of first two state vectors $s_1$ and $s_2$ resulting from feeding the RNN model with the input sequence $x^{(1)}$?

Enter a list of two numbers corresponding to $s_1$:

Enter a list of two numbers corresponding to $s_2$:

2B)
Map all the sequences to their corresponding state vector representations (that is, the final state vector that results from feeding the entire sequence into the model):

Enter a list of two numbers for the final state vector given input $x^{(1)}$:

Enter a list of two numbers for the final state vector given input $x^{(2)}$:

Enter a list of two numbers for the final state vector given input $x^{(3)}$:

2C)
Suppose we train a linear classifier that operates on the state vector representations of these sequences $x^{(1)}$, $x^{(2)}$, and $x^{(3)}$ (run through our RNN). Could the linear classifier separate these three examples for all possible labelings of positive and negative examples?

Could the linear classifier separate these three examples regardless of how they were labeled?

--
No
Yes

2D)
We can think of the RNN model as giving rise to an equivalent feed-forward neural network for any input sequence $(x_1,x_2,\ldots,x_n)$ run through it. Choose True or False for each statement based on whether it is correct for this feed-forward interpretation:

The parameters of each layer are different:

--
True
False

Each input $x_i$ is fed into a different hidden layer:

--
True
False

The number of layers is proportional to the length of the input sequence:

--
True
False

Each hidden layer would have two units in case of our RNN above:

--
True
False

3) State

We consider a language model based on sequences of characters.
We construct a classifier on the state

$$
s_t = {\rm tanh}\left(\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} s_{t-1} + W^{s, x} x_t\right)
$$

such that positive classification is decided if $[1, 1] \cdot s_T > 0$
and negative classification is if decided $[1, 1] \cdot s_T < 0$ (for
$s_T$ at the end of the word).

Find $W^{sx}$ such that 'draw' and
'saw' result in positive classification and 'paw' and 'raw' result in
negative classification. Assume $s_0 = [0, 0]$.

Assume our alphabet
only has the letters 'a', 'd', 'p', 'r', 's', and 'w' and that when we
make a one-hot encoding, we do it in that order.

3A)
What is the shape of $W^{sx}$?

Enter a Python list of numbers corresponding to the shape of the matrix:

3B)
Provide a $W^{sx}$.

Enter the matrix as a list of lists; one list for each row of the matrix:

<img src="data:image/gif;base64,R0lGODlhEAAQAPIGAMLCwkJCQgAAAGJiYoKCgpKSkv///wAAACH/C05FVFNDQVBFMi4wAwEAAAAh/hpDcmVhdGVkIHdpdGggYWpheGxvYWQuaW5mbwAh+QQJCgAGACwAAAAAEAAQAAADMmi63P4wyklrAyEAGoQInAdOmGYBw7AxwLoMGcG2rkHEQFHQLTsQOd2mB9ERCpTWzpEAACH5BAkKAAYALAAAAgAKAA4AAAMraAYRoNAEIUJUs97VHgTD4EVDQ2xEM2wgMV5AUbyKLKNEvoxA3P8sYNCQAAAh+QQJCgAGACwAAAAACgAOAAADLWi6EAFrBSGCAmQ0as1wROFABuEM0TUQ5FUU7fK+aRkWNYDFqV4bOl8v+BMuEgAh+QQJCgAGACwAAAAADgAKAAADKmi6QAMrrhECkaaVVl+FRiFuAwEEghAoYxGhqgI0oPxlNSbPOcb3PqAkAQAh+QQJCgAGACwCAAAADgAKAAADKWhqUAUrLuekApA+MiDD4BYExAVGwzgsmNR0lgWMXmwEghDYCq7zDFoCACH5BAkKAAYALAYAAAAKAA4AAAMqaADWros9GEuRUBE7jeUTYGEhMZANEQREN6xDJ54PsKJGIAhBp/OyWyMBACH5BAkKAAYALAYAAgAKAA4AAAMpaKoA+609Fie1C5Tipt7WRhRWw0ED0T1DEAyMq7mEEghCAKTdnZcySwIAIfkEBQoABgAsAgAGAA4ACgAAAytoumwALb4X2YR1U[Truncated]