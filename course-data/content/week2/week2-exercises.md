For these exercises, it will be helpful to review the notes on Linear Classifiers and the Perceptron. You may also find it helpful to write some test code with a local python installation or in a google colab notebook.
1) Classification

Consider a linear classifier through the origin in 4 dimensions, specified by

$$
\theta = (1, -1, 2, -3)
$$

Which of the following points $x$ are classified as positive, i.e. $h(x; \theta) = +1$?

* $(1, -1, 2, -3)$

* $(1, 2, 3, 4)$

* $(-1, -1, -1, -1)$

* $(1, 1, 1, 1)$

Enter a Python list with a subset of the numbers 1, 2, 3, 4:

Submit
View Answer

You have infinitely many submissions remaining.

2) Classifier vs Hyperplane

Consider another parameter vector

$$
\theta' = (-1, 1, -2, 3)
$$

Ex2a

Does $\theta'$ represent the same hyperplane as $\theta$ does?

--
yes
no

Submit
View Answer

You have infinitely many submissions remaining.

Ex2b

Does $\theta'$ represent the same classifier as $\theta$ does?

--
yes
no

Submit
View Answer

You have infinitely many submissions remaining.

3) Linearly Separable Training

As discussed in lecture and in the lecture notes, note that $\mathcal{E}_n (\theta, \theta_0)$ refers to the training error of the linear classifier specified by $\theta, \theta_0$, and $\mathcal{E} (\theta, \theta_0)$ refers to its test error. What does the fact that the training data are linearly separable imply?
Select "yes" or "no" for each of the following statements:
Ex3a

There must exist $\theta, \theta_0$ such that E(θ,θ0)=0

--
yes
no

Submit
View Answer

You have infinitely many submissions remaining.

Ex3b

There must exist $\theta, \theta_0$ such that En(θ,θ0)=0

--
yes
no

Submit
View Answer

You have infinitely many submissions remaining.

Ex3c

A separator with 0 training error exists

--
yes
no

Submit
View Answer

You have infinitely many submissions remaining.

Ex3d

A separator with 0 testing error exists, for all possible test sets

--
yes
no

Submit
View Answer

You have infinitely many submissions remaining.

Ex3e

The perceptron algorithm will find $\theta, \theta_0$ such that En(θ,θ0)=0

--
yes
no

Submit
View Answer

You have infinitely many submissions remaining.

4) Separable Through Origin?

Provide two points, $(x_0, x_1)$ and $(y_0, y_1)$ in two dimensions that are linearly separable but not linearly separable through the origin. If you get stuck try drawing a picture and review the notes on offsets.

Enter a Python list with two entries of the form [[ x0, x1], label] where label is 1 or -1. (So each entry represents a point with 2 dimensions and its label)

Submit
View Answer

You have infinitely many submissions remaining.