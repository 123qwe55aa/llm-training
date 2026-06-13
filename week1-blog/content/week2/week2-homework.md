This homework will aid in your understanding of the Perceptron Algorithm by having you:

* practice applying the perceptron algorithm to toy data

* implement the perceptron algorithm and one of its variants

* apply your perceptron implementation on larger, more interesting data sets

In addition to the Numpy functions and features mentioned in hw 1, here are some you should be familiar with for this assignment:

* np.argmax

* np.array_split, look at the axis argument

* np.concatenate, look at the axis argument

* np.zeros

* numpy.ndarray.shape

* numpy logic functions
e.g. np.array([[1, 2],[3, 4]]) == 3

For this homework, it will be helpful to review the notes on perceptron.

For problems 7-10, here is a code file that will be useful for debugging on your computer; alternatively, you may find it helpful to use this preformatted google colab notebook.

Note: for all of the problems in this assignment, you are allowed to use for loops.

1) Perceptron Mistakes

Let's apply the perceptron algorithm (through the origin) to a small training set containing three points:

$i$
Data Points $x^{(i)}$
Labels $y^{(i)}$

$1$
$[1, -1]$
1

$2$
$[0, 1]$
-1

$3$
$[-1.5, -1]$
1

Given that the algorithm starts with $\theta^{(0)} = [0,0]$, the first point that the algorithm sees is always a mistake. The algorithm starts with some data point (to be specified in the question), and then cycles through the data until it makes no further mistakes.

1.1) Take 1

1.1a) How many mistakes does the algorithm make until convergence if the algorithm starts with data point $x^{(1)}$?

Number of mistakes is:

(1) (2)

(3) (4)

1.1b) Which of the above plot(s) correspond to the progression of the hyperplane as the algorithm cycles? Ignore the initial 0 weights, and include an entry only when $\theta$ changes.

Please provide the plot number(s) in the order of progression as a Python list.

1.1c) How many mistakes does the algorithm make if it starts with data point $x^{(2)}$ (and then does $x^{(3)}$ and $x^{(1)}$)?

Number of mistakes is:

1.1d) Again, if it starts with data point $x^{(2)}$ (and then does $x^{(3)}$ and $x^{(1)}$), which plot(s) correspond to the progression of the hyperplane as the algorithm cycles? Ignore the initial 0 weights.

Please provide the plot number(s) in the order of progression as a Python list.

1.2) Take 2

Now assume that $x^{(3)} = [-10, -1]$, with label $1$.

1.2a) How many mistakes does the algorithm make until convergence if cycling starts with data point $x^{(1)}$?

Number of mistakes is

1.2b) How many mistakes if it starts with data point $x^{(2)}$?

Number of mistakes is

2) Initialization

2.1) If we were to initialize the perceptron algorithm with $\theta = [1000, -1000]$, how would it affect the number of mistakes made in order to separate the data set from question 1?

It would have small or no effect on the number of mistakes.
It would significantly decrease the number of mistakes.
It would significantly increase the number of mistakes.

2.2) Provide a value of $\theta^{(0)}$ for which running the perceptron algorithm (through origin) on the data set from question 1 returns a different result than using $\theta^{(0)} = [0,0]$. The data set is repeated below:

$i$
Data Points $x^{(i)}$
Labels $y^{(i)}$

$1$
$[1, -1]$
1

$2$
$[0, 1]$
-1

$3$
$[-1.5, -1]$
1

Enter the 2 coordinates of $\theta$ as a Python list or the string 'none' if none exists.'

3) Dual View

The following table shows a data set and the number of times each point is misclassified during a run of the perceptron algorithm (with offset). $\theta$ is initialized to the zero vector and $\theta_0$ is initialized to 0.

$i$
$x^{(i)}$
$y^{(i)}$
times misclassified

$1$
$[-3, 2]$
1
2

$2$
$[-1, 1]$
-1
4

$3$
$[-1, -1]$
-1
2

$4$
$[2, 2]$
-1
1

$5$
$[1, -1]$
-1
0

3.1) What is the post training $\theta$?

Provide it as a python list of the form $[a, b]$.

3.2) What is the post training $\theta_0$?

Provide it as a number.

4) Decision Boundaries

4.1) AND

Consider the AND function defined over three binary variables:
$f(x_1,x_2,x_3)=(x_1 \wedge x_2 \wedge x_3)$.

If you are unfamiliar with the AND function, it simply returns 1 if all three variables are true (value of 1) and 0 otherwise.

We aim to find a $\theta$ such that, for any $x = [x_1, x_2, x_3]$, where $x_i \in \{0, 1\}$:

$$
\theta\cdot x + \theta_0 > 0 \textrm{ when } f(x_1, x_2, x_3) = 1, \textrm{ and }
$$

$$
\theta\cdot x + \theta_o \le 0 \textrm{ when } f(x_1, x_2, x_3) = 0.
$$

4.1a) For each of the combination of values of $(x_1, x_2, x_3)$, that is, $[(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0),(1,0,1), (1,1,0),(1,1,1)]$ enter the values of $f(x_1,x_2,x_3)$.

Please enter the values of $f(x_1,x_2,x_3)$ as a Python list.

4.1b) Assuming $\theta_0 = 0$ (no offset), enter $\theta$ as a Python
list of length 3 or enter 'none' as a Python string (with quotes) if none
exists.

Enter a Python list of 3 numbers or the string 'none'

4.1c) Assuming $\theta_0$ is non-zero (offset), enter a $\theta$ and $\theta_0$ as a Python
list of length 4 ($\theta_0$ last) or enter 'none' as a Python string (with quotes) if none
exists.

Enter a Python list with 4 numbers or the string 'none'.

4.2) Families

You are given the following labeled data points:

* Positive examples: $[-1, 1]$ and $[1, -1]$,

* Negative examples: $[1, 1]$ and $[2, 2]$.

For each of the following parameterized families of classifiers, find the parameters of a family member that can correctly classify the above data, or think about why no such family member exists.

Is there a classifier of the following forms that can correctly classify the above data?

Recall from lecture that we consider a point lying exactly on the separator to be classified as negative.

4.2a) Inside or outside of an origin-centered circle with radius $r$

Enter a value for $r$ or the string 'none' if none exists.'

4.2b) Inside or outside of a circle centered on $x_0$ with radius $r$

Enter a list with 3 entries for coordinates of $x_0$ and $r$ ([x0_1,x0_2,r]) or the string 'none' if none exists.'

4.2c) On one side of a line through the origin with normal $\theta$ (recall normal vector points into positive half-space).

Enter a list with 2 entries for coordinates of $\theta$ or the string 'none' if none exists.'

4.2d) On one side of a line with normal $\theta$ and offset $\theta_0$ (recall normal vector points into positive half-space).

Enter a list with 3 entries for coordinates of $\theta$ and $\theta_0$ ([theta_1, theta_2, theta0])or the string 'none' if none exists.'

4.2e)

Which of the above are families of linear classifiers?
4.2a
4.2b
4.2c
4.2d

5) Separation

Indicate if the following datasets are:

* not linearly separable

* linearly separable without an offset

* linearly separable only with a non-zero offset

HINT: You shouldn't have to work through the perceptron algorithm to figure this out.

5.1) Dataset 1:

$i$
Data Points $x^{(i)}$
Labels $y^{(i)}$

$1$
$[1, -1]$
-1

$2$
$[1, 1]$
1

$3$
$[2, -1]$
1

$4$
$[2, 1]$
-1

This dataset is:

--
Not linearly separable
Linearly separable without an offset
Linearly separable only with a non-zero offset

5.2) Dataset 2:

$i$
Data Points $x^{(i)}$
Labels $y^{(i)}$

$1$
$[1, -1]$
1

$2$
$[1, 1]$
1

$3$
$[2, -1]$
-1

$4$
$[2, 1]$
-1

This dataset is:

--
Not linearly separable
Linearly separable without an offset
Linearly separable only with a non-zero offset

5.3) Dataset 3:

$i$
Data Points $x^{(i)}$
Labels $y^{(i)}$

$1$
$[1, -1, 1]$
1

$2$
$[1, 1, 1]$
1

$3$
$[2, -1, 1]$
-1

$4$
$[2, 1, 1]$
-1

This dataset is:

--
Not linearly separable
Linearly separable without an offset
Linearly separable only with a non-zero offset

5.4) Compare datasets 2 and 3, and see if you can generalize what you learned from those two examples.
Which of the following transformations allow one to transform any dataset
that is only linearly separable with an offset to a dataset that is linearly
separable without an offset, such that the transformation doesn't depend on the values
of the datapoints? Select all which are valid:

This dataset is:
Remove the final dimension of each of the datapoints
Add an extra dimension to all of the datapoints using any (nonzero) numbers
Add an extra dimension to all of the datapoints using the same (nonzero) number for each point
Add an extra dimension to all of the datapoints using a 1 for each point
Add an extra dimension to all of the data points, using 0 for each point
A transformation with these properties is impossible

