In the first part of this assignment, we will consider several general
issues in representing features and their impact on classification.
In the second part of the assignment, we will experiment with these
strategies in the context of realistic data sets. Please make sure your read the lecture
notes covering feature representations for this assignment.

Feature Transformations

A code file that is required for this assignment can be found
here, and a colab notebook here.

1) Scaling

Consider a linearly separable dataset with two features:

data = ([[200, 800, 200, 800],
[0.2, 0.2, 0.8, 0.8]])
labels = [[-1, -1, 1, 1]]

Consider the separator defined by $\theta = (0, 1), \theta_0 = -0.5$.

In order to apply the perceptron mistake bound (see notes), we transform our problem from $\theta^T x + \theta_0 = 0$ to some $\theta'^T x = 0$. We do this by appending $\theta_0$ to $\theta$, and appending $1$ to $x$, as follows:

$$
\theta'^T x = \begin{bmatrix} \theta_1 & \theta_2 & ... & \theta_0 \end{bmatrix} \cdot \begin{bmatrix} x_1 \\ x_2 \\ ... \\ 1\end{bmatrix} = 0
$$

In this phrasing, our new "$\theta$" is (0, 1, -0.5). For a separator through the origin, recall that the margin of the data set is the minimum of $\gamma = y^{(i)}(\theta^T x^{(i)}) / \lVert \theta \rVert$ over all data points $(x^{(i)}, y^{(i)})$.

For the following questions, assume we are working in the transformed (3d) feature space, with perceptron through the origin, and where if the data has bounded magnitude $R$, then the theoretical upper bound on mistakes made by perceptron is $(\frac{R}{\gamma})^2$, for a separable data set.

You are free to use your perceptron algorithm implemented in the previous homework to answer the following questions. (Some parts require more runs of the perceptron algorithm than one could reasonably perform by hand.)

1A)

What is the margin $\gamma$ of this data set with respect to that separator (up to 3 decimal places)?

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1B)

What is the theoretical bound on the number of mistakes perceptron will make on this problem?

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1C)

How many mistakes does perceptron through origin have to make in order to find a perfect separator on the data provided above, in the order given? (Try it on your computer, not by hand!)

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1D)

If we were to multiply both original features of all of the points by $.001$, and considered the separator through origin $\theta = (0, 1, -0.0005)$, what would the margin of the new dataset be?

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1E)

How would the performance of the perceptron (as predicted by the mistake bound) change?

--
Fewer mistakes
Approximately the same mistakes
More mistakes

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1F)

If we multiplied just the first original feature (first row of the data) by .001, and used our original separator, what would the new margin be?

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1G)

What would the mistake bound be in this case?

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1H)

Run the perceptron algorithm on this data; how many mistakes does it make?

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2) Encoding Discrete Values

Some data sets have features that take on discrete values drawn from a set. Examples might be:

* which section of a class a student is in (1, 2, 3, 4)

* manufacturer of a cell phone (Samsung, Xiaomi, Sony, Apple, LG, Nokia)

* which laboratory performed a particular medical test

Sometimes they already have an obvious encoding into integers; other times, they don't but it's easy to make one (e.g., Samsung = 1, Xiaomi = 2, Sony = 3, Apple = 4, LG = 5, Nokia = 6)

2A)
Let's consider the case of the cell phones, using the encoding above, and imagine there is some prediction problem, such as predicting whether the phone will last three years, for which we have the data set:

data = [[2, 3, 4, 5]]
labels = [[1, 1, -1, -1]]

What value of $\theta$ and $\theta_0$ would we get when running perceptron on this data? You are free to use the perceptron implemented in homework $2$.

Enter a Python list with two floats, one for $\theta$ and one for $\theta_0$.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2B)
What prediction would we make about other phone types based on this classifier?

Enter a Python list with two labels (1 or -1), the first one for a Samsung phone and the second for a Nokia phone.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2C)

Are these predictions meaningful given the training data we used?

--
Yes
No

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2D)
It is common to encode a feature which takes on a value from a set of discrete values, not as a single multi-valued feature, but
using a one hot encoding.

Here, assume you have a feature $f$ which can take on any value from the set $\{1,2,...,k\}$. If $f$ takes on value $i$, then we represent it as a vector of length $k$ of all zeros, except for a +1 at the $i$th coordinate.

Write a function one_hot that takes as input $x$, a single feature value (between 1 and $k$), and $k$, the total possible number of values this feature can take on, and transform it to a numpy column vector of $k$ binary features using a one-hot encoding (remember vectors have zero-based indexing).

For example, one_hot(3,7) should return a column vector of length $7$ with the entry at index $2$ taking value $1$ (indices start at $0$) and other entries taking value $0$.

12345import numpy as npdef one_hot(x, k): passXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Run Code
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2E)
What happens if we use one-hot encoding on the data set part 2A) above, and put it into the perceptron? Recall that for a classifier $h(x)$, the prediction is $+1$ if $h(x) > 0$ and $-1$ otherwise. Further note that the perceptron algorithm makes an update whenever $y^{(i)}(\theta^Tx^{(i)}+\theta_0)\le 0$.

2E i)
What is the separator produced by the perceptron algorithm?

Enter a Python list with 7 floats, six for $\theta$ and one for $\theta_0$.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2E ii)
What are the predictions for Samsung and Nokia?

Enter a Python list with two labels (1 or -1), the first one for a Samsung phone and the second for a Nokia phone.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2E iii)
What are the distances for the Samsung and Nokia data points from the separator?

Enter a Python list with two distances, the first one for a Samsung phone and the second for a Nokia phone.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2F)
Now, what if we have this dataset:

data = [[1, 2, 3, 4, 5, 6]]
labels = [[1, 1, -1, -1, 1, 1]]

Is it linearly separable in the original encoding?

--
Yes
No

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2G)
Is it linearly separable in the one-hot encoding? If so, provide the separator found by the perceptron.

Enter a Python list with 7 floats, six for $\theta$ and one for $\theta_0$ or 'none'

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2H)
Enter an assignment of data values to labels (with distinct data points) that is not linearly separable using the one-hot encoding, or enter None if no such assignment exists.

Enter a Python list with 6 tuples (value, label) or 'none'

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

3) Polynomial Features

One systematic way of generating non-linear transformations of your input features is to consider the polynomials of increasing order. Given a feature vector $x = [x_1, x_2, ..., x_d]^T$, we can map it into a new feature vector that contains all the factors in a polynomial of order $d$. For example, for $x = [x_1, x_2]^T$ and order 2, we get

$$
\phi(x) = [1, x_1, x_2, x_1x_2, x_1^2, x_2^2]^T
$$
and for order 3, we get
$$
\phi(x) = [1, x_1, x_2, x_1x_2, x_1^2, x_2^2, x_1^2x_2, x_1x_2^2, x_1^3, x_2^3]^T.
$$
In the code file, we have defined make_polynomial_feature_fun that, given the order, returns a feature transformation function (analogous to $\phi$ in the description). You should use it in doing this problem.

3A)

Enter a list of 6 integers indicating the number of polynomial features of degrees [1, 10, 20, 30, 40, 50] for a 2-dimensional feature vector.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

3B)
Consider this data-set of four points in two-dimensional space:

data = ([[1, 1, 2, 2],
[1, 2, 1, 2]])
labels = [[-1, 1, 1, -1]]

It is standardly called the "exclusive-or" or "xor" problem. These points are not linearly separable, and you could interpret each point as being a pair of truth values, with their label being the XOR of the values.
In the code file, we have defined 4 sample data sets, (1) super_simple_separable_through_origin, (2) super_simple_separable, (3) xor, and (4) xor_more. On your own machine, you should run the code we have provided (test_with_features) for various orders of polynomial features and enter below the order of the smallest feature that separates the data. Make sure that you have included your implementation of perceptron in that file or you can use the implementation we have provided. You may need to adjust the number of iterations that the perceptron runs.

The separators are displayed when the code runs; it's instructive to watch them to see the range of separators that these non-linear transformations produce. Note that the separators are drawn by evaluating the feature transformations on a grid of points in the feature space and using the separator to classify them. (Note: If you have issues with the graphic not moving forward, try pressing the keys within your terminal.)

Enter a Python list of integers indicating the smallest polynomial order for which a separator exists for each of the four datasets in the code file (in order).

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

Experiments

A code and data folder that will be necessary for doing this homework can
be found at the top of the page. In the file code_for_hw3_part2.py, include your
learner code from HW 2. You will
want to modify the evaluation algorithms so that they take a T
argument to pass to the learners.

The rest of this assignment will require running the code on your computer; we will be asking only for the results of your runs.

4) Evaluating algorithmic and feature choices for AUTO data

We now want to build a classifier for the auto data,
with a focus on the numeric data.
In the code_for_hw3_part2.py, we have supplied you
with the load_auto_data function, which can read the relevant .tsv file.
It returns a list of dictionaries, one for each data item.

We then specify what feature function to use for each column in the data.
The file hw3_part2_main.py has an example that constructs
the data and label arrays using raw features for all the columns.

In the list features of hw3_part2_main.py,
you will find a list of feature name, feature function tuples.
There are three options for feature functions: raw, standard and one_hot.
raw uses the original value;
standard subtracts out the mean value and divides by the standard deviation;
and one_hot will one-hot encode the input, as described in the notes.

The function auto_data_and_labels processes the dictionaries and
return data, labels.
data has
dimension $(d, 392)$, where $d$ is the total number of features specified,
and labels has dimension $(1, 392)$.
The data in the file is sorted by class, but it will be shuffled when loaded.

We have included staff implementations of perceptron and average perceptron in code_for_hw3_part2.py.
Using the feature arrays and these implementations, you will be able to compute $\theta$ and $\theta_0$.

We have also included staff implementations of eval_classifier and xval_learning_alg (in the same code file).
You should use these functions to report accuracies.

4.1) Making choices

We know of two algorithm classes: perceptron and averaged perceptron (which we implemented in HW 1).
We have a several parameters that specify the settings for these learning algorithms.

A) Which parameters should we use for the learning algorithm? In the perceptron and averaged perceptron, there is a single parameter, $T$, the number of iterations.

B) Which features should we use? We have lots of choices here: we can use any subset of the data columns and for each column we have choices of how to compute features.

C) We will use expected accuracy, estimated by 10-fold cross-validation (we have included the definition in the code file), to make these choices of parameters.

*
We will try two types of algorithms: perceptron and averaged perceptron.

*
We will try 3 values of $T$: $T=1$, $T=10$, $T=50$.

*
We will try 2 feature sets:

* [cylinders=raw, displacement=raw, horsepower=raw, weight=raw, acceleration=raw, origin=raw]

* [cylinders=one_hot, displacement=standard, horsepower=standard, weight=standard, acceleration=standard, origin=one_hot]

Perform 10-fold cross-validation for all combinations of the two algorithms, three $T$ values, and the two choices of feature sets. It will be worthwhile investing in a piece of code to carry out all of the evaluations, in case you need to do this more than once.

In general, you should shuffle the dataset before evaluating, but for this exercise, please use hw3.xval_learning_alg, which shuffles the dataset for you, so that your results match ours.

4.1C i)

Enter accuracies (perceptron, averaged perceptron) for T=1, feature set 1:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

4.1C ii)

Enter accuracies (perceptron, averaged perceptron) for T=1, feature set 2:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

4.1C iii)

Enter accuracies (perceptron, averaged perceptron) for T=10, feature set 1:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

4.1C iv)

Enter accuracies (perceptron, averaged perceptron) for T=10, feature set 2:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

4.1C v)

Enter accuracies (perceptron, averaged perceptron) for T=50, feature set 1:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

4.1C vi)

Enter accuracies (perceptron, averaged perceptron) for T=50, feature set 2:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

Now we have the data we need to make rational choices.

4.1D) Which algorithm class is typically more effective?

Pick one:

--
Perceptron
Averaged Perceptron

Submit
View Answer <button class="btn btn-catsoop" onclick="!queue.get('location') ?
"Enter Table Number",
"Please enter your table number:",
true,
true).then(function(text) {
queue.set('location', text);
queue.add('help', {
location: queue.get('location'),
assignment: {
name: 'q000026',
page: catsoop.this_path,
path: catsoop.path_info,
display_name: 'Tutor Question 7 in Section 4.1',
},
})

queue.set('_visible', true);
}) : (queu[Truncated]