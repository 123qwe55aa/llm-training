For this lab:

* You will need to understand the material in the notes on
recommender systems.

Recommender Systems

1) Complete data

We will explore the effect of $k$ (the rank of the approximation) in representing relationships between raters and ratings by looking at complete ratings matrices and the ability to fit them using different models.

Consider the following data set:

$$
Y = \begin{bmatrix} 5 & 1 \\ 1 & 5 \end{bmatrix}
$$

1A)
Can this matrix be perfectly represented with $k = 1$? Provide $U$, $V^T$ matrices if it can such that $Y = UV^T$, or prove (at checkoff) that it cannot be done.

Enter a Python list of $U$ and $V^T$, with $U$ as a 2x1 matrix and $V^T$ as a 1x2 matrix, e.g., [[[1],[2]], [[3, 4]]], or the quoted string 'None' if none exists:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

Now consider this data set:

$$
Y = \begin{bmatrix} 5 & 10 & 15 \\ 2 & 4 & 6 \\ 1 & 2 & 3 \\ \end{bmatrix}
$$

1B)
Can this matrix be perfectly represented with $k = 1$? Provide $U$, $V^T$ matrices if it can such that $Y = UV^T$, or prove (at checkoff) that it cannot be done.

Enter a Python list of $U$ and $V^T$, with $U$ as a 3x1 matrix and $V^T$ as a 1x3 matrix, e.g., [[[1],[2],[3]], [[4, 5, 6]]], or the quoted string 'None' if none exists:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2) Two kinds of people, two kinds of movies

Here's our new movie database:

1) Llama Drama

2) Rocking Robot

3) Rama Llama Ding Dong

4) Rossum's Universal Robots

5) Me Llamo Llama

6) Robbie and Rosie

7) Three L Llama

We have complete preferences for four people, Amy, Bob, Cathy, and
Dana, corresponding (in that order) to the rows of $Y$. The entries of
the matrix are their ratings out of 5 stars (where 5 stars is strongly
positive and 1 star is strongly negative):

$$
Y = \begin{bmatrix} 5 & 1 & 5 & 1 & 5 & 1 & 5\\ 1 & 5 & 1 & 5 & 1 & 5 & 1\\ 5 & 5 & 5 & 5 & 5 & 5 & 5\\ 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix}
$$

2A)

What kind of movies does Amy like?

--
Robot
Llama

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2B)

Who likes all movies?

--
Amy
Bob
Cathy
Dana

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2C)
It turns out that we can come up with several exact factorizations of the data matrix $Y$ in this case! One nice factorization has:

U = [[ 1 5]
[ 5 1]
[ 5 5]
[ 1 1]]

What is the matrix $V^T$ which makes $Y = UV^T$?

Enter $V^T$ as a python list of lists:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2D)

What aspects of $U$ and $V$ model Amy's high scores on Llama movies?
I. Movies about Robots have a weight of zero on their second $V$ component (column $V_2$). Amy has higher weight on the second component.

II. Llama movies have a value of zero for the first feature and positive value for the second feature, and Amy also assigns low weight to the first feature and high weight to the second feature. Therefore, the dot product between Amy's features (first row of $U$) and movie features (columns of $V^T$) is high for Llama movies.

III. The dot product between Amy's features (first row of $U$) and
movie features (columns of $V^T$) corresponding to Llama movies is
higher than the dot product between the other people's features (other rows of $U$) and
movie features corresponding to Llama movies.

--
I
II
III

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2E)

What aspects of $U$ and $V$ model Dana's low ratings of all movies?
I. Movies about Llamas have a lower weight on their first $V$ component (column $V_1$), and higher on the second.

II. The dot product between Dana's features (last row of $U$) and the features of any movie in our database is 1.

III. The first entry of $U$ corresponding to Dana is equal to the second.

--
I
II
III

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

In practice, we often won't be able to immediately come up with a factorization as nice as the one above. Consider the following factorization that exactly reconstructs the data, i.e., $Y = UV^T$:

U = [[ 128.27790923 37.21907314]
[ -24.10214086 132.5681096 ]
[ 86.81314031 141.48931895]
[ 17.36262806 28.29786379]]
V.T = [[ 0.03494581 -0.00298991 0.03494581 -0.00298991 0.03494581 -0.00298991 0.03494581]
[ 0.01389677 0.03717287 0.01389677 0.03717287 0.01389677 0.03717287 0.01389677]]

2F) What aspects of your answers for the previous two parts are unchanged? What aspects are different? Does this new factorization provide counterexamples explaining why the incorrect choices for the previous two answers were indeed incorrect? Be prepared to discuss this at checkoff time.

2G)

Does factorizing preference matrices this way require the engineer to define any special features of people or movies?

--
Yes
No

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

3) Half-fixed

Notation

* $n$ is number of people

* $m$ is number of movies

* $k$ is the rank of the approximation

* $U$ is $n$ by $k$

* $V$ is $m$ by $k$

* $u^{(a)}$ is row $a$ of $U$, but transposed into a $k$ by 1 column vector. Corresponds to
the importance (weight) of the $k$ movie features on person $a$'s movie ratings.

* $v^{(i)}$ is row $i$ of $V$, also transposed into a $k$ by 1 column vector. Corresponds to the $k$ features of movie $i$.

* $Y$ is the (sparse) $n$ by $m$ data matrix

* $D$ is the set of $(a, i)$ pairs for which there is a rating in $Y$

Let's consider optimizing our objective:

$$
J(U, V) = \frac{1}{2}\sum_{(a, i) \in D} (Y_{ai} - {u^{(a)}}\cdot v^{(i)})^2 + \frac{\lambda}{2} \sum_{a = 1}^n \lVert u^{(a)} \rVert^2 + \frac{\lambda}{2} \sum_{i = 1}^m \lVert v^{(i)} \rVert^2.
$$

Assume someone came up with a really great $k$-dimensional set of features for characterizing movies (e.g., how violent, how loud, how sappy, how many llamas, how many robots) and hired a bunch of movie experts with time on their hands to watch and describe all the movies in terms of these features.

3A)

Would this information be contained in $U$ or $V$?

--
U
V

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

Given a $k$-dimensional movie feature representation $V$, and not allowing it to vary, we get a new objective that is only on the matrix $U$:

$$
J(U) = \frac{1}{2}\sum_{(a, i) \in D} (Y_{ai} - {u^{(a)}}\cdot v^{(i)})^2 + \frac{\lambda}{2} \sum_{a = 1}^n \lVert u^{(a)} \rVert^2.
$$

Be prepared to show your understanding of this simplification at checkoff time.

3B)
What is the gradient of this new objective with respect to one row, $u^{(a)}$, of $U$?
Warm up by being sure you know the dimension of $u^{(a)}$ and of $\nabla_{u^{(a)}} J(U)$.

The form of the gradient is:

$$
\nabla_{u^{(a)}} J(U) = \sum_{\{i \mid (a, i) \in D\}} g(Y_{ai}, u^{(a)}, v^{(i)}) + \lambda u^{(a)}
$$

Note that the gradient of the regularization term, $\lambda u^{(a)}$, is separated from $g(Y_{ai}, u^{(a)}, v^{(i)})$.

$g(Y_{ai}, u^{(a)}, v^{(i)}) =$
$u^{(a)}\cdot v^{(i)}$
$-u^{(a)}\cdot v^{(i)}$
$Y_{ai} - u^{(a)}\cdot v^{(i)}$
$-(Y_{ai} - u^{(a)}\cdot v^{(i)})$
$(Y_{ai} - u^{(a)}\cdot v^{(i)})v^{(i)}$
$-(Y_{ai} - u^{(a)}\cdot v^{(i)})v^{(i)}$

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

Note that in practice, it is impossible to have all users rate all movies; companies like Netflix recommend movies to its users who have rated only a subset of movies.

To make this concrete, consider an incomplete version of the data from the previous question:

$$
Y = \begin{bmatrix} ? & 1 & ? & 1 & 5 & 1 & 5\\ 1 & 5 & 1 & ? & ? & 5 & 1\\ 5 & 5 & 5 & 5 & ? & ? & 5\\ 1 & ? & 1 & 1 & 1 & 1 & ? \end{bmatrix}
$$

Now our goal is to find $U$, $V^T$ that minimize $J(U,V)$ given the incomplete matrix $Y$ above. To do so, we will use the alternating least squares method, where we

* Fix (or initialize) $V$ and find the optimal $U$ that minimizes $J(U)$ given this $V$.

* Fix $U$ to the value obtained from 1, and find the optimal $V$ that minimizes $J(V)$ given this $U$.

* Repeat from 1, with $V$ fixed to value obtained from 2.

To implement the alternating least squares method with our incomplete
matrix $Y$, we will define some notation that will make the gradient
$\nabla_{u^{(a)}} J(U)$ easier to write in matrix form, and similarly
for $\nabla_{v^{(i)}} J(V)$.

Let $I_a$ be the set of all movies that were rated by person
$a$. To be precise, $I_a = \{i \mid (a, i) \in D\}$. Also, define $B_a$ to be the matrix of relevant data:
$B_a$ is constructed by taking $V$ and removing every row $i$ that is
not in $I_a$. In other words, $B_a$