For this homework, you will need to understand the notes on recommender systems.

1) Matrix factorization by rank

Consider the following data set:

$$
Y = \begin{bmatrix} 4 & 3 & 1\\ 1 & 3 & -2\\ 5 & 2 & 3 \end{bmatrix}
$$

We would like to factorize $Y = UV^T$ such that $U$ and $V$ are both $3 \times k$ matrices.

Hint: See if any columns are linear combinations of others. Try subtracting the first two columns.

1A)
Can this matrix be factored using $k = 1$? Provide $U$, $V^T$ matrices if it can (such that $Y = UV^T$), or enter 'None' if it cannot be done.

Enter a Python list of $U$ and $V^T$, with $U$ as a 3x1 matrix and $V^T$ as a 1x3 matrix, e.g., [[[1],[2],[3]], [[4, 5, 6]]], or the quoted string 'None' if none exists:

1B)
Can this matrix be factored using $k=2$? Provide $U$, $V^T$ matrices if it can (such that $Y = UV^T$), or or enter 'None' if it cannot be done.

Enter a Python list of $U$ and $V^T$, with $U$ as a 3x2 matrix and $V^T$ as a 2x3 matrix, e.g., [[[1, 2],[3, 4],[5, 6]], [[7, 8, 9],[10, 11, 12]], or the quoted string 'None' if none exists:

2) Some movies are more equal than others

A code and data folder that will be useful for doing the remainder of this assignment can
be found here. Download
this to your computer. Alternatively, the code is available in a colab notebook here.

In Lab 12, we formulated recommender systems without offsets: there was no equivalent of $\theta_0$ in the regression problem we solved. But offsets are very useful in this problem to account for some reviewers that are just generally grumpy (or enthusiastic) or some movies that are just generally awful (or great). For this section we will work to extend the results from the lab to include offsets.

Rather than trying to predict what rating Amy will give a movie based on the properties of that movie, it might be more effective to predict how much "more highly than usual" Amy will rate the movie. To do this, we introduce a vector $b_u$ to characterize offsets for users and a vector $b_v$ to characterize offsets for movies. Now, the objective can be written as

$$
J(U, V) = \frac{1}{2}\sum_{(a, i) \in D} (Y_{ai} - {u^{(a)}}\cdot v^{(i)} - b_u^{(a)} - b_v^{(i)})^2 + \frac{\lambda}{2} \sum_{a = 1}^n \lVert u^{(a)} \rVert^2 + \frac{\lambda}{2} \sum_{i = 1}^m \lVert v^{(i)} \rVert^2
$$

We will stick with just one half of the problem for now: finding $U$ and $b_u$ given a fixed $V$ and $b_v$. We'll concentrate on computing a new estimate for $u^{(a)}$ and $b_u^{(a)}$. Continuing the notation from lab, the regression problem that we have to solve becomes:

$$
-B_a^T (Z_a - B_a u^{(a)} - b_v - b_u^{(a)}) + \lambda u^{(a)} = 0
$$

where $b_v$ is the $l_a$ by 1 vector of offsets for the movies and $b_u^{(a)}$ is the offset for user $a$. This is a linear regression problem in which the targets can be viewed as $Z_a - b_v$, and where there is an offset $b_u^{(a)}$ that is not regularized. This is, then, the standard ridge regression set-up. We have seen how to solve this problem via gradient descent before, but it can also be solved analytically by essentially turning it into a regular linear regression problem. Using this approach to computing the optimal $u^{(a)}$ requires first centering the data, then doing linear regression, then doing a little bit of work to recover the offset $b_u^{(a)}$. In the code, we have supplied a procedure (ridge_analytic) for doing exactly this.

Let's see how it works out in our running example. For illustrative purposes, we'll add two more movies:

8) 6.036 lecture videos

9) The Zzzz files

Our new preference data is:

$$
Y = \begin{bmatrix} ? & 1 & ? & 1 & 5 & 1 & 5 & 5 & 1\\ 1 & 5 & 1 & ? & ? & 5 & 1 & 5 & 1\\ 5 & 5 & 5 & 5 & ? & ? & 5 & ? & 1\\ 1 & ? & 1 & 1 & 1 & 1 & ? & 5 & ? \end{bmatrix}
$$

We will assume the following value for $b_v$:

$$
b_v = \begin{bmatrix} 3 & 3 & 3 & 3 & 3 & 3 & 3 & 5 & 1 \end{bmatrix}^T
$$

And let

$$
V = \begin{bmatrix} 10 & 1 & 10 & 1 & 10 & 1 & 10 & 5 & 5\\ 1 & 10 & 1 & 10 & 1 & 10 & 1 & 5 & 5 \end{bmatrix}
$$

We can now compute the optimal $u^{(a)}$ and $b_u^{(a)}$ for $a=0$:

Z = np.array([[1], [1], [5], [1], [5], [5], [1]])
b_v = np.array([[3], [3], [3], [3], [3], [5], [1]])
B = np.array([[1, 10], [1, 10], [10, 1], [1, 10], [10, 1], [5, 5], [5, 5]])
# Solution with offsets, using ridge_analytic provided in code file
u_a, b_u_a = ridge_analytic(B, (Z - b_v), 1)
u_a, b_u_a
(array([[ 0.22024566],
[-0.22193986]]),
array([[ 0.00762389]]))
# Solution using previous model, with no offsets
u_a_no_b = np.dot(np.linalg.inv(np.dot(B.T, B) + 1 * np.identity(2)), np.dot(B.T, Z))
u_a_no_b
array([[ 0.50148126],
[ 0.0562376 ]])

The code above computes $u^{(a)}$ and $b_u^{(a)}$ for Amy. Use them for the next few parts. You can assume that Llama movies always satisfy $v^{(i)} = \begin{bmatrix} 10 \\ 1 \end{bmatrix}$
and Robot movies satisfy $v^{(i)} = \begin{bmatrix} 1 \\ 10 \end{bmatrix}$.

2A)
How will Amy feel about a brand new Llama movie (not in the existing movie data) that gets bad ratings from almost everyone ($b_v^{(i)}= 1$), in the two models?

Enter a Python list with two floats [a,b] (with two decimal places) where a is the prediction without offset and b is the prediction with offset:

2B)
What about a brand new Robot movie (not in the existing movie data) that gets good ratings from almost everyone ($b_v^{(i)}= 5$), in the two models?

Enter a Python list with two floats [a,b] (with two decimal places) where a is the prediction without offset and b is the prediction with offset:

2C)
What about a brand new Robot movie (not in the existing movie data) that gets average ratings from almost everyone ($b_v^{(i)}= 3$), in the two models?

Enter a Python list with two floats [a,b] (with two decimal places) where a is the prediction without offset and b is the prediction with offset:

3) Implementing recommender systems

Now we'll look in detail at two implementations of matrix factorization. One is the alternating least squares algorithm, and the other is the stochastic gradient descent algorithm.

We will assume that the input data is made up of (a, i, r) triples, where a is a user index. i is an item index and r is a rating. Here is a small example:

ratings_small = \
[(0, 0, 5), (0, 1, 3), (0, 3, 1),
(1, 0, 4), (1, 3, 1),
(2, 0, 1), (2, 1, 1), (2, 3, 5),
(3, 0, 1), (3, 3, 4),
(4, 1, 1), (4, 2, 5), (4, 3, 4)]

We will assume that predictions are made using user and item offsets, that is,

$$
y = {u^{(a)}}\cdot v^{(i)} + b_u^{(a)} + b_v^{(i)}
$$

In this expression $b_u$ is a vector of user offsets and $b_v$ is a vector of item offsets.

3.1) Alternating Least Squares (ALS)

Below is a function that provides the "outer loop" of the alternating least squares algorithm.

* Define n and m from the data.

* Initialize a list of lists us_from_v where us_from_v[i] contains the indices and ratings of users who rated item i.

* Similarly, initialize a list of lists vs_from_u where vs_from_u[a] contains the indices and ratings of items rated by user a.

* Initialize the set of parameters x (note that the u,v entries are set randomly while the user and item offsets are set to 0) where
* x[0] = u, a list of column vectors (initialized randomly) such that u[a] corresponds to $u^{(a)}$ as defined above.

* x[1] = b_u, a column vector (initialized with 0s) equal to $b_u$ as defined above.

* x[2] = v, a list of column vectors (initialized randomly) such that v[i] corresponds to $v^{(i)}$ as defined above.

* x[3] = b_v, a column vector (initialized with 0s) equal to $b_v$ as defined above.

* Then we alternate minimizations.

* And report the results: the error between predicted scores and a held-out set of actual scores on the same users and items.

def mf_als(data_train, data_validate, k=2, lam=0.02, max_iter=100, verbose=False):
# size of the problem
n = max(d[0] for d in data_train)+1 # users
m = max(d[1] for d in data_train)+1 # items
# which entries are set in each row and column and the rating
us_from_v = [[] for i in range(m)]
vs_from_u = [[] for a in range(n)]
for (a, i, r) in data_train:
us_from_v[i].append((a, r))
vs_from_u[a].append((i, r))
# Initial guess at u, b_u, v, b_v
# Note that u and v are lists of column vectors (rows of U, V).
x = ([np.random.normal(1/k, size=(k,1)) for a in range(n)],
np.zeros(n),
[np.random.normal(1/k, size=(k,1)) for i in range(m)],
np.zeros(m))
# Alternation, modifies the contents of x
for i in range(max_iter):
update_U(data_train, vs_from_u, x, k, lam)
update_V(data_train, us_from_v, x, k, lam)
# The root mean square errors measured on validate set
if data_validate != None:
print('ALS result for k =', k, ': rmse =', rmse(data_validate, x))
return x

Here is an example of vs_from_u for the small data set given above:

vs_from_u = \
[[(0, 5), (1, 3), (3, 1)],
[(0, 4), (3, 1)],
[(0, 1), (1, 1), (3, 5)],
[(0, 1), (3, 4)],
[(1, 1), (2, 5), (3, 4)]]

The only part that's missing are the update_U and update_V
procedures. These are very similar, so we'll just do update_U,
where we hold the $v^{(i)}$ constant and solve for the $u^{(a)}$. We
have seen above that each of the steps is solving a ridge regression
problem, that is, finding a set of coefficients for a linear function,
$(\theta, \theta_0)$, so as to minimize the mean sum of squared errors
on data given by $(X,Y)$ (with regularization on the magnitude of
$\theta$).

We have given you a function ridge_analytic(X,Y,lam) that solves the ridge regression problem and returns (th, th0) as usual, where th is a column vector and th0 is a float.

3.1)
Now, write the function update_U(data, vs_from_u, x, k, lam)

* data is a list of (a, i, r) triples

* vs_from_u is a list of lists as defined above

* x is a list of parameters as defined above

* k is an integer indicating the length of the individual u and v vectors

* lam is the regularization parameter

The function should update the entries in x corresponding to the u vectors and the b_u entries. It should also return x, so the Tutor can check it.
Note that if there are no ratings from a particular user, we don't want to update that user's entries.

12345def update_U(data, vs_from_u, x, k, lam): (u, b_u, v, b_v) = x # Your code here return xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

3.2) Stochastic Gradient Descent (SGD)

Alternatively, we can use Stochastic Gradient Descent directly on the objective function

$$
J(U, V) = \frac{1}{2}\sum_{(a, i) \in D} (Y_{ai} - {u^{(a)}}\cdot v^{(i)} - b_u^{(a)} - b_v^{(i)})^2 + \frac{\lambda}{2} \sum_{a = 1}^n \lVert u^{(a)} \rVert^2 + \frac{\lambda}{2} \sum_{i = 1}^m \lVert v^{(i)} \rVert^2
$$