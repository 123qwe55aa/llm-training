For these exercises, you should read the notes on recommender systems.
1) Scaling parameters in collaborative filtering

Consider a collaborative filtering problem, where the
data matrix $X$ is a product $UV^T$, where $U$ is $n\times k$ and $V$ is $m\times k$:

1A)
Let $k = 3$. If we were to take the matrix $U$ and multiply the
first column by 2, the second column by 3 and the third column by 4,
to make a new matrix $U'$, what would we have to do to $V$ to get a
$V'$ so that $U'{V'}^T = UV^T$?

Make sure to remember that $V$ is transposed in the product above!

Choose option(s) which would work:
Multiply the three rows of $V$ by $1/2$, $1/3$, and $1/4$, respectively
Multiply the three rows of $V$ by $2$, $3$, and $4$, respectively
Nothing can be done to $V$ to provide the desired outcome
Multiply the three columns of $V$ by $1/2$, $1/3$, and $1/4$, respectively
Multiply all elements of $V$ by $1/2$, $1/3$, and $1/4$, respectively

View Answer

You have 4 submissions remaining.

2) Rotten tomatoes

Alex finds a solution to a collaborative filtering problem with 4
users, 4 movies, and $k = 2$, specified by matrices $U$ (for users) and $V$ (for movies):

$$
U = \begin{bmatrix} 1 & 2 \\ 3 & 1 \\ 2 & 5 \\ 1 & 3 \\ \end{bmatrix} \;\;\;\;\;\;\;\; V^T = \begin{bmatrix} 2 & 1 & 5 & 3 \\ 3 & 4 & 2 & 1 \\ \end{bmatrix}
$$

Assume the biases $b_U^{(a)} = 0$ and $b_V^{(i)} = 0$ for all $a$ and $i$.

2A)
Assuming that the first entry has index 1, what is the predicted value
for user 3's rating of movie 1?

Enter the answer as a number:

View Answer

3) Low-rank matrix factorization

We will consider the problem of low-rank matrix factorization, which aims to minimize

$$
J(U, V) = \frac{1}{2}\sum_{(a,i) \in D} (Y_{ai} - [UV^T]_{ai})^2 + \frac{\lambda}{2}\sum_{a=1}^n \sum_{j=1}^k U_{aj}^2 + \frac{\lambda}{2}\sum_{i=1}^m \sum_{j=1}^k V_{ij}^2
$$

Let $Y$ be defined as

$$
Y = \begin{bmatrix} 5 & ? & 7 \\ ? & 2 & ? \\ 4 & ? & ? \\ ? & 3 & 6 \end{bmatrix}
$$

where "?" indicates the element is missing from $D$, i.e., $(a,i) \notin D.$ We let $k = \lambda = 1.$

Assume $U$ and $V$ are set as $U=[6,0,3,6]^T$ and $V=[4,2,1]^T$.

1A)
Compute $X$, the matrix of predicted rankings given these values.

Enter $X$ as a list of lists of integers (representing the rows of the matrix):

View Answer

1B)
Compute the value of the squared error term in $J(U,V)$, including the $\frac{1}{2}$ factor, for the current estimate $X$:

Enter the answer as a number:

View Answer

1C)
Compute the total regularization term in $J(U,V)$ for the current values of $U$, $V$ and $\lambda$.

Enter the answer as a number:

View Answer