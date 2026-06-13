Readiness Assessment

Our purpose here is to enable you to judge your readiness for 6.036 (and
consider taking other subjects before 6.036, or to immediately launch
self-study to address gaps) in the following areas:

* Linear algebra (e.g. 18.06)

* Differential calculus (e.g. 18.03)

* Python programming (e.g. 6.0001)

1) Linear Algebra

We will be using linear algebra in many places throughout 6.036
to represent and operate on information. Our purpose here is to check
your understanding of a few key concepts and skills you'll need.
(This is not a complete set -- just a sampling of a few points
to help you judge readiness and need for further study!)

For the time being we'll be using Python lists to represent vectors
and arrays. Once we start in 6.036, we'll be using the Python numpy
module, but do not assume you know anything about numpy below.

1.1) Row and column vectors

In this section, we'll consider $n$-dimensional column vectors
$c = \begin{bmatrix}c_0\\ c_1 \\ \vdots \\ c_i \\ \vdots \\ c_{n-1}\end{bmatrix}$
and $m$-dimensional row vectors
$r = \begin{bmatrix}r_0 & r_1 & \cdots & r_j & \cdots & r_{m-1}\end{bmatrix}$,
with the subscript indicating the i-th and j-th element within the
vector, respectively.
In both cases, we'll represent column and row vectors for now as Python lists,
so will also be explicit about whether a vector is a row vector
or a column vector. For example, we might specify in Python
r1 = [2, 4, 7, 1] to be a row vector with four elements,
and c1 = [1, 3, -1] to be a column vector with three elements.

(Q1) Given $a = \begin{bmatrix}1 & 5 & -3 & 2\end{bmatrix}$, and
$b = \begin{bmatrix}8 & 2 & 4 & 7\end{bmatrix}$, what is $a + b$?

Enter your result as a Python list:

View Answer

100.00%

We are given $c^{(1)} = \begin{bmatrix}1 \\ 5 \\ -3 \\ 2\end{bmatrix}$, and
$c^{(2)} = \begin{bmatrix}8 \\ 2 \\ 4 \\ 7\end{bmatrix}$.
Note that the superscript $^{(1)}$ just labels the vector, i.e., is part of the
name or label for that vector. Thus $c^{(1)}$ and $c^{(2)}$ are just two different
$n = 4$ length column vectors.

(Q2) What is $c^{(1)} + 10c^{(2)}$?

Enter your result as a Python list:

View Answer

100.00%

(Q3) What is $c^{(1)}\ * c^{(2)}$? Here $*$ indicates element-wise multiplication.

Enter your result as a Python list:

View Answer

100.00%

The transpose is notated here by the superscript $^T$. For example, if
$a$ is the row vector $a = \begin{bmatrix}1 & 5 & -3 & 2\end{bmatrix}$,
then $a^T$ is the column
vector $a^T = \begin{bmatrix}1 \\ 5 \\ -3 \\ 2\end{bmatrix}$.

(Q4) Given the vector $a$ as above and
$b^T = \begin{bmatrix}8 & 2 & 4 & 7\end{bmatrix}$, what is $a$+$b$?

Enter your result as a Python list, or None if the operation is invalid:

View Answer

100.00%

1.2) Vector dot product

You should be familiar with vector dot products of vectors. For example,
given two column vectors $a$ and $b$, we might be interested in the
dot product $a \cdot b$ of those two vectors.

(Q5) Given $a = \begin{bmatrix}1 & 5 & -3 & 2\end{bmatrix}^T$, and
$b = \begin{bmatrix}8 & 2 & 4 & 7\end{bmatrix}^T$, what is $a \cdot b$?

Enter your result as a Python value, or None if the operation is invalid:

View Answer

100.00%

(Q6) What is $a^T b$? Enter your result as a Python list (if a vector) or as a Python number (if a single number), or as None if the operation is invalid:

0.00%

Solution: 20

Explanation:
For two column vectors (or two row vectors)
$a$ and $b$, the dot product can also be
written as the matrix product $a^Tb$, or $b^Ta$, resulting in the same scalar
value of 20 for the given $a$ and $b$.

1.3) 2-D arrays and matrices

For now, we'll represent two-dimensional arrays (or matrices)
in Python as a list of lists,
i.e., a list of rows. Thus the matrix

$M = \begin{bmatrix}1 & 2 & 3 \\ -2 & 3 & 7\end{bmatrix}$
is an $n$ x $m$ dimensional matrix ($n = 2$ or two rows and $m = 3$ or three
columns in this case), and will have a Python representation
as [[1, 2, 3], [-2, 3, 7]]. Matrix and vector multiplication
(notably different than element-wise multiplication) is an
important operation in linear algebra that you should be familiar with.

In the problems below, when we ask for the "Python value" of a given
matrix multiplication or other operation, enter a Python list if the
result is a vector; enter a Python list of lists (list of rows) if the
result is a two-dimensional matrix (having more than one rows or
columns); enter a single number if the result is a scalar; and if the
expression does not result in a valid vector or matrix, enter the
Python value None.

In addition to $M$ as defined above, assume we have a vector
$v = \begin{bmatrix}1 \\ 0 \\ -1\end{bmatrix}$.

(Q7) What kind of value is the result of $Mv$?

--
A scalar
A row vector
A column vector
A matrix with multiple columns and rows
An invalid operation

View Answer

100.00%

(Q8) What is $Mv$? Enter your result as a Python value:

View Answer

100.00%

(Q9) What kind of value is the result of $vM$?

--
A scalar
A row vector
A column vector
A matrix with multiple columns and rows
An invalid operation

View Answer

100.00%

(Q10) Given $w = \begin{bmatrix}1 & 5 & -3 & 2\end{bmatrix}^T$, and
$x = \begin{bmatrix}8 & 2 & 4 & 7\end{bmatrix}^T$, what is $w^T x$?

Enter your result as a Python value:

View Answer

100.00%

(Q11) Given $w = \begin{bmatrix}1 & 5 & 0\end{bmatrix}^T$, and
$x = \begin{bmatrix}8 & 2 \end{bmatrix}^T$, what is $w x^T$?

Enter your result as a Python value:

View Answer

100.00%

(Q12) Which of these is equivalent to $(AB)^T$?

* $AB^T$

* $A^TB$

* $A^TB^T$

* $BA$

* $B^TA$

* $B^TA^T$

* None of these

Selection:

--
1
2
3
4
5
6
7

View Answer

100.00%

(Q13) Does the matrix $X$ below have an inverse?
$X = \begin{bmatrix}1 & 3 & 2 \\ 2 & 6 & 7\end{bmatrix}$

--
Yes
No

View Answer

100.00%

(Q14) Does the matrix $X$ below have an inverse?
$X = \begin{bmatrix}1 & 3 & 2 \\ 2 & 6 & 7 \\ 3 & 9 & 6\end{bmatrix}$

--
Yes
No

View Answer

100.00%

(Q15) We have the following relationship: $Wx = y$. Assuming
$W$ is full rank and square and the dimensions of $W$, $x$, and $y$ are
appropriate, which of the following expressions gives us a solution for $x$?

* $x = W^Ty$

* $x = y/W$

* $x = (W^TW)^{-1}y$

* $x = W^{-1}y$

* $x = (W^TW)^{-1}Wy$

* None of these

Selection:

--
1
2
3
4
5
6

View Answer

100.00%

2) Differential Calculus (Gradients)

Let $f$ be a function that takes a vector $v = \begin{bmatrix}v_1 & v_2 & \cdots & v_n\end{bmatrix}^T$
as input and returns the scalar value $1 \cdot v_1 + 2 \cdot v_2 + 3 \cdot v_3 + \cdots + n \cdot v_n$.

(Q16) What kind of an object is $\frac{\partial f(v)}{\partial v_3}$?

--
A scalar
A row vector
A column vector
A matrix with multiple columns and rows
An invalid operation

View Answer

100.00%

(Q17) What is the value of $\frac{\partial f(v)}{\partial v_3}$?

View Answer

100.00%

(Q18) What kind of an object is $\nabla_v f(v)$?

--
A scalar
A row vector
A column vector
A matrix with multiple columns and rows
An invalid operation

View Answer

100.00%

(Q19) What is the value of $\nabla_v f(v)$?

* The scalar $n$

* The row vector $\begin{bmatrix}0 & 0 & \cdots & 1\end{bmatrix}$ (consisting of $n$ total elements)

* The column vector $\begin{bmatrix}0 & 0 & \cdots & 1\end{bmatrix}^T$ (consisting of $n$ total elements)

* The row vector $\begin{bmatrix}1 & 2 & \cdots & n\end{bmatrix}$ (consisting of $n$ total elements)

* The column vector $\begin{bmatrix}1 & 2 & \cdots & n\end{bmatrix}^T$ (consisting of $n$ total elements)

* An $n$ x $n$ matrix with diagonal elements $1, 2, \cdots, n$.

* None of these

--
1
2
3
4
5
6
7

View Answer

100.00%

(Q20) If you dropped a marble down on a
(smooth) mountain range whose height over an n-dimensional landscape
was given by $f$, in what direction would the marble roll?

* Most strongly in the positive $v_1$ direction

* Most strongly in the negative $v_1$ direction

* Most strongly in the positive $v_n$ direction

* Most strongly in the negative $v_n$ direction

* It would not move

* None of these

--
1
2
3
4
5
6

View Answer

100.00%

3) Python

We will also be using Python, and packages built on Python, extensively
in 6.036. Here we have a small number of small Python coding problems
that you should be able to implement, given adequate prior Python experience.

3.1) Basic functions

Given two lists of numbers, write a procedure that returns a list of
the element-wise sum of the number in those two lists. In the
following, no imports should be used.

(Q21) Implement add_two_lists below.

```python
def add_two_lists(a, b): pass return [x + y for x, y in zip(a, b)]
```

View Answer

100.00%

 Your score on your most recent submission was: 100.00%

Show/Hide Detailed ResultsTest Results:

Test 01

The test case was:

#Your Code Here
ans = add_two_lists([1, 2, 3], [4, 5, 6])

Our solution produced the following value for ans:

[5, 7, 9]

Your submission produced the following value for ans:

[5, 7, 9]

Given two column vectors (each represented as a list of numbers),
write a procedure dot that returns the (scalar) dot product of two input
vectors, each represented as a list of numbers.

(Q22) Implement dot below.

```python
def dot(v1, v2): pass return sum(x * y for x, y in zip(v1, v2))
```

View Answer

100.00%

 Your score on your most recent submission was: 100.00%

Show/Hide Detailed ResultsTest Results:

Test 01

The test case was:

#Your Code Here
ans = dot([1, 2, 3], [4, 5, 6])

Our solution produced the following value for ans:

32

Your submission produced the following value for ans:

32

Test 02

The test case was:

#Your Code Here
ans = dot(list(range(100)), list(range(10,110)))

Our solution produced the following value for ans:

377850

Your submission produced the following value for ans:

377850

3.2) Functions as objects

Write a function add_n that takes a single numeric argument n, and returns
a function. The returned function should take a vector v as an argument and
return a new vector with the value for n added to each element of vector v.
For example, add_n(10)([1, 5, 3]) should return [11, 15, 13].

(Q23) Implement add_n below.

```python
def add_n(n): 
def add_to_vector(v): return [x + n for x in v] return add_to_vector
```

View Answer

100.00%

 Your score on your most recent submission was: 100.00%

Show/Hide Detailed ResultsTest Results:

Test 01

The test case was:

#Your Code Here
ans = add_n(10)([1, 5, 3])

Our solution produced the following value for ans:

[11, 15, 13]

Your submission produced the following value for ans:

[11, 15, 13]

Test 02

The test case was:

#Your Code Here
ans = add_n(2)(list(range(-5, 25, 3)))

Our solution produced the following value for ans:

[-3, 0, 3, 6, 9, 12, 15, 18, 21, 24]

Your submission produced the following value for ans:

[-3, 0, 3, 6, 9, 12, 15, 18, 21, 24]

3.3) Arrays as lists of lists

Write a function array_mult that takes two two-dimensional arrays
and performs a matrix multiplication, return a new two-dimensional array.
Each array should be represented as a list of lists, i.e., as a list of
rows, as discussed earlier. For example,

>>> M1 = [[1, 2, 3], [-2, 3, 7]]
>>> M2 = [[1,0,0],[0,1,0],[0,0,1]]
>>> array_mult(M1, M2)
[[1, 2, 3], [-2, 3, 7]]

>>> M3 = [[1], [0], [-1]]
>>> array_mult(M1, M3)
[[-2], [-9]]

(Q24) Implemement array_mult below.

```python
def array_mult(A, B): pass return [ [ sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0])) ] for i in range(len(A)) ]
```

View Answer

100.00%

 Your score on your most recent submission was: 100.00%

Show/Hide Detailed ResultsTest Results:

Test 01

The test case was:

#Your Code Here
M1 = [[1, 2, 3], [-2, 3, 7]]
M2 = [[1,0,0],[0,1,0],[0,0,1]]
ans = array_mult(M1, M2)

Our solution produced the following value for ans:

[[1, 2, 3], [-2, 3, 7]]

Your submission produced the following value for ans:

[[1, 2, 3], [-2, 3, 7]]

Test 02

The test case was:

#Your Code Here
M1 = [[1, 2, 3], [-2, 3, 7]]
M3 = [[1], [0], [-1]]
ans = array_mult(M1, M3)

Our solution produced the following value for ans:

[[-2], [-9]]

Your submission produced the following value for ans:

[[-2], [-9]]