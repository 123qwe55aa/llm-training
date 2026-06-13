# Week 1 Exercises — MIT 6.036

These warmup exercises are divided into two sections:

1. **Hyperplanes** — some linear algebra foundational to machine learning
2. **NumPy intro** — an introduction to numpy

---

## 1) Hyperplanes

We will be using the notion of a hyperplane a great deal. A hyperplane is useful for classification, as discussed in the notes.

![Hyperplane diagram](content/images/example1.png)

### Notational conventions

- $x$: a point in $d$-dimensional space (represented as a column vector of $d$ real numbers), $\mathbb{R}^d$
- $\theta$: a point in $d$-dimensional space (represented as a column vector of $d$ real numbers), $\mathbb{R}^d$
- $\theta_0$: a single real number

We represent $x$ and $\theta$ as column vectors, that is, $d \times 1$ arrays. Remember dot products? We write dot products in one of two ways: $\theta^T x$ or $\theta \cdot x$. In both cases:

$$\theta^T x = \theta \cdot x = \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_d x_d$$

In a $d$-dimensional space, a hyperplane is a $d-1$ dimensional subspace that can be characterized by a normal to the subspace and a scalar offset. For example, any line is a hyperplane in two dimensions, an infinite flat sheet is a hyperplane in three dimensions, but in higher dimensions, hyperplanes are harder to visualize. Fortunately, they are easy to specify.

> **Hint:** When doing the two-dimensional problems below, start by drawing a picture. That should help your intuition. If you get stuck, take a look at this [geometry review for planes and hyperplanes](http://faculty.bard.edu/belk/math213/PlanesAndHyperplanes.pdf).

---

### 1.1) Through origin

In $d$ dimensions, any vector $\theta \in \mathbb{R}^d$ can define a hyperplane. Specifically, the hyperplane through the origin associated with $\theta$ is the set of all vectors $x \in \mathbb{R}^d$ such that $\theta^T x = 0$. Note that this hyperplane includes the origin, since $x = 0$ is in the set.

![Labelled hyperplane diagram](content/images/example1-labelled.png)

**Ex1.1a.** In two dimensions, $\theta = [\theta_1, \theta_2]$ can define a hyperplane. Let $\theta = [1, 2]$. Give a vector that lies on the hyperplane given by the set of all $x \in \mathbb{R}^2$ such that $\theta^T x = 0$:

Vector on hyperplane: `__________`

> Enter your answer as a Python list of numbers.

**Ex1.1b.** Using the same hyperplane, determine a vector that is normal to the hyperplane.

Vector normal to hyperplane: `__________`

**Ex1.1c.** Now, in $d$ dimensions, supply the simplified formula for a unit vector normal to the hyperplane in terms of $\theta$ where $\theta \in \mathbb{R}^d$.

> In this question and the subsequent ones that ask for a formula, enter your answer as a Python expression. Use `theta` for $\theta$, `theta_0` for $\theta_0$, `x` for any array $x$, `transpose(x)` for transpose of an array, `norm(x)` for the length (L2-norm) of a vector, and `x @ y` to indicate a matrix product of two arrays.

Formula for unit vector normal to hyperplane: `__________`

---

### 1.2) General hyperplane, distance to origin

Now, we'll consider hyperplanes defined by $\theta^T x + \theta_0 = 0$, which do not necessarily go through the origin. Distances from points to such general hyperplanes are useful in machine learning models, such as the perceptron ([as described in the notes](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week2/perceptron/4)).

Define the **positive side** of a hyperplane to be the half-space defined by $\{x \mid \theta^T x + \theta_0 > 0\}$, so $\theta$ points toward the positive side.

![General hyperplane diagram](content/images/example2.png)

**Ex1.2a.** In two dimensions, let $\theta = [3, 4]$ and $\theta_0 = 5$. What is the signed perpendicular distance from the hyperplane to the origin? The distance should be positive if the origin is on the positive side of the hyperplane, 0 on the hyperplane and negative otherwise.

> Hint: Draw a picture.

Distance = `__________`

**Ex1.2b.** Now, in $d$ dimensions, supply the formula for the signed perpendicular distance from a hyperplane specified by $\theta, \theta_0$ to the origin. If you get stuck, take a look at this [walkthrough of point-plane distances](http://mathworld.wolfram.com/Point-PlaneDistance.html).

Formula for signed distance to origin: `__________`

> *(Note: The original CAT-SOOP exercises continue with subsections 1.3–1.9 (On which side?, Separating hyperplane, Practice with formula, Decision boundary, Which side II, Midterm practice, Another midterm practice), but those were not successfully scraped before the MIT server's TLS certificate expired.)*

---

## 2) NumPy intro

NumPy is a package for doing a variety of numerical computations in Python. We will use it extensively. It supports writing very compact and efficient code for handling arrays of data. We will start every code file that uses numpy with `import numpy as np`, so that we can reference numpy functions with the `np.` precedent.

You can find general documentation on NumPy [here](https://docs.scipy.org/doc/numpy/user/quickstart.html), and we also have a [6.036-specific numpy tutorial](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/welcome/6_036_Information_You_Should_Know/6).

The fundamental data type in numpy is the multidimensional array, and arrays are usually generated from a nested list of values using the `np.array` command. Every array has a `shape` attribute which is a tuple of dimension sizes.

In this class, we will use two-dimensional arrays almost exclusively. That is, we will use 2D arrays to represent both matrices and vectors!

We will represent a **column vector** as a $d \times 1$ array and a **row vector** as a $1 \times d$ array. So for example, we will represent the three-element column vector,

$$x = \begin{bmatrix} 1 \\ 5 \\ 3 \end{bmatrix}$$

as a $3 \times 1$ numpy array:

```python
x = np.array([[1], [5], [3]])
```

or by using the transpose of a $1 \times 3$ array (a row vector):

```python
x = np.transpose(np.array([[1, 5, 3]]))   # note the "double" brackets
```

or using the `.T` attribute:

```python
x = np.array([[1, 5, 3]]).T
```

> **Important:** In this assignment we will not accept answers that use loops. One reason for avoiding loops is efficiency, but the more important reason is that using higher-level constructs leads to simpler code that is easier to debug.

### NumPy functions you should know

| Function | Description |
|---|---|
| `np.array` | Create an array |
| `np.transpose` / `a.T` | Transpose an array |
| `np.ndarray.shape` | Get array dimensions |
| `np.dot` / `a @ b` | Matrix/dot product |
| `np.sign` | Sign function |
| `np.sum` | Sum (with `axis` and `keepdims` arguments) |
| `+`, `-`, `*`, `/` | Elementwise operators |

Note: In Python, `np.dot(a, b)` is the matrix product `a @ b`.

---

### 2.1) Array

Provide an expression that sets `A` to be a $2 \times 3$ numpy array (2 rows by 3 columns), containing any values you wish.

```python
import numpy as np
A = 0   # your code here
```

### 2.2) Transpose

Write a procedure that takes an array and returns the transpose of the array. You may use `np.transpose` or `.T`, but you may not use a loop.

```python
import numpy as np
def tp(A):
    pass   # your code here
```

### 2.3) Shapes

Let `A` be a $4 \times 2$ numpy array, `B` be a $4 \times 3$ array, and `C` be a $4 \times 1$ array. For each of the following expressions, indicate the shape of the result as a tuple of integers (recall python tuples use parentheses, e.g. `(x,)` for a single-element tuple) or `"none"` if it is illegal.

| Ex | Expression | Result shape |
|---|---|---|
| 2.3a | `C * C` | |
| 2.3b | `np.dot(C, C)` | |
| 2.3c | `np.dot(np.transpose(C), C)` | |
| 2.3d | `np.dot(A, B)` | |
| 2.3e | `np.dot(A.T, B)` | |
| 2.3f | `D = np.array([1,2,3])` | (shape of `D`) |
| 2.3g | `A[:,1]` | |
| 2.3h | `A[:,1:2]` | |

### 2.4) Row vector

Write a procedure that takes a list of numbers and returns a 2D numpy array representing a **row vector** containing those numbers.

```python
import numpy as np
def rv(value_list):
    pass   # your code here
```

### 2.5) Column vector

Write a procedure that takes a list of numbers and returns a 2D numpy array representing a **column vector** containing those numbers. You may use the `rv` procedure.

```python
import numpy as np
def cv(value_list):
    pass   # your code here
```

### 2.6) Length

Write a procedure that takes a column vector and returns the vector's Euclidean length (magnitude) as a scalar. You may **not** use `np.linalg.norm`, and you may **not** use a loop.

```python
import numpy as np
def length(col_v):
    pass   # your code here
```

### 2.7) Normalize

Write a procedure that takes a column vector and returns a unit vector in the same direction. You may **not** use a loop. Use your `length` procedure from above (you do not need to redefine it here).

```python
import numpy as np
def normalize(col_v):
    pass   # your code here
```

### 2.8) Indexing

Write a procedure that takes a 2D array and returns the **final column** as a two dimensional array. You may **not** use a loop.

```python
import numpy as np
def index_final_col(A):
    pass   # your code here
```

### 2.9) Representing data

Alice has collected weight and height data of 3 people:

| Weight | Height |
|---|---|
| 150 | 5.8 |
| 130 | 5.5 |
| 120 | 5.3 |

She wants to put this into a numpy array such that each row represents one individual's height and weight in the order listed. Write code to set `data` equal to the appropriate numpy array:

```python
import numpy as np
data = 0   # your code here
```

### 2.10) Matrix multiplication

Now Alice wants to compute the sum of each person's height and weight as a column vector by multiplying `data` by another numpy array. She has written the following **incorrect** code and needs your help to fix it:

```python
import numpy as np
def transform(data):
    return (np.dot(np.array([1, 1]), data))
```

What's wrong, and what should the correct implementation be?

---

> **Note:** The interactive answer-checking system on the original MIT CAT-SOOP server is currently unavailable due to an expired TLS certificate on `introml_oll.odl.mit.edu` (expired 2026-06-12). You can work through these problems locally using Python/numpy and check your answers manually against the notes.
