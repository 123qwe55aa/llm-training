# Linear Classifiers

> MIT 6.036 Week 1 课程原始讲义的网页阅读版。内容由课程 PDF 文本层转换并做轻量排版清理。

## Classification

A binary classifier is a mapping from Rd → {−1, +1}. We'll often use the letter h (for hypothesis) to stand for a classifier, so the classification process looks like:

x → h → y

Real life rarely gives us vectors of real numbers; the x we really want to classify is usually something like a song, image, or person. In that case, we'll have to define a function φ(x), whose domain is Rd, where φ represents features of x, like a person's height or the amount of bass in a song, and then let the h : φ(x) → {−1, +1}. In much of the following, we'll omit explicit mention of φ and assume that the x(i) are in Rd, but you should always have in mind that some additional process was almost surely required to go from the actual input examples to their feature representation.

In supervised learning we are given a training data set of the form:

Dn = { (x(1), y(1)), …, (x(n), y(n)) }

We will assume that each x(i) is a d × 1 column vector. The intended meaning of this data is that, when given an input x(i), the learned hypothesis should generate output y(i).

What makes a classifier useful? That it works well on new data; that is, that it makes good predictions on examples it hasn't seen. But we don't know exactly what data this classifier might be tested on when we use it in the real world. So, we have to assume a connection between the training data and testing data; typically, they are drawn independently from the same probability distribution.

> My favorite analogy is to problem sets. We evaluate a student's ability to generalize by putting questions on the exam that were not on the homework (training set).

Given a training set Dn and a classifier h, we can define the training error of h to be:

En(h) = (1/n) Σᵢ 𝟙[h(x(i)) ≠ y(i)]

For now, we will try to find a classifier with small training error (later, with some added criteria) and hope it generalizes well to new data, and has a small test error

E(h) = (1/n′) Σᵢ 𝟙[h(x(i)) ≠ y(i)]

on n′ new examples that were not used in the process of finding the classifier.

## Learning algorithm

A hypothesis class H is a set (finite or infinite) of possible classifiers, each of which represents a mapping from Rd → {−1, +1}.

A learning algorithm is a procedure that takes a data set Dn as input and returns an element h of H; it looks like:

Dn → learning alg (H) → h

We will find that the choice of H can have a big impact on the test error of the h that results from this process. One way to get h that generalizes well is to restrict the size, or "expressiveness" of H.

## Linear classifiers

We'll start with the hypothesis class of linear classifiers. They are (relatively) easy to understand, simple in a mathematical sense, powerful on their own, and the basis for many other more sophisticated methods.

A linear classifier in d dimensions is defined by a vector of parameters θ ∈ Rd and scalar θ₀ ∈ R. So, the hypothesis class H of linear classifiers in d dimensions is the set of all vectors in Rd+1. We'll assume that θ is a d × 1 column vector.

> Let's be careful about dimensions. We have assumed that x and θ are both d × 1 column vectors. So θᵀx is 1 × 1, which in math (but not necessarily numpy) is the same as a scalar.

Given particular values for θ and θ₀, the classifier is defined by:

h(x; θ, θ₀) = sign(θᵀx + θ₀) = { +1 if θᵀx + θ₀ > 0; −1 otherwise }

Remember that we can think of θ, θ₀ as specifying a hyperplane. It divides Rd, the space our x(i) points live in, into two half-spaces. The one that is on the same side as the normal vector is the positive half-space, and we classify all points in that space as positive. The half-space on the other side is negative and all points in it are classified as negative.

**Example:** Let h be the linear classifier defined by θ = [−1, 1.5]ᵀ, θ₀ = 3.

The diagram below shows several points classified by h. In particular, let x(1) = [3, 2]ᵀ and x(2) = [4, −1]ᵀ.

h(x(1); θ, θ₀) = sign([−1, 1.5] · [3, 2]ᵀ + 3) = sign(3) = +1

h(x(2); θ, θ₀) = sign([−1, 1.5] · [4, −1]ᵀ + 3) = sign(−2.5) = −1

Thus, x(1) and x(2) are given positive and negative classifications, respectively.

θᵀx + θ₀ = 0

> **Study Question:** What is the green vector normal to the hyperplane? Specify it as a column vector.

> **Study Question:** What change would you have to make to θ, θ₀ if you wanted to have the separating hyperplane in the same place, but to classify all the points labeled '+' in the diagram as negative and all the points labeled '-' in the diagram as positive?

## Learning linear classifiers

Now, given a data set and the hypothesis class of linear classifiers, our objective will be to find the linear classifier with the smallest possible training error.

This is a well-formed optimization problem. But it's not computationally easy!

We'll start by considering a very simple learning algorithm. The idea is to generate k possible hypotheses by generating their parameter vectors at random. Then, we can evaluate the training-set error on each of the hypotheses and return the hypothesis that has the lowest training error (breaking ties arbitrarily).

> It's a good idea to think of the "stupidest possible" solution to a problem, before trying to get clever. Here's a fairly (but not completely) stupid algorithm.

```
RANDOM-LINEAR-CLASSIFIER(Dn, k, d)
  for j = 1 to k
    randomly sample (θ(j), θ₀(j)) from (Rd, R)
  j* = arg minⱼ∈{1,…,k} En(θ(j), θ₀(j))
  return (θ(j*), θ₀(j*))
```

> This might be new notation: arg minₓ f(x) means the value of x for which f(x) is the smallest. Sometimes we write arg minₓ∈X f(x) when we want to explicitly specify the set X of values of x over which we want to minimize.

> **Study Question:** What do you think happens to En(h), where h is the hypothesis returned by RANDOM-LINEAR-CLASSIFIER, as k is increased?

> **Study Question:** What properties of Dn do you think will have an effect on En(h)?

## Evaluating a learning algorithm

How should we evaluate the performance of a classifier h? The best method is to measure test error on data that was not used to train it.

How should we evaluate the performance of a learning algorithm? This is trickier. There are many potential sources of variability in the possible result of computing test error on a learned hypothesis h:

- Which particular training examples occurred in Dn
- Which particular testing examples occurred in Dn′
- Randomization inside the learning algorithm itself

Generally, we would like to execute the following process multiple times:

1. Train on a new training set
2. Evaluate resulting h on a testing set that does not overlap the training set

Doing this multiple times controls for possible poor choices of training set or unfortunate randomization inside the algorithm itself.

One concern is that we might need a lot of data to do this, and in many applications data is expensive or difficult to acquire. We can re-use data with cross validation (but it's harder to do theoretical analysis).

```
CROSS-VALIDATE(D, k)
  divide D into k chunks D₁, D₂, …, Dₖ (of roughly equal size)
  for i = 1 to k
    train hᵢ on D \ Dᵢ (withholding chunk Dᵢ)
    compute "test" error Eᵢ(hᵢ) on withheld data Dᵢ
  return (1/k) Σᵢ Eᵢ(hᵢ)
```

It's very important to understand that cross-validation neither delivers nor evaluates a single particular hypothesis h. It evaluates the algorithm that produces hypotheses.
