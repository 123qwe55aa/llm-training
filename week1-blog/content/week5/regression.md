# Regression

> Week 5: Regression · MIT 6.036 courseware archive

## Notes – Chapter 7: Regression

Notes – Chapter 7: Regression
You can sequence through the Regression lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 7: Regression
notes as a PDF file.

## Lecture: Regression and the ordinary least squares problem

Lecture: Regression and the ordinary least squares problem
Lecture: Regression and the ordinary least squares problem
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Introduction to regression

Introduction to regression
Now we will turn to a slightly different form of machine-learning problem, called
“Regression," in common parlance, means moving backwards. But this is forward progress!
regression
. It is still supervised learning, so our data will still have the form
[mathjax]S_ n = \left\{ \left(x^{(1)}, y^{(1)}\right), \dots , \left(x^{(n)}, y^{(n)}\right)\right\} \; \; .[/mathjax]
But now, instead of the [mathjaxinline]y[/mathjaxinline] values being discrete, they will be real-valued, and so our hypotheses will have the form
[mathjax]h: \mathbb {R}^ d \rightarrow \mathbb {R} \; \; .[/mathjax]
This is a good framework when we want to predict a numerical quantity, like height, stock value, etc., rather than to divide the inputs into categories.
The first step is to pick a loss function, to describe how to evaluate the quality of the predictions our hypothesis is making, when compared to the “target" [mathjaxinline]y[/mathjaxinline] values in the data set. The choice of loss function is part of modeling your domain. In the absence of additional information about a regression problem, we typically use
squared error
(SE):
[mathjax]\text {Loss}(\text {guess}, \text {actual}) = (\text {guess} - \text {actual})^2 \; \; .[/mathjax]
It penalizes guesses that are too high the same amount as it penalizes guesses that are too low, and has a good mathematical justification in the case that your data are generated from an underlying linear hypothesis, but with Gaussian-distributed noise added to the [mathjaxinline]y[/mathjaxinline] values.
We will consider the case of a linear hypothesis class,
[mathjax]h(x;\theta , \theta _0) = \theta ^ Tx + \theta _0 \; \; ,[/mathjax]
remembering that we can get a rich class of hypotheses by performing a non-linear feature transformation before doing the regression. So, [mathjaxinline]\theta ^ Tx + \theta _0[/mathjaxinline] is a linear function of [mathjaxinline]x[/mathjaxinline], but [mathjaxinline]\theta ^ T\varphi (x) + \theta _0[/mathjaxinline] is a non-linear function of [mathjaxinline]x[/mathjaxinline] if [mathjaxinline]\varphi[/mathjaxinline] is a non-linear function of [mathjaxinline]x[/mathjaxinline].
We will treat regression as an optimization problem, in which, given a data set [mathjaxinline]{\cal D}[/mathjaxinline], we wish to find a linear hypothesis that minimizes mean squared error. Our objective, often called
mean squared error
, is to find values for [mathjaxinline]\Theta = (\theta , \theta _0)[/mathjaxinline] that minimize
[mathjax]J(\theta , \theta _0) = \frac{1}{n}\sum _{i = 1}^ n\left(\theta ^ Tx^{(i)} + \theta _0 - y^{(i)}\right)^2 \; \; ,[/mathjax]
resulting in the solution:
[mathjax]\theta ^*, \theta _0^* = {\rm arg}\min _{\theta , \theta _0} J(\theta , \theta _0)\; \; .[/mathjax]
(1.1)
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:13 PM (revision f808f068e)

## Lecture: Regression - ordinary least squares solution using optimization

Lecture: Regression - ordinary least squares solution using optimization
Lecture: Regression - ordinary least squares solution using optimization
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Regression - OLS analytical solution setup

Lecture: Regression - OLS analytical solution setup
Lecture: Regression - OLS analytical solution setup
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Regression - OLS analytical solution using gradients

Lecture: Regression - OLS analytical solution using gradients
Lecture: Regression - OLS analytical solution using gradients
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Analytical solution: ordinary least squares

Analytical solution: ordinary least squares
One very interesting aspect of the problem finding a linear hypothesis that minimizes mean squared error (this general problem is often called
ordinary least squares
(
ols
)) is that we can find a closed-form formula
What does “closed form" mean? Generally, that it involves direct evaluation of a mathematical expression using a fixed number of “typical" operations (like arithmetic operations, trig functions, powers, etc.). So equation
is not in closed form, because it's not at all clear what operations one needs to perform to find the solution.
for the answer!
Everything is easier to deal with if we assume that the [mathjaxinline]x^{(i)}[/mathjaxinline] have been augmented with an extra input dimension (feature) that always has value 1, so we may ignore [mathjaxinline]\theta _0[/mathjaxinline]. (See chapter
, section
for a reminder about this strategy).
We will approach this just like a minimization problem from calculus homework: take the derivative of [mathjaxinline]J[/mathjaxinline] with respect to [mathjaxinline]\theta[/mathjaxinline], set it to zero, and solve for [mathjaxinline]\theta[/mathjaxinline]. There is an additional step required, to check that the resulting [mathjaxinline]\theta[/mathjaxinline] is a minimum (rather than a maximum or an inflection point) but we won't work through that here. It is possible to approach this problem by:
Finding [mathjaxinline]\partial {J}/\partial {\theta _ k}[/mathjaxinline] for [mathjaxinline]k[/mathjaxinline] in
We will use [mathjaxinline]d[/mathjaxinline] here for the total number of features in each [mathjaxinline]x^{(i)}[/mathjaxinline], including the added 1.
[mathjaxinline]1, \ldots , d[/mathjaxinline],
Constructing a set of [mathjaxinline]k[/mathjaxinline] equations of the form [mathjaxinline]\partial {J}/\partial {\theta _ k} = 0[/mathjaxinline], and
Solving the system for values of [mathjaxinline]\theta _ k[/mathjaxinline].
That works just fine. To get practice for applying techniques like this to more complex problems, we will work through a more compact (and cool!) matrix view.
Study Question:
Work through this and check your answer against ours below.
We can think of our training data in terms of matrices [mathjaxinline]X[/mathjaxinline] and [mathjaxinline]Y[/mathjaxinline], where each column of [mathjaxinline]X[/mathjaxinline] is an example, and each “column" of [mathjaxinline]Y[/mathjaxinline] is the corresponding target output value:
[mathjax]X = \begin{bmatrix} x_1^{(1)} &  \dots &  x_1^{(n)}\\ \vdots &  \ddots &  \vdots \\ x_ d^{(1)} &  \dots &  x_ d^{(n)}\end{bmatrix} \; \; \;  Y = \begin{bmatrix} y^{(1)} &  \dots &  y^{(n)}\end{bmatrix}\; \; .[/mathjax]
Study Question:
What are the dimensions of [mathjaxinline]X[/mathjaxinline] and [mathjaxinline]Y[/mathjaxinline]?
In most textbooks, they think of an individual example [mathjaxinline]x^{(i)}[/mathjaxinline] as a row, rather than a column. So that we get an answer that will be recognizable to you, we are going to define a new matrix and vector, [mathjaxinline]W[/mathjaxinline] and [mathjaxinline]T[/mathjaxinline], which are just transposes of our [mathjaxinline]X[/mathjaxinline] and [mathjaxinline]Y[/mathjaxinline], and then work with them:
[mathjax]W = X^ T = \begin{bmatrix} x_1^{(1)} &  \dots &  x_ d^{(1)}\\ \vdots &  \ddots &  \vdots \\ x_1^{(n)} &  \dots &  x_ d^{(n)}\end{bmatrix} \; \;  T = Y^ T = \begin{bmatrix} y^{(1)}\\ \vdots \\ y^{(n)}\end{bmatrix} \; \; .[/mathjax]
Study Question:
What are the dimensions of [mathjaxinline]W[/mathjaxinline] and [mathjaxinline]T[/mathjaxinline]?
Now we can write
[mathjax]J(\theta ) = \frac{1}{n}\underbrace{(W\theta - T)^ T}_{1 \times n}\underbrace{(W\theta - T)}_{n \times 1} = \frac{1}{n}\sum _{i=1}^ n \left(\left(\sum _{j=1}^ d W_{ij}\theta _ j \right) - T_ i\right)^2[/mathjax]
and using facts about matrix/vector calculus, we get
[mathjax]\nabla _{\theta }J = \frac{2}{n}\underbrace{W^ T}_{d \times n}\underbrace{(W\theta - T)}_{n \times 1}\; \; .[/mathjax]
Setting to 0 and solving, we get:
[mathjaxinline]\displaystyle  \frac{2}{n}W^ T(W\theta - T)[/mathjaxinline]
[mathjaxinline]\displaystyle  = 0[/mathjaxinline]
[mathjaxinline]\displaystyle W^ TW\theta - W^ T T[/mathjaxinline]
[mathjaxinline]\displaystyle  = 0[/mathjaxinline]
[mathjaxinline]\displaystyle W^ TW\theta[/mathjaxinline]
[mathjaxinline]\displaystyle  = W^ T T[/mathjaxinline]
[mathjaxinline]\displaystyle \theta[/mathjaxinline]
[mathjaxinline]\displaystyle  = (W^ TW)^{-1} W^ T T[/mathjaxinline]
And the dimensions work out!
[mathjax]\theta = \underbrace{\left(W^ TW\right)^{-1}}_{d \times d}\underbrace{W^ T}_{d \times n}\underbrace{T}_{n \times 1}[/mathjax]
So, given our data, we can directly compute the linear regression that minimizes mean squared error. That's pretty awesome!
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:45 PM (revision 4f166135)

## Lecture: Regression - beauty of the closed form OLS solution

Lecture: Regression - beauty of the closed form OLS solution
Lecture: Regression - beauty of the closed form OLS solution
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Regression - regularization by ridge regression

Lecture: Regression - regularization by ridge regression
Lecture: Regression - regularization by ridge regression
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Regression - analytical minimization of the ridge regression objective

Lecture: Regression - analytical minimization of the ridge regression objective
Lecture: Regression - analytical minimization of the ridge regression objective
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Regularizing linear regression

Regularizing linear regression
Well, actually, there are some kinds of trouble we can get into. What if [mathjaxinline]\left(W^ TW\right)[/mathjaxinline] is not invertible?
Study Question:
Consider, for example, a situation where the data-set is just the same point repeated twice: [mathjaxinline]x^{(1)} = x^{(2)} = (1, 2)^ T[/mathjaxinline]. What is [mathjaxinline]W[/mathjaxinline] in this case? What is [mathjaxinline]W^ TW[/mathjaxinline]? What is [mathjaxinline](W^ TW)^{-1}[/mathjaxinline]?
Another kind of problem is
overfitting
: we have formulated an objective that is just about fitting the data as well as possible, but as we discussed in the context of margin maximization, we might also want to
regularize
to keep the hypothesis from getting
too
attached to the data.
We address both the problem of not being able to invert [mathjaxinline](W^ TW)^{-1}[/mathjaxinline] and the problem of overfitting using a mechanism called
ridge regression
. We add a regularization term [mathjaxinline]\| \theta \| ^2[/mathjaxinline] to the
ols
objective, with trade-off parameter [mathjaxinline]\lambda[/mathjaxinline].
Study Question:
When we add a regularizer of the form [mathjaxinline]\| \theta \| ^2[/mathjaxinline], what is our most “preferred" value of [mathjaxinline]\theta[/mathjaxinline], in the absence of any data?
Here is the ridge regression objective function:
[mathjax]J_{\text {ridge}}(\theta , \theta _0) = \frac{1}{n}\sum _{i = 1}^ n\left(\theta ^ Tx^{(i)} + \theta _0 - y^{(i)}\right)^2 + \lambda \| \theta \| ^2[/mathjax]
Larger [mathjaxinline]\lambda[/mathjaxinline] values pressure [mathjaxinline]\theta[/mathjaxinline] values to be near zero. Note that we don't penalize [mathjaxinline]\theta _0[/mathjaxinline]; intuitively, [mathjaxinline]\theta _0[/mathjaxinline] is what “floats" the regression surface to the right level for the data you have, and so you shouldn't make it harder to fit a data set where the [mathjaxinline]y[/mathjaxinline] values tend to be around one million than one where they tend to be around one. The other parameters control the orientation of the regression surface, and we prefer it to have a not-too-crazy orientation.
There is an analytical expression for the [mathjaxinline]\theta , \theta _0[/mathjaxinline] values that minimize [mathjaxinline]J_\text {ridge}[/mathjaxinline], but it's a little bit more complicated to derive than the solution for
ols
because [mathjaxinline]\theta _0[/mathjaxinline] needs special treatment. If we decide not to treat [mathjaxinline]\theta _0[/mathjaxinline] specially (so we add a 1 feature to our input vectors), then we get:
[mathjax]\nabla _{\theta }J_\text {ridge} = \frac{2}{n}W^ T(W\theta - T) + 2 \lambda \theta \; \; .[/mathjax]
Setting to 0 and solving, we get:
[mathjaxinline]\displaystyle  \frac{2}{n}W^ T(W\theta - T) + 2 \lambda \theta[/mathjaxinline]
[mathjaxinline]\displaystyle  = 0[/mathjaxinline]
[mathjaxinline]\displaystyle \frac{1}{n}W^ TW\theta - \frac{1}{n}W^ TT + \lambda \theta[/mathjaxinline]
[mathjaxinline]\displaystyle  = 0[/mathjaxinline]
[mathjaxinline]\displaystyle \frac{1}{n}W^ TW\theta + \lambda \theta[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{1}{n}W^ TT[/mathjaxinline]
[mathjaxinline]\displaystyle W^ TW\theta + n \lambda \theta[/mathjaxinline]
[mathjaxinline]\displaystyle  = W^ TT[/mathjaxinline]
[mathjaxinline]\displaystyle (W^ TW + n \lambda I)\theta[/mathjaxinline]
[mathjaxinline]\displaystyle  = W^ TT[/mathjaxinline]
[mathjaxinline]\displaystyle \theta[/mathjaxinline]
[mathjaxinline]\displaystyle  = (W^ TW + n \lambda I)^{-1}W^ TT[/mathjaxinline]
Whew! So,
[mathjax]\theta _{\text {ridge}} = \left(W^ TW + n\lambda I\right)^{-1}W^ TT[/mathjax]
This is called “ridge" regression because we are adding a “ridge" of [mathjaxinline]\lambda[/mathjaxinline] values along the diagonal of the matrix before inverting it.
which
becomes invertible when [mathjaxinline]\lambda > 0[/mathjaxinline].
Study Question:
Derive this version of the ridge regression solution.
Talking about regularization
In machine learning in general, not just regression, it is useful to distinguish two ways in which a hypothesis [mathjaxinline]h \in {\cal H}[/mathjaxinline] might contribute to errors on test data. We have
There are technical definitions of these concepts that are studied in more advanced treatments of machine learning. Structural error is referred to as
bias
and estimation error is referred to as
variance
.
note
When we increase [mathjaxinline]\lambda[/mathjaxinline], we tend to increase structural error but decrease estimation error, and vice versa.
Study Question:
Consider using a polynomial basis of order [mathjaxinline]k[/mathjaxinline] as a feature transformation [mathjaxinline]\phi[/mathjaxinline] on your data. Would increasing [mathjaxinline]k[/mathjaxinline] tend to increase or decrease structural error? What about estimation error?
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:13 PM (revision f808f068e)

## Lecture: Regression - ridge regression using gradient descent

Lecture: Regression - ridge regression using gradient descent
Lecture: Regression - ridge regression using gradient descent
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Optimization via gradient descent

Optimization via gradient descent
Inverting the [mathjaxinline]d \times d[/mathjaxinline] matrix [mathjaxinline]\left(W^ TW\right)[/mathjaxinline] takes
Well, actually, Gauss-Jordan elimination, a popular algorithm, takes [mathjaxinline]O(d^3)[/mathjaxinline] arithmetic operations, but the bit complexity of the intermediate results can grow exponentially! There are other algorithms with polynomial bit complexity. (If this just made no sense to you, don't worry.)
[mathjaxinline]O(d^3)[/mathjaxinline]
time, which makes the analytic solution impractical for large [mathjaxinline]d[/mathjaxinline]. If we have high-dimensional data, we can fall back on gradient descent.
Study Question:
Why is having large [mathjaxinline]n[/mathjaxinline] not as much of a computational problem as having large [mathjaxinline]d[/mathjaxinline]?
Recall the ridge objective
[mathjax]J_{\text {ridge}}(\theta , \theta _0) = \frac{1}{n}\sum _{i = 1}^ n\left(\theta ^ Tx^{(i)} + \theta _0 - y^{(i)}\right)^2 + \lambda \| \theta \| ^2[/mathjax]
and its gradient with respect to [mathjaxinline]\theta[/mathjaxinline]
[mathjax]\nabla _{\theta }J = \frac{2}{n}\sum _{i = 1}^ n\left(\theta ^ Tx^{(i)} + \theta _0 - y^{(i)}\right)x^{(i)} + 2\lambda \theta[/mathjax]
and partial derivative with respect to [mathjaxinline]\theta _0[/mathjaxinline]
[mathjax]\frac{\partial J}{\partial \theta _0} = \frac{2}{n}\sum _{i = 1}^ n\left(\theta ^ Tx^{(i)} + \theta _0 - y^{(i)}\right) \; \; .[/mathjax]
Armed with these derivatives, we can do gradient descent, using the regular or stochastic gradient methods from chapter
.
Even better, the objective functions for
ols
and ridge regression are
convex
, which means they have only one minimum, which means, with a small enough step size, gradient descent is
guaranteed
to find the optimum.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:45 PM (revision 4f166135)

## Lecture: Regression - stochastic gradient descent

Lecture: Regression - stochastic gradient descent
Lecture: Regression - stochastic gradient descent
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Regression - OLS and gradient descent demo example

Lecture: Regression - OLS and gradient descent demo example
Lecture: Regression - OLS and gradient descent demo example
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Regression - structural error and estimation error

Lecture: Regression - structural error and estimation error
Lecture: Regression - structural error and estimation error
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Video transcripts

### Lecture: Regression and the ordinary least squares problem


LESLIE KAELBLING: It is time for us
to change our machine learning problem.
So so far, we've been talking about classification.
Classification is a kind of supervised learning
where we get inputs in some vector space,
and we try to map them into, say, a plus 1 or minus 1.
So we want to say, these are the positive examples,
these are the negative examples.
Today, we're going to start talking about regression.
It's going to use a lot of the same tools
that we already have, but it's a slightly different problem
setting.
So in regression, we still have data
that looks like a set of x1, y1 pairs, x and y n.
So that's all as it used to be.
And we're still going to assume that our xi's are
in R to the d.
But now we're going to assume that our yi's are real valued.
So instead of trying to classify something,
we're going to think about predicting a continuous value
for something.
Maybe it's the temperature or the stock price
or the percentage of the class that
gets an A or something like that, so some continuous value.
And actually, almost everything we talk about
can be generalized to the case of R to some other dimension.
That is to say instead of predicting
a single real-valued output, we could actually
predict a whole vector valued-output.
Today, we'll just focus on this case.
But it's a very straightforward extension
to go to the multi-valued output,
multi-dimensional output.
OK.
So our hypotheses, then, so our hypothesis space,
right, a hypothesis is going to take an element in R to the d
and give us a real value.
And so far, we've been concentrating
on linear classifiers and learned
that if we make a nonlinear transformation to the input
space, we can end up with a nonlinear classifier.
So for similar reasons, we're going
to focus on linear regression.

So in linear regression, our hypothesis class
H is going to be defined, again, with a vector theta, right,
so a vector theta in R to the d and a theta 0.
That's going to be our hypothesis class.
And the hypothesis is going to be
h of x given theta and theta 0.
This is going to look very familiar.

It's just going to be that, right?
So so far, for classification, we
did this dot product with a vector of parameter values
and an offset.
And then we took the sine of it or did something
to convert it into a plus 1 or a minus 1.
Now we're just going to leave it like this
and say, oh, we're going to predict
that if the input is this vector x,
the output is this real number.

Does that make sense so far?

OK.
So then-- let's see-- so we set up a machine learning problem.
What are our steps?
Our steps are to see what data we have,
to pick a hypothesis class.
Next step is to pick a loss function.
OK.
So we have data, hypothesis class.
So what's the loss function that we're going to use?
OK.
So loss function looks like loss of guess and actual, right?
So we want to say, oh, if our hypothesis predicts
this guess but this is really what we wanted the answer
to be, here's how unhappy we are.
And again, this could be a domain-dependent choice.
You might know something about your domain,
and whether it's better or worse to predict
values that are too high or values that are too low,
or you might have some feeling about how this ought to go,
what kind of predictions you want.
In the absence of any particular domain-dependent understanding
of your problem, the classical loss function
here is to do guess minus actual squared.
And this is called, not surprisingly, squared loss.

It would be completely reasonable to make
this a absolute value.
It makes things not differentiable in a way that's
sort of inconvenient.
But it's not hopeless.
Lots of people do look at the case
where you do absolute value instead of square.
But we're going to stick with square because things are nice,
and it's very classic.
And it's well-motivated for a reason
that I'll talk about a little bit
once I start drawing pictures.
It has the property that it's equally unhappy
if you predict something that's a little too high
or a little too low.
It doesn't care whether you were high or low.
It's the magnitude of the difference that matters.
So this is not the only--
when we talk about regression, this
is the definition of a regression problem, right?
But you can pick your hypothesis class--
all different choices of hypothesis class.
That's still a regression problem.
And you can pick your loss function,
different choices of lost.
It's still a regression problem.
And we're just going to work through a very classical setup
for regression, but there's, of course,
different ways you could do it
OK.
So linear regression, squared loss.

So this gives us, actually, if we take linear regression
and squared loss together, this gives us a problem that's
very classical in statistics.
It's so classical it's called ordinary.
So it's sometimes called ordinary least squares,
or even, to its friends, OLS.
So ordinary least squares--
it's called least squares because we're
going to try to find the hypothesis that
minimizes this squared error.


### Lecture: Regression and the ordinary least squares problem


LESLIE KAELBLING: So last time, we
talked about how the next step in the kind of typical design
of machine learning algorithm is to set up an optimization
problem.
So since we have a data set, and a loss function,
and a hypothesis class, we can set up an optimization problem.
And first, we'll do it with our regularization,
and then we can add the regularization in.
So we do it without regularization.
We want to write down an objective, so here's our J.
And we're going to have our big--
like capital theta is going to be made up of this theta
vector and the offset, right?
Those are the parameters that we're
going to have to optimize.
So when we write down J, J is like a pessimists thing.
How unhappy are we about theta and theta knot, right?
This is how mad.
OK, so how mad?
Well, we're in a generally take an average over the data set.
We're going to consider all our data points,
and we're going to say how mad we are about this is how mad we
are about the prediction that we make on each of these points.
And so it's going to be the squared loss,
so our guess is going to be theta transpose
XI plus theta knot.
And our actual is YI and squared.

This could be our basic objective function.

So now, the most awesome thing about this problem
is that it has a direct solution.
OK, so let's talk about, what's the solution?
Well, what we really want to find,
what we want is theta star or theta knot
star, which are the arg max over theta and theta
knot of J theta, theta knot.
OK, so this is what we would like to find, right?
The stars-- I can't remember if I've
talked about this before yet.
But generally speaking, we write something with a star.
It's like the best one.
It's the star, right?
So the best possible theta and theta knot are the ones
that for us that maximize the value of J.
And often, we get this far writing our problem down.
And then we have to think about some complicated algorithmic
method of trying to compute or estimate the theta star.
But for this particular problem set up,
we're going to be able to actually derive
a very beautiful formula that will give us
kind of a direct method of calculating theta and theta
star.
Question?

Thank you, yes.

Good, we are trying to minimize.


### Lecture: Regression and the ordinary least squares problem

LESLIE KAELBLING: All right, our approach
is going to be an approach kind of good old fashioned calculus.
Which is to say, see if we can take
the derivative of this thing with respect to the thetas,
set it to zero, and solve.
OK, and you can do it by actually--
so there's a system of equations, right?
So one thing you could do is you can
take this objective function, compute
the derivative with respect to each individual theta
parameter, and that gives you--
and set it to zero.
That'll give you a system of equations.
And if you solve that system of equations,
you'll get the theta star values.
We're going to do that, but we're
going to do it in matrix form.
Partly, because it's beautiful, and it fits on the board.
And we're less likely to make mistakes.
Partly, because that's the way we're
going to approach doing some derivations for neural networks
also.
But if any of the what I do here on the board is mysterious,
you can always go back and do it piece
by piece in terms of the individual partial derivatives.
It's kind of a nice shorthand to get used to it.
OK, so to get a setup for the shorthand,
we have to define some matrices.
So let's think about a big matrix x, and make it a big x.

And this is a matrix that we're going
to make from all the x's in our data
set, so it's going to have x1 one, x.

OK, yeah, let me do it this way, x1d.
All right, so r for us when we were thinking about our data
points, they're column vectors.
So this is the x1.
This is the first column vector, our first training example x.
And then we go to xn, xnd, so that's our last training
example, right?
So you could just take all the x's in your data
set, mash them together into a big matrix.
This gives us x.
The dimensions of x are d by n.

Similarly, if we take our data set,
and we grab all the y values--
So the y values are scalar.
But we can put them into a vector,
and we can put them in a vector that has this shape.

And this is one by n, so we could
operate on this x and that y.

I'm going to do a maneuver, which
I do also explicitly in the notes
to make the problem that we solve here
look like the problems that you might see in some other more
classical textbooks.
Again, like different people pick access
to be the rows versus the columns.
And so it's hard to be compatible with everybody,
but there is a particular way that lots of people
do this OLS.
And it requires us to rotate our matrices,
so I'm going to do that.
OK, so this is not a huge deal.
But it'll make things look more like everybody else.
So I'm going to define a new matrix called
w, which is just our big old x over there transposed.
And that is going to have each data point in a row.

And similarly, I'm going to define t.
I don't know if it's the best choice of letters,
but to be y transpose, big y over there transposed.
So that means now it's going to be a column vector.

All right, and x now is--
I mean, w is n by d.
That was d by n.
This is n by d, and this is n by one.
OK, so we're going to work with w and t.
But they're just transposes of those other guys,
so now each training example is a row.
And this is the label that goes with it.

And so intuitively, what we want to do
is find a vector theta so that each row if we multiply it
by that vector theta, we get the y over here.
That's what we really want.
OK, but let's see.
What are we trying to do?
What we're trying to do is find some thetas that minimize
the j, so now what I'm going to do
is rewrite j using this matrix w and this vector t.
OK, so how do we do that?
So, what's j?
I'm also going to assume--

wait, I'm sorry.
I can't see you talking, and therefore, I can't hear you.
OK, good.
AUDIENCE: [INAUDIBLE]

LESLIE KAELBLING: Beauty, OK, good.
Exactly, so you just totally set up
the thing I was going to say.
Which is we're going to assume for the purposes of this
that we have made one of our features be all ones,
so that we're not going to explicitly deal
with the theta knot.
I mean, you could do it with the theta knot,
but it makes the board more complicated.
So we'll just assume that we don't
have to deal with theta knot.
OK, good.
So let's rewrite j of theta now.
It's still one over n.

OK, but what's going on here?
So this is like--

it's a sum of some differences, right?
It's a sum over the data points, but we
can get a sum over the data points
by doing a matrix operation.
So we're going to think of this theta w theta.
So if you think of w theta, so you take this matrix w.
You multiply it by the vector theta.
This is going to give us a vector of predictions,
and let's do the dimensions.
So if w is n by d, and theta is d by one,
then w theta is going to be n by one.

That's a vector of predictions.
Now we're interested in how different our predictions are
from the values that we really wanted,
and so we can subtract the actual values
that we really wanted.
So this is our predictions.
This is our actuals.
That's also n by one.
That's good.
That's good.
That means we can subtract them, and everything is cool.
Now we need to square and sum to get that thing over there.
So a way to square and sum is to write it like this.

That's one way to think about it, right?
So this is a vector of sort of just the differences,
and this is a vector of differences.
And this is like taking the dot product of the vector
of differences with itself.
So that will sum and square, so this is the same as that
if you just leave out the theta knot.
Does that seem good?
So it's getting more compact.
Basically, the view is the fewer summations and indices
that you need to write down, the more likely
it is that you will do your derivation right.
At least for me, that's true.
OK, so here's our objective function.

So we'd like to find that theta that minimizes that, so this
is ultimately a scalar, right?
You're good with that, right?
Because this is n by one.
But if we transpose it, it's one by n.
We got one by n times then by one.
That's a scalar, so that's the thing we want to minimize.


### Lecture: Regression and the ordinary least squares problem


LESLIE KAELBLING: We are going to take the derivative
and set it to zero.
The derivative that we need is really a gradient.
OK, remember-- OK, so this is a gradient.
And just remember that it's going to have the same shape
as theta.

And theta up there--
theta is d by one, d by one.

J is a scalar.
Theta is a d by 1 vector.
And the gradient is also going to be d by 1.
That's what a gradient is.
And it is just the vector of partial derivative.
So it's partial j, partial theta 1, partial j, partial theta d.

And if you want to work it out that way,
and you should at home with your pencil,
you will get the same answer as we got
doing it in a different way.
OK, so let's just kind of back up and apply.
So another way to think about that is that it's a square.
It's this times that.
And if you, in regular old calculus,
if you needed to take the derivative of a square,
you end up with 2 times the thing
you had times the derivative of the stuff inside.
The derivative of the stuff inside with respect to theta.
I'm just going to write down as W.
Now, you wouldn't obviously know to transpose W
or to put it over here, I can give you an argument
for why that's right.
But just back up for a minute and say
look, what's the derivative of this with respect to theta?
If these were scalers, they would totally be W.
And it's going to work out that way, OK, in vectors, too.
And the way to convince yourself is
to do it component by component.
So you can verify that this is the right answer by taking
the derivative by hand.
Let me just-- I'm not going to do it on the board
though because you can all do it.
I am just going to verify the shapes work out OK though.
So shapes.
Let's see.
So what is the shape of W?
The shape of W is n by d.
So this thing is going to be d by n.

We already figured out the dimension of this.
This is n by 1.
And so d by n times n by 1 is d by 1,
which is the shape we want.
So at least the shape is right.
And you can verify the elements.

So again, what is this vector?
It's a vector.
What does this vector mean?
So each element here, each element
says how much would j change?
At what rate would j change if I were
to change parameter theta I?
So if j doesn't really depend very much on theta I,
this will be small [INAUDIBLE].
If it doesn't depend very much on theta 1, it'll be small.
If it depends a lot on theta 1, it'll be big.

We good with this?

OK, awesome.
So now what do we do?
What do we do?
Well, we can try to set to 0 and solve.
Remember that there's two steps in one
of these calculus things.
So one is that you need-- so we know
that this is a critical point.

And if you were to just write that down
in your calculus test, you would get minus 2
because you also have to verify that it's a minimum.
It might be an inflection or it might be a maximum and so on.
And you say you wish you'd also verify
that it's a minimum that-- we're not going to do that.
You can trust me.
Or you can work it out on your own if you want to.
But it's minimum.
OK, good.
So now what we're going to do is set this thing
to 0 and solve for theta.
It's kind of fun.
We can ignore those scalar bits there.
So we're going to get W transpose W theta
minus W transpose T equals 0.

So we get W transpose W theta equals W transpose T.
And if you remember like anything about doing
matrix algebra stuff out like this,
it's that you can't change the order of anything
unless you follow the rules.
And you can't divide.
But we can take the inverse of this.
So we can make--
and we can multiply both sides by it.
So W transpose W inverse W transpose W theta
is W transpose W inverse W transpose
T. And then by design, we see that this times this
is the identity.
So we get theta is equal to W transpose W inverse WT.

So this is the solution.

This is not [INAUDIBLE] this is not just any old theta.
That is theta star.
Yeah, did I goof?
STUDENT: [INAUDIBLE].

LESLIE KAELBLING: Say again.
STUDENT: [INAUDIBLE].
LESLIE KAELBLING: It is the best data.
STUDENT: [INAUDIBLE].
LESLIE KAELBLING: That's right.


### Lecture: Regression and the ordinary least squares problem


LESLIE KAELBLING: Let me just talk
about how awesome and unusual this is for a minute, right?
So this is like--
it's what people call--
I always used to not know what this meant.
So I wrote a little note about it.
It's what people call a closed form
solution, which means roughly--
the way I think about it is, it means roughly
that like the mathematical expression that's
written on the board, you could just straight up
compute it with a pencil.
Like it tells you what to do.
Admittedly, inverting this matrix, especially if it's big,
is not going to be nice.
And we're going to come to that in a minute,
but it's like a thing you could do on your paper.
You know what operations you'd have to do.
Before when we just wrote that we wanted the theta star that
minimized this, we'd be like, oh, OK that's good.
But I don't know how to find it.
What's cool here is that this just tells you what to do,
so it's a closed form solution.
So that's kind of pretty cool, and it's
very rare for something like this to come out.
So we have to admire it while we have it.
OK, so having admired it, we have two worries.
One worry is that this matrix, W transpose W,
might not be invertible.
That's a thing to worry about.
Another worry is it might be really big.

Not necessarily, no.
It is symmetric, but it is not necessarily positive definite.
OK, so it might be not invertible,
and that's a problem.
Because it's hard to invert a thing that's not invertible,
and it might also be big.
So generally speaking, the computational complexity
of inverting a matrix is on the order of dimension cubed.
So that's expensive, right?
And if you're like, cool, machine learning people,
we're going to eventually be worried
about inverting the matrix.
So first, we'll worry about just the math part of it,
whether it's invertible or not.
And then we'll take another step,
and deal with the computational problems,


### Lecture: Regression and the ordinary least squares problem

LESLIE KAELBLING: We're going to kind of kill
two birds with one stone here.
So one thing to notice is that we're
worried about whether this is going to be something that's
well defined.
And note, also, that we didn't do any regularization.
So we didn't tell it anything about what we wanted the theta
parameters to be except for that they minimize the error.
And generally, what it's going to mean for that matrix
to not be invertible is to say that those parameters are not
yet well specified, that there's no good answer
to the question of what is the theta star that minimizes j.
So to make the problem well-formed in the case
that it's underspecified, we can add a regularizer.
So the regularizer also will keep our solution
from getting too complicated in the way we
talked about last time.
And it will also make the problem well-formed
in the case where it's underspecified,
which is what leads to that matrix not being invertible.
OK, so we're going to look at regularization.

And what we're going to do is take an approach-- again,
this is not the only one, but it's very classic one, which
is called ridge regression.
I'll tell you-- we'll see in a minute
why it's called ridge regression.

It will look very familiar to you.

I'm going to write out the theta 0 parameter again.

OK.
And again, let's be clear that there's two terms.

So this is just the objective function
we had before, but with a constant, right?
So we have a constant, magic parameter.

And this, which note that it can also
be written as theta transpose theta-- that'll
be useful in a minute.
It's just the dot product of theta with itself.

OK, so this says all other things being equal,
I would like the theta vector to be small norm.
I would like to minimize the error.
But I also want to put some pressure on theta to be small.
And it turns out that just putting pressure on theta
is enough in the case that things are underspecified
to say, well, if the data hasn't told you what the thetas need
to be, then I will tell you how I
would like you to break any ties, in some sense,
by applying this pressure on the theta.
OK, so one thing to notice and that you might wonder--
it's a good thing to wonder about.
You might wonder-- so we have theta.
And I called out theta 0 again, so back
to the original formulation.
And the reason I did that is because I want to explicitly
not penalize theta 0.
And the argument is kind of like this--
so a regression problem looks like--
here's our x in one dimension, right?
Here's our x.
Here's our y.
And what we're doing is computing some function of x.
Now, in the case that it's linear, of course,
it's just a linear function of x.
And if it's just a linear function,
well, then it could just only be a line.
Theta 0 governs how it's transposed up and down,
sort of how it floats above the x plane.
And theta governs the slope.
And generally speaking, we don't want to penalize--
like if our data is here, we would like the guy at least
to go up and down for free, which we don't want
to try to drag it down to 0.
We might try to tell it to not have a big slope
because a big slope is a kind of funny thing
in that having your data too close together
can end up with a big slope in a way you don't want.
But generally speaking, we don't want to just tell it, oh, you
should be down near 0, which is what would happen
if we penalized theta 0 also.
So in regression, we're coming up with a function of y
as a function of x.
Generally speaking, x will be a giant high-dimensional plane
that's hard to think about.
And y is a function that floats above that hyperplane.
So we would like to penalize the slope, but not the offset.
OK.
So that's the ridge regression objective.


### Lecture: Regression and the ordinary least squares problem


LESLIE KAELBLING: I'm going to write down
the answer to a simplified version of the problem,
so that you can see the form, and so we
can talk about why it's called ridge regression.
OK.
So, here's point number one, which is actually optimizing
this is a little bit tricky.
You can do it.

So optimizing j ridge with theta and theta 0 is a little tricky.

It is not beyond your capacity.
But it is not what I want to focus on.
It's a homework assignment in 6.8, 6.7.
So you can go take 6.8, 6.7 if you want to do that.
We're going to do--

we're just going to consider the case with theta only.
So I just argued to you that we shouldn't
penalize theta 0, the offset.
But now I'm going to show you the solution which
involves penalizing the offset, because the other one,
it involves subtracting something off the data,
moving it down here, doing the regression down here,
and then putting it back up there again.
It's just like more hassle than we want to think about.
So we're just going to, for right now, consider
the case with theta only.
It will give you the idea.
OK.
And the idea is, if we consider the case with theta only,
we end up--
I'm just going to write the gradient down.
We end up with the gradient having the form--

This part that we saw already, right.
That was the gradient for the first term.
That's the same as the plain old gradient we had before.
And this comes-- you look at that
and you say, oh well, we've got the lambda
and we had the squared thing.
So we've got a 2.
And then the derivative of a squared
is the thing just all by itself.
And that gives us the theta.
And again, you can verify using the fact
that this is a vector of partial derivatives,
you can verify that that's the right answer for that term.
OK.
So this is the gradient.
And then similarly, we can set it to 0 and solve.
But this time I'm not going to do the algebra out
on the board.
And we're going to get theta ridge,
theta star ridge is this.

OK.
So what is different is this.
So this is a scalar.
This is our lambda, which is how hard we're
regularizing this thing.
And I is the identity matrix, right, that just has 1s.
So I is this guy.
1, 1, 1, 1, and 0's everywhere.
So what we're doing is we're taking our W transpose W,
and we are adding to it's something along the diagonal,
right.
And this thing, this guy, it's the ridge.

So we're making our matrix more diagonally dominant,
and by doing this we can make it invertible.
We can make this whole thing inevitable.
So if this guy wasn't before, by adding this in we
can make it invertible.
So we've cured two problems in a sense.
We've cured the problem that just kind of formally
and mathematically this might not be invertible.
And we have also added a regularizer,
something that penalizes the magnitude of theta and can
kind of--
we'll see some demonstrations --kind
of keep it from over fitting.
Yep.
STUDENT: How do we know that it's not invertible?
LESLIE KAELBLING: So you would like to show that at this point
it's positive definite, I think.
So offline I can work this out for you.
I cannot do it on the board.
But generally speaking, if people
talk about matrices being diagonally dominant,
basically once the diagonal is heavy
you're good to go, kind of, and that's the way to do that.

OK.
So, yeah?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: So, let me tell you the strategy.
If you want to do the tricky thing, what you do
is you compute the average y value, just in your data set.
You just compute the average y value.
You take all your data points and you subtract off
the average y value.
So they are near the mean.
Then you do the thing I'm doing, not taking the offset
into account.
And then you push them all back up again.
So it's just a little messy.
I just don't want to write that all over the board.

Yeah.
OK.

OK, good.
So, now what?
So now we've cured one problem.
We do still have, and if your matrices are not too big,
you can just do this, right.
You can analytically compute the solution.
So that's nice.
But we do potentially have a problem,
which is that this might be hard to invert.
So it's sort of on the order of n
cubed operations to compute the inverse.
And so if n is big that's not a thing you want to be doing.
So we're going to have kind of two more topics.
One is to think about the role of the regularizer.
I think I'm going to leave that for a minute.
And the other one is to think about how to optimize
this in a different way.
So I'll do the regularization later,


### Lecture: Regression and the ordinary least squares problem


LESLIE KAELBLING: What's another strategy for computing
the optimal theta here?
STUDENT: Gradient descent.
LESLIE KAELBLING: Gradient descent.
Yeah, that's going to be the answer
to basically every question I asked
about optimization, right?
So we can apply gradient descent to this problem.

And actually, we can do it with confidence
because again, this objective is convex.
And so that means if we apply standard gradient descent
with a small enough step size, which actually might
be too small to be palatable.
But if we do it with a small enough step size,
we would be guaranteed to get as close as we
want to to the optimal.
OK, so let's just remind ourselves
of what the gradient descent rule would look like.
How would we do gradient descent?
It's actually super easy.
So here's our ridge objective.

And actually what I want to do is write--
OK, so-- no, this is fine.

So if we just look at this, that's our gradient.
So gradient descent here--
how would it go?
We would set theta to be some initial vector.
I don't know.
It could be the vector of all zeros.

Later on, that's not going to be a good initialization choice.
But for this problem, it's OK.
So we use that theta to be the vector of all zeros.
And then we're going to loop.

And we're going to say our new theta is our old theta
minus some step size.

So the gradient-- we're on somewhere in some space.
Maybe it's easiest to think about it
in two dimensions I think.
So the gradient-- it's a bucket.
We're lucky, right?
Convex.
It's going to be like this.
We're standing somewhere.
We compute where we're standing.
From where we're standing, what's
the direction of steepest descent?
We take a small step and re-evaluate.
Note that this depends on the particular theta value
that we have at the moment.
If we want to be careful, if we wanted to actually--
let's do this.
So this is going to be theta 0.

And we can say our theta t plus 1 is theta t.
And this is theta t.
And we'd have to look and see what we have to do.
We have to set t is equal to 0.
And t is equal to t plus 1.
This is just-- we didn't have to write it that way.
This is just a way to decorate what's
going on so that you can see that we're
taking the gradient of the ridge function at a point,
at a particular theta value.
And the particular theta value is the theta t
that we're standing at.
So we're right now we're standing
at this place on the hill, theta t.
We figure out the steepest direction.
We take a step down the hill.
And then that gives us our new theta t plus 1.

Does that seem good?
Does everybody see how you could do that?

OK, now, each iteration-- it's hard to prove.
I mean, you can think about people
can prove the rate at which this converges depending
on something about the shape of the bowl
and what you do with your epsilon and so on.
But you can see that each iter--
so it's not totally clear how many iterations
you're going to have to run.
But you can see that the time it takes to compute the gradient,
it's not so bad.
But it's not so great either, right?
It's going to be--
let's see.
We're going to have to do this multiplication, right?
And W transpose-- let's just look at the dimensions again.
So W transpose is d by n, d by n.
And this thing is n by 1.
So we're still going to have to do
a lot of n size manipulation.
Basically, we're going to consider all the data points
to compute the gradient step.

So if I have a million points in my data set, I will have to--
n will be a million.
And I'll have to do something with vectors of size a million
to compute one gradient step.
So way better than doing something
on the order of a million cube, so that's like nicer.
And hopefully, we won't take that many iterations.
But it's still kind of expensive to even compute this gradient
exactly.

So we're going to look at a cure for that.
STUDENT: So when you have order d cubed
as the functionality of the problem
rather than the number of [INAUDIBLE] you have?
LESLIE KAELBLING: That is the bug in the nodes because--

no, no, no, no, oh, no, you're right.
You're right.
You're right.
STUDENT: So my question is--
LESLIE KAELBLING: I see.
STUDENT: --if you have a low dimensionality problem
but a lot of points--
LESLIE KAELBLING: It might not be so bad.
STUDENT: That would be better.
Whereas if you have high dimensionality, no matter
how many points you have--
LESLIE KAELBLING: Almost.
OK, because unfortunately, you still have to compute.
So if have a lot of points, this is going to be expensive.
STUDENT: But you only have to do it once.
LESLIE KAELBLING: Right, so you're going to be order--
you're going to be linear in n and cubic
in d because you still have to touch all your data to compute
this.
STUDENT: Right, I guess my question
is, if you have a low dimensionality problem,
it may be faster to do this other thing--
LESLIE KAELBLING: It might be.
It might-- oh, I see, good.
So we're going to actually in the end
consider three different methods.
OK, no, I'm with you.
So this is like the analytic method.
And it's going to be order d cubed m
because we're going to have to look at all the data.
This guy is like--
well, one iteration is going to be order dn.
We're not quite sure how many iterations it's going to take.
That's kind of the hard math problem.
But maybe not so many, OK.
So you're right that this might not be desperate
if D is not too big.
We're still, in both cases, have a dependence on n.
And in a minute, we're going to try to relax that.
That's good.
But thank you for disentangling that.
I had missed that.
Yep.
STUDENT: This might not be an acceptable approach
but for sparse matrices, there was a reason that [INAUDIBLE]
Although the exponent of the volume is [INAUDIBLE]..

LESLIE KAELBLING: Right, OK, good.
So the point is an interesting one,
which is if your matrix happens to be sparse,
then you might be able to do the inverse in some nice way.
Generally speaking, this matrix--
I mean, I don't really know conditions under which it might
be sparse because at this point, you've taken your data,
transposed it, multiplied it by itself.
Even if W is sparse to start with, yeah, I'm not sure.
That's an interesting idea, though.
I mean, it is a totally good impulse to ask the question,
are there some structural assumptions
about my problem that could give me a better way
to address this?
And so some kinds of sparsity independence
can really-- could potentially help.
So that's a good idea.
I'm not sure if it works here.
But it's a good idea.


### Lecture: Regression and the ordinary least squares problem


LESLIE KAELBLING: Let's consider a method
that, if we have an enormous data set,
might be a better strategy.
And this is going to pay off for us in the future too.
So let's look at the form of the ridge objective
that we wrote up there.
And let's-- we can look at the gradient.
But let's now write the gradient out as a sum.

And I'll show you why.
So we can write the gradient.

And let's never mind theta 0.

We can put it in, but I'm just going to keep the board simple.

OK.
So there's the gradient.
And it has the form of a sum, right?
The gradient itself has the form of a sum.
And in the case that the gradient-- and this
is a general trick you can do.
In the case that the gradient has the form of a sum,
you might not have to compute the whole gradient every time
you want to take a step.
So now we're going to introduce a new algorithm and a new idea
called stochastic gradient.
And we could have applied this in ridge regression.
We just didn't have time to talk about it last time.
So I'm introducing it here and now.
But it's not specific to regression.

OK, so when the objective has the form
of a sum or an expectation, which is a kind of a sum,
we can sample randomly an element
of the summation, so one term, and do a small step
in that direction.

This is a stochastic gradient descent algorithm, right?
Our gradient here is the sum over a bunch of terms, one
for each point in the data set.
And instead of summing them all and getting
the absolutely perfect definition of the gradient,
which seems like the thing you ought to do, instead,
what we're going to do is we're going to randomly pick an i.
And we're just going to look at the gradient for point i,
and we're going to take a step.
Now, that's not necessarily in the gradient direction.
Generally speaking, it won't be.
The average of those gradients will be the gradient.
And so the idea is kind of like, well, if I sat here
and I computed for all my data points-- by computer, for all
my data points which direction to go and averaged them,
that would tell me totally the right gradient.
And I could take a bigger step in that direction.
Instead of taking a bigger step in what's
totally absolutely surely the right direction,
I'm going to take a small step in a sample,
in a direction that's a sample, that's
a component of the whole true right answer.
And the idea is that we'll get a kind of averaging effect.
But we'll be moving as we go, all right?
Averaging effect, but we'll be moving as we go.
And so the reason to do this is that--
there's two reasons.
One is it's computationally easier
to compute the gradient for just one point
than the whole sum over n.

And furthermore, in many problems,
especially when we have lots and lots of data,
you don't really need the sum over all the n
in order to get a good estimate of which way to go.
So you could start moving sooner.
So we start moving sooner and we don't
have to do quite as much computation per iteration.
So we can write this algorithm down.

I'll write SGD, Stochastic Gradient Descent.

And so, how is it going to go in general?
In general, we'll set theta 0 to be some initial values.

And we're going to loop.

And we're going to randomly select i from 1 to n.
So that means we're going to pluck--
you can think of it as plucking a training example out
of the training data.
And then set theta t to be theta t minus i.
Oh, I did t plus 1 over there.
I'll calculate it this way.
t plus 1, theta t minus eta of t--
I'll talk about this in a minute.
Gradient with respect to theta, f--
I'm going to write it this way.

So this is in the general case, right?
So in the general case, where our f
is our objective function, when it has the form of a sum,
i is 1 to n of f i theta, right?
So regular old gradient descent.
Before, when we looked at it, we had just the f, just the one f.
And now I'm saying, in the special case that your f can
be written like this--

in the special case that your f can
be written like this, as a sum of things,
then you can do this instead.

OK.
So this algorithm looks a lot like the algorithm
that we had before for gradient descent.
But you might ask, why is eta function of t,
and what can we prove about this?
So those things are related.

Is the basic setup OK with everybody,
not talking yet about eta?
AUDIENCE: Yeah.
LESLIE KAELBLING: OK.
It's not a big change.
OK, so let's talk about eta.
So the theorem that we had before
was that if our function was convex
and we had a small enough step size, then
we were guaranteed to get close to the optimum.
Here, the theorem requires that eta get smaller over time.

And you can kind of understand that, right?
Because we need to be averaging right?
We would really like to keep computing the sum,
but we're just taking a step in the direction
of each component at a time.
And if we're taking big steps, then we're
letting f1 tell us what to do, and then
f7, and then f93, and then f2.
We're letting them all tell us what to do
and taking big steps in a bunch of different directions.
If what we really want to be doing
is taking a step in the direction of the sum
so that they're all nicely equally weighted,
we have to be making smaller and smaller steps
so that, in the limit, we are taking steps, in some sense,
in the direction of the sum.
So the rule for convergence--

so the theorem is that it converges almost surely--
so, almost surely, I have to tell you--
it's a small joke about this.
So this is a detailed technical term in probability theory.
It doesn't mean that we're not so sure about this.
But actually, one of my students submitted a paper
to a conference.
And there was a theory about something
happening almost surely.
And the reviewers complained about us not being
careful and precise in our wording.
But it is a careful and precise thing,
which I'm not going to define for you right now.
Anyway, just read it as most of the time, almost always--
converges almost surely to a local optimum.

Yeah, and now there's a condition on etas.
And it's a kind of a funny condition.
I'll write it down.
You don't worry too much about it.
But it's interesting to see.

t squared.

So this says, you keep moving.
Your etas don't just die away completely to nothing, right?
So that if you sum over the infinite lifetime,
it goes to infinity.
But this says they're not too big, right?
So the squares die away.
And a criterion that works--
so a fine idea is eta of t is 1 over t.

Generally, it turns out that this
is kind of slow in practice.
This satisfies the requirements of the theorem,
but it's kind of slow in practice.
We'll talk in a couple weeks about more clever ways
to mess around with the step size.


### Lecture: Regression and the ordinary least squares problem


LESLIE KAELBLING: Let us look at some examples
and then talk a little bit about regularization.
So here's some data.
So you can see blue points or data.
Now, this is a regression problem, right.
We're not trying to divide the pluses from the minuses.
Instead, we have an input, which is x.
So our little d is 1.
We're in a one dimensional problem.
We're trying to predict y as a function of x.
We had those blue training examples, and in this case,
actually, let me do the t of 1.
Hang on.
OK, so we should start at this.
Here's linear, plain old linear regression.
And in fact, I computed it using the analytical solution
wherever it went to.
Here, right, OK.
So those data points, I stacked them into a matrix.
I stacked the outputs into a matrix.
I did that calculation, and it gave me
the parameters of that line.
And that is the line of best fit according to the least squares
criterion to that data.
But we can do the same trick that we
before with changing the order of using
like a polynomial basis.
So here, I did it, but the inputs to the regression
are 1 and x and x squared.
And you can see that it tries the best it can to fit
a quadratic to that data.
I can increase the order.
You guys will play with us in lab.
If I increase the order, it starts to like kind of fit
that.
The data was generated with a sign,
it turns out, a sign plus some noise.
So it's not going to be well fit using exactly a polynomial,
but that's not so bad.

The other thing that we can do is put it in a regularizer.
So now, that was t5, but if I put
in a little bit of regularization,
right, so I set lambda to 0.1, and now it's
kind of flattened it out again.
It said, well, don't do that.
Don't go quite so close to those data points.

This is an example, if I do t8, plain old t8 gets kind of wild,
kind of wild and crazy, right.
So what's going on here?
What's going on here?
What would you call this kind of behavior?
Overfitting, yay, OK, good.
Because look at what prediction it's making for like 0.95,
right?
It thinks it's like down here in the basement
somewhere which it totally doesn't have data for.
It says, that's just nuts, right.
So it's overfitting because you know, it's a polynomial,
it can get close to these points.
It almost can get them exactly, but we're not
so happy about that.
If we take this, though, and we regularize it
so we give it like 0.1, lambda 0.1, then
we're back to something like this.
So if you think that that's too lame, we could give it 0.1.
Let's see what happens.
Even 0.1, lambda 0.1, gets us here.
So it's kind of interesting.
The other thing that we can do which is kind of interesting
is watch gradient descent.
So if we do tgd 5 so this is using a fifth order polynomial
basis--
and actually, let me fix it.
Draw a pause.
Good.

What it's doing here is the gradient descent with a step
size a little bit too big.

But you can see on each iteration,
the fit is getting better.
The error is actually going down.
And now it's like kind of starting to curve itself up.
So if we let that run for a while,
it might get to some kind of solution that we like.
So that is just running this algorithm.


### Lecture: Regression and the ordinary least squares problem


LESLIE KAELBLING: The last thing I want to do
is talk a little bit more about the sources of error
in making a prediction.
If we pick a hypothesis class, so we
get to pick a hypothesis class and an objective function,
and then we do some kind of an optimization,
then we get out a solution.

So we optimize, and that gives us some thetas.

And in the end, we're interested in, what we really care about
is, test error of theta, right.
So we did this actually with H and J and D, right.
We had a data set.
We picked a hypothesis class.
We picked an objective function.
We did some optimization.
For right now, let's assume that the optimizer is awesome.
It's perfect.
Potentially there's a problem with the optimizer
not giving us the answer we want.
For right now, let's assume that the optimizer is working, OK.
It gives us this theta.
We take this theta, we deliver it to our customer.
The customer uses the theta and then actually observes
some performance, let's say, or maybe you
tested on held out data.
And so an interesting question, an important question
to think about is, what can contribute
to theta behaving badly.
So you can you tell me some things
that could contribute to theta behaving badly?

What's a case where you think you'll get bad test error?
Yeah.
Yup.
AUDIENCE: [INAUDIBLE]

LESLIE KAELBLING: So OK, good.
So OK, good.
So I'm going to interpret that to mean something that we
call structural error, which--
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: OK, Good.
That was awesome.
Structural error is when, imagine that your data,
it totally looks like this.
You know, it's like the light readings every day.
Imagine your data looks like that.
Really, you got a lot of data.
It's really nice and it looks just like this.
And you do a linear regression.
Aha, Good, right.
So you know, you picked H, and you said, oh,
I know about linear regression.
I'm going to do linear regression.
And then like no matter how good your algorithm is,
and no matter how much data you have,
you are not going to get good performance, all right.
So a structural error is the error
you get when you can't represent a good hypothesis in H, right.
So that's an example.

OK, then there's like an opposite problem,
an opposite problem which actually is illustrated here.
So an opposite problem, we'll call estimation error.

OK, so the problem here is not that a ninth order polynomial
is not rich enough.
It's not that it can't represent a good hypothesis.
It can, right.
It can represent a pretty good hypothesis on that data,
plus like a bunch of other stuff.
But that we don't have enough data to kind of nail
down or specify theta well.

So imagine, let's go back to this.
Actually, let's erase some of it so
that it looks like our problem.
I don't know, kind of like that.
Imagine that we had this.
So here we have just a few points
drawn from kind of a sine wave, and when we do the ninth order
polynomial, it's like free to try
to go through all the points.
But if I had a ton of data, actually, I
should do this demo next time, and I ran that same fit,
I would get something that looks like this, because that data,
I can perfectly represent this curve using
that class of solutions, that class of hypotheses.
And here, I have enough data to say, darn it,
this is how you should be.

So this is like the essential tension of machine learning.
If you pick a really rich hypothesis class, one
that can represent all kinds of answers,
then you will not have too much structural error
because you can represent the complexity of the problem
that you need, but you risk not having enough data
to nail it down in a good way.
So you risk having estimation error.
On the other hand, fitting a line to that data,
you can do that pretty nicely, pretty reliably,
without too much data.
But it might not be able to give you, even
express an answer of the kind that you want.
So these are important ideas.
Again, they're not tied to regression at all.
This is just a moment where we have some time
to talk about it, and we're now seeing regularization
a couple of times.
And so regularization is a way that we try to kind of navigate
between these things.
Adding regularizers helps us deal with the fact
that we might not be estimating the parameters very well.
They provide some extra constraint
on the parameter estimation, and keeps them
from being goofy when they have kind of the option
to be like that.
So keep this tension in mind as you navigate forward
in all the machine learning stuff we do.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/030eba9b066b079e2b16dc863c18ea39/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Regression.pdf
