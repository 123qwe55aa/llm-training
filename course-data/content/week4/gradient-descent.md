# Gradient Descent

> Week 4: Margin Maximization · MIT 6.036 courseware archive

## Notes – Chapter 6: Gradient Descent

Notes – Chapter 6: Gradient Descent
You can sequence through the Gradient Descent lecture video and note segments (go to Next page). Currently, the lectures videos are segments from a previous semester discussing gradient descent in the context of margin maximization, and so do not match up perfectly with the Fall 2019 content. The revised Fall 2019 lecture video will be posted here one or two days after the live lecture.
You can download the revised (Fall 2019)
Chapter 6: Gradient Descent
notes as a PDF file.

## Lecture: Gradient descent optimization - algorithm in one dimension

Lecture: Gradient descent optimization - algorithm in one dimension
Lecture: Gradient descent optimization - algorithm in one dimension
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Gradient descent - introduction

Gradient descent - introduction
In the previous chapter, we showed how to describe an interesting objective function for machine learning, but we need a way to find the optimal [mathjaxinline]\Theta ^* = {\rm arg}\min _{\Theta } J(\Theta )[/mathjaxinline]. There is an enormous, fascinating, literature on the mathematical and algorithmic
Which you should consider studying some day!
foundations of optimization
, but for this class, we will consider one of the simplest methods, called
gradient descent.
Intuitively, in one or two dimensions, we can easily think of [mathjaxinline]J(\Theta )[/mathjaxinline] as defining a surface over [mathjaxinline]\Theta[/mathjaxinline]; that same idea extends to higher dimensions. Now, our objective is to find the [mathjaxinline]\Theta[/mathjaxinline] value at the lowest point on that surface. One way to think about gradient descent is that you start at some arbitrary point on the surface, look to see in which direction the “hill" goes down most steeply, take a small step in that direction, determine the direction of steepest descent from where you are, take another small step,
Here's a very old-school humorous description of gradient descent and other optimization algorithms using analogies involving kangaroos:
ftp://ftp.sas.com/ pub/neural/kangaroos.txt
etc.
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:08 PM (revision f808f068e)

## Lecture: Gradient descent optimization - local optima

Lecture: Gradient descent optimization - local optima
Lecture: Gradient descent optimization - local optima
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## One dimension

One dimension
We will start by considering gradient descent in one dimension. Assume [mathjaxinline]\Theta \in \mathbb {R}[/mathjaxinline], and that we know both [mathjaxinline]J(\Theta )[/mathjaxinline] and its first derivative with respect to [mathjaxinline]\Theta[/mathjaxinline], [mathjaxinline]J'(\Theta )[/mathjaxinline]. Here is pseudo-code for gradient descent on an arbitrary function [mathjaxinline]f[/mathjaxinline]. Along with [mathjaxinline]f[/mathjaxinline] and [mathjaxinline]f'[/mathjaxinline], we have to specify the initial value for parameter [mathjaxinline]\Theta[/mathjaxinline], a
step-size
parameter [mathjaxinline]\eta[/mathjaxinline], and an
accuracy
parameter [mathjaxinline]\epsilon[/mathjaxinline]:
Note that there are many other reasonable ways to decide to terminate, including when [mathjaxinline]\left| \Theta ^{(t)} - \Theta ^{(t-1)} \right| <\epsilon[/mathjaxinline] or when [mathjaxinline]\left|f(\Theta ^{(t)}) - f(\Theta ^{(t-1)}) \right| <\epsilon[/mathjaxinline].
Theorem:
If [mathjaxinline]J[/mathjaxinline] is
A function is convex if the line segment between any two points on the graph of the function lies above or on the graph.
convex,
for any desired accuracy [mathjaxinline]\epsilon[/mathjaxinline], there is some step size [mathjaxinline]\eta[/mathjaxinline] such that gradient descent will converge to within [mathjaxinline]\epsilon[/mathjaxinline] of the optimal [mathjaxinline]\Theta[/mathjaxinline].
However, we must be careful when choosing the step size to prevent slow convergence, oscillation around the minimum, or divergence.
The following plot illustrates a convex function [mathjaxinline]f(x) = (x - 2)^2[/mathjaxinline], starting gradient descent at [mathjaxinline]\theta _{init} = 4.0[/mathjaxinline] with a step-size of [mathjaxinline]1/2[/mathjaxinline]. It is very well-behaved!
Study Question:
What happens in this example with very small [mathjaxinline]\eta[/mathjaxinline]? With very big [mathjaxinline]\eta[/mathjaxinline]?
If [mathjaxinline]J[/mathjaxinline] is non-convex, where gradient descent converges to depends on [mathjaxinline]\theta _{\it init}[/mathjaxinline]. When it reaches a value of [mathjaxinline]\theta[/mathjaxinline] where [mathjaxinline]f'(\theta ) = 0[/mathjaxinline] and [mathjaxinline]f"(\theta ) > 0[/mathjaxinline], but it is not a minimum of the function, it is called a
local minimum
or
local optimum
.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:36 PM (revision 4f166135)

## Lecture: Gradient descent optimization - algorithm in multiple dimensions

Lecture: Gradient descent optimization - algorithm in multiple dimensions
Lecture: Gradient descent optimization - algorithm in multiple dimensions
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Multiple dimensions

Multiple dimensions
The extension to the case of multi-dimensional [mathjaxinline]\Theta[/mathjaxinline] is straightforward. Let's assume [mathjaxinline]\Theta \in \mathbb {R}^ m[/mathjaxinline], so [mathjaxinline]J: \mathbb {R}^ m \rightarrow \mathbb {R}[/mathjaxinline]. The gradient of [mathjaxinline]J[/mathjaxinline] with respect to [mathjaxinline]\Theta[/mathjaxinline] is
[mathjax]\nabla _\Theta J = \begin{bmatrix}  \partial J / \partial \Theta _1 \\ \vdots \\ \partial J / \partial \Theta _ m \end{bmatrix}[/mathjax]
The algorithm remains the same, except that the update step in line 5 becomes
[mathjax]\Theta ^{(t)} = \Theta ^{(t-1)} - \eta \nabla _\Theta J[/mathjax]
and we have to change the termination criterion. The easiest thing is to replace the test in line 6 with [mathjaxinline]\left|f(\Theta ^{(t)}) - f(\Theta ^{(t-1)}) \right| < \epsilon[/mathjaxinline], which is sensible no matter the dimensionality of [mathjaxinline]\Theta[/mathjaxinline].
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:36 PM (revision 4f166135)

## Lecture: Gradient descent optimization - parameters and demo

Lecture: Gradient descent optimization - parameters and demo
Lecture: Gradient descent optimization - parameters and demo
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Application to logistic regression objective

Application to logistic regression objective
We can now solve the optimization problem for our linear logistic classifier as formulated in chapter
. We begin by stating the objective and the gradient necessary for doing gradient descent. In our problem where we are considering linear separators, the entire parameter vector is described by parameter vector [mathjaxinline]\theta[/mathjaxinline] and scalar [mathjaxinline]\theta _0[/mathjaxinline] and so we will have to adjust them both and compute gradients of [mathjaxinline]J[/mathjaxinline] with respect to each of them. The objective and gradient (note that we have replaced the constant [mathjaxinline]\lambda[/mathjaxinline] with [mathjaxinline]\frac{\lambda }{2}[/mathjaxinline] for convenience), are,
The following step requires passing familiarity with matrix derivatives. A foolproof way of computing them is to compute partial derivative of [mathjaxinline]J[/mathjaxinline] with respect to each component [mathjaxinline]\theta _ i[/mathjaxinline] of [mathjaxinline]\theta[/mathjaxinline].
note
letting [mathjaxinline]g^{(i)} = \sigma (\theta ^ Tx^{(i)}+\theta _0)[/mathjaxinline]
[mathjaxinline]\displaystyle  J_\text {lr}(\theta ,\theta _0)[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{1}{n}\sum _{i=1}^ n \mathcal{L}_\text {nll} (g^{(i)}, y^{(i)}) + \frac{\lambda }{2}\left\lVert \theta \right\rVert ^2[/mathjaxinline]
[mathjaxinline]\displaystyle \nabla _\theta J_\text {lr}(\theta , \theta _0)[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{1}{n}\sum _{i=1}^ n \left(g^{(i)} - y^{(i)}\right) x^{(i)} + \lambda \theta[/mathjaxinline]
[mathjaxinline]\displaystyle \frac{\partial J_\text {lr}(\theta , \theta _0)}{\partial \theta _0}[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{1}{n}\sum _{i=1}^ n \left(g^{(i)} - y^{(i)} \right) \; \; .[/mathjaxinline]
Note that [mathjaxinline]\nabla _\theta J[/mathjaxinline] will be of shape [mathjaxinline]d \times 1[/mathjaxinline] and [mathjaxinline]\frac{\partial J}{\partial \theta _0}[/mathjaxinline] will be a scalar since we have separated [mathjaxinline]\theta _0[/mathjaxinline] from [mathjaxinline]\theta[/mathjaxinline] here.
Study Question:
Convince yourself that the dimensions of all these quantities are correct, under the assumption that [mathjaxinline]\theta[/mathjaxinline] is [mathjaxinline]d \times 1[/mathjaxinline]. How does [mathjaxinline]d[/mathjaxinline] relate to [mathjaxinline]m[/mathjaxinline] as discussed for [mathjaxinline]\Theta[/mathjaxinline] in the previous section?
Study Question:
Compute [mathjaxinline]\nabla _\theta \left\lVert \theta \right\rVert ^2[/mathjaxinline] by finding the vector of partial derivatives [mathjaxinline](\partial \left\lVert \theta \right\rVert ^2 / \partial \theta _1, \ldots , \partial \left\lVert \theta \right\rVert ^2 / \partial \theta _ d)[/mathjaxinline]. What is the shape of [mathjaxinline]\nabla _\theta \left\lVert \theta \right\rVert ^2[/mathjaxinline]?
Study Question:
Compute [mathjaxinline]\nabla _\theta \mathcal{L}_\text {nll}( \sigma (\theta ^ T x + \theta _0), y)[/mathjaxinline] by finding the vector of partial derivatives [mathjaxinline](\partial \mathcal{L}_\text {nll}( \sigma (\theta ^ T x + \theta _0), y)/ \partial \theta _1, \ldots , \partial \mathcal{L}_\text {nll}( \sigma (\theta ^ T x + \theta _0), y) / \partial \theta _ d)[/mathjaxinline].
Study Question:
Use these last two results to verify our derivation above.
Putting everything together, our gradient descent algorithm for logistic regression becomes
Study Question:
Is it okay that [mathjaxinline]\lambda[/mathjaxinline] doesn't appear in line 7?
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:08 PM (revision f808f068e)

## Stochastic Gradient Descent

Stochastic Gradient Descent
When the form of the gradient is a sum, rather than take one big(ish) step in the direction of the gradient, we can, instead,
The word “stochastic" means probabilistic, or random; so does “aleatoric," which is a very cool word. Look up aleatoric music, sometime.
randomly
select one term of the sum, and take a very small step in that direction. This seems sort of crazy, but remember that all the little steps would average out to the same direction as the big step if you were to stay in one place. Of course, you're not staying in that place, so you move, in expectation, in the direction of the gradient.
Most objective functions in machine learning can end up being written as a sum over data points, in which case, stochastic gradient descent (
sgd
) is implemented by picking a data point randomly out of the data set, computing the gradient as if there were only that one point in the data set, and taking a small step in the negative direction.
Let's assume our objective has the form
[mathjax]f(\Theta ) = \sum _{i = 1}^ n f_ i(\Theta )\; \; .[/mathjax]
Here is pseudocode for applying
sgd
to an objective [mathjaxinline]f[/mathjaxinline]; it assumes we know the form of [mathjaxinline]\nabla _\Theta f_ i[/mathjaxinline] for all [mathjaxinline]i[/mathjaxinline] in [mathjaxinline]1\ldots n[/mathjaxinline]:
Note that now instead of a fixed value of [mathjaxinline]\eta[/mathjaxinline], it is indexed by the iteration of the algorithm, [mathjaxinline]t[/mathjaxinline]. For
sgd
to converge to a local optimum as [mathjaxinline]t[/mathjaxinline] increases, the step size has to decrease as a function of time.
Theorem:
If [mathjaxinline]J[/mathjaxinline] is convex, and [mathjaxinline]\eta (t)[/mathjaxinline] is a sequence satisfying
[mathjax]\sum _{t = 1}^{\infty }\eta (t) = \infty \; \; \text {and}\; \;  \sum _{t = 1}^{\infty }\eta (t)^2 < \infty \; \; ,[/mathjax]
then SGD converges
We have left out some gnarly conditions in this theorem, and note that
almost surely
is a technical term from probability theory, not a lack of courage on the part of the author.
almost surely
to the optimal [mathjaxinline]\Theta[/mathjaxinline].
One “legal" way of setting the step size is to make [mathjaxinline]\eta (t) = 1/t[/mathjaxinline] but people often use rules that decrease more slowly, and so don't strictly satisfy the criteria for convergence.
Study Question:
If you start a long way from the optimum, would making [mathjaxinline]\eta (t)[/mathjaxinline] decrease more slowly tend to make you move more quickly or more slowly to the optimum?
There are two intuitions for why
sgd
might be a better choice algorithmically than regular
gd
(which is sometimes called
batch
gd
(
bgd
)):
If your [mathjaxinline]f[/mathjaxinline] is actually non-convex, but has many shallow local optima that might trap
bgd
, then taking
samples
from the gradient at some point [mathjaxinline]\Theta[/mathjaxinline] might “bounce" you around the landscape and out of the local optima.
Sometimes, optimizing [mathjaxinline]f[/mathjaxinline] really well is not what we want to do, because it might overfit the training set; so, in fact, although
sgd
might not get lower training error than
bgd
, it might result in lower test error.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:36 PM (revision 4f166135)

## Video transcripts

### Lecture: Gradient descent optimization - algorithm in one dimension


PROFESSOR: So, we have an objective function,
and this is the point at which we could say, oh, yeah,
we are machine learning people, and it's our job to formulate
the objective function.
And then we find our friends who do
optimization and say, here, please optimize this for us.
It's not that clean in machine learning because,
actually quite frequently, and already, we
try to make the optimization problems kind of nice
for our friends, the optimizers.
So, 0-1 loss, it turns out, is a real pain for optimization.
It's discrete, and if you want optimize 0-1 loss,
you often don't have too many choices,
except for to like consider all the hypotheses or something
horrible like that.
So it's really combinatorial and kind of hard to optimize.
Here, we've made something that's a little smoother,
and that's going to let us use a set of optimization techniques
that work better.
So, it turns out that you can't be a responsible machine
learning person without knowing something about optimization.
So, we're going to learn something about optimization,
so that we can think about how to optimize this objective.

I'm going to keep this board here and do some more SVM
stuff there in a minute.
Let's talk about gradient descent.
So, if you read an optimization textbook-- actually,
this algorithm we're going to--
I'm going to teach you about optimization is like the way
that everything, approximate-- actually it's
a thing that we do almost all the time,
it's the way neural networks get optimized.
In machine learning, it's a very popular algorithm.
In optimization-land it's like ridiculously naive,
but it works well on very large data sets,
it's easy to think about and implement.
So, it's what we're going to study.
So, something called gradient descent.

And the idea--

I'm just going to-- what I'm going
to do is write down an algorithm and then
we'll see what it does.
Gradient descent.
And, for right now, we're going to do it in one dimension,
but then we'll do it in more dimensions.

So, the idea is we have a bunch of parameters of prime data
in it eta.
That's eta.
That's epsilon.
Have to learn Greek.
So, our goal is, given some function f--

so we have a function of one parameter.
So f of some parameter theta.
So here's theta.
Here's f.

We want to find theta star, which is the arg
min over thetas of f of theta.
This is what we wish we could do.
We want to find-- we said, originally,
that was our problem, we would like to find the theta that
minimizes our function f.
Now, what we find is that this algorithm won't always
solve our problem.
But sometimes it will, often enough that it's useful.
So, we have some function f, and we
assume that we know the form of the function f.
Like we have f in our hands as a computer program,
we can put in theta's get out values.

So that's f.
So, we have some function f.
F prime is the derivative.
So we know the derivative of f, with respect to theta.
Now, it turns out that you don't actually
have to have a computer program for computing the derivative,
there'll be a homework exercise about that.
You can estimate-- in calculus you learned
how to estimate the derivative by taking
a little step on each side.
Remember that?
Estimating the slope, all that.
Good. , So you can estimate f prime,
but we'll just imagine that you know f prime for right now.
So you were able to take the derivative of your f.
There's now software that can take
the derivatives of really really, really
interesting and complicated functions,
like whole computer programs.
Pretty cool.
OK, so you have f and f prime, and you have an initial theta.
So, you might say, I'm going to start here.
This is theta in it.

This is f-- I don't feel like drawing f prime.
And then you have a couple parameters, which you'll
see when I write the code down.
So, here's what you're going to do instead.
It's an iterative algorithm, we start out
we're going to let theta be equal to theta in it.

And we don't need t, who needs t.
And then we're just going to loop--

let theta get theta, minus eta.
Theta minus eta.

OK.
It's a pretty simple algorithm, not too many lines of code.
Let's think what it does.

So it says we start out with some theta value here.
We'll talk about how to pick it and when and why it matters.
We start out with some theta value,
and we compute the derivative at that point.
So, it's easy to think about here, there's a derivative.

And the gradient, right as it goes in the uphill direction,
so the gradient goes in the uphill direction.
This eta-- OK, so this eta, eta is something called step size.

It is positive.
If it were 0, nothing would happen.
If it were negative, you'd go in the wrong direction.
And generally small.
Then you ask me what does small mean,
and I'll talk about that in a minute,
too, but just think small for right now just means small.
So little small step take a small step
take a small step away and in the downhill direction.
So, the gradient is going to be pointing this way,
so I'll take a small step here.

I'll add to my original theta, while I
subtract from my theta a small step
in the direction of the gradient.
I'll figure out the gradient here, go this way,
go like this, go like this, come down here, keep going.
It might be that I come over here by mistake,
a little bit too far.

I'll do some stuff in this problem,
we'll go back and forth and we might bounce around
a little bit but we'll get near the optimal theta.
So, really, we're just going down the hill in small steps.
That's it.
Downhill, small steps.
What
Are the kinds of trouble you can get into?
Well, what if my step size was really big,
like what if my step size was, I don't know--
the units depend on the units of the thing
that you're looking at.
But if my step size is much too big, I might be standing here
and have the right idea about what direction
to go but too much too far.
If I go a little bit too far, I might go over to here.
And then here, the gradient is going
to point in this direction, but maybe
be smaller magnitude than that.
So then maybe I might go over here.
So, sometimes, you can get in a situation
where you like, hopscotch, but you still arrive roughly
where you want to go.
Sometimes, if it's a really too big, the step I take,
I step all the way over here.
And then my next step goes over here,
and then you're usually off to the races.
So then your next step is over there and the next one is NaN.
If you haven't seen NaN, if you've
done num-pi for a little while, you've probably seen NaN,
but it's a great perturbation to some people, the first time.
Like you multiply two numbers and the answer
comes out to be this thing called a NaN.
N A N stands for "not a number."
it means, usually, positive or negative infinity.
Something we can't work with anymore.
So, if you pick a step size too big,
NaN will be the answer that you get, or the program will


### Lecture: Gradient descent optimization - algorithm in one dimension


LESLIE KAELBLING: Let's look at one more situation
where we can have a problem.
So let me draw a different f.

So this is my f.
And theta star is right there.
Probably.
I mean, if it goes up after this,
this is going to be the best theta value.
That's the theta value that minimizes f.
But if I start here, if my theta [INAUDIBLE]
is here, very likely, and my step size is small enough,
I'll probably wind up there.
Or if I start here, I'll wind up here.
And it's only if I start here or over
that I'm going to end up in the right place.
So with gradient descent, we cannot promise to find a global
optimum in general.
We can promise to find--
so what can we find?

We can say that we will find a local optimum
under some conditions.
In particular, for any local optimum--

this is kind of a kicker.
For any local optimum, there exists a value of eta
such that you will arrive at a theta within--
so for any local optimum, let's call it theta--
I don't want to call it theta star,
we'll call it theta hat because it might not be the best one--
theta within epsilon of theta hat.

So imagine that your function looks like this.

The optimum is down in a little valley.
Or the valley might be so little that it's
like this and the rest of your function looks like this.
You have to have a little tiny step size to get down
in that little crevice.
And lots of machine learning problems,
their objective function looks like this like.
You can find the optimum, sometimes,
by calculation or something.
Or if you can plot the function, you
see that gradient methods don't necessarily get there.
So this is a great method.
We use it all the time.
But it's really important to be aware of the kinds of problems
that you can have.
So problems you can have are you arrive
at a local optimum but not the global optimum you wanted.
Problem you can have is your step size is too big
and it goes off into infinity.
Another problem is your step size is too big
and it doesn't go off into infinity
but it never quite manages to step down this little hole.

So it's tricky.
Later on we'll talk about some strategies
that adapt to step size.
So they kind of like start out with a bigger step size.
And then they make the step size smaller over time.
And sometimes that makes things better.

This is why you can't not know about the optimization method
as a machine learning person.
So a lot of people are just like, oh, man, just teach me
TensorFlow and then I'll do it.
And why do I have to hear all this stuff about gradients
and optimization and so on?
And the answer is because lots of times it won't work.
And you have to know something about what's
going on in order to have a hope of understanding why it's not
working.
So that's why we're forcing you to learn all this stuff.
[INAUDIBLE]
But what's nice is that it's a totally generic algorithm.
It's not about machine learning.
It's not about this loss function.
It just says, you give me an f and an f prime
and I'll do the thing for you.
And so that's very nice.
And so that means, also, if there is better optimization
methods, as long as they have the same calling conventions,
you can use them.
Although it's kind of on you to know about their weaknesses.
So two more things to do today.
So one is to talk about, before we get back to here,
is to talk about what happens when theta is something
other than one dimensional.


### Lecture: Gradient descent optimization - algorithm in one dimension


LESLIE KAELBLING: So now imagine theta is a column vector--
d by 1 column vector, say.
It might not always be the dimension of your data, right?
Because maybe a more complicated hypotheses or something.
But let's just say it's d by 1.

So that's a vector.
There's a vector.
And so now, in a vector space, we're going to--
or, the thing that we subtract from theta
can't be just a number, right?
It has to be a whole vector.
So it's easy to think about in--
easier to think about in two dimensions, right?
In two dimensions, here's my theta.
I have a two-dimensional theta space.
My J is some function over here.
And now, when I take a step, I'm going to use two--
a two-dimensional vector is going to tell me
what direction to go in, right?
So here, the derivative is kind of boring, right?
It just-- it tells us which direction to go in and how far
so that we're sensitive to the magnitude of the derivative.
In higher dimensions, the gradient gives us a direction.
Remember, I learned to ski-- not very well,
but I took skiing lessons kind of, like, as an adult.
And the ski instructor was on and on
and on about this thing called the fall line, which
I couldn't understand.
But you were supposed to keep your skis across the fall line.
That was the rule.
But what I realized was that what he meant
was that you should be orthogonal to the gradient.
But he didn't say it that way.
I don't know why.
Right.
So the fall line is the gradient.
It's the direction of steepest descent.
Sometimes this gets called the method of steepest descent.
So we're just going to go down the direction where
it's steepest, OK?
As a skier, you don't necessarily want to do that.
But it's good to know what it is, in any case.
OK.
So in multiple dimensions, what we're
going to write here is eta--
still eta scalar.

OK.
The gradient of J with respect to theta.
OK.
So this is where, you know, 1802--
for some of you, it's been longer than for others.
Let's just remember kind of what's going on here
and not panic, right?
J is a scalar.
Theta is a vector.

This gradient here is just a vector of partial derivatives
dj d theta 1 through dj d theta d.

That's all.
And if-- we'll do some things where we take
shortcuts, because we understand something about derivatives
of gradients and stuff.
But you can always fall back on this fact
and compute the elements of this vector if you have to.
OK.
So basically, all we have to do is replace the line
up there where I said theta equals theta minus 8 f prime--
f prime?
Yeah, f prime?
So we do this.
Oh, I switched-- that's OK.
You guys-- you're robots to this, right?

All right.
Is that good?
Does that make sense?
The dimensions work out.
Everything is cool.

All right.
So now, let's just--
yeah.
Let's just figure out what J prime is here.
That's going to pose one more small piece of trouble, one
more--
very small piece trouble, but it's a piece of trouble.
OK.

And now, remember, this is, again, where I suffer from my--
some pun that we made earlier on, which is we
have theta and theta 0.
So our vector theta there, our whole parameter vector,
is actually made up of these two guys.
So we're going to have to compute the gradient of J with
respect to our theta vector-- this one--
and the gradient of J with respect to this guy.
And that's going to be our whole big gradient vector.
OK.
So what is the gradient of J with respect to theta?
Let's just-- I'm going to do this in shortcut version
on the board.
You can verify it by components if you want to.
OK.
But let's just remember some rules of calculus,
like the derivative of the sum.
We multiply by a constant that we don't
have to do anything about that.
Derivative of the sum is sum of derivatives.
That's totally cool.
OK.
Now we have a function.
Not sure what to do about that, but we remember the chain rule.
We take the derivative of the guts of the stuff
that we're passing in.
So the derivative of that stuff with respect
to theta transpose--
what is that going to be?
Well, it's going to be yi xi.

And then we're going to have some L prime of yi times
theta transpose xi plus theta 0.
So we don't know what that is yet.
We'll come back to that.
And the derivative of this, I'm just
going to assert for right now is lambda theta.
Yes.

OK.

That was a 802 homework, so you can go work it out.

And so that was fine, and we didn't
break any rules of calculus.
Everything's cool.
And the dimensions are OK, right?
Always check the dimensions, right?
So this thing needs to be a column vector.
So let's just be sure that we still have a column vector.

This is a scalar.
This is a column vector.

This is a loss function, right?
So the loss function is just a scalar, and it's a scalar--
so this is going to be the derivative
of the loss with respect to the stuff inside.
So this is just going to be a scalar.
And this is a vector.
This is a column vector.
So we have a bunch column vectors
times scales plus column vector, so that's all column vector,
so we're good.
OK.
So now I just want to talk about this.
OK.
Remember what L prime is.
It looks like this.

So this is where you might be a little bit nervous.
So if this is L, what's L prime?
So what's the derivative of that function?
It's OK.
So if you're, like, a relaxed person, you say, oh, it's OK.
It's minus 1 here, right?

And then it's 0.
So it's minus 1 up to there.
Everything's good.
Let's just ignore that point for a minute.
It's minus 1.
All good.
Minus 1.
And then ignoring that point for just a minute, it's 0, right?
Minus 1, 0.
There's a problem with that point, right?
The derivative doesn't really exist.

We're just going to pretend it's 0.
It'll be OK.
There's a story about subgradients which you can
read about, if it worries you.
We're OK, but I don't want to get into it right now.
So we're just going to pretend it's 0 there.
You can pretend it's 1.
That's OK, too.
It's just that this little bendy point, just assume that you're
on one side or the other.
But know that, actually, it can manifest
as some kinds of funniness when we're doing optimization.
But mostly, we'll just pretend it's OK.

Sometimes people actually change this loss function
so that it's smooth right there.
And then, ah, the calculus is happy.
And actually, sometimes, the optimization
is a little bit better.
So it's a thing to kind of be aware of.
All right.
So here, we know J. We know the gradient of J
with respect to theta.
I'm just going to write down one more thing, which is--
because we have one more parameter, right?
So we have to know dj d theta 0.
And just for completeness, I'll write that down.

Notice that theta 0 doesn't appear in the regularizer,
right?
So this is just--

ah.

Good.
OK.
So if we stack up this gradient with this one more value,
then we have that whole gradient that we need,
and we can do gradient descent.
And if we picked the right parameters,


### Lecture: Gradient descent optimization - algorithm in one dimension

LESLIE KAELBLING: OK, let's actually see this in action.
OK.
So here is a problem.

I'm doing this-- so this data is separable
and I'm doing it with a very small lambda.
There are thought questions in the notes
and also actual homework questions in next week's
homework that are really about understanding what
happens with lambda values.
And you should study and learn and understand that stuff.
OK.
So the data is separable and I'm using a very small lambda.
Because the data is separable, that
means I will be able to find somebody who
doesn't misclassify the data.
And having a tiny lambda means I would like you to maximize
the margin as much as you can.
But don't compromise your ability to separate the points.
So that's kind of what's going on.
So here we're going to show the steps of gradient descent.
So I'm doing gradient descent.
The separator you can see just peeking out here in the corner.
And then these parameters are being adjusted now.
This isn't the first upfront algorithm.
It's that algorithm running on this objective function.
And so it kind of wiggles its way over.
And it's going to adjust a little bit more.
But it arrives basically at the max margin separator.

The twitchiness is interesting.
OK.
If I try a more difficult problem--
so let me do this one.

I'm not going to make myself hit Carriage Return
and I'm going to resize the window
while it's going if it lets me.
So it's going right now.
It's just-- so these points, the magnitudes are bigger.
And I had to make the step size smaller for it to work nicely.
So it's going along.
Let me do it without drawing and we'll see the convergence plot.
Oh, stop.

OK.
So there, it ran and converged.
And what I'm plotting here, just so you
can get used to the idea, is this is the objective function.
So this is the step for the iteration
through the algorithm.
And this is the objective function.
And we can see the objective function comes down very
steeply, and then it's all kind of wiggly for a while,
and then it smooths out.
I would be able-- if I had picked a smaller step size,
it would not be jaggy.
It can go right there, I think.
But that is very slow.
So there's another game you often
play which is like, oh, if I really want to be sure
that the guy is well-behaved, and it's going nice
and in an orderly way.
It's the optimum.
A step size that will do that for you
is often just too painfully slow.
So sometimes you live with a certain amount
of jagginess and weirdness in order
to trade off against the steps taking too long to converge.
So step size is a parameter of the algorithm.
It's interesting.
Now we have three kinds of parameters.
Let me just-- I have two minutes so I can talk about three kinds
of parameters just so that we--
you kind of have to keep them straight
and know what to do about them.
So our thetas are parameters of the answer.
The thetas are the answer that we're giving out.

The lambda is a hyperparameter.
It's a parameter that describes the trade-off we're making.
It describes something about what we're asking
the optimization algorithm for.
We're saying, oh, optimization algorithm,
please give me the optimum of that function.
But then kind of unfortunately, the optimization algorithm
is not miraculous, and it has a parameter, which is the eta.

Generally, eta is not a thing--
it is a thing you could try to do
cross-validation to pick eta, but normally you
wouldn't do that.
Normally, the eta is something that
affects the ability to optimize just on the set that you have,
right?
OK.
So, good.
So eta affects about the ability to solve your optimization
problem.
Lambda affects whether the optimization problem
you're solving is going to give you good performance on data
that you haven't seen.
And theta is actually the thing that
describes the hypothesis you give out to the world.
So I recognize that that's pretty confusing, too.
Three kinds of parameters that matter
at three different stages in some sense of what's going on.
So that's another thing to go back and ponder about.
Like, what different roles do eta, lambda, and theta play?


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/d81d9ec0bd142738b069ce601382fdb7/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Gradient_Descent.pdf
