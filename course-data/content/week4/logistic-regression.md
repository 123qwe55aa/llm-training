# Logistic Regression

> Week 4: Margin Maximization · MIT 6.036 courseware archive

## Notes – Chapter 5: Logistic Regression

Notes – Chapter 5: Logistic Regression
You can sequence through the Logistic Regression lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 5: Logistic Regression
notes as a PDF file.

## Lecture: Logistic Regression

Lecture: Logistic Regression
Lecture: Machine learning as optimization - framework
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Logistic regression - setting and sigmoid function

Lecture: Logistic regression - setting and sigmoid function
Lecture: Logistic regression - setting and sigmoid function
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Machine learning as optimization

Machine learning as optimization
The perceptron algorithm was originally written down directly via cleverness and intuition, and later analyzed theoretically. Another approach to designing machine learning algorithms is to frame them as optimization problems, and then use standard optimization algorithms and implementations to actually find the hypothesis. Taking this approach will allow us to take advantage of a wealth of mathematical and algorithmic technique for understanding and solving optimization problems, which will allow us to move to hypothesis classes that are substantially more complex than linear separators.
We begin by writing down an
objective function
[mathjaxinline]J(\Theta )[/mathjaxinline], where [mathjaxinline]\Theta[/mathjaxinline] stands for
all
the parameters in our model. Note that we will sometimes write [mathjaxinline]J(\theta , \theta _0)[/mathjaxinline] because when studying linear classifiers, we have used these two names for parts of our whole collection of parameters, so [mathjaxinline]\Theta = (\theta , \theta _0)[/mathjaxinline]. We also often write [mathjaxinline]J(\Theta ; {\cal D})[/mathjaxinline] to make clear the dependence on the data [mathjaxinline]{\cal D}[/mathjaxinline]. The objective function describes how we feel about possible hypotheses [mathjaxinline]\Theta[/mathjaxinline]: we will generally look for values for parameters [mathjaxinline]\Theta[/mathjaxinline] that minimize
You can think about [mathjaxinline]\Theta ^*[/mathjaxinline] here as “the theta that minimizes [mathjaxinline]J[/mathjaxinline]".
the objective function:
[mathjax]\Theta ^* = {\rm arg}\min _{\Theta } J(\Theta )\; \; .[/mathjax]
A very common form for an ML objective is
[mathjax]J(\Theta ) = \left(\frac{1}{n} \sum _{i=1}^ n \underbrace{\mathcal{L}(h(x^{(i)}; \Theta ), y^{(i)})}_\text {loss}\right) + \underbrace{\lambda } _\text {constant} \underbrace{R(\Theta )}_\text {regularizer}.[/mathjax]
(1.1)
The
loss
tells us how unhappy we are about the prediction [mathjaxinline]h(x^{(i)}; \Theta )[/mathjaxinline] that [mathjaxinline]\Theta[/mathjaxinline] makes for [mathjaxinline](x^{(i)}, y^{(i)})[/mathjaxinline]. A common example is the 0-1 loss, introduced in chapter
:
[mathjax]L_{01}(h(x; \Theta ), y) = \begin{cases}  0 &  \text { if } y = h(x; \Theta )\\ 1 &  \text { otherwise} \end{cases}\; \; ,[/mathjax]
which gives a value of 0 for a correct prediction, and a 1 for an incorrect prediction. In the case of linear separators, this becomes:
[mathjax]L_{01}(h(x;\theta , \theta _0), y) = \begin{cases}  0 &  \text { if } y(\theta ^ Tx + \theta _0) > 0 \\ 1 &  \text { otherwise} \end{cases}\; \; .[/mathjax]
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:08 PM (revision f808f068e)

## Lecture: Linear logistic classifier - hypothesis class

Lecture: Linear logistic classifier - hypothesis class
Lecture: Linear logistic classifier - hypothesis class
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Regularization

Regularization
If all we cared about was finding a hypothesis with small loss on the training data, we would have no need for regularization, and could simply omit the second term in the objective. But remember that our ultimate goal is to
perform well on input values that we haven't trained on!
It may seem that this is an impossible task, but humans and machine-learning methods do this successfully all the time. What allows
generalization
to new input values is a belief that there is an underlying regularity that governs both the training and testing data. We have already discussed one way to describe an assumption about such a regularity, which is by choosing a limited class of possible hypotheses. Another way to do this is to provide smoother guidance, saying that, within a hypothesis class, we prefer some hypotheses to others. The regularizer articulates this preference and the constant [mathjaxinline]\lambda[/mathjaxinline] says how much we are willing to trade off loss on the training data versus preference over hypotheses.
This trade-off is illustrated in the figure below. Hypothesis [mathjaxinline]h_1[/mathjaxinline] has 0 training loss, but is very complicated. Hypothesis [mathjaxinline]h_2[/mathjaxinline] mis-classifies two points, but is very simple. In absence of other beliefs about the solution, it is often better to prefer that the solution be “simpler," and so we might prefer [mathjaxinline]h_2[/mathjaxinline] over [mathjaxinline]h_1[/mathjaxinline], expecting it to perform better on future examples drawn from
To establish some vocabulary, we say that [mathjaxinline]h_1[/mathjaxinline] is
overfit
to the training data.
this same distribution.
Another nice way of thinking about regularization is that we would like to prevent our hypothesis from being too dependent on the particular training data that we were given: we would like for it to be the case that if the training data were changed slightly, the hypothesis would not change by much.
A common strategy for specifying a
regularizer
is to use the form
[mathjax]R(\Theta ) = \left\lVert \Theta - \Theta _{\it prior}\right\rVert ^2[/mathjax]
when we have some idea in advance that [mathjaxinline]\theta[/mathjaxinline] ought to be near some value [mathjaxinline]\Theta _{\it prior}[/mathjaxinline].
Learn about Bayesian methods in machine learning to see the theory behind this and cool results!
note
In the absence of such knowledge a default is to
regularize toward zero
:
[mathjax]R(\Theta ) = \left\lVert \Theta \right\rVert ^2\; \; .[/mathjax]
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:08 PM (revision f808f068e)

## A new hypothesis class: linear logistic classifiers

A new hypothesis class: linear logistic classifiers
For classification, it is natural to make predictions in [mathjaxinline]\{ +1, -1\}[/mathjaxinline] and use the [mathjaxinline]0-1[/mathjaxinline] loss function. However, even for simple linear classifiers, it is very difficult to find values for [mathjaxinline]\theta , \theta _0[/mathjaxinline] that minimize simple training error
[mathjax]J(\theta , \theta _0) = \frac{1}{n} \sum _{i=1}^ n \mathcal{L}(\text {sign}(\theta ^ Tx^{(i)} + \theta _0), y^{(i)})\; \; .[/mathjax]
This problem is NP-hard,
The “probably" here is not because we're too lazy to look it up, but actually because of a fundamental unsolved problem in computer-science theory, known as “P vs NP."
which probably
implies that solving the most difficult instances of this problem would require computation time
exponential
in the number of training examples, [mathjaxinline]n[/mathjaxinline].
What makes this a difficult optimization problem is its lack of “smoothness":
There can be two hypotheses, [mathjaxinline](\theta , \theta _0)[/mathjaxinline] and [mathjaxinline](\theta ', \theta _0')[/mathjaxinline], where one is closer in parameter space to the optimal parameter values [mathjaxinline](\theta ^*, \theta _0^*)[/mathjaxinline], but they make the same number of misclassifications so they have the same [mathjaxinline]J[/mathjaxinline] value.
All predictions are categorical: the classifier can't express a degree of certainty about whether a particular input [mathjaxinline]x[/mathjaxinline] should have an associated value [mathjaxinline]y[/mathjaxinline].
For these reasons, if we are considering a hypothesis [mathjaxinline]\theta ,\theta _0[/mathjaxinline] that makes five incorrect predictions, it is difficult to see how we might change [mathjaxinline]\theta ,\theta _0[/mathjaxinline] so that it will perform better, which makes it difficult to design an algorithm that searches through the space of hypotheses for a good one.
For these reasons, we are going to investigate a new hypothesis class:
linear logistic classifiers
. These hypotheses are still parameterized by a [mathjaxinline]d[/mathjaxinline]-dimensional vector [mathjaxinline]\theta[/mathjaxinline] and a scalar [mathjaxinline]\theta _0[/mathjaxinline], but instead of making predictions in [mathjaxinline]\{ +1, -1\}[/mathjaxinline], they generate real-valued outputs in the interval [mathjaxinline](0, 1)[/mathjaxinline]. A linear logistic classifier has the form
[mathjax]h(x; \theta , \theta _0) = \sigma (\theta ^ T x + \theta _0)\; \; .[/mathjax]
This looks familiar! What's new?
The
logistic
function, also known as the
sigmoid
function, is defined as
[mathjax]\sigma (z) = \frac{1}{1+e^{-z}}\; \; ,[/mathjax]
and plotted below, as a function of its input [mathjaxinline]z[/mathjaxinline]. Its output can be interpreted as a probability, because for any value of [mathjaxinline]z[/mathjaxinline] the output is in [mathjaxinline](0, 1)[/mathjaxinline].
Study Question:
Convince yourself the output of [mathjaxinline]\sigma[/mathjaxinline] is always in the interval [mathjaxinline](0, 1)[/mathjaxinline]. Why can't it equal 0 or equal 1? For what value of [mathjaxinline]z[/mathjaxinline] does [mathjaxinline]\sigma (z) = 0.5[/mathjaxinline]?
What does a linear logistic classifier (LLC) look like? Let's consider the simple case where [mathjaxinline]d = 1[/mathjaxinline], so our input points simply lie along the [mathjaxinline]x[/mathjaxinline] axis. The plot below shows LLCs for three different parameter settings: [mathjaxinline]\sigma (10x + 1)[/mathjaxinline], [mathjaxinline]\sigma (-2x + 1)[/mathjaxinline], and [mathjaxinline]\sigma (2x - 3).[/mathjaxinline]
Study Question:
Which plot is which? What governs the steepness of the curve? What governs the [mathjaxinline]x[/mathjaxinline] value where the output is equal to 0.5?
But wait! Remember that the definition of a classifier from chapter
is that it's a mapping from [mathjaxinline]\mathbb {R}^ d \rightarrow \{ -1, +1\}[/mathjaxinline] or to some other discrete set. So, then, it seems like an LLC is actually not a classifier!
Given an LLC, with an output value in [mathjaxinline](0, 1)[/mathjaxinline], what should we do if we are forced to make a prediction in [mathjaxinline]\{ +1, -1\}[/mathjaxinline]? A default answer is to predict [mathjaxinline]+1[/mathjaxinline] if [mathjaxinline]\sigma (\theta ^ T x + \theta _0) > 0.5[/mathjaxinline] and [mathjaxinline]-1[/mathjaxinline] otherwise. The value [mathjaxinline]0.5[/mathjaxinline] is sometimes called a
prediction threshold
.
In fact, for different problem settings, we might prefer to pick a different prediction threshold. The field of
decision theory
considers how to make this choice from the perspective of Bayesian reasoning. For example, if the consequences of predicting [mathjaxinline]+1[/mathjaxinline] when the answer should be [mathjaxinline]-1[/mathjaxinline] are much worse than the consequences of predicting [mathjaxinline]-1[/mathjaxinline] when the answer should be [mathjaxinline]+1[/mathjaxinline], then we might set the prediction threshold to be greater than [mathjaxinline]0.5[/mathjaxinline].
Study Question:
Using a prediction threshold of 0.5, for what values of [mathjaxinline]x[/mathjaxinline] do each of the LLCs shown in the figure above predict [mathjaxinline]+1[/mathjaxinline]?
When [mathjaxinline]d = 2[/mathjaxinline], then our inputs [mathjaxinline]x[/mathjaxinline] lie in a two-dimensional space with axes [mathjaxinline]x_1[/mathjaxinline] and [mathjaxinline]x_2[/mathjaxinline], and the output of the LLC is a surface, as shown below, for [mathjaxinline]\theta = (1, 1), \theta _0 = 2[/mathjaxinline].
Study Question:
Convince yourself that the set of points for which [mathjaxinline]\sigma (\theta ^ T x + \theta _0) = 0.5[/mathjaxinline], that is, the separator between positive and negative predictions with prediction threshold [mathjaxinline]0.5[/mathjaxinline] is a line in [mathjaxinline](x_1, x_2)[/mathjaxinline] space. What particular line is it for the case in the figure above? How would the plot change for [mathjaxinline]\theta = (1, 1)[/mathjaxinline], but now with [mathjaxinline]\theta _0 = -2[/mathjaxinline]? For [mathjaxinline]\theta = (-1, -1), \theta _0 = 2[/mathjaxinline]?
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:08 PM (revision f808f068e)

## Lecture: Linear logistic classifier - negative log likelihood loss function

Lecture: Linear logistic classifier - negative log likelihood loss function
Lecture: Linear logistic classifier - negative log likelihood loss function
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Loss function for logistic classifiers

Loss function for logistic classifiers
We have defined a class, LLC, of hypotheses whose outputs are in [mathjaxinline](0, 1)[/mathjaxinline], but we have training data with [mathjaxinline]y[/mathjaxinline] values in [mathjaxinline]\{ +1, -1\}[/mathjaxinline]. How can we define a loss function? Intuitively, we would like to have
low loss if we assign a low probability to the incorrect class.
We'll define a loss function, called
negative log-likelihood
(NLL), that does just this. In addition, it has the cool property that it extends nicely to the case where we would like to classify our inputs into more than two classes.
In order to simplify the description, we will assume that (or transform so that) the labels in the training data are [mathjaxinline]y \in \{ 0, 1\}[/mathjaxinline], enabling them to be interpreted as probabilities of being a member of the
Remember to be sure your [mathjaxinline]y[/mathjaxinline] values have this form if you try to learn an LLC using NLL!|
class of interest.
We would like to pick the parameters of our classifier to maximize the probability assigned by the LCC to the correct [mathjaxinline]y[/mathjaxinline] values, as specified in the training set. Letting guess [mathjaxinline]g^{(i)} = \sigma (\theta ^ Tx^{(i)} + \theta _0)[/mathjaxinline], that probability is
[mathjax]\prod _{i = 1}^ n \begin{cases}  g^{(i)} &  \text {if $y^{(i)} = 1$} \\ 1 - g^{(i)} &  \text {otherwise} \end{cases}\; \; ,[/mathjax]
under the assumption that our predictions are independent. This can be cleverly rewritten, when [mathjaxinline]y^{(i)} \in \{ 0, 1\}[/mathjaxinline], as
[mathjax]\prod _{i = 1}^ n {g^{(i)}}^{y^{(i)}}(1 - g^{(i)})^{1 - y^{(i)}}\; \; .[/mathjax]
Study Question:
Be sure you can see why these two expressions are the same.
Now, because products are kind of hard to deal with, and because the log function is monotonic, the [mathjaxinline]\theta , \theta _0[/mathjaxinline] that maximize the log of this quantity will be the same as the [mathjaxinline]\theta , \theta _0[/mathjaxinline] that maximize the original, so we can try to maximize
[mathjax]\sum _{i = 1}^ n \left( {y^{(i)}}\log {g^{(i)}} + (1 - y^{(i)})\log (1 - g^{(i)})\right)\; \; .[/mathjax]
We can turn the maximization problem above into a minimization problem by taking the negative of the above expression, and write in terms of minimizing a loss
[mathjax]\sum _{i = 1}^ n \mathcal{L}_\text {nll}(g^{(i)}, y^{(i)})[/mathjax]
where [mathjaxinline]\mathcal{L}_\text {nll}[/mathjaxinline] is the
negative log-likelihood
loss function:
[mathjax]\mathcal{L}_\text {nll}(\text {guess},\text {actual}) = -\left(\text {actual}\cdot \log (\text {guess}) + (1 - \text {actual})\cdot \log (1 - \text {guess})\right) \; \; .[/mathjax]
This loss function is also sometimes referred to as the
log loss
or
You can use any base for the logarithm and it won't make any real difference. If we ask you for numbers, use log base [mathjaxinline]e[/mathjaxinline].
cross entropy
.
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:08 PM (revision f808f068e)

## Lecture: Machine learning as optimization - gradient descent in one dimension

Lecture: Machine learning as optimization - gradient descent in one dimension
Lecture: Machine learning as optimization - gradient descent in one dimension
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Machine learning as optimization - gradient descent in multiple dimensions

Lecture: Machine learning as optimization - gradient descent in multiple dimensions
Lecture: Machine learning as optimization - gradient descent in multiple dimensions
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Logistic classification as optimization

Logistic classification as optimization
We can finally put all these pieces together and develop an objective function for optimizing regularized negative log-likelihood for a
That's a lot of fancy words!
linear logistic classifier.
In fact, this process is usually called “logistic regression," so we'll call our objective [mathjaxinline]J_\text {lr}[/mathjaxinline], and define it as
[mathjax]J_\text {lr}(\theta , \theta _0; {\cal D}) = \left(\frac{1}{n} \sum _{i=1}^ n \mathcal{L}_\text {nll}(\sigma (\theta ^ T x^{(i)} + \theta _0), y^{(i)})\right) + \lambda \left\lVert \theta \right\rVert ^2\; \; .[/mathjax]
Study Question:
Consider the case of linearly separable data. What will the [mathjaxinline]\theta[/mathjaxinline] values that optimize this objective be like if [mathjaxinline]\lambda = 0[/mathjaxinline]? What will they be like if [mathjaxinline]\lambda[/mathjaxinline] is very big? Try to work out an example in one dimension with two data points.
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:08 PM (revision f808f068e)

## Lecture: One-dimensional linear regression - demo

Lecture: One-dimensional linear regression - demo
Lecture: One-dimensional linear regression - demo
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Two-dimensional linear regression - demo

Lecture: Two-dimensional linear regression - demo
Lecture: Two-dimensional linear regression - demo
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Linear logistic classifier - a few comments about regularization

Lecture: Linear logistic classifier - a few comments about regularization
Lecture: Linear logistic classifier - a few comments about regularization
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Video transcripts

### Lecture: Machine learning as optimization - framework


LESLIE KAELBLING: Our plan for today
is to keep talking about supervised learning,
and keep talking about classification,
but introduce a way of thinking about solving machine
learning problems and a way of deriving algorithms.
We'll start out just looking at linear separators,
like we're doing with a perceptron,
but this basic underlying set of ideas now,
will go forward and be the foundations of much more
complicated learning methods, that we'll look at later.
And so, fundamentally, the idea here
is that instead of trying to intuit cool learning
algorithms, like perceptron, we're
going to come up with a machinery that
will let us drive machine learning
algorithms for arbitrarily complicated problems.
And the way we're going to think about it
is to think about turning a machine learning problem
into an optimization problem.
So optimization is an important field in applied math,
and numerical methods, and computer science.
It's a way of taking some function
and trying to use computational methods to find
a minimum or a maximum of that function.
So we're going to see if we can turn machine learning
problems into optimization problems,
and then take advantage of the huge amount of stuff
that people know already about how
to solve optimization problems.
So that's our plan.
So we're going to do ML as optimization.

So fundamentally, when you do optimization, what you do
is you write down an objective function.
So usually-- I don't know, people use different things--
J is a common one.
J of theta.
So J is going to be our objective function.

That's the J and the theta here, are some parameters.

Right now, we're going to do a little bit
of a notational shift.
You might have noticed when I wrote it in the notes.
It's really hard for me to do it font-wise on the board.
But I used a capital theta here.
When we talk about optimization, machine learning
as optimization, we'll talk about parameters
generically as theta.
That capital theta just means all the parameters
in my problem.
So if what we're doing is linear separators,
this capital theta will actually stand
for our theta vector and the theta 0, if we have that.
Later on, it might stand for actually a whole set
of weight matrices, or something like that, in a neural network.
So theta is all the parameters of my problem.
So what I want to do is say, OK, my problem
as a machine learning person is that I want
to find a set of parameters,
Right, so we're still looking for a hypothesis
and the hypothesis class.
And we'll see what the class looks like.
But what we want to do is find some parameters
of a hypothesis in my class.
And we want to find ones that optimize
some objective function J.
So generally, the optimization problem
then is that we want to find a theta
star, which is the argmin over thetas, of J of theta.
Right, that Is to say I want to find the theta that
minimizes J of theta.
So that's going to be the thing that we're trying to do.
So this is kind of a general way of thinking
about an optimization problem.
In machine learning, we can set up
a bunch of different optimization problems.
But by far the most typical kind of objective
function that we'll write down for machine learning looks
like this.
So it typically looks like, we want
to find some parameters theta that
minimize the sum of two terms.
So we're going to have a sum over our data points.
I'm going to write this L and come back to it.

[INAUDIBLE] Did I close this?
No, I didn't.
One more parenthesis.
So this is the general form of the thing that we'll look at.
So the first term here is an average
over all of our data points.
And this L is our loss function.

We talked about 0-1 loss before.
We talked about it, right.
So this here, h of xi theta.
That's the prediction we would make
if theta were our parameters.
So this is our guess, our guess of the answer for question x.
And y is what our data set tells us the answer should be.
Right, so this is a loss function
that takes the guess and the actual value
and tells us how sad we are that we made this guess,
when that was really the answer.
That's the loss function.
How unhappy are we with this guess?
So that's this whole first part.
What's another name for this whole first term?

Uhh, I can wait.
What's another name for this term?

SUBJECT: [INAUDIBLE]
LESLIE KAELBLING: Yes.
The training error, good.
I know, it's early.
And many of you don't normally come.
So thank you for coming.
That's good.
So good.
So this is a training error.
This is the sum over our training
set of how unhappy we are about the predictions we're
making on the training set.
And we talked a little bit, at the very beginning,
about how training error-- it's all good and well to minimize
the training error, but in fact, our problem is to predict well,
not just on the training data, but really on testing data
that we haven't seen yet.
So we're as-- the same time that we're
introducing this idea of machine learning as an optimization
problem, we're going to introduce this extra term.
And we'll-- over several weeks-- we'll keep using a term like
this and see how it affects the kind of predictions we make.
But let me just talk to you about it little bit.
So this thing, this term right here, is called a regularizer.

And generally speaking, it's some kind
of penalty on the thetas.
It's some way of saying, well you
know what, I would like you to pick some thetas that work well
on the data.
But don't go crazy with that.
Don't try your hardest to fit the data exactly.
Maybe you should just relax a little bit.
Maybe it will work better if you don't
try super hard to fit the data.
So we'll see some examples of that today.
We'll see examples next week.
We'll keep seeing examples of what
this regularizer might be like.
This lambda is a constant.

And it's a knob that you can turn.
So knobs that you can turn in machine learning
are often called hyperparameters.
We talked about that before, hyperparameters.

And lambda governs the degree to which
you want to try to fit your data,
you're training data, really well,
versus the degree to which you might want
your hypothesis to be simple.
So that's going to be a knob that we'll play with.
And we'll practice again ways of setting the theta.
A common way of writing down a regularizer,
and the one that we'll pursue--
well, I'll come back to the regularizer.
We'll just leave it at that for now.

So the cool thing is that if we can turn a machine learning
problem into an optimization problem, like this, then
we can apply general purpose techniques.
So the plan for the lecture today
is that I'm going to introduce a loss
function and a hypothesis class that
gives us another algorithm for finding linear separators.
So that's going to be the first part.
And then the second part is that we're
going to study a simple algorithm for actually
optimizing this objective.
So that's the plan.

So and I had a terminological crisis.
So I'm going to introduce two names for the same thing.
And hopefully you'll see why.


### Lecture: Machine learning as optimization - framework


LESLIE KAELBLING: So we're going to talk about something which
we call in the notes linear logistic classifiers,
because this is a better name for what it is.

But if you go out in the world and read about this topic,
it will be called logistic regression,
which is a less great name but the name
that people use mostly.

So we're going to look at kind of a hypothesis class
and a loss function.
Often, the design of the hypothesis class and the design
of a loss function go together in some way,
so we'll look at a pair of these together that makes sense.
OK.
So first of all, let's just think about,
instead of inventing a new loss function,
let's look at the last loss function that we
thought very much about.
So before, or when we talked about Perceptron,
we thought about 0-1 Loss.

And 0-1 Loss, just as a reminder--
so Loss 01 of a guess and an actual was 1 if the guess
didn't equal the actual, and 0 otherwise.

So there's nothing wrong at all conceptually about saying,
my hypothesis class is going to be linear separators,
so just as we saw before, with the theta and the theta 0.
And I'm going to use this loss function,
and never mind that regularizer stuff for a minute.
We'll come back to it.
You could just say, an interesting question to ask
would be, can I find a theta and a theta 0 that
minimizes this loss function?
Is that something Perceptron can do?
Let me ask you this question.
Does the Perceptron algorithm--
does it minimize 0-1 Loss on a data set?

Or under what conditions does the Perceptron minimize
the 0-1 Loss on the data set?
Yeah?

SUBJECT: [INAUDIBLE]
LESLIE KAELBLING: Good.
It promises it's going to minimize this loss
if the minimal loss is 0.
That is to say, if the data is linearly separable,
Perceptron promises to find you a linear separator.
But Perceptron doesn't promise anything,
average Perceptron doesn't promise anything,
if the data is not linearly separable.
So if you have a data set that looks plus, plus, plus, plus,
plus, minus, minus, minus, minus, minus, oh,
and there's one plus one over there,
you might be happy to have a separator like that.
Perceptron doesn't promise to give you that.
That separator would make error 1.
Sorry, my pluses and minuses are not very clear.
This would make error loss 1, total loss 1 on that data set.
That's not such a bad answer, but we can't rely on Perceptron
to do that.
OK.
So we could say, all right, well,
maybe, since you're so fancy talking about optimization
here, we can just set ourselves the problem of minimizing
0-1 Loss on the data and we could find the separator
that minimizes the loss.
And the answer is that that's like a completely well-formed
problem.
You could say, what is a separator that
minimizes this loss function?
The problem is now, it's computationally very difficult
to find.
So in fact, computer scientists don't
know any good, efficient algorithms for finding
the minimal hypothesis.
So one of the things that's interesting
is, writing down the J function, that's a machine learning
problem in a sense.
We have to think about our problem, what kind
of hypothesis class do we want, and so on.
Once we write down the J and the loss function--
that helps write down the J--
then it becomes a computer science problem, an algorithm
problem.
And then it turns out that this problem is NP hard, the problem
of minimizing 0-1 Loss.
And so there's not going to be an efficient algorithm
for solving it.
So that makes us sad, but that's kind of a fundamental truth
of computer science.
So then what do we do?
What do we do?
What do we do?
Well, we try to make the problem easier
in a way that gives us computational leverage,
but doesn't change the actual statement of the problem
too much.
So now I want to illustrate, actually,
the idea in one dimension.
We're going to spend some time today
with one-dimensional classification problems, which
are really simple.
But it helps us think about what's going on.

We'll do it this way.
If we had some data like this in one dimension,
then a linear separator is just some point, really,
in between here.
And 0-1 Loss is like, well, either I
get it completely right or completely wrong.

Another way that we can think about this
is that instead of our guess being 0 or 1,
we can let our guess be something that's smooth.
So that's what we're going to do.
The 0-1 Loss function-- you can think of it as basically either
I have it right, so my loss is 0, or I have it wrong,
so my loss is 1, or you can say either
my output of my classifier is 0, or else the output
of my classifier is 1.
That makes it hard to deal with.
We're going to think about smoothing that thing out
in a way.
So here's our new hypothesis class, so
a linear logistic classifier.

It's going to take an x and a theta and a theta 0,
and the prediction it makes is going to be this.

OK.
So it used to be this square thing is the sine function.
And when we were talking before about just
the linear classifiers we talked about
before, when we talked about Perceptron, we said,
oh, we'll take the sine function, which is either--
actually, might as well address this now.
The sine function either gives you value minus 1 or plus 1.
That was the sine function.
And our old linear hypotheses said, I'm
going to take the sign of this.
It gave us plus 1 or minus 1.
So now, instead of the sine, we're
going to use this function sigma,
so that's the sigmoid function.

And sigma of some value z is 1 over 1--
and I always have to look-- plus e to the minus z.
And that sigmoid function, if this is z, looks like this.
It goes between 0 and 1.
So if this z is negative, and it crosses,
actually, right here at 0.5.
So when z is 0, this is 0.5.
When z is negative, then we get a value between 0 and 0.5.
When z is positive, we get a value between 0.5 and 1.
So you can just kind of interpret it
as something kind of like the sine function, but it's smooth.

So that's sigma, sigmoid function.
And why are we doing this?
Well, the reason that we're doing this
is because it gives us a way to talk about whether--
if we're making, let's say, a wrong classification,
how wrong is it?
Before, when we just had this step function,
if we were just barely wrong, or totally desperately wrong,
the loss would be the same.
And that would mean that an algorithm that's
trying to find a good place to put the separator
wouldn't have any idea about whether making
an incremental change would make things better.
So what we're going to do is try to find loss functions that
have the property that they're smooth,
and so that the smoothness will actually
help us algorithmically to find a good solution.
That's why we're doing what we're doing.
Not because it gives us a better class of hypotheses--
in fact, in the end, it gives us the same class of hypotheses.
But because this loss function gives the algorithm a way
to say warmer, warmer, warmer, warmer,
so it can search in a way to find a better solution.

So that's what we're up to.
Any questions about this stuff so far?

Yeah?
SUBJECT: [INAUDIBLE]
LESLIE KAELBLING: Oh, about the lambda up there?
Positive.
I can't leap up there and get that back down.
Just positive constant.
Yeah.

OK, good.
So this is a logistic linear classifier.
Almost.
It's not quite a classifier.
So this is going to give us out a value that's
in the interval 0, 1.
It can't give you actually exactly 0
unless you have infinite parameters, or exactly 1
unless you have infinite parameters.
But it gives you something between 0 and 1.
And in a minute, we're going to actually interpret it
as a probability.

And we can think of it as the probability that x is positive.

So the closer to 1 it is, the more sure
we are that x should be positive.
The closer to 0 it is, the more sure
we are that x should be negative.
When it's 50-50, we really don't know.


### Lecture: Machine learning as optimization - framework


LESLIE KAELBLING: So if you want to think
of this as a classifier, we have to pick a threshold, right,
to make it into a classifier--
to make a classifier actually--

if you have to place a bet on whether something
is positive or negative, to make an actual classifier,
then you should predict plus 1 when
sigma theta transpose x plus theta 0 is greater than 0.5.
And I talk in the notes a little bit
about how in some kinds of cases,
you might want to set that threshold differently, right.
It might be too risky to predict a negative in a case
where predicting a negative, when it's not such a good idea.
Where the risk of predicting a negative when
it should be positive is bad, then maybe you
want to reduce the threshold.
So here's a question for you.
If I want to predict positive when sigma of this stuff
is bigger than 0.5, what's the condition on the inner stuff?

Right, what has to be true of this stuff inside
for that to be true, that inequality to be true?
It has to be positive, right?
So this is the same as when theta transpose x plus theta 0
is bigger than 0.
So we're using our same old hypothesis class.
There is a nice Piazza question about that.
It is our same old hypothesis class,
we're just setting things up differently for two reasons.
One is to make the optimization easier.
And the other one is that it's actually useful.
There's actually sort of more information
when you get these values out.
So sometimes in applications, it's
useful to get this continuous quantity that
goes between 0 and 1.
You might take advantage of that.
You might say, oh, if I'm very sure it's positive,
I'll do something.
If I'm very sure it's negative, I'll do something.
But if I'm right near the 0.5, then maybe I
should not place a bet yet, maybe
I should get more information or do something.
So sometimes there some useful information in the fact
that the signal varies, OK.
OK, good.
So what does this look like in two dimensions?
What does this look like in two dimensions?
So here's a two dimensional problem.

And so now in this problem, right,
our x's have two dimensions.
So this is the first dimension of the data.
And this is the second dimension of the data.

Well, OK, actually we've never really looked too much
at the one dimensional case.
All right, so we'll do one dimension and then
two dimensions.
OK, so one dimension, let's come over here and do one dimension.
Imagine our data is like this-- negative, negative, positive,
positive, and it's just in one dimension.
But I'm going to kind of make a little bit of a cheat here.
I'm also going to draw the z-axis, right, so z--
or actually, I mean, I'd rather the sigma,
the output of the classifier.
And so if we have a nice logistic classifier
in one dimension, then one way to see what's going on
is that the output value for points over here,
right, so this is a function of x, right.
And this function here that I'm plotting
is sigma of theta x plus theta 0, right.
So for some value of theta and theta 0,
I'll get an output curve that looks like that.
The separator is still wherever this crosses 0.5, right.
So if this is 0.5, then 0.5, then that's our separator,
right.
So we still have a linear separator,
which is a point when it's in one dimension.
But the sigma value actually has this variation.
Now depending on the values of theta and theta 0,
we can slide it over, so maybe it goes over like that.
We can actually flip it over, so that it goes up
on the left side.
We can make it more or less steep.
And that all depends on the theta values.

So that's what it looks like in one dimension.
In two dimensions, it's kind of hard to see,
because I don't have the third dimension.
But for this data, if you found a good hypothesis,
a good linear logistic classifier for this,
the analogous sort of figure to that,
our separator might be this line, right.
So what that means is that the sigma function crosses 0.5
at this line.
So what is that?
So think of sigma just as sigma here-- so x
was our theta and sigma was this extra dimension, right.
It was the output function of the data.
In two dimensions, you think of a sigma
as growing out of the board, right.
And in the negative part, it should be kind of like 0,
so it should be kind of flat.
And then it's going to kind of come out and then
be like 1 over here, right.
So it's like taking that thing and sweeping it.
I don't know.
It's hard for me to illustrate that.
There's a little picture in the notes.
So this is some kind of surface over the xy, over the x1, x2.
But always where it crosses your threshold,
it will still induce a linear separator back
in your original space.
So that's an important thing to get.
All right, so that's kind of what this loss function is.
So let's think about just kind of how
we can see some of the losses.
How do we kind--
actually, this is the last one.
So this Is the prediction, right,
so this is the prediction.
So this is the prediction.
Now we need a last one.
OK, so our hypothesis class is going
to be these logistic linear classifiers.
And they're going to take a parameter vector theta, right,
which is in D dimensions, same as before,
and a theta 0, right.
So that's going to be our hypothesis class.
And now we need to derive a loss function.


### Lecture: Machine learning as optimization - framework


LESLIE KAELBLING: The way we're going
to think about the loss function here is probabilistically.
So I would like to define a loss function on all my data,
which is somehow inversely related.
I'm just going to say in English because it's
going to take a minute to get it to math.
That's going to be inversely related to the probability
that theta theta naught assign to my data.

OK, so this is what I'd like to do, right?
So that's like saying, if I can think
of the output of these sigmoids as a probability,
right, the probability that this is of class 1,
then I can kind of think about my classifier
as assigning a probability to individual data points
and therefore to the whole data set.
And so what I'd like to do, motivation for this setup,
is that I would like to find theta theta naught that
assign high probability to my data set.
So that's what we're going to try to do.
So what is the probability?
So what is the probability that theta theta naught
assigned to my data?
Well, let's define just for shorthand
guess i to be sigmoid of theta transpose Xi plus theta naught.
So that's if we pick theta theta naught as our hypothesis,
then this is the guess we're going to make.
And it's interpretable as a probability.
And so this probability that theta theta naught assigns
to my data, if you think of your data points
as being independent, it's a product.
You guys are by now used to sums, right?
Sums, maybe people have not yet seen this,
this just stands for multiplying a whole bunch of things
together.
Product of what?
So I'm going to write this down and we can think about it.

I will say that is to say if yi equals 0.

OK, so what is this about?
Let's think about this.
Let's just actually think about this part.
So you're like, how well does my hypothesis predict my data?
Right?
How much probability does my hypothesis assign to the data?
Well, the way we want to think about it
is two different cases, right?
We want to say, well, if it's supposed to be a positive data
point then how well I predict that is the g.I.
It's the actual guests that I make, right?
Because it's higher if I think it's positive and lower
if I think it's negative.
On the other hand, if the point is supposed to be negative.

Then I want to score myself according to 1 minus g.
If it's supposed to be negative, I
want my prediction to be near 0, right?
If the output is supposed to be zero,
I would like my prediction to be near zero.
So our data will have labels 1 and 0.

We could actually do this all with labels 1 and minus 1.
But it's just easier while we're thinking
about this loss function and this hypothesis class
to switch to 1 and 0.
For data has labels 1 and 0, then this
is a way of saying how happy we are with the predictions
that we're making.
OK, so we could say that we'd like
to optimize our theta and theta naught to make this quantity as
big as possible.
And we're going to do that.
Except mathematically, products are kind
of a little bit of a hassle.
For a whole bunch of reasons, we like sums better than products.
Not because they're more virtuous mathematically
really but just because they're more practically
easy to deal with.
For instance, multiplying a lot of small numbers
together can get you a really, really, really, really
small number.
And that can get you into trouble.
And some of the manipulations we'd
like to do, taking derivatives of products
is another kind of nightmare.
And we need to take derivatives.
So sums are convenient.
So what can we do?
What will you do?
Well, what we can do is we can say,
hmm, I would like to, in the end,
pick thetas that optimize this.
But it would be much more convenient
if I could have sums.
So the log function is monotonic right?
Log is monotonic.

So as x increases, the log increases.
And that means that if you find some value that
makes the log of x as big or as small as possible, well,
that would make x as big or as small as possible also.
So we're going to take logs.
And that's going to make our life easier.
So that's the trick.
It's not a trick that you would be used to
or that you would automatically think of.
But an impulse, whenever you're faced
with a big product of things, is to see if you can take the log.
Usually, it makes your life easier.
So if all we're trying to do is optimize,
it will give us a different value.
But the arg max will be the same, right?
So we're trying to find the parameters that
make some objective function as good as possible.
If you take them a monotonic kind of transformation
of your objective and you find the thing that maximizes that,
it will be the thing that maximize the original.
So that's why this maneuver is OK.
Yep?
STUDENT: [INAUDIBLE]

LESLIE KAELBLING: Oh, good.
OK, good.
Thank you.
I should have said that more clearly.
Excellent.
Why the product anyway?
OK, so we said that you didn't really
have to have a background in probability for this class.
And you really don't.
But let me use a little bit of background in probability
to motivate this.
The idea is that if you're making independent predictions
about a bunch of events, then the probability
that they're all correct, in some sense,
is the product of the probabilities
of the individual predictions.
So that's the reason.
Good point.

OK, so here we go.
Here's our thing we'd like to optimize.
I'm going to do one maneuver still in the products land.
And then we're going to take the log.
And then pretty soon, we'll have an objective function.
OK.
So the other thing, there's two things
that are not beautiful about the way this is written.
One is that it's got products.
And the other one is that it's got if statements in it.
Taking derivatives of stuff with if statements is also not
convenient.
And we're eventually, again, going
to want to take the derivative of this.
So what do we do?
So let me rewrite it still with the product
but without the if statement using a very fun trick.

OK.
So I want to argue that this is the same as that.
So this is gi raised to the yi power times 1
minus gi raised to the 1 minus yi power.
And because the yi's are either 0 or 1,
this is effectively that if statement.
Because one of these exponents is going to be a 0 and one
will be 1.
And the exponent that's 0 just means
that this whole-- if yi is 0, it turns this whole thing
into a 1, which means it gets ignored.
So if yi is zero, this whole thing gets ignored.
And we just have 1 minus gi to the 1,
which is 1 minus gi and vice versa.
So cool trick.
We good with that trick?
OK, now if we take the log of this thing,
so we take the log now, we'll get a sum, right,
because the log of a product is a sum.
So sum of 1 to n.
Now the log of that thing, right,
remember when you take the log of a product, it's the sum.
And when you take the log of an exponential,
you can take the exponent out to the front
and then make a log, right?
So we end up with y to the i--
let me put in parentheses here--
log g.
It's not y to the i.
It's yi.
It's the ith y, our gi, plus 1 minus yi log 1 minus gi.
And then we can close paren.

OK.

Got that?
All right, so all we did was take the log of this thing.
And the trick is still here, right?
The 1, 0 trick because you can tell either like if yi is 1,
then this just all reduces to log of g.
And if yi is 0, then this whole thing
reduces to log of 1 minus g.
So it's still an if statement hidden
in multiplication and addition.
OK, this thing is going to be our loss function almost.
So one thing is--
OK, so we said we wanted to maximize
the probability that our model assigned to the data.
But over there, we were kind of pessimistic
and said, well, we want to think of it in terms of loss.
Machine learning people are pessimistic.
They're always talking about loss
not maximizing but minimizing the loss.
So to minimize the loss, we have to actually
if we want to make this into a loss function,
we're just going to negate it.

OK, and so this is going to be where the way
we can think about it is i equals 1 to n
of this loss function, which we'll call NLL of gi yi.
And just so we have it written down nicely, the loss function,
the NLL stands for Negative Log Likelihood.
Likelihood is a word for probability.
Log is a word for log.
And negative is negative.
OK so NLL of a guess and a y or a guess and an actual
is minus y log g plus 1 minus y log 1 minus g.

OK, that's a loss function.
If you go reading blog posts and stuff,
you might hear people call this log loss or cross entropy.

But it's still the same kind of loss.
It still functions the same way as that 0, 1 loss.
It's still something that says, if this is my guess
and this is what the answer should have been,
that is how unhappy I am.

Questions about this?

Yep?
STUDENT: I'm not sure if this makes sense, but I feel like,
so you're multiplying the probabilities
of all the points.
LESLIE KAELBLING: Mm-hmm.
STUDENT: [INAUDIBLE] points, you just automatically just
approach 0 [INAUDIBLE]
LESLIE KAELBLING: So OK, so the question is
and your intuition is right.
He says, well, if you're multiplying the probabilities
that you get a whole bunch of points right,
isn't this thing going to go to 0 as you get more points?
And it totally is.
And it's going to be numerically super
hard to deal with this product of probabilities,
which is why we take the log.
By taking the log, it becomes numerically much nicer.
And in the end, even if your data set is really big,
that quantity will be small.
But remember, we're trying to just-- what this
is about is not that quantity as much
as the parameters that will make it as big as possible.
So that's yeah, but that's good, good, good thinking.
Other questions?

OK.

OK, good.
I will do some demonstrations.
But I want to talk about gradient descent first.
And then we'll watch gradient descent
operate on this loss function.
OK.
But let me just kind of recap what we did.
We defined a new kind of hypothesis
that gives out a probability.
And then because we did that, we needed a loss function
that would operate on this output that can be interpreted
as a probability.
And so we made this loss function, right?
So this loss function is not sensible,
not particularly sensible.
I mean, you could use it.
Oh, let's see.
If the g's are 0s and 1s, does this turn into 0, 1 loss?

You can do that as a homework problem.
We'll add it.
OK.


### Lecture: Machine learning as optimization - framework


LESLIE KAELBLING: So now let's say
we have this hypothesis class, these logistic linear
classifiers, and we have this loss function.
And so now we have an objective.
Right now, we can ignore the lambda and the regularizer.
We have an objective.
And so now what we'd like to do is find the thetas
that minimize the objective.
We think that's going to give us the best classifier.
And we would like to not have to be smart about inventing
a new algorithm.
We would like to just use some old algorithm
that optimization people know.
So this is the simplest possible optimization algorithm.
No.
The simplest possible one we already
explored at the beginning, which is maybe
like generate hypotheses at random
and see how good they are.
You could still do that.
But here's something slightly better.
People who really work in optimization
have much cleverer things to do.
But this one is simple, and efficient,
and easy to understand, so we'll use it to start with.
Actually, we'll use it most of the time,
mostly because it's efficient.
OK, so gradient descent.

So fundamentally, the idea is maybe
we have some function of some parameter values.
So you might say, oh, this is some function value
f of some input value x.
You'd think about it that way if we were just
thinking about how to find the minimum of a function.

And I'll leave it like that.
And what gradient descent does-- it's an iterative algorithm.
It guesses an x to start with.

Not smart, just somehow guesses an x.
Comes up here, looks at the derivative.
So in this case, the derivative is pointing this way.
Well, the derivative actually is pointing this way, right?
It says, oh, to get bigger, you should go up.
So the derivative is pointing this way.
So the derivative's going that way.
And we are going to take a step in the negative direction
of the derivative, which means we're
going to take a step this way.
Get a new x.
Come up here.
See what the derivative is like.
It's pointing that way.
The derivative, again, is pointing this way,
we're going to go in the negative derivative.
Take a step.
Come over here.
See what it looks like.
Negative derivative is this way.
Take a step.
Come up here.
Negative derivative's like this.
Take a step.
Oh, look.
There, the derivative seems to be pointing this way.
All right.
I'm going to go this way, and back and forth,
and eventually, I'm going to arrive at an answer.
So that's an intuitive idea.
So we sit somewhere, compute the derivative,
take a step, somewhere, compute the derivative, take a step.
So let me just write it down in one dimension.
One dimension is awesome.
So 1DGD-- 1D Gradient Descent.
For this, you need an initial theta,
some function you're trying to optimize,
the derivative of some function you're trying to optimize,
and a little epsilon, which I'll talk about.
So the f here-- this is the f.
Theta init, or it could be x init, either way.
I guess I wrote this code as if it's f of theta.

It's just a formal parameter.
It doesn't matter.
I mean, we're going to use this technique
to optimize loss functions on hypotheses because we're
machine learning people, but this technique
is super useful for minimizing all kinds of functions.
It's not just about this.
So how does the algorithm go?
We set our initial theta value to be theta init,
and we set some time index to be 0, and we loop.

And we'll increment the t.

And we do the important step, which is this.
We set our new theta values to be our old theta values minus--
uh-oh.
We needed one more parameter up here.

I'll talk about eta in a minute.

OK.
This is the only important, really--
well, there are two important lines in this program,
but this is the most important one.
So what is this?
This says, your theta was this value that was moving around.
You're trying to find a good theta value.
So you get your finger on this data,
and you say, what's the derivative right there
at that theta?
So if we're doing this in one dimension,
the derivative is a scalar.
It might be plus 3, or minus 2, or something like.
The derivative's a scalar.
And what we're going to do is we're
going to take our old theta value
and subtract something times the derivative
to get a new theta value.
This thing that looks like an n is called eta, actually.
Eta.
And it's really a step size parameter.

It says, how big are the steps that I should take?
In particular, I'm going to find out this derivative
and multiply it by some step size.
So I'm going to do this.
Go around in a loop until I don't want
to go around in a loop anymore.
And the question is, when is that?
So when do I not want to go around the loop?
And there's a bunch of ways you can terminate,
but this is the one that will generalize best to other cases.
Until f of theta at time t minus f of theta at time t minus 1
is small.
So epsilon is like a tolerance.
It's like a small number.
So basically, the idea is that you're
going to keep doing this thing, trying to find the optimum,
until the value's not changing very much,
and then you're done.

That's the outer rhythm.
Oh, and then what do we do?
We return theta.
Return theta t.

OK.
Questions about the algorithm?
Yeah?
STUDENT: [INAUDIBLE] function [INAUDIBLE] of x be?
LESLIE KAELBLING: What if the function is not convex?
Awesome.
Let's see.
What can we say about this algorithm?
So does it always work?
What does it do?
What does it mean to work?
Right, because you might have a function that looks like that.
And if your initial value is here,
if your initial x is here, it might
be that you'll find your way to here,
but you might not find your way to here.
So let's establish a little terminology.
This is a global minimum--

minimum.
OK, global minimum.
And this is a local minimum.

The slope is 0, and the gradient is positive in each direction.
If you move a little bit this way, you go up.
If you move a little bit this way, you go up.

So that makes a local minimum.
And you might not be able to get out of here.
So what we can promise you is that if your function has
a single local minimum, we'll go there.

And in general--
I'll also write the conditions down,
but we can promise that we can get you to a local minimum.
We cannot promise that we will get you to a global minimum.
So the theorem is, if f is convex,
which means basically somehow shaped like this--
you can read in more detail in the notes about that--
then for any desired--
it's a little bit tricky, this theorem--
accuracy epsilon, there is some step size,
such that GD will converge to a theta
within epsilon of the optimum.

What's sneaky about this theorem is that there is some eta.

The lab actually is going to have you mess
around with step size a lot and get some intuition for what
happens.
But fundamentally, the issue is that if you pick eta
to be very small, then you take like teeny little steps,
and it takes forever.
But if you pick eta to be too big,
you risk taking enormous steps and then the whole thing
doesn't converge.
Like, it goes nuts.
So you'll have to play around with this,
get a feeling for it.
But there is a fundamental kind of difficulty,
which is there is no obvious way to know
what your eta should be.
So it's another thing that you have
to kind of play around with.

So that's a little bit of trouble.
OK.
We are almost done talking about gradient descent,
but this is gradient descent in one dimension.
We call it derivative descent.
Yep?
STUDENT: If epsilon is [INAUDIBLE]

LESLIE KAELBLING: Theta.

Right.

Yeah, and I've written this down really informally.
If you want the formal definition, post on Piazza
and we'll find you a good one.
Yeah.
Other questions?

This is cool if we have a one-dimensional parameter
family, which is not usual for us.
Usually our theta is a whole vector of parameters.
And remember now here, when I'm going
to talk about theta, when I talked about j of theta
over there, theta is like all our parameters.
So for the linear hypothesis class,
it's going to be our normal theta vector plus also
the theta 0.
So it's all of our parameters.
We want to find values of all of our parameters
together that minimize our objective function.


### Lecture: Machine learning as optimization - framework


LESLIE KAELBLING: So this was good for one dimension, what
if we have multiple dimensions?
So let's do the multiple dimension case.

So if we have multiple dimensions, so
then just general gradient descent, I'm actually--
I actually don't hardly need to write this down again,
but maybe I will just so that we can be sure.
We'll look at the types of things.
So lets-- I'll write down all the things
and we'll look at their shapes.

So first thing is if you have a function--
imagine it's a function that takes let me not use
d, let me use, I don't know, m.
It's not necessarily the dimension of our data.
We-- you have a function that takes m arguments.
Let's imagine that, our function takes m arguments.
So our thetas are d by m--
m-- no, d-- excuse me, m by 1.
So theta has dimension m by 1.

So theta has m by 1.
Our function takes a m by 1 vector
really is the way to think about it and gives back a scalar.
So let's just assume for right now
that the f output is a scalar, for all
of this we'll assume that.
So now the question is what about the gradient?
And many of you looked at gradients and some of you
more recently than others, so let's
just remember the definition because it's
not super terrifying even though the symbol's cool.
So what is this?
So the gradient of f with respect to theta--
so, first of all, really important
is to know its dimension.
Its dimension is m by 1.

So it's going to be a vector of the same shape as theta
and it's really just a vector of the partial derivative.
So it's the partial derivative of f with respect
to parameter 1.
And then the next element is partial f, parameter 2,
and then partial f, partial parameter m.

So that's the gradient, just the vector of partial derivatives.
And sometimes we'll use some fancy gradient calculus.
Actually, we use it off and on throughout the class.
It makes writing stuff on the board super compact
and beautiful.
If you-- it doesn't make sense to you,
you can always, always, always go back and just compute
the entries of this vector.
And usually you just have to compute one or two,
because they just depend on indices in some systematic way.
So this is really-- this is a gradient.
But you could think of it as a direction.
It's the gradient-- ah, when--
I remember, I think--
I don't know, I was in high school or college--
maybe college, probably-- and I was learning to ski
and I was taking a ski lesson and the ski instructor
was trying to explain this idea that there
was the steepest way down the hill
and he called it the fall line.
And the idea is you're supposed to keep your skis,
so that you don't fall down.
Anyway, so keeping your skis orthogonal to the gradient
would have been a very simple way
to say the thing he was trying to say,
but he had to invent all these words.
Finally, I understood, oh, OK, orthogonal to the gradient,
we're good.
So the gradient is the steepest way down the hill--
or steepest way up the hill actually-- excuse me,
the gradient is the steepest way up the hill.
The direction is the same, but the sine is different.
So that's the gradient.
So good.
So now we have theta init.
So theta init is gonna be also m by 1.
And that's just the point we're going to start at on the hill.
And then we're going to execute the same algorithm.
These two guys are still scalars.
And the only thing that's different--
so I'm not going to write all the bookkeeping
stuff-- the only thing that's different
is that now we say theta t is theta t minus 1 minus eta.
Now instead of the derivative, we
say the gradient of f at theta t minus 1.
I mean-- so it's the gradient with respect to theta,
but at this particular place.

So we're standing here on the mountain,
we figure out what's the steepest way down,
we take a step.
Now we look and we see where is the steepest way down,
this way, and we take a step, and so on.
So that's gradient sense-- now really gradient descent.

Good?
Questions about this?
Termination is the same.

All right, let's see some examples.
So what do we have put together now?
We have a hypothesis class which gives us
out numbers between 0 and 1.
We have a loss function that consumes
numbers between 0 and 1, which gives us
an objective function, j.
And then now we have an algorithm
for trying to find the minimum.


### Lecture: Machine learning as optimization - framework


LESLIE KAELBLING: I spent this morning making a new demo--
1-dimensional linear regression.
So this is going to be a picture that's like that one
that I drew up there in pencil.

We have two negative examples, one here at 0 and one at 1,
and two positive examples, one at 2 and one at 3.
And I initialized-- this is after one step already.
I initialized that at 0, 0, my parameters.
There's two parameters.
D is equal to 1, so theta is 1-dimensional,
and we have theta 0, so we have two parameters.
So we're searching for two parameters.
So I'm running this algorithm gradient descent.
My theta vector is 2-dimensional because I have two parameters.
And what I'm going to do is just show steps of gradient descent.
So this is a sigmoid.
It's not doing a very good job.
Right now-- actually I probably should've drawn the separator,
so right here is 0.5 on my plot.
So if I were taking--
so this sigmoid is my current hypothesis.
This is the output.
This is sigmoid if theta times x plus theta 0.
And if I thresholded it at 0.5--

that 0.5 is right here on my axes
and so thresholding it at 0.5 would be here.
So what would-- how many points do I get right
if this is my hypothesis?

1?
Ah, I think I get three right.
Yeah, right, because if I put my threshold here,
I'm saying everybody that way is a negative guy and everybody
this way is positive.
And so I get this negative one right and these two positives
right, but I get this negative one wrong.
So that's not exactly a good answer, but it's not terrible.
I take a step of gradient descent
and it moves over a little.
So all I'm doing--
I'm running that algorithm on those two parameters
and what is it doing?
It's sliding-- the threshold of sliding over.
So, by now, the place where I cross 0.5 is just about here.
So I'm still getting this guy wrong,
but I almost have it right.
I know it's moving over farther.

So it's moving the threshold over.
What else is it doing?
Besides moving the threshold over?
It's making a steeper.
Why is it making it steeper?
Why do we want it to be steep?

Why is steep good?
Yeah.
SUBJECT: [INAUDIBLE]
LESLIE KAELBLING: Good.

As it gets steeper, it's making the g values, those guesses--
the blue is my g.
it's making the g values closer to 0 and 1.
And if I can separate my data, I want, ideally,
to maximize this probability.
I would like my g values to be 0 and 1.
Now I can ever get them actually to 0 and 1.
And the fact is that if you run an optimizer
forever on this problem without a regularizer--
we'll put a regularizer in here in a minute.
If I run it forever, it will actually eventually give you
an answer which you may not have seen before.
A computer program sometimes gives you--
explodes and gives you and says something about NaN--
N-A-N-- that means Not a Number and that means your numbers
got so big that the computer can't
represent them conveniently anymore, so that's big.
And it will want to do that.
It will want to make your weights infinitely big.
But here we go, we're doing pretty good.
Anyway, it's going to run like this for a while longer.
I told it to go 100 at most.
So here's my other plot.
This is a plot.
Now the x-axis is the iterations,
so here I just did 100.
So here my gradient descent method
takes another parameter, which is the maximum number
of iterations to run.
It's always a good idea to have this thing terminate
for some reason other than this reason.
Because this reason you might not make true,
you're going too slowly, or your step size
is too big, or something.
So anyway-- so mine terminates, I can tell it to stop.
So it ran 100 steps.
So this is after each iteration--
this is the f of x value.
So we're trying to minimize f of x.
And my f of x in this particular example
is that summed, negative, log likelihood
of my four data points.
Does this make sense?
We can play around a little bit with it
and then we'll do 2-dimensions.
So let's try-- so that was step size two,
I'm going to set the t to be bigger, like 1,000.
Actually looks let's just see--
and I'm going to make it not stop.
So draw is false and pause is false.

Ah, ah, ah-- sorry, one second.
What's the name of that parameter called?
Oh, I didn't pass it in.
This is what I get for hacking something in the morning
and not making the step the total--
all right, get-- never mind, we'll
look at it in two dimensions.
That was the 1D thing that I did this morning.
So here we are--
actually, let me set the lambda, though.
That's interesting.

So if we set--

actually let's set the step size to be big.
So if we step-- set the step size to be 10.
We can start over.
So that-- originally we were doing steps size 1 [INAUDIBLE]
10.
Whoo, it's taking big steps.
But, look, it's, oh, it's settled down.
So that's good.
So that went very quickly to a really awesome separator.
And now it's deepening up a little bit,
but it's hard to see.
If I set the step size to, I don't know-- let's try 20.

So step size 20, we start off in the same place.

Oh, that's even better.
Maybe-- that makes me think that a really big step
size would be good.
We'll try 40.
So after one step, step size 40, look how steep it is.
So it's the magnitudes--
you can figure and see this yourself.
You can play with a pencil.
It's the magnitude of theta that governs the steepness.
So after one step.
It's gotten big and now--
yeah, OK, that's good.
So we have some regularization in there
that's keeping it from going crazy,
let's see if we say [INAUDIBLE] is 0.

Even steps size 100-- this guy is very robust
actually Normally a step size of 100
would not work out very well.
But it manages to go to a place where
the gradient is so small that it's just
very happy where it is.


### Lecture: Machine learning as optimization - framework


LESLIE KAELBLING: Now let's do two dimensions.
So in two dimensions, I can't draw.
I could do Surface Plus, but that's too tricky.
So we're just going to see the separator now.
So let's do T3 100 true true [INAUDIBLE] 1.

OK.
So this is going to look familiar to you.

We saw the same data set when we looked at perceptrons.
And it's after one iteration the separator is up there.
And remember, really it's a whole sheet
that's at level sort of like one on the side where
the normal is pointing, and down at level
zero on the other side.
And we're doing gradient descent.
Now there are three parameters, right?
Theta is three-dimensional for us here, the big theta,
because we have the two parameters in our little theta.
Because we got two-dimensional x's, two-dimensional inputs,
and then the offset.
So now we're doing this in three dimensions.
OK.
I took a step, and the separator ran away.
We'll see if it comes back.
Oh, it's back.
OK, here we go.
So it's working its way over here.
At this point--
OK, what's interesting at this point,
if this were a perceptron, it would be like, OK, I'm done.
Right?
But this guy's not done, right?
It really wants to totally really optimize this thing.
And so what it's going to do is actually keep moving itself
so that it balances the error due to the positive guys
and the error due to the negative guys.
It really wants to minimize the sum of the errors.
So this is one thing that's already nicer
in a way than the perceptron.
So first of all, we get this general instance
of an optimization algorithm.
But we'll tend to do a kind of balancing between the positives
and the negatives.
What's happening to the normal as I take steps?

It's getting longer.
What does that correspond to?
It's getting steeper.
Right?
So that's a really important thing
to have in your head, right?
So the normal getting bigger is going
to really correspond to this thing getting steeper,
which I'm not drawing here.
OK, let's look at another example, just for fun.

So we can do.
So here's another data set.
And this is perceptron actually.
So perceptron on this data set, just so you remember
what perceptron looks like.

The separator hasn't shown up yet.
There it is.
OK.
And now it's done, right?
Perceptron-- it ran for four iterations, found a separator,
and said, I am good.
So now if we run logistic regression on the same data--
so T1-- so this is a logistic regression on the same data.

Oh, yeah.
I ran it non-interactively.
Because it takes-- given the step size I gave it,
it takes quite a few iterations.
So how do you feel about these two separators?
So that one is logistic regression, and this one is--
no, excuse me.
That one's perceptron, and this one is logistic regression.
Somebody want to say something about what
might be good about the logistic regression one?

Do you like it better?
Yeah?
It might do better on test set.
Why might it do better?
Yeah?
The margin is bigger, right?
Again, we talked about this a little bit last time.
We had the sense that maybe if you wiggled around the data
points here a little bit, you might be worried
that you would catch a negative on the other side of this guy
maybe, right?
So you get a sense that it might not be so robust.
In this case, it actually wants to kind of assign
good probability to all the data points.
So that's sort of a good thing.
Do you think that this one has converged?
Or do you think it's going to keep moving?
The logistic regression line.

Going to keep moving.
Your intuition is that it's going
to keep moving until it's kind of like here, right?
So let's run it for a lot of iterations.

Is that more than we want to wait?
No, it should finish pretty soon.

OK.
It did move.

But possibly-- but look at the convergence plot.
OK, so here's the convergence plot.
Like oh, all those extra iterations
didn't really help very much.
Right?
So again, what does this plot?
This is the number of iterations,
and this is the objective function.
So you can barely see--
there's a couple little dots on this margin
where it was learning, and then now it's just [INAUDIBLE]..
So that was 10,000 iterations.
It didn't get any better.
At first, you might look at this and think,
I am not sure I like that answer.
It feels like it should be closer to the green ones maybe.
Is that what it feels like to you?
So why is it not closer to the green ones,
to the positive ones?

There are three of those guys.
And they are ganging up on the negative one.
Right?
So it wants to make-- so it's adding up the log probability
that we're assigning to the positive ones, and the log
probability minus log probability
that we're assigning to the negative one, log
minus probability.
And so by scooting away from the positive ones,
it has a higher value of its curve over there.
And so that last guy is far over,
and it really wants those guys to be positive.
So it doesn't maximize the margin.
There's another objective function, another loss
function, that's about actually trying
to maximize the margin of the hypothesis
with respect to the data set.
And that can be desirable in some circumstances,
although it's harder to optimize.
So logistic regression does not maximize the margin.
And here's an example of why not, right?
Because if I put a whole bunch more positive ones over there,
it would actually move the separator even farther away
from them, because it wants them to be
as positive as they can be.


### Lecture: Machine learning as optimization - framework


LESLIE KAELBLING: So what we're going
to do this week is play with gradient descent,
implement it, understand some stuff about how it terminates,
play with a step size to see how that affects how it behaves.
And we're also going to look at logistic regression.
An important thing that I possibly deliberately
left out is that in order to implement logistic regression
using gradient descent, we need to know f, which is good.
f is our loss function, right. f is the sum of the losses there.
But also we need to know the gradient of f.
And so that's a thing that you will have
to calculate with a pencil.
Let me say one more thing about regularization, right.
So in the linear case, so regularization--
often our regularizer, so R of theta, the most standard one--

is something like the norm theta or actually norm theta squared.

And what that does is it says, well, all thing--
so try to fit your data, but keep the norm of the parameters
small.
And one thing that already does for us
here in logistic regression is that it keeps the optimization
from going nuts.
If you say, assign high probability to my points,
but oh, by the way, don't let the weights go crazy,
don't let the thetas get too big,
then that'll keep your optimization
problem well formed.
It will also mean that it might try a little bit less
hard to assign the most probability it
can to some distant points.
And that will be a useful thing.
We'll see a lot more about regularization next time,
when we'll see it in a context that's a lot clearer.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/920d2b98ba33e5f1ef544c1e54d9a69d/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Logistic_regression.pdf
