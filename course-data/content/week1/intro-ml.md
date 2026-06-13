# Introduction to ML

> Week 1: Basics · MIT 6.036 courseware archive

## Notes – Chapters 1 and 2

Notes – Chapters 1 and 2
You can sequence through the Introduction and Linear Classifier lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 1: Introduction to ML
and
Chapter 2: Linear classifiers
notes as PDF files.

## Lecture: Introduction to ML - perspective and history

Lecture: Introduction to ML - perspective and history
Lecture: Introduction to ML - perspective and history
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Introduction to ML

Introduction to ML
The main focus of machine learning is
making decisions or predictions based on data
. There are a number of other fields with significant overlap in technique, but
This story paraphrased from a post on 9/4/12 at
andrewgelman.com
difference in focus:
in economics and psychology, the goal is to discover underlying causal processes and in statistics it is to find a model that fits a data set well. In those fields, the end product is a model. In machine learning, we often fit models, but as a means to the end of making good predictions or decisions.
As machine-learning (ML) methods have improved in their capability and scope, ML has become the best way, measured in terms of speed, human engineering time, and robustness, to make many applications. Great examples are face detection and speech recognition and many kinds of language-processing tasks. Almost any application that involves understanding data or signals that come from the real world can be best addressed using machine learning.
and often undervalued
One crucial
aspect of machine learning approaches to solving problems is that human engineering plays an important role. A human still has to
frame
the problem: acquire and organize data, design a space of possible solutions, select a learning algorithm and its parameters, apply the algorithm to the data, validate the resulting solution to decide whether it's good enough to use, etc. These steps are of great importance.
The conceptual basis of learning from data is the
Bertrand Russell is my hero. –lpk
problem of induction
:
Why do we think that previously seen data will help us predict the future? This is a serious philosophical problem of long standing. We will operationalize it by making assumptions, such as that all training data are IID (independent and identically distributed) and that queries will be drawn from the same distribution as the training data, or that the answer comes from a set of possible answers known in advance.
In general, we need to solve these two problems:
estimation:
When we have data that are noisy reflections of some underlying quantity of interest, we have to aggregate the data and make estimates or predictions about the quantity. How do we deal with the fact that, for example, the same treatment may end up with different results on different trials? How can we predict how well an estimate may compare to future results?
generalization:
How can we predict results of a situation or experiment that we have never encountered before in our data set?
We can describe problems and their solutions using six characteristics, three of which characterize the problem and three of which characterize the solution:
Problem class:
What is the nature of the training data and what kinds of queries will be made at testing time?
Assumptions:
What do we know about the source of the data or the form of the solution?
Evaluation criteria:
What is the goal of the prediction or estimation system? How will the answers to individual queries be evaluated? How will the overall performance of the system be measured?
Model type:
Will an intermediate model be made? What aspects of the data will be modeled? How will the model be used to make predictions?
Model class:
What particular parametric class of models will be used? What criterion will we use to pick a particular model from the model class?
Algorithm:
What computational process will be used to fit the model to the data and/or to make predictions?
Without making some assumptions about the nature of the process generating the data, we cannot perform generalization. In the following sections, we elaborate
Don't feel you have to memorize all these kinds of learning, etc. We just want you to have a very high-level view of (part of) the breadth of the field.
on these ideas.
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:30:51 PM (revision f808f068e)

## Lecture: Introduction to ML - estimation and generalization

Lecture: Introduction to ML - estimation and generalization
Lecture: Introduction to ML - estimation and generalization
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Supervised learning - setting

Lecture: Supervised learning - setting
Lecture: Supervised learning - setting
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Problem class

Problem class
There are many different
problem classes
in machine learning. They vary according to what kind of data is provided and what kind of conclusions are to be drawn from it. Five standard problem classes are described below, to establish some notation and terminology.
In this course, we will focus on classification and regression (two examples of supervised learning), and will touch on reinforcement learning and sequence learning.
Supervised learning
The idea of
supervised
learning is that the learning system is given inputs and told which specific outputs should be associated with them. We divide up supervised learning based on whether the outputs are drawn from a small finite set (classification) or a large finite or continuous set (regression).
Classification
Training data [mathjaxinline]{\cal D}_ n[/mathjaxinline] is in the form of a set of pairs [mathjaxinline]\{ (x^{(1)}, y^{(1)}), \ldots , (x^{(n)}, y^{(n)})\}[/mathjaxinline] where [mathjaxinline]x^{(i)}[/mathjaxinline] represents an object to be classified, most typically a [mathjaxinline]d[/mathjaxinline]-dimensional vector of real and/or discrete values, and [mathjaxinline]y^{(i)}[/mathjaxinline] is an element of a discrete
Many textbooks use [mathjaxinline]x_ i[/mathjaxinline] and [mathjaxinline]t_ i[/mathjaxinline] instead of [mathjaxinline]x^{(i)}[/mathjaxinline] and [mathjaxinline]y^{(i)}[/mathjaxinline]. We find that notation somewhat difficult to manage when [mathjaxinline]x^{(i)}[/mathjaxinline] is itself a vector and we need to talk about its elements. The notation we are using is standard in some other parts of the machine-learning literature.
set of values.
The [mathjaxinline]y[/mathjaxinline] values are sometimes called
target values
.
A classification problem is
binary
or
two-class
if [mathjaxinline]y^{(i)}[/mathjaxinline] is drawn from a set of two possible values; otherwise, it is called
multi-class
.
The goal in a classification problem is ultimately, given a new input value [mathjaxinline]x^{(n+1)}[/mathjaxinline], to predict the value of [mathjaxinline]y^{(n+1)}[/mathjaxinline].
Classification problems are a kind of
supervised learning
, because the desired output (or class) [mathjaxinline]y^{(i)}[/mathjaxinline] is specified for each of the training examples [mathjaxinline]x^{(i)}[/mathjaxinline].
Regression
Regression is like classification, except that [mathjaxinline]y^{(i)} \in \mathbb {R}^ k[/mathjaxinline].
Unsupervised learning
Unsupervised
learning doesn't involve learning a function from inputs to outputs based on a set of input-output pairs. Instead, one is given a data set and generally expected to find some patterns or structure inherent in it.
Density estimation
Given samples [mathjaxinline]x^{(1)}, \ldots , x^{(n)} \in \mathbb {R}^ D[/mathjaxinline]
IID stands for
independent and identically distributed
, which means that the elements in the set are related in the sense that they all come from the same underlying probability distribution, but not in any other ways.
drawn IID
from some distribution [mathjaxinline]\Pr (X)[/mathjaxinline], the goal is to predict the probability [mathjaxinline]\Pr (x^{(n+1)})[/mathjaxinline] of an element drawn from the same distribution. Density estimation sometimes plays a role as a “subroutine" in the overall learning method for supervised learning, as well.
Clustering
Given samples [mathjaxinline]x^{(1)}, \ldots , x^{(n)} \in \mathbb {R}^ D[/mathjaxinline], the goal is to find a partitioning (or “clustering") of the samples that groups together samples that are similar. There are many different objectives, depending on the definition of the similarity between samples and exactly what criterion is to be used (e.g., minimize the average distance between elements inside a cluster and maximize the average distance between elements across clusters). Other methods perform a “soft" clustering, in which samples may be assigned 0.9 membership in one cluster and 0.1 in another. Clustering is sometimes used as a step in density estimation, and sometimes to find useful structure in data.
Dimensionality reduction
Given samples [mathjaxinline]x^{(1)}, \ldots , x^{(n)} \in \mathbb {R}^ D[/mathjaxinline], the problem is to re-represent them as points in a [mathjaxinline]d[/mathjaxinline]-dimensional space, where [mathjaxinline]d < D[/mathjaxinline]. The goal is typically to retain information in the data set that will, e.g., allow elements of one class to be discriminated from another.
Dimensionality reduction is a standard technique which is particularly useful for visualizing or understanding high-dimensional data. If the goal is ultimately to perform regression or classification on the data after the dimensionality is reduced, it is usually best to articulate an objective for the overall prediction problem rather than to first do dimensionality reduction without knowing which dimensions will be important for the prediction task.
Reinforcement learning
In reinforcement learning, the goal is to learn a mapping from input values [mathjaxinline]x[/mathjaxinline] to output values [mathjaxinline]y[/mathjaxinline], but without a direct supervision signal to specify which output values [mathjaxinline]y[/mathjaxinline] are best for a particular input. There is no training set specified
a priori
. Instead, the learning problem is framed as an agent interacting with an environment, in the following setting:
The agent observes the current state, [mathjaxinline]x^{(0)}[/mathjaxinline].
It selects an action, [mathjaxinline]y^{(0)}[/mathjaxinline].
It receives a reward, [mathjaxinline]r^{(0)}[/mathjaxinline], which depends on [mathjaxinline]x^{(0)}[/mathjaxinline] and possibly [mathjaxinline]y^{(0)}[/mathjaxinline].
The environment transitions probabilistically to a new state, [mathjaxinline]x^{(1)}[/mathjaxinline], with a distribution that depends only on [mathjaxinline]x^{(0)}[/mathjaxinline] and [mathjaxinline]y^{(0)}[/mathjaxinline].
The agent observes the current state, [mathjaxinline]x^{(1)}[/mathjaxinline].
[mathjaxinline]\ldots[/mathjaxinline]
The goal is to find a policy [mathjaxinline]\pi[/mathjaxinline], mapping [mathjaxinline]x[/mathjaxinline] to [mathjaxinline]y[/mathjaxinline], (that is, states to actions) such that some long-term sum or average of rewards [mathjaxinline]r[/mathjaxinline] is maximized.
This setting is very different from either supervised learning or unsupervised learning, because the agent's action choices affect both its reward and its ability to observe the environment. It requires careful consideration of the long-term effects of actions, as well as all of the other issues that pertain to supervised learning.
Sequence learning
In sequence learning, the goal is to learn a mapping from
input sequences
[mathjaxinline]x_0, \ldots , x_ n[/mathjaxinline] to
output sequences
[mathjaxinline]y_1, \ldots , y_ m[/mathjaxinline]. The mapping is typically represented as a
state machine
, with one function [mathjaxinline]f[/mathjaxinline] used to compute the next hidden internal state given the input, and another function [mathjaxinline]g[/mathjaxinline] used to compute the output given the current hidden state.
It is supervised in the sense that we are told what output sequence to generate for which input sequence, but the internal functions have to be learned by some method other than direct supervision, because we don't know what the hidden state sequence is.
Other settings
There are many other problem settings. Here are a few.
In
semi-supervised
learning, we have a supervised-learning training set, but there may be an additional set of [mathjaxinline]x^{(i)}[/mathjaxinline] values with no known [mathjaxinline]y^{(i)}[/mathjaxinline]. These values can still be used to improve learning performance if they are drawn from [mathjaxinline]\Pr (X)[/mathjaxinline] that is the marginal of [mathjaxinline]\Pr (X, Y)[/mathjaxinline] that governs the rest of the data set.
In
active
learning, it is assumed to be expensive to acquire a label [mathjaxinline]y^{(i)}[/mathjaxinline] (imagine asking a human to read an x-ray image), so the learning algorithm can sequentially ask for particular inputs [mathjaxinline]x^{(i)}[/mathjaxinline] to be labeled, and must carefully select queries in order to learn as effectively as possible while minimizing the cost of labeling.
In
transfer
learning (also called
meta-learning
), there are multiple tasks, with data drawn from different, but related, distributions. The goal is for experience with previous tasks to apply to learning a current task in a way that requires decreased experience with the new task.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Lecture: Supervised learning - hypotheses

Lecture: Supervised learning - hypotheses
Lecture: Supervised learning - hypotheses
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Assumptions

Assumptions
The kinds of assumptions that we can make about the data source or the solution include:
The data are independent and identically distributed.
The data are generated by a Markov chain.
The process generating the data might be adversarial.
The “true" model that is generating the data can be perfectly described by one of some particular set of hypotheses.
The effect of an assumption is often to reduce the “size" or “expressiveness" of the space of possible hypotheses and therefore reduce the amount of data required to reliably identify an appropriate hypothesis.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Lecture: Evaluating predictions - loss functions

Lecture: Evaluating predictions - loss functions
Lecture: Evaluating predictions - loss functions
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Evaluating hypotheses - training set error

Lecture: Evaluating hypotheses - training set error
Lecture: Evaluating hypotheses - training set error
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Evaluation criteria

Evaluation criteria
Once we have specified a problem class, we need to say what makes an output or the answer to a query good, given the training data. We specify evaluation criteria at two levels: how an individual prediction is scored, and how the overall behavior of the prediction or estimation system is scored.
The quality of predictions from a learned model is often expressed in terms of a
loss function
. A loss function [mathjaxinline]L(g, a)[/mathjaxinline] tells you how much you will be penalized for making a guess [mathjaxinline]g[/mathjaxinline] when the answer is actually [mathjaxinline]a[/mathjaxinline]. There are many possible loss functions. Here are some frequently used examples:
0-1 Loss
applies to predictions drawn
If the actual values are drawn from a continuous distribution, the probability they would ever be equal to some predicted [mathjaxinline]g[/mathjaxinline] is 0 (except for some weird cases).
from finite domains.
[mathjax]L(g, a) = \begin{cases}  0 &  \text {if $g = a$} \\ 1 &  \text {otherwise} \end{cases}[/mathjax]
Squared loss
[mathjax]L(g, a) = (g - a)^2[/mathjax]
Linear loss
[mathjax]L(g, a) = |g - a|[/mathjax]
Asymmetric loss
Consider a situation in which you are trying to predict whether someone is having a heart attack. It might be much worse to predict “no" when the answer is really “yes", than the other way around.
[mathjax]L(g, a) = \begin{cases}  1 &  \text {if $g = 1$ and $a = 0$} \\ 10 &  \text {if $g = 0$ and $a = 1$} \\ 0 &  \text {otherwise} \end{cases}[/mathjax]
Any given prediction rule will usually be evaluated based on multiple predictions and the loss of each one. At this level, we might be interested in:
Minimizing expected loss over all the predictions (also known as risk)
Minimizing maximum loss: the loss of the worst prediction
Minimizing or bounding regret: how much worse this predictor performs than the best one drawn from some class
Characterizing asymptotic behavior: how well the predictor will perform in the limit of infinite training data
Finding algorithms that are probably approximately correct: they probably generate a hypothesis that is right most of the time.
There is a theory of rational agency that argues that you should always select the action that
minimizes the expected loss
. This strategy will, for example, make you the most money in the long run, in
Of course, there are other models for action selection and it's clear that people do not always (or maybe even often) select actions that follow this rule.
a gambling setting.
Expected loss is also sometimes called
risk
in the machine-learning literature, but that term means other things in economics or other parts of decision theory, so be careful...it's risky to use it. We will, most of the time, concentrate on this criterion.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Model type

Model type
Recall that the goal of a machine-learning system is typically to estimate or generalize, based on data provided. Below, we examine the role of model-making in machine learning.
No model
In some simple cases, in response to queries, we can generate predictions directly from the training data, without the construction of any intermediate model. For example, in regression or classification, we might generate an answer to a new query by averaging answers to recent queries, as in the
nearest neighbor
method.
Prediction rule
This two-step process is more typical:
“Fit" a model to the training data
Use the model directly to make predictions
In the
prediction rule
setting of regression or classification, the model will be some hypothesis or prediction rule [mathjaxinline]y = h(x ; \theta )[/mathjaxinline] for some functional form [mathjaxinline]h[/mathjaxinline]. The idea is that [mathjaxinline]\theta[/mathjaxinline] is a vector of one or more parameter values that will be determined by fitting the model to the training data and then be held fixed. Given a new [mathjaxinline]x^{(n+1)}[/mathjaxinline], we would then make the prediction [mathjaxinline]h(x^{(n+1)}; \theta )[/mathjaxinline].
We write [mathjaxinline]f(a ; b)[/mathjaxinline] to describe a function that is usually applied to a single argument [mathjaxinline]a[/mathjaxinline], but is a member of a parametric family of functions, with the particular function determined by parameter value [mathjaxinline]b[/mathjaxinline]. So, for example, we might write [mathjaxinline]h(x ; p) = x^ p[/mathjaxinline] to describe a function of a single argument that is parameterized by [mathjaxinline]p[/mathjaxinline].
note
The fitting process is often articulated as an optimization problem: Find a value of [mathjaxinline]\theta[/mathjaxinline] that minimizes some criterion involving [mathjaxinline]\theta[/mathjaxinline] and the data. An optimal strategy, if we knew the actual underlying distribution on our data, [mathjaxinline]\Pr (X,Y)[/mathjaxinline] would be to predict the value of [mathjaxinline]y[/mathjaxinline] that minimizes the
expected loss
, which is also known as the
test error
. If we don't have that actual underlying distribution, or even an estimate of it, we can take the approach of minimizing the
training error
: that is, finding the prediction rule [mathjaxinline]h[/mathjaxinline] that minimizes the average loss on our training data set. So, we would seek [mathjaxinline]\theta[/mathjaxinline] that minimizes
[mathjax]\mathcal{E}_ n(\theta ) = \frac{1}{n}\sum _{i = 1}^ n L(h(x^{(i)};\theta ), y^{(i)})\; \; ,[/mathjax]
where the loss function [mathjaxinline]L(g, a)[/mathjaxinline] measures how bad it would be to make a guess of [mathjaxinline]g[/mathjaxinline] when the actual value is [mathjaxinline]a[/mathjaxinline].
We will find that minimizing training error alone is often not a good choice: it is possible to emphasize fitting the current data too strongly and end up with a hypothesis that does not generalize well when presented with new [mathjaxinline]x[/mathjaxinline] values.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Model class and parameter fitting

Model class and parameter fitting
A model
class
[mathjaxinline]{\cal M}[/mathjaxinline] is a set of possible models, typically parameterized by a vector of parameters [mathjaxinline]\Theta[/mathjaxinline]. What assumptions will we make about the form of the model? When solving a regression problem using a prediction-rule approach, we might try to find a linear function [mathjaxinline]h(x ; \theta , \theta _0) = \theta ^ T x + \theta _0[/mathjaxinline] that fits our data well. In this example, the parameter vector [mathjaxinline]\Theta = (\theta , \theta _0)[/mathjaxinline].
For problem types such as discrimination and classification, there are huge numbers of model classes that have been considered...we'll spend much of this course exploring these model classes, especially neural networks models. We will almost completely restrict our attention to model classes with a fixed, finite number of parameters. Models that relax this assumption are called “non-parametric" models.
How do we select a model class? In some cases, the machine-learning practitioner will have a good idea of what an appropriate model class is, and will specify it directly. In other cases, we may consider several model classes. In such situations, we are solving a
model selection
problem: model-selection is to pick a model class [mathjaxinline]{\cal M}[/mathjaxinline] from a (usually finite) set of possible model classes;
model fitting
is to pick a particular model in that class, specified by parameters [mathjaxinline]\theta[/mathjaxinline].
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Lecture: Learning algorithms

Lecture: Learning algorithms
Lecture: Learning algorithms
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Algorithm

Algorithm
Once we have described a class of models and a way of scoring a model given data, we have an algorithmic problem: what sequence of computational instructions should we run in order to find a good model from our class? For example, determining the parameter vector [mathjaxinline]\theta[/mathjaxinline] which minimizes [mathjaxinline]\mathcal{E}_ n(\theta )[/mathjaxinline] might be done using a familiar least-squares minimization algorithm, when the model [mathjaxinline]h[/mathjaxinline] is a function being fit to some data [mathjaxinline]x[/mathjaxinline].
Sometimes we can use software that was designed, generically, to perform optimization. In many other cases, we use algorithms that are specialized for machine-learning problems, or for particular hypotheses classes.
Some algorithms are not easily seen as trying to optimize a particular criterion. In fact, the first algorithm we study for finding linear classifiers, the perceptron algorithm, has this character.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Video transcripts

### Lecture: Introduction to ML - perspective and history

LESLIE KAELBLING: Let's do machine learning then.
So here we are.
This is machine learning.
Now unless you live under a rock,
you've been reading about machine
learning in the newspaper and everything.
And so the question is what in the world is it,
how do we do it?
That's what we're going to do in this class.
And the way I think about machine learning especially
is that fundamentally, it's about getting data in some form
and aggregating it in a way that lets us
make some kind of predictions.
So it might be that we would like
to predict what's going to happen to the stock market
or to the weather, or a robot might need to predict what
would happen if it turns left.
Some kinds of predictions can also be actions.
So in the robotics centers that I work in machine
learning applied to robotics, so I use a lot of robot examples.
So it might also be that you're going
to make predictions about what would be a good action to take.
So there's all different kinds of things you could predict,
but fundamentally, for us, it's about taking data and not
so much just analyzing it and getting insight from it.
But taking data and using that data to actually
kind of do some job.
And so the question is well, you have
to describe what the job is that you want to do
and how to measure whether you're doing a good job of it.
So what's happened--
I mean, so machine learning, as so many things,
it's been the cool thing and then
kind of not so good and then cool again and not so good.
This is at least the third epoch of coolness
of machine learning as it's way cooler than it used to be,
so that's good.
And really what's stunning this time around
is that it works well and reliably
in a large set of applications.
And basically now for any problem
that involves dealing with some kind of signals, images,
or speech, or language, the only way
really that we make those applications anymore
is through using machine learning.
So a story that I like to tell is that I don't know, say,
20 years ago was maybe the beginning
of this, of machine learning taking off in image analysis.
So at that time, people were interested in finding faces
in images.
So now everybody has got a camera in their pocket,
and the camera in their pocket can find the faces
and the images no problem.
It draws little green boxes around everybody's face.
So 20 years ago, people were trying really hard to do that,
and lots of very smart people were
trying to write programs that would find faces and images.
And they would say, "Oh, well, maybe we
should look for eyes and a nose and understand
how they should be related."
Lots and lots of really smart people
did lots of work trying to find faces in images,
and mostly it didn't work very well.
And then people turned their attention
to the problem of instead of trying for the humans
to write the program to do the job,
the humans started writing the program that we're taking data,
this is an image of a face, this is an image of a not face.
Here's another face, here's another not face and pour that
into some program which the humans still have to write.
And then out from that would come
something that could tell a face from a not face.
So we're still writing programs.
And the human and the engineering input
is very important, but it's sort of now
happening at one level of abstraction up.
Instead of trying to write the program to recognize the face,
we try to write the program that can analyze the data to decide
how to recognize the face.
So it's been enormously successful


### Lecture: Introduction to ML - perspective and history


LESLIE KAELBLING: It's tempting.
There's this a kind of romantic idea about machine learning,
which is that it's this magic box,
and you pour data in one end and out
comes super awesome programs to do things.
And it's not so easy, right?
And so-- and the human has a big role
to play in setting up a machine learning program or a machine
learning solution to a problem.
And we're going to be talking about many aspects of this, not
all the aspects, but I really want
to emphasize what you have to do to get this to work.
So what do you have to do to make
a machine learning application?
Well, you have to somehow, you have to get data,
so that's not always easy.
There can be all kinds of subtleties about that
and process it in a way that a machine learning
algorithm can deal with it.
You have to think about a space of possible solutions.

So what could possibly-- if we're
going to use a computer program to find
an answer to a question, we do have
to at least think about what is the space of possible answers.
We have to characterize what makes a good solution, sort
of the objective.
So how could I tell whether one solution
is better than another?
We have to decide that, too.
We'll talk about these things in much more detail today.
I should also say there will be detailed notes posted.
They're posted already.
And so don't feel the need to slavishly write down
stuff I write on the board because it really
is all written somewhere else.
You're welcome to, but just saying that.
So we have space of possible solutions,
figure out an objective, what makes one solution better
than another one, come up with an algorithm,
so some kind of an algorithm to actually find a good solution.

Then we have to run it.
And we have to validate the results.

Really the only part of this process
that the computer does for us completely on its own is this.
So we have to approach the problem,
understand it well enough, get the data, do all these steps.
We run an algorithm, we figure out what happens.
So we'll think about how we do all these things.

So when you get data, what do you do with it?
So what are the interesting problems
involved in going from data to some kind of prediction?
So one aspect of machine learning
is actually the part of it that's
been studied by statisticians for years
and is very important.
So one part of the problem that we have to solve is estimation.

All right, so maybe I'm interested in understanding,
I don't know, what the temperature in this room is,
and I take a bunch of measurements.
And I get one that says 63.2, and another one
says 74.1, and another one that says 69.0.
And so I get all these measurements,
and I have to aggregate that data in some way that gives me,
I don't know, some estimate of the temperature in the room
and maybe some estimate of my own certainty of what's
going on.
So I might estimate--
I might look at the variance.
I might use the variance of this data
to estimate my uncertainty about the actual mean.
So those are the kinds of questions that get
answered in statistics a lot.
And these kinds of statistical questions are important to us
in machine learning, but the part that we really actually
most focus on is the problem of generalization.
And the problem of generalization
might be I measured the temperature
in this room for the last three days,
or the last three months, or the last three years,
and I would like to make a prediction now
about the temperature in this room tomorrow.
And making a prediction about the temperature in this room
tomorrow is an act of hubris.
It's an act of saying--
you're asking me to make a prediction about a situation
that I don't have any data about directly.
I have data about related situations.
I have data about what it was like in this room yesterday
and two weeks ago and so on.
But I'm going to have to use that data now
to answer a question that's a brand
new question, a question about which I
have no examples in particular.
So generalization is about going from data that's
about situations, questions, problems,
cases that are not the same as the one
that you want to make the prediction about.
And so that's super hard.
What do you think gives us a license to do that?
So there's a whole lot of work in philosophy
about generalization and induction and so on
and a lot of work in statistics trying to think about that.
But this is going to be the thing that we always
have to keep in mind is that we're getting data about one
thing, and we're using it to make predictions
about other things in general.


### Lecture: Introduction to ML - perspective and history


LESLIE KAELBLING: There is an enormous catalog.
I kind of wrote a long list in the introductory set
of the notes about different settings of machine learning.
Within this class, we'll study many of them,
but not all of them.
There's kind of too many to do in an introductory class.
I'm not going to go through that catalog right now.
What we're going to do is just dive in
and start thinking about what's probably the most typical, most
generally well understood useful setting, which is what people
call supervised learning.

So in supervised learning, the idea
is that you're given a data set, you're given some data set,
and it's organized into a set of pairs.
And I'll try to introduce notation that will
be mostly consistent about.

So in supervised learning, we're given a data set of pairs.
This x in each one of these pairs,
we can think of it as an input and the y as an output.

And so the idea is that what we want to do
is basically learn a mapping, learn some kind of relationship
so that in the future when we're given an x, we can predict a y.
So this might be these are the x is
the vital signs of a patient, and the y is whether or not
they're having a heart attack, or how many days we expect
them to be in the hospital.
The x might be an image, and the y might be whether it's a face
or whose face is it, or how face like is
it, or something like that.
So in supervised learning, it's called supervised
because we're given question answer pairs, question, answer,
question, answer, question, answer.
And we have to figure out a way from that
to answer future questions.
Typically, not always, typically, and for us,
really most of the time, the x is-- the xi will be--
you can think of them as vectors of real numbers
in D dimensions.
D could be big.
D could be small, but vectors in D dimensions.

And y can be a variety of different things.
What we're going to do in the first few weeks
here is consider the case where the y is discrete.
And in particular to start with, we're
going to consider y as in plus 1, minus 1.
So when y is in plus 1, minus 1, this
is called the classification problem.
So classification is a kind of supervised learning,
and in particular because there is only
two elements in that set, it's a binary classification problem.

So this is the setup for trying to decide whether this image
contains a face or it doesn't.
So we're given an x, and we're going to say yes or no.
So we often call this a positive example
if the y is plus and a negative example if the y is minus.
So we're given a bunch of examples, xy pairs.

But let's talk just for one more minute about the x's and where
they come from.
So I said humans had to do a bunch of work.
So one piece of the work that humans have to do
is really come up with a representation.
We'll talk about this more next week, but I just--
we have to kind of think about it already from the beginning.
So really, x's might be--
the actual x's might be patients or songs.
Now you can't take a song and put it
in the data set abstractly.
You can only-- you have to characterize it somehow.
So sometimes we'll talk about a feature mapping
so that you might say, well, I have
real x's, real secret magic x's which are people or songs.
And what I have to do is go from the person or the song
into some actual feature representation.

And it's that feature representation
that's in R to the D. The song is not
an element in a vector space.
The song is just a song, but we have
some feature representation that could take a song
and give us a vector.
And that's a thing that you will have to design.
If you're going to go out in the world
and apply machine learning to some problem,
you will have to do something to figure out
how it is that you're going to go
from songs to some kind of representation
you can put in a data set, songs, or people, or whatever,
cars.
So we will now that I've introduced this
as a thing to worry about, we're going to mostly not
talk about it.
Well, we will come back to it off and on, but I'm going
to write on the board all the time that we're mapping
x's in R to the D to y's.
But really when you think about that,
those x's come from some process that a human did,
which is to take an actual thing in the world, a day or a room,
and map it into some vector in some fixed dimensional space.
Lots of the current work in machine learning
is about doing stuff that doesn't involve
fixed dimensional vector spaces, but we, in this class,
are not going to worry about that.

So we had some domain of interest.
We have examples.
We've coded up the examples in some way
so that the inputs are elements in R to the D,
and the outputs are our plus 1 and minus 1.
So that's the setup for classification
that we're going to think about.


### Lecture: Introduction to ML - perspective and history


LESLIE KAELBLING: OK.
And so, then, the next thing we're going to do
is think about a space of possible solutions.
So what is it that we're looking for?
Ultimately, we said we wanted to make predictions.
But what we're going to do now is think
about the learning algorithm.
So to kind of think about the learning algorithm,
generally speaking, what we need to do
is come up with a hypothesis class.
So a hypothesis is going to be a guess,
an idea about the relationship between the x's and the y's.
So a hypothesis-- we'll say, hypothesis
looks like some rule that would let us take an x
and generate a y.
I'll put my parentheses over there.
Because I'll stick something in there in a minute.
So hypothesis class h is some set of possible functions.
And we're interested in these little h's.
A little h in the set of possible h's is a function
that would take an x and give me a y.
So for my particular problem, my x's are
going to be in [? r ?] to the D. And my y's
are going to be Boolean, plus or minus 1.
In other cases, they might be something else.
OK.
So hypothesis class.
Generally speaking, the way we think about hypothesis class
is by describing them in terms of some set of parameters.
And this is going to be a little too abstract.
But I'll make it concrete pretty soon.
So generally, what we do is we say,
well, here's a class of hypotheses--
a kind of a way of describing a relationship between
x's and y's--
and it has some parameters.
And these parameters will actually
pick a particular hypothesis from this [INAUDIBLE]..
So we'll talk about this form of a hypothesis.
But basically, then, what you want
to think about it is that a hypothesis is
some kind of a box.
An x comes in.
A y comes out.
And we'd like to find a hypothesis that
agrees, in some way, with our data, right,
that does a good job, somehow, with our data
and we think is going to perform well in the future.


### Lecture: Introduction to ML - perspective and history


LESLIE KAELBLING: So if we're going to try to do a good job
on our [? data-- ?] we said, OK, we
have to say what's a space of possible solutions--
the next thing we have to do is talk about what
makes one purported solution, one hypothesis
better than another one.
OK.
So before we talk about what makes a hypothesis better
than another one, we need to talk
about what makes one prediction better than another one, OK?
So let's start with that.
So there is this idea of a Loss Function.

Oh, and I have to apologize--
not really, but I'm going to--
there's some particular notation that we're using in the notes,
then, that I'll write on the board.
If you read another textbook or another blog post,
every single textbook and blog post you read
will do something mildly different.
So this is a new field which is growing like crazy.
Everybody does everything slightly differently.
And you just have to be aware of that.
So I will do things in some way that, hopefully, is internally
consistent.
But beyond that, I can't promise that's how every other book you
read will do it.
OK.
So loss function is function.
It takes in elements of the y set.
So in our case, if we're doing classification,
these two arguments will be elements
of the plus or minus 1, right?
So g is in plus 1 minus 1. a is in plus 1 minus 1.
And g is--
I use the letter g for Guess and a for Actual.

And the idea is--
excuse me-- that this is, how sad are we that we predicted g
when a was the true answer?


### Lecture: Introduction to ML - perspective and history


LESLIE KAELBLING: So we said we were trying to figure out
what makes a hypothesis good.
And so far, what we did was we said what
makes a single prediction good.
So now we have to talk about what makes a whole-- how do
we evaluate a whole hypothesis?
And so what makes a hypothesis good?
Well, the fact is that what makes a hypothesis good
is its performance.
What we would really like is small loss on new data.

All right?
So I always use studying in a class as my example for this,
right?
So you might study really hard to get a hypothesis
in your head that performs very well on your training data,
right?
So that's like learning the homework problems.
But the homework-- I tell you now, is--
the homework problems are not going to be on the test, right?
So you have to learn a hypothesis that generalizes,
right?
You have to learn a hypothesis that's
going to perform well on questions that you have never
seen before.
And so that's having a small loss on new data.
OK, so good.
So we'd like to have a hypothesis that
has small loss on new data.
But I don't know how to write down an objective function
that a computer can optimize that says, please optimize
your performance on data but I don't know what it is.
All right, so that's not an easy thing.
I don't know how to write that problem down.
So generally speaking, what we can do--
and this will not be the final answer.
This will be the answer that we work with for right now.
And then we'll sneak up on trying
to make this more like that.
But in the short term, a proxy for asking for a hypothesis
to have small loss on new data is
asking it to have small loss on the training data.

Right, so I'm writing this down as a proxy.
But I want to say, you know, not the whole story.
This isn't really the right answer.
But it's part of the right answer,
and it's good enough for right now.

So small loss on training data, this
is a thing that we can at least start to write down
on the board, right?
So we would say--
and so we might say, well, if I have n training examples,
I can define the training set error.

OK, so this is the training set error.
This is the sum over my training examples.
I said we had n training examples.
They're xy pairs.
They come in pairs.
It's important to keep them together.
So we say, OK, how good is this hypothesis?
Well, I sum up over all my training examples.
And I say, h of xi--
what is that, h of xi?
That is the prediction that my hypothesis would
make if it's given this input.
So this is the prediction that I make.

That's my guess.
And the yi is the actual.
And I say, how sad am I that I made this prediction when
that was the actual answer?
I add those all up, divide by n, that's my training error.

So we'll write it this way.
So just so that we kind of establish the idea that there
are these two quantities and they are not the same,
we can also talk about test error.

And our notation will be this.

And it's going to be the sum over some--

it's going to look sort of like the same thing.

But it's summed over some, let's say, additional chunk of data.
Right, so you might have a big set of data.
You might say, oh, well, this n to n plus--
data n to n-prime are the ones that--
oh, n plus 1 to n-prime, if we're being careful
and we're doing 1 indexing.
n plus 1 n-prime, that's the data that my learning
algorithm didn't get to see.
It's my testing data.
So really, this is what I get to work with.
But this is what I would like to optimize.
And it'll be a theme in the class.
Over and over we'll talk about how it is that optimizing this
might or might not cause us to optimize that.
But we'll focus on the training error here for now.
Yep?
STUDENT: Is there a particular reason
why you don't average in the test error, you got 1 over n?
LESLIE KAELBLING: Yeah, good.
OK, so that's a great question.
So-- oh, oh, oh, this!
Oh, that's a typo.
That's the reason why.
Good.
I thought the question was, why don't-- you could also ask,
why don't I average my training performance on my testing
performance?
And the answer is, well, you could do that.
But that's not really a good estimate
of how well you'll perform on the wild and brand-new data.
But no, that was just a typo.
So thank you.
Yes, do please--
I will make mistakes.
If I use slides, it just becomes really boring
and it doesn't give you a chance to find me--
catch me out.
So--
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: There aren't any slides, no.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: There's written notes.
There's copious written notes.
There aren't any actual slides.
Yeah.
OK, so anyway, if you find a mistake anywhere,
do let me know.
I love it when you find mistakes.


### Lecture: Introduction to ML - perspective and history


LESLIE KAELBLING: OK.
So here's our enterprise, then.
We have a data set.
We're interested, for right now, on finding
a hypothesis that will have good training [INAUDIBLE]..
All right.
So that's going to be the job that we try to do.
And so, that's the job of the learning algorithm.
So now let me write what the learning algorithm is, right?
So this is a hypothesis.
That's not a learning algorithm.
A learning algorithm is a different kind of a creature.

So a learning algorithm, you can think of it as a box.
What comes in is a data set.
And what comes out is a hypothesis, right?
And you can think of this as a learning algorithm.

And maybe you can think of it as, potentially,
being parameterized by a class of possible hypotheses
it could generate.
So learning algorithm.
That's the computer program that we're going to write, right?
We're not going to write down hypotheses, directly.
We're going to write down algorithms that consume data
sets and give out hypotheses.
So that's a learning algorithm.
Consume a data set, give out a hypothesis.
And what we'd like to do-- the way we would like to frame--
so the question is, how do we come up
with learning algorithms?
Actually, so this is a good thing.
How do we come up with learning algorithms?
So there's kind of two strategies.
And one is to be a clever human, which many of us are.

And the other one is to use optimization methods.

So the first algorithm that we-- we're
going to talk about a couple algorithms today, actually.
One is dumb.
Uh.
You can also come up with that algorithm.
So just-- you can be dumb.
So that's good.
So we're going to, today--
hah.
Good.
Awesome.
Today, we're going to, first of all,
think about an algorithm that's dumb.
My favorite thing, when given a problem,
is to come up with a really dumb algorithm.
It's a good idea.
It helps you at least be sure that you
understand your problem, even if you don't know the answer yet.
So we're going to do the dumb algorithm.
And then we're going to look at an algorithm
that a very clever human invented,
just kind of directly.
But after that, basically, all of the ways
that we think about solving machine
learning problems, coming up with machine learning
algorithms, is going to be, actually,
to use optimization methods, right?
So what we're going to do is write down
an objective function, like the training error,
come up with a hypothesis class, and then use
computational optimization methods
to try to find the best element of the hypothesis class,
according to our objective function.
So that's really kind of the standard method.
And that's where I feel like there's
some kind of a boundary--
a machine learning person has to be partly
a kind of a statistical analysis and generalization person
and partly a computer scientist, right?
And it's up until the point that you frame it as an optimization
problem that you're really kind of worrying
about the statistical and generalization aspects.
And once you frame your question as an optimization problem,
it becomes a problem in numerical algorithms.
And so, and sort of, then, your role changes a little bit.
And you become an optimization person.
Now, of course, that boundary is not rigid.
And generally, the best practitioners
of machine learning are mindful of the kinds of optimization
problems we know how to solve.
And they work at trying to formulate their machine
learning problem as an optimization
that we have good methods for.
Or they invent new methods to solve those problems
that they come up with.
But that's a kind of a way to think about the overall problem
that we try to address.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/2481f8f2964716032b134db99e369b81/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Introduction.pdf
- https://openlearninglibrary.mit.edu/assets/courseware/v1/9d904854b4ae0878cfdcedcdceabf937/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Linear_classifiers.pdf
