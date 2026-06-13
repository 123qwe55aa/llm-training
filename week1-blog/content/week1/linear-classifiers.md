# Linear classifiers

> Week 1: Basics · MIT 6.036 courseware archive

## Notes – Chapter 2: Linear classifiers

Notes – Chapter 2: Linear classifiers
You can sequence through the Linear Classifier lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 2: Linear classifiers
notes as a PDF file.

## Lecture: Linear classifiers

Lecture: Linear classifiers
Lecture: Linear classifiers
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Classification

## Classification
A binary
classifier
is
Actually, general classifiers can have a range which is any discrete set, but we'll work with this specific case for a while.
a mapping from [mathjaxinline]\mathbb {R}^ d \rightarrow \{ -1, +1\}[/mathjaxinline].
We'll often use the letter [mathjaxinline]h[/mathjaxinline] (for hypothesis) to stand for a classifier, so the classification process looks like:
[mathjax]x \rightarrow \boxed {h} \rightarrow y \; \; .[/mathjax]
Real life rarely gives us vectors of real numbers; the [mathjaxinline]x[/mathjaxinline] we really want to classify is usually something like a song, image, or person. In that case, we'll have to define a function [mathjaxinline]\varphi (x)[/mathjaxinline], whose domain is [mathjaxinline]\mathbb {R}^ d[/mathjaxinline], where [mathjaxinline]\varphi[/mathjaxinline] represents
features
of [mathjaxinline]x[/mathjaxinline], like a person's height or the amount of bass in a song, and then let the [mathjaxinline]h: \varphi (x) \rightarrow \{ -1, +1\}[/mathjaxinline]. In much of the following, we'll omit explicit mention of [mathjaxinline]\varphi[/mathjaxinline] and assume that the [mathjaxinline]x^{(i)}[/mathjaxinline] are in [mathjaxinline]\mathbb {R}^ d[/mathjaxinline], but you should always have in mind that some additional process was almost surely required to go from the actual input examples to their feature representation.
In
supervised learning
we are given a training data set of the form
[mathjax]{\cal D}_ n = \left\{ \left(x^{(1)}, y^{(1)}\right), \dots , \left(x^{(n)}, y^{(n)}\right)\right\}  \; \; .[/mathjax]
We will assume that each [mathjaxinline]x^{(i)}[/mathjaxinline] is a [mathjaxinline]d \times 1[/mathjaxinline]
column vector
. The intended meaning of this data is that, when given an input [mathjaxinline]x^{(i)}[/mathjaxinline], the learned hypothesis should generate output [mathjaxinline]y^{(i)}[/mathjaxinline].
What makes a classifier useful? That it works well on
new
data; that is, that it makes good predictions on
My favorite analogy is to problem sets. We evaluate a student's ability to
generalize
by putting questions on the exam that were not on the homework (training set).
examples it hasn't seen.
But we don't know exactly what data this classifier might be tested on when we use it in the real world. So, we have to
assume
a connection between the training data and testing data; typically, they are drawn independently from the same probability distribution.
Given a training set [mathjaxinline]{\cal D}_ n[/mathjaxinline] and a classifier [mathjaxinline]h[/mathjaxinline], we can define the
training error
of [mathjaxinline]h[/mathjaxinline] to be
[mathjaxinline]\displaystyle  \mathcal{E}_ n(h) = \frac{1}{n}\sum _{i = 1}^{n}\begin{cases}  1 &  h(x^{(i)}) \ne y^{(i)} \\ 0 &  \text {otherwise}\end{cases} \; \; .[/mathjaxinline]
For now, we will try to find a classifier with small training error (later, with some added criteria) and hope it
generalizes well
to new data, and has a small
test error
[mathjaxinline]\displaystyle  \mathcal{E}(h) = \frac{1}{n'}\sum _{i = n + 1}^{n + n'}\begin{cases}  1 &  h(x^{(i)}) \ne y^{(i)} \\ 0 &  \text {otherwise}\end{cases}[/mathjaxinline]
on [mathjaxinline]n'[/mathjaxinline] new examples that were not used in the process of finding the classifier.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Learning algorithm

Learning algorithm
A
hypothesis class
[mathjaxinline]{\cal H}[/mathjaxinline] is a set (finite or infinite) of possible classifiers, each of which represents a mapping from [mathjaxinline]\mathbb {R}^ d \rightarrow \{ -1, +1\}[/mathjaxinline].
A
learning algorithm
is a procedure that takes a data set [mathjaxinline]{\cal D}_ n[/mathjaxinline] as input and returns an element [mathjaxinline]h[/mathjaxinline] of [mathjaxinline]{\cal H}[/mathjaxinline]; it looks like
[mathjax]{\cal D}_ n \longrightarrow \boxed {\text {learning alg (${\cal H}$)}} \longrightarrow h[/mathjax]
We will find that the choice of [mathjaxinline]{\cal H}[/mathjaxinline] can have a big impact on the test error of the [mathjaxinline]h[/mathjaxinline] that results from this process. One way to get [mathjaxinline]h[/mathjaxinline] that generalizes well is to restrict the size, or “expressiveness" of [mathjaxinline]{\cal H}[/mathjaxinline].
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Linear classifiers

Linear classifiers
We'll start with the hypothesis class of
linear classifiers
. They are (relatively) easy to understand, simple in a mathematical sense, powerful on their own, and the basis for many other more sophisticated methods.
A linear classifier in [mathjaxinline]d[/mathjaxinline] dimensions is defined by a vector of parameters [mathjaxinline]\theta \in \mathbb {R}^ d[/mathjaxinline] and scalar [mathjaxinline]\theta _0 \in \mathbb {R}[/mathjaxinline]. So, the hypothesis class [mathjaxinline]{\cal H}[/mathjaxinline] of linear classifiers in [mathjaxinline]d[/mathjaxinline] dimensions is the
set
of all vectors in [mathjaxinline]\mathbb {R}^{d+1}[/mathjaxinline]. We'll assume that [mathjaxinline]\theta[/mathjaxinline] is an [mathjaxinline]d \times 1[/mathjaxinline] column vector.
Given particular values for [mathjaxinline]\theta[/mathjaxinline] and [mathjaxinline]\theta _0[/mathjaxinline], the
Let's be careful about dimensions. We have assumed that [mathjaxinline]x[/mathjaxinline] and [mathjaxinline]\theta[/mathjaxinline] are both [mathjaxinline]d \times 1[/mathjaxinline] column vectors. So [mathjaxinline]\theta ^ T x[/mathjaxinline] is [mathjaxinline]1 \times 1[/mathjaxinline], which in math (but not necessarily numpy) is the same as a scalar.
classifier is defined by
[mathjaxinline]\displaystyle  h(x; \theta , \theta _0) = \text {sign}(\theta ^ T x + \theta _0) = \begin{cases}  +1 &  \text {if $\theta ^ Tx + \theta _0 > 0$} \\ -1 &  \text {otherwise}\end{cases} \; \; .[/mathjaxinline]
Remember that we can think of [mathjaxinline]\theta , \theta _0[/mathjaxinline] as specifying a hyperplane. It divides [mathjaxinline]\mathbb {R}^ d[/mathjaxinline], the space our [mathjaxinline]x^{(i)}[/mathjaxinline] points live in, into two half-spaces. The one that is on the same side as the normal vector is the
positive
half-space, and we classify all points in that space as positive. The half-space on the other side is
negative
and all points in it are classified as negative.
Example:
Let [mathjaxinline]h[/mathjaxinline] be the linear classifier defined by [mathjaxinline]\theta = \begin{bmatrix}  -1 \\ 1.5 \end{bmatrix}, \theta _0 = 3[/mathjaxinline].
The diagram below shows several points classified by [mathjaxinline]h[/mathjaxinline]. In particular, let [mathjaxinline]x^{(1)} = \begin{bmatrix}  3 \\ 2 \end{bmatrix}[/mathjaxinline] and [mathjaxinline]x^{(2)} = \begin{bmatrix}  4 \\ -1 \end{bmatrix}[/mathjaxinline].
[mathjaxinline]\displaystyle  h(x^{(1)}; \theta , \theta _0)[/mathjaxinline]
[mathjaxinline]\displaystyle = \text {sign}\left(\begin{bmatrix}  -1 &  1.5 \end{bmatrix}\begin{bmatrix}  3 \\ 2 \end{bmatrix} + 3\right) = \text {sign}(3) = +1[/mathjaxinline]
[mathjaxinline]\displaystyle h(x^{(2)}; \theta , \theta _0)[/mathjaxinline]
[mathjaxinline]\displaystyle = \text {sign}\left(\begin{bmatrix}  -1 &  1.5 \end{bmatrix}\begin{bmatrix}  4 \\ -1 \end{bmatrix} + 3\right) = \text {sign}(-2.5) = -1[/mathjaxinline]
Thus, [mathjaxinline]x^{(1)}[/mathjaxinline] and [mathjaxinline]x^{(2)}[/mathjaxinline] are given positive and negative classfications, respectively.
Study Question:
What is green vector normal to the hyperplane? Specify it as a column vector.
Study Question:
What change would you have to make to [mathjaxinline]\theta , \theta _0[/mathjaxinline] if you wanted to have the separating hyperplane in the same place, but to classify all the points labeled '+' in the diagram as negative and all the points labeled '-' in the diagram as positive?
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Lecture: The random linear classifier algorithm

Lecture: The random linear classifier algorithm
Lecture: The random linear classifier algorithm
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Learning linear classifiers

Learning linear classifiers
Now, given a data set and the hypothesis class of linear classifiers, our objective will be to find the linear classifier with the smallest possible training error.
This is a well-formed optimization problem. But it's not computationally easy!
We'll start by considering a very
It's a good idea to think of the “stupidest possible" solution to a problem, before trying to get clever. Here's a fairly (but not completely) stupid algorithm.
simple learning algorithm.
The idea is to generate [mathjaxinline]k[/mathjaxinline] possible hypotheses by generating their parameter vectors at random. Then, we can evaluate the training-set error on each of the hypotheses and return the hypothesis that has the lowest training error (breaking ties arbitrarily).
This might be new notation: [mathjaxinline]{\rm arg}\min _{x} f(x)[/mathjaxinline] means the value of [mathjaxinline]x[/mathjaxinline] for which [mathjaxinline]f(x)[/mathjaxinline] is the smallest. Sometimes we write [mathjaxinline]{\rm arg}\min _{x \in {\cal X}} f(x)[/mathjaxinline] when we want to explicitly specify the set [mathjaxinline]{\cal X}[/mathjaxinline] of values of [mathjaxinline]x[/mathjaxinline] over which we want to minimize.
A note about notation.
Study Question:
What do you think happens to [mathjaxinline]\mathcal{E}_ n(h)[/mathjaxinline], where [mathjaxinline]h[/mathjaxinline] is the hypothesis returned by Random-Linear-Classifier, as [mathjaxinline]k[/mathjaxinline] is increased?
Study Question:
What properties of [mathjaxinline]{\cal D}_ n[/mathjaxinline] do you think will have an effect on [mathjaxinline]\mathcal{E}_ n(h)[/mathjaxinline]?
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Evaluating a learning algorithm

Evaluating a learning algorithm
How should we evaluate the performance of a
classifier
[mathjaxinline]h[/mathjaxinline]? The best method is to measure
test error
on data that was not used to train it.
How should we evaluate the performance of a
learning algorithm
? This is trickier. There are many potential sources of variability in the possible result of computing test error on a learned hypothesis [mathjaxinline]h[/mathjaxinline]:
Which particular
training examples
occurred in [mathjaxinline]{\cal D}_ n[/mathjaxinline]
Which particular
testing examples
occurred in [mathjaxinline]{\cal D}_{n'}[/mathjaxinline]
Randomization inside the learning
algorithm
itself
Generally, we would like to execute the following process multiple times:
Train on a new training set
Evaluate resulting [mathjaxinline]h[/mathjaxinline] on a testing set
that does not overlap the training set
Doing this multiple times controls for possible poor choices of training set or unfortunate randomization inside the algorithm itself.
One concern is that we might need a lot of data to do this, and in many applications data is expensive or difficult to acquire. We can re-use data with
cross validation
(but it's harder to do theoretical analysis).
It's very important to understand that cross-validation neither delivers nor evaluates a single particular hypothesis [mathjaxinline]h[/mathjaxinline]. It evaluates the
algorithm
that produces hypotheses.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:04 PM (revision 4f166135)

## Video transcripts

### Lecture: Linear classifiers


LESLIE KAELBLING: Time to get one level more specific.
So we've been talking pretty generally about algorithms
and hypotheses and so on.
So now, for about the first few weeks,
we're going to talk about--
well, or at least for the first two weeks,
we'll talk about linear classifiers.
So let's get more specific.

So a linear classifier, now, this
is really a choice of hypothesis, class.
So what we need to do, if we're doing classification,
we need to find a way to take R to the d-- big,
ole d dimensional space--
and divide it into some subspace that we call positive
and some subspace that we call negative.
And the simplest way to do that is
by putting a linear separator in there.
OK, so a linear classifier has this form.

I'll write it, and then I'll explain the whole story.

OK, so there's a bunch of things to talk
about here in what I wrote.
I'll make this easier to see.

So the first thing is what in the heck is this notation?
So many of you are not used to some kind of function
that has a semicolon in it, so let
me just talk about that first.
Right?
So what we want to do here is talk about--
so the hypothesis class, h, is the class
of all possible linear separators.
Actually, let me say what is a linear separator?
Here we go.
What is a linear separator in two dimensions?
So imagine we have two dimensions, x1 and x2, so
our points, so d is equal to 2.
And we have some positive examples,
and we have some negative examples.
We might say, ah, here's my hypothesis.
My hypothesis is this line in two dimensions,
and I'm going to think about the normal to this line.
I'm going to say, if the point is on the side
that the normal points to, then I am a positive point,
and if it's on the other side, I am a negative point.
So that is a linear classifier.
And that line, right, what line is that?
That line is the set of points for which theta
dot x plus theta naught is equal to 0.
That's what the line is.
So what is theta?
Theta is going to be a vector in R to the D,
and theta naught is going to be in R.
And just to get the dimensions right, many of you,
this will be your first experience
with writing computer programs that
do manipulations of vectors and matrices and so on.
And especially at the beginning, the dimensions
will be a little bit of a nightmare.
And we're going to get you practicing right away on using
NumPy and playing with representations of these things
and being clear about the dimensions,
but you always want to be really clear about the dimensions.
We're going to treat our data points and our theta
as column vectors.
So xi is going to be d by 1, and theta is d by 1.
So let's just check dimensions here for a minute.
Right?
So if we take theta transpose, that's 1 by d, and a 1
by d thing times a d by 1 thing is a 1
by 1 thing, that is to say a scalar.
And then, we can add another scalar to it,
and we're going to get out of scalar, right?
So this quantity here is going to be some real number,
And then sine of a real number is
going to give us plus 1 or minus 1,
depending on the sign of this thing.
So that's what that means, right?
OK.
So back to what is this crazy thing with a semicolon.
So now, what this means is, if you give me
some values for theta.
So if d's equal to 2, this is like two numbers.
You give me some values for theta and a value
for theta naught, that defines the line.

Right?
It defines the line in not high school algebra terms, right?
So if you try to remember y equals mx plus b,
just remove that from your head.
Right?
So we like theta 1, x1 plus theta 2, x2 plus theta 0
equals 0.
That's what we're going to think about is
the equation for the line.
Why?
Because it extends to high dimensions.
I'm going to draw pictures on the board in two dimensions
usually.
That's all I can manage, but all of our software,
all of our thinking, should be for big d.
d could be 1,000 or 10,000.
It's hard to think about that space.
I can think about two and three, that's usually,
but really, we're going to think about these things
in high dimensions.
This way of thinking about planes
extends to high dimensions.
So even though I draw a picture that your high school
algebra applies to, don't think about it that way.
Think about it like this.
So that's the equation for that line.

OK.
So now, if you specify a theta and a theta
naught, in my example here in two dimensions,
that's like drawing a particular lane.
And once you've drawn that particular lane,
now you can ask the question for this point x.
So now, I have a new point.
I have a new x here, and I want to ask the question is my x,
I'm going to predict a positive value or a negative value
for this x?
And the way I'm going to answer that question is I'm
going to take the coordinates of this point x,
take the dot product with my theta and theta 0,
and ask if it's bigger than 0 or not.

Does that makes sense?
I take questions about this kind of--
there's a ton of practice exercises and problems
and so on to just help think about hyperplanes and normals
and stuff like that.
But fundamentally, these things on this side of the semicolon
define the particular hypothesis.
So you could say, actually, h, linear classifiers defines
the hypothesis class.
Once I pick the theta and the theta naught,
I've picked a particular hypothesis-- a particular line,
a particular hyperplane.
And now, given those choices of parameters,
I can ask, for this new input, would I
predict that it's positive or negative?
So that's the setup.


### Lecture: Linear classifiers


LESLIE KAELBLING: We're going to talk about two algorithms
for finding a linear classifier, given a data set.
So given the data set D, what H--
So really, if you think about it this way, what--
once I've specified the big class script H,
which is linear classifiers--
what specifies the particular h in this case
is really this vector theta and theta naught.
That specifies the hypothesis--
the particular hypothesis.

All right, so let's talk about an algorithm.
So I said there were several ways to do this,
and we're going to start with [INAUDIBLE],,
my favorite method.
OK, so here's an algorithm.
Now, this is our first machine learning algorithm,
and you'll implement it in the homework assignment--
random linear classifier.

It takes in a data set and a K. I'll explain K in a minute.
OK, so what's it going to do?
I'll write it down.
I won't talk about it.

I have to put a superscript here-- theta J. OK,
there we go.
So there's our first machine learning algorithm.
Let's talk about it.

OK, so what are we going to do?
First of all what, this algorithm takes as input
is two things.
And already, there's something interesting about the two
things.
So this D is a data set.
It's a training data set, like the kind that we've
talked about already.
K is an integer--
positive integer-- and it's a kind of a thing
that we call a hyperparameter.
So this is cool already machine learning is cool.
Regular old computer programming has parameters,
but we have hyperparameters.

And by hyperparameter, what we mean
is that it's something that affects how the machine
learning algorithm works.
It's a parameter of the machine learning algorithm.
It's not a priority of the hypothesis.
It's a parameter of the machine learning algorithm.

So we'll see what role k plays here, after we
talk through the algorithm.
So what we're going to do is we're
going to go around K times.
And K times, we're going to pick a random theta and theta
naught.
So we're going to try this one, and that one, and this one.
We're going to pick a bunch of crazy separators
by using a random number generator.

And then what are we going to do?
So partly, I like to teach you guys
this algorithm because this is the kind of notation
that some people haven't run into, and we use it a lot.
So arg min-- so now, OK, we generated
a bunch of hypotheses.
And now, well, what do you say?
Our job is to try to do as well as we can to minimize
our error on the training data.
So if we want to minimize our error on the training data,
and we just generated a bunch of random hypotheses, what we're
going to do is we're going to compute the training set
error--
the training error for each of our hypotheses,
and then let j* be the index--
the value of j that makes this as small as possible.
So j* is the index of the best of these guys,
and then that's the one we're going to return right.
How does this work?
We generate a bunch of random hypotheses,
we see which one scores the best, and we return it.
Cool.
So what should happen, as k gets bigger and bigger and bigger?
STUDENT: It'll give us the optimum.
LESLIE KAELBLING: We should get a better answer,
and eventually, we should hit the optimum.

Here's a plot.
You might imagine that, as we increase k, and if we plot,
En--

well, if we only try one, it's not going to work very well.
But as we try more, it's going to work better,
and eventually it's not going to work any better, let's say--
something like that.
So k is a parameter.
It's a parameter that governs how well this algorithm is
going to work.
If we make it really, really, really big,
we'll get a good answer.
But if we don't, we might not.
Yeah?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Oh, man.
Awesome.
Thank you.

Yeah?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Say that again.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: It depends.

Notice, the loss function comes to the party here twice.
The loss function is embedded in our training error.
Our training or depends on our loss function.
So this says, pick the hypothesis
that has the lowest average loss on our training data.
So it's going to come here in playing
a role in which hypothesis we pick,
and then it's kind of come here when we draw this plot,
because this is what we actually care about,
at least for right now.
OK, so there's an algorithm--
machine learning algorithm-- certified, not broken--
maybe not the best thing, but not totally crazy.
So we're going to ask you guys to implement this machine
learning algorithm this week.
But it has the right shape.
It has the right kind of input.
It's the right kind of outputs.
It does the job.
One thing-- those of you who are sticklers for notation--
I know there are a few of you in here--
let me say something.
So I defined error on a hypothesis base.
And here, I've written error in terms of these parameters.
But this is just slight laziness.
We say that the error here is really the error of h.
And this lets me bring up another notation
all conundrum, which is why I don't write it that way.
Statisticians use this notation with the little dot
in the center.
You might run into it.
And it means the function of this argument
that you get by filling in the other parameters like this.
Computer scientists have a better notation for this.

But this is-- only computer scientists
know the lambda thing, and not all computer scientists.
Every computer scientist should know the lambda thing.
If you don't know the lambda thing, we will teach it to you.
But anyway, what really--
the question is, how do I name a hypothesis?
How do I name the hypothesis that I get,
when I specify a theta and a theta naught.
So this is one way to name that function, that hypothesis.
This is another way to name it.
Here, we were slightly lazy, and just put the parameters
in there.
Yep?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: That's a great question.
So whenever anybody draws a plot,
it's really easy to nod your head.
But it is really good to not nod your head and say, why?
And the question was, why would it look like this?
It seems like, if you have a few points,
it's easier than if you have a lot of points.
Notice what I'm plotting here.
I am keeping the same training set the whole time,
and I am measuring how good my hypothesis is
on that training set.
What I'm varying here is this hyperparameter of my algorithm.
So we could say, you know what, you
can run this crazy algorithm that
generates one hypothesis at random for k equals 1,
and returns it.
Now, if I generate one hypothesis at random
and return it, it's going to suck.
But if I generate 100,000, the best one might be good.
So this is a function of how many I try.
Good.
I'm glad you were looking at that.

So that's our not so smart but not totally stupid algorithm.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/9d904854b4ae0878cfdcedcdceabf937/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Linear_classifiers.pdf
- https://openlearninglibrary.mit.edu/assets/courseware/v1/dc2ca4ef11bd7be3faec0efc56f46cca/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Linear_classifers.pdf
