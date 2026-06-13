# The Perceptron

> Week 2: Perceptrons · MIT 6.036 courseware archive

## Notes – Chapter 3: Perceptron

Notes – Chapter 3: Perceptron
You can sequence through the Perceptron lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 3: Perceptron
notes as a PDF file.

## Lecture: The perceptron algorithm

Lecture: The perceptron algorithm
Lecture: The perceptron algorithm
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## An algorithm by any other name...

An algorithm by any other name...
First of all, the
Well, maybe “neocognitron," also the name of a real ML algorithm, is cooler.
coolest algorithm name!
It is based on the 1943 model of neurons made by McCulloch and Pitts and by Hebb. It was developed by Rosenblatt in 1962. At the time, it was not interpreted as attempting to optimize any particular criteria; it was presented directly as an algorithm. There has, since, been a huge amount of study and analysis of its convergence properties and other aspects of its behavior.
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:30:57 PM (revision f808f068e)

## Lecture: The perceptron algorithm in action - an example

Lecture: The perceptron algorithm in action - an example
Lecture: The perceptron algorithm in action - an example
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Algorithm

Algorithm
Recall that we have a training dataset [mathjaxinline]{\cal D}_ n[/mathjaxinline] with [mathjaxinline]x \in \mathbb {R}^ d[/mathjaxinline], and [mathjaxinline]y\in \{ -1, +1\}[/mathjaxinline]. The Perceptron algorithm trains a binary classifier [mathjaxinline]h(x; \theta , \theta _0)[/mathjaxinline] using the following algorithm to find [mathjaxinline]\theta[/mathjaxinline] and [mathjaxinline]\theta _0[/mathjaxinline] using [mathjaxinline]\tau[/mathjaxinline]
We use Greek letter [mathjaxinline]\tau[/mathjaxinline] here instead of [mathjaxinline]T[/mathjaxinline] so we don't confuse it with transpose!
iterative steps:
Let's check dimensions. Remember that [mathjaxinline]\theta[/mathjaxinline] is [mathjaxinline]d \times 1[/mathjaxinline], [mathjaxinline]x^{(i)}[/mathjaxinline] is [mathjaxinline]d \times 1[/mathjaxinline], and [mathjaxinline]y^{(i)}[/mathjaxinline] is a scalar. Does everything match?
note
Intuitively, on each step, if the current hypothesis [mathjaxinline]\theta , \theta _0[/mathjaxinline] classifies example [mathjaxinline]x^{(i)}[/mathjaxinline] correctly, then no change is made. If it classifies [mathjaxinline]x^{(i)}[/mathjaxinline] incorrectly, then it moves [mathjaxinline]\theta , \theta _0[/mathjaxinline] so that it is “closer" to classifying [mathjaxinline]x^{(i)}, y^{(i)}[/mathjaxinline] correctly.
Note that if the algorithm ever goes through one iteration of the loop on line 4 without making an update, it will never make any further updates (verify that you believe this!) and so it should just terminate at that point.
Study Question:
What is true about [mathjaxinline]\mathcal{E}_ n[/mathjaxinline] if that happens?
Example:
Let [mathjaxinline]h[/mathjaxinline] be the linear classifier defined by [mathjaxinline]\theta ^{(0)} = \begin{bmatrix}  1 \\ -1 \end{bmatrix}, \theta _0^{(0)} = 1[/mathjaxinline]. The diagram below shows several points classified by [mathjaxinline]h[/mathjaxinline]. However, in this case, [mathjaxinline]h[/mathjaxinline] (represented by the bold line) misclassifies the point [mathjaxinline]x^{(1)} = \begin{bmatrix}  1 \\ 3 \end{bmatrix}[/mathjaxinline] which has label [mathjaxinline]y^{(1)} = 1[/mathjaxinline]. Indeed,
[mathjax]y^{(1)}\left(\theta ^ T x^{(1)} + \theta _0\right) = \begin{bmatrix}  1 &  -1 \end{bmatrix} \begin{bmatrix}  1 \\ 3 \end{bmatrix} + 1 = -1 < 0[/mathjax]
By running an iteration of the Perceptron algorithm, we update
[mathjax]\theta ^{(1)} = \theta ^{(0)} + y^{(1)}x^{(1)} = \begin{bmatrix}  2 \\ 2 \end{bmatrix}[/mathjax]
[mathjax]\theta _0^{(1)} = \theta _0^{(0)} + y^{(1)} = 2[/mathjax]
The new classifier (represented by the dashed line) now correctly classifies that point, but now makes a mistake on the negatively labeled point.
A really important fact about the perceptron algorithm is that, if there is a linear classifier with 0 training error, then this algorithm will (eventually) find it! We'll look at a proof of this in detail, next.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:15 PM (revision 4f166135)

## Lecture: Evaluating learning algorithms - validation

Lecture: Evaluating learning algorithms - validation
Lecture: Evaluating learning algorithms - validation
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Offset

Offset
Sometimes, it can be easier to implement or analyze classifiers of the form
[mathjaxinline]\displaystyle  h(x; \theta ) = \begin{cases}  +1 &  \text {if } \theta ^ Tx > 0 \\ -1 &  \text {otherwise.} \end{cases}[/mathjaxinline]
Without an explicit offset term ([mathjaxinline]\theta _0[/mathjaxinline]), this separator must pass through the origin, which may appear to be limiting. However, we can convert any problem involving a linear separator
with
offset into one with
no
offset (but of higher dimension)!
Consider the [mathjaxinline]d[/mathjaxinline]-dimensional linear separator defined by [mathjaxinline]\theta = \begin{bmatrix}  \theta _1 &  \theta _2 &  \cdots &  \theta _ d \end{bmatrix}[/mathjaxinline] and offset [mathjaxinline]\theta _0[/mathjaxinline].
to each data point [mathjaxinline]x \in {\cal D}[/mathjaxinline], append a coordinate with value +1, yielding
[mathjax]x_{\rm new} = \begin{bmatrix}  x_1 &  \cdots &  x_ d &  +1 \end{bmatrix}^ T[/mathjax]
define
[mathjax]\theta _{\rm new} = \begin{bmatrix}  \theta _1 &  \cdots &  \theta _ d &  \theta _0 \end{bmatrix}^ T[/mathjax]
Then,
[mathjaxinline]\displaystyle  \theta _{\rm new}^ T \cdot x_{\rm new}[/mathjaxinline]
[mathjaxinline]\displaystyle = \theta _1x_1 + \cdots + \theta _ dx_ d + \theta _0 \cdot 1[/mathjaxinline]
[mathjaxinline]\displaystyle = \theta ^ Tx + \theta _0[/mathjaxinline]
Thus, [mathjaxinline]\theta _{\rm new}[/mathjaxinline] is an equivalent ([mathjaxinline](d+1)[/mathjaxinline]-dimensional) separator to our original, but with no offset.
Consider the data set:
[mathjaxinline]\displaystyle  X[/mathjaxinline]
[mathjaxinline]\displaystyle  = [[1], [2], [3], [4]][/mathjaxinline]
[mathjaxinline]\displaystyle Y[/mathjaxinline]
[mathjaxinline]\displaystyle  = [[+1], [+1], [-1], [-1]][/mathjaxinline]
It is linearly separable in [mathjaxinline]d = 1[/mathjaxinline] with [mathjaxinline]\theta = [-1][/mathjaxinline] and [mathjaxinline]\theta _0 = 2.5[/mathjaxinline]. But it is not linearly separable through the origin! Now, let
[mathjax]X_{\rm new} = \begin{bmatrix} \begin{bmatrix}  1 \\ 1 \end{bmatrix}\begin{bmatrix}  2 \\ 1 \end{bmatrix}\begin{bmatrix}  3 \\ 1 \end{bmatrix}\begin{bmatrix}  4 \\ 1 \end{bmatrix}\end{bmatrix}[/mathjax]
This new dataset is separable through the origin, with [mathjaxinline]\theta _{\rm new} = [-1, 2.5]^ T[/mathjaxinline].
We can make a simplified version of the perceptron algorithm if we restrict ourselves to separators
We list it here because this is the version of the algorithm we'll study in more detail.
through the origin:
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:15 PM (revision 4f166135)

## Lecture: Perceptron - overview of plan

Lecture: Perceptron - overview of plan
Lecture: Perceptron - overview of plan
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Perceptron through origin algorithm

Lecture: Perceptron through origin algorithm
Lecture: Perceptron through origin algorithm
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Theory of perceptron - Linear separability

Lecture: Theory of perceptron - Linear separability
Lecture: Theory of perceptron - Linear separability
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Theory of perceptron - margin of a dataset

Lecture: Theory of perceptron - margin of a dataset
Lecture: Theory of perceptron - margin of a dataset
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Perceptron convergence theorem

Lecture: Perceptron convergence theorem
Lecture: Perceptron convergence theorem
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Proof sketch of the perceptron convergence theorem

Lecture: Proof sketch of the perceptron convergence theorem
Lecture: Proof sketch of the perceptron convergence theorem
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Theory of the perceptron

Theory of the perceptron
Now, we'll say something formal about how well the perceptron algorithm really works. We start by characterizing the set of problems that can be solved perfectly by the perceptron algorithm, and then prove that, in fact, it can solve these problems. In addition, we provide a notion of what makes a problem difficult for perceptron and link that notion of difficulty to the number of iterations the algorithm will take.
Linear separability
A training set [mathjaxinline]{\cal D}_ n[/mathjaxinline] is
linearly separable
if there exist [mathjaxinline]\theta , \theta _0[/mathjaxinline] such that, for all [mathjaxinline]i = 1, 2, \dots , n[/mathjaxinline]:
[mathjax]y^{(i)}\left(\theta ^ Tx^{(i)} + \theta _0\right) > 0 \; \; .[/mathjax]
Another way to say this is that all predictions on the training set are correct:
[mathjax]h(x^{(i)}; \theta , \theta _0) = y^{(i)} \; \; .[/mathjax]
And, another way to say this is that the training error is zero:
[mathjax]\mathcal{E}_ n(h) = 0 \; \; .[/mathjax]
Convergence theorem
The basic result about the perceptron is that, if the training data [mathjaxinline]{\cal D}_ n[/mathjaxinline] is linearly separable, then the perceptron algorithm is guaranteed to find
If the training data is
not
linearly separable, the algorithm will not be able to tell you for sure, in finite time, that it is not linearly separable. There are other algorithms that can test for linear separability with run-times [mathjaxinline]O(n^{d/2})[/mathjaxinline] or [mathjaxinline]O(d^{2n})[/mathjaxinline] or [mathjaxinline]O((n^{d-1}\log {n})[/mathjaxinline].
a linear separator.
We will more specifically characterize the linear separability of the dataset by the
margin
of the separator. We'll start by defining the margin of a point with respect to a hyperplane.
First, recall that the distance of a point [mathjaxinline]x[/mathjaxinline] to the hyperplane [mathjaxinline]\theta , \theta _0[/mathjaxinline] is
[mathjax]\frac{\theta ^ Tx + \theta _0}{\left\lVert \theta \right\rVert }\; \; .[/mathjax]
Then, we'll define the
margin
of a
labeled point
[mathjaxinline](x, y)[/mathjaxinline] with respect to hyperplane [mathjaxinline]\theta , \theta _0[/mathjaxinline] to be
[mathjax]y \cdot \frac{\theta ^ Tx + \theta _0}{\left\lVert \theta \right\rVert }\; \; .[/mathjax]
This quantity will be positive if and only if the point [mathjaxinline]x[/mathjaxinline] is classified as [mathjaxinline]y[/mathjaxinline] by the linear classifier represented by this hyperplane.
Study Question:
What sign does the margin have if the point is incorrectly classified? Be sure you can explain why.
Now, the
margin
of a
dataset
[mathjaxinline]{\cal D}_ n[/mathjaxinline] with respect to the hyperplane [mathjaxinline]\theta , \theta _0[/mathjaxinline] is the
minimum
margin of any point with respect to [mathjaxinline]\theta , \theta _0[/mathjaxinline]:
[mathjax]\min _ i \left(y^{(i)} \cdot \frac{\theta ^ T x^{(i)} + \theta _0}{\left\lVert \theta \right\rVert }\right)\; \; .[/mathjax]
The margin is positive if and only if all of the points in the data-set are classified correctly. In that case (only!) it represents the distance from the hyperplane to the closest point.
Example:
Let [mathjaxinline]h[/mathjaxinline] be the linear classifier defined by [mathjaxinline]\theta = \begin{bmatrix}  1 \\ -1 \end{bmatrix}, \theta _0 = 1[/mathjaxinline].
The diagram below shows several points classified by [mathjaxinline]h[/mathjaxinline], one of which is misclassified. We compute the margin for each point:
[mathjax]y^{(1)} \cdot \frac{\theta ^ T x^{(1)} + \theta _0}{\left\lVert \theta \right\rVert } = 1 \cdot \frac{-2 + 1}{\sqrt {2}} = -\frac{\sqrt {2}}{2}[/mathjax]
[mathjax]y^{(2)} \cdot \frac{\theta ^ T x^{(2)} + \theta _0}{\left\lVert \theta \right\rVert } = 1 \cdot \frac{1 + 1}{\sqrt {2}} = \sqrt {2}[/mathjax]
[mathjax]y^{(3)} \cdot \frac{\theta ^ T x^{(3)} + \theta _0}{\left\lVert \theta \right\rVert } = -1 \cdot \frac{-3 + 1}{\sqrt {2}} = \sqrt {2}[/mathjax]
Note that since point [mathjaxinline]x^{(1)}[/mathjaxinline] is misclassified, its margin is negative. Thus the margin for the whole data set is given by [mathjaxinline]-\frac{\sqrt {2}}{2}[/mathjaxinline].
Theorem:
[Perceptron Convergence] For simplicity, we consider the case where the linear separator must pass through the origin. If the following conditions hold:
there exists [mathjaxinline]\theta ^*[/mathjaxinline] such that [mathjaxinline]y^{(i)} \frac{\theta ^{*T}x^{(i)}}{\left\lVert \theta ^*\right\rVert } \geq \gamma[/mathjaxinline] for all [mathjaxinline]i = 1, \ldots , n[/mathjaxinline] and for some [mathjaxinline]\gamma > 0[/mathjaxinline] and
all the examples have bounded magnitude: [mathjaxinline]\left\lVert x^{(i)}\right\rVert  \leq R[/mathjaxinline] for all [mathjaxinline]i = 1, \ldots n[/mathjaxinline],
then the perceptron algorithm will make at most [mathjaxinline]\left(\frac{R}{\gamma } \right)^2[/mathjaxinline] mistakes.
We initialize [mathjaxinline]\theta ^{(0)} = 0[/mathjaxinline], and let [mathjaxinline]\theta ^{(k)}[/mathjaxinline] define our hyperplane after the perceptron algorithm has made [mathjaxinline]k[/mathjaxinline] mistakes. We are going to think about the angle between the hypothesis we have now, [mathjaxinline]\theta ^{(k)}[/mathjaxinline] and the assumed good separator [mathjaxinline]\theta ^*[/mathjaxinline]. Since they both go through the origin, if we can show that the angle between them is decreasing usefully on every iteration, then we will get close to that separator.
So, let's think about the [mathjaxinline]\cos[/mathjaxinline] of the angle between them, and recall, by the definition of dot product:
[mathjax]\cos \left(\theta ^{(k)}, \theta ^*\right) = \frac{\theta ^{(k)} \cdot \theta ^*}{ \left\lVert \theta ^*\right\rVert \left\lVert \theta ^{(k)}\right\rVert }[/mathjax]
We'll divide this up into two factors,
[mathjax]\cos \left(\theta ^{(k)}, \theta ^*\right) = \left(\frac{\theta ^{(k)} \cdot \theta ^*}{\left\lVert \theta ^*\right\rVert }\right)\cdot \left(\frac{1}{\left\lVert \theta ^{(k)}\right\rVert }\right)\; \; ,[/mathjax]
(1.1)
and start by focusing on the first factor.
Without loss of generality, assume that the [mathjaxinline]k^{th}[/mathjaxinline] mistake occurs on the [mathjaxinline]i^{th}[/mathjaxinline] example [mathjaxinline]\left(x^{(i)}, y^{(i)}\right)[/mathjaxinline].
[mathjaxinline]\displaystyle  \frac{\theta ^{(k)} \cdot \theta ^*}{\left\lVert \theta ^*\right\rVert }[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{\left(\theta ^{(k-1)} + y^{(i)}x^{(i)}\right)\cdot \theta ^*}{\left\lVert \theta ^*\right\rVert }[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{\theta ^{(k-1)}\cdot \theta ^*}{\left\lVert \theta ^*\right\rVert } + \frac{ y^{(i)}x^{(i)}\cdot \theta ^*}{\left\lVert \theta ^*\right\rVert }[/mathjaxinline]
[mathjaxinline]\displaystyle \geq \frac{\theta ^{(k-1)}\cdot \theta ^*}{\left\lVert \theta ^*\right\rVert } + \gamma[/mathjaxinline]
[mathjaxinline]\displaystyle \geq k\gamma[/mathjaxinline]
where we have first applied the margin condition from [mathjaxinline](a)[/mathjaxinline] and then applied simple induction.
Now, we'll look at the second factor in equation
. We note that since [mathjaxinline]\left(x^{(i)},y^{(i)}\right)[/mathjaxinline] is classified incorrectly, [mathjaxinline]y^{(i)}\left({\theta ^{(k-1)}}^ Tx^{(i)}\right) \leq 0[/mathjaxinline]. Thus,
[mathjaxinline]\displaystyle  \left\lVert \theta ^{(k)}\right\rVert ^2[/mathjaxinline]
[mathjaxinline]\displaystyle = \left\lVert \theta ^{(k-1)} + y^{(i)}x^{(i)}\right\rVert ^2[/mathjaxinline]
[mathjaxinline]\displaystyle = \left\lVert \theta ^{(k-1)}\right\rVert ^2 + 2y^{(i)} {\theta ^{(k-1)}}^ Tx^{(i)} + \left\lVert x^{(i)}\right\rVert ^2[/mathjaxinline]
[mathjaxinline]\displaystyle \leq \left\lVert \theta ^{(k-1)}\right\rVert ^2 + R^2[/mathjaxinline]
[mathjaxinline]\displaystyle \leq kR^2[/mathjaxinline]
where we have additionally applied the assumption from [mathjaxinline](b)[/mathjaxinline] and then again used simple induction.
Returning to the definition of the dot product, we have
[mathjax]\cos \left(\theta ^{(k)}, \theta ^*\right) = \frac{\theta ^{(k)} \cdot \theta ^*}{\left\lVert \theta ^{(k)}\right\rVert  \left\lVert \theta ^*\right\rVert } = \left(\frac{\theta ^{(k)} \cdot \theta ^*}{\left\lVert \theta ^*\right\rVert }\right) \frac{1}{\left\lVert \theta ^{(k)}\right\rVert } \geq (k\gamma )\cdot \frac{1}{\sqrt {k}R} = \sqrt {k}\cdot \frac{\gamma }{R}[/mathjax]
Since the value of the cosine is at most 1, we have
[mathjaxinline]\displaystyle  1[/mathjaxinline]
[mathjaxinline]\displaystyle \geq \sqrt {k} \cdot \frac{\gamma }{R}[/mathjaxinline]
[mathjaxinline]\displaystyle k[/mathjaxinline]
[mathjaxinline]\displaystyle \leq \left(\frac{R}{\gamma }\right)^2.[/mathjaxinline]
This result endows the margin [mathjaxinline]\gamma[/mathjaxinline] of [mathjaxinline]{\cal D}_ n[/mathjaxinline] with an operational meaning: when using the Perceptron algorithm for classification, at most [mathjaxinline](R/\gamma )^2[/mathjaxinline] classification errors will be made, where [mathjaxinline]R[/mathjaxinline] is an upper bound on the magnitude of the training vectors.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:15 PM (revision 4f166135)

## Video transcripts

### Lecture: The perceptron algorithm


LESLIE KAELBLING: Let's try a smarter algorithm.
And where this algorithm comes from
is a clever human, named Rosenblatt.
And one of the best things about this algorithm-- oh, it's
good for a bunch of reasons-- but one of the best things
is its name.

Like, the Perceptron.
And it comes from the 50s, right?
The perceptron is such a 50s name.
And it's so cool.
So we're going to study the perceptron algorithm.
The perceptron algorithm is a machine learning algorithm.
It has the same kind of spec as that one over there.
It also takes in a training data set.
And it also has a hyperparameter, this one, T.
And it's also going to give out a hypothesis.
It's going to give out a linear classifier.
It's going to give out a theta and a theta naught.
But it has a really different character.
OK.
So here, let's do it.
So we start by initializing theta to the vector of 0's.
The fact is you can initialize it any way you want to.
But you might as well initialize it to be 0.
And theta naught is just going to be 0.

How long is this vector of 0's?
D, right?
D is the dimension of our data.
D is the dimension of our theta.
It's not how many data points we have.
It's what space those data points live in.
OK.
So theta is in D.
OK.
And so, now, we're going to write a For loop--
we're going to write two For loops.
So For t is 1 to T and For i is 1 to n.
N is the number of examples in our training set.

What are we going to do?
OK.
So if yi times theta transpose x theta naught--

And then when it's all done, we're
going to return theta and theta naught.

OK.
All right.
Chalk on the board.
What does it mean?
We'll have to think about it.
And I'll do a demo, which will, hopefully, also
make it clearer.
All right.
So what does it mean?

OK.
So first of all, that bigger outer structure
is that t times, we're going to go through our data set.
That's this.
And then this is going through the data set.
So we just have our data in a file, let's say.
And we're going to go through one example at a time.
And this is, like, the most important thing
to have a good intuition for.
So let's just unpack this expression.
So first, let me ask you, what is that?
STUDENT: Classifier.
LESLIE KAELBLING: It's a classifier.
In that particular output, how can we
interpret that particular value?
How do we interpret the value of that?

Someone is speaking something interesting.
But I can't hear it.

OK.
I don't know.
There's some motion here.
Some people have some ideas.
But I can't quite hear them.
OK.
What is this?
This is the inner piece, here, right?
It's the predict-- in particular--
it's for this particular-- so we're working through our file,
right?
We're working through the data.
Right now, we have our finger on example i.
And this is the stuff, right, the thing
that we're going to take the sign of to make our prediction,
right?
And in particular, if this stuff in the box
is positive, then the current hypothesis,
right-- so you can think of, at any given
moment in this algorithm, you have a current hypothesis.
The current hypothesis is defined by theta and theta
naught, right?
So right at this moment, I have a hypothesis.
I have a particular separator.
And I'm asking, well, is this point
x, is it on the positive side or the negative side
of my separator, right?
So if this quantity is positive, then what
we would do for this particular training example, x,
is we would predict plus 1 if this is positive?
You buy that?
And we would predict minus 1 if it's negative.
OK, cool.
So the sign of this is the prediction that we would make.
What's yi?

It's the correct prediction.
It's the prediction we should make.
If we're right, then this is positive and this is positive,
or this is negative and this is negative.
In either case, if we're right, this whole thing
is positive, right?
Either the answer should be negative.
And we're saying it's negative.
Or it should be positive.
And we're saying it's positive.
So if we're right, we are happy.
Everything is awesome.
This If statement doesn't trigger.
And we go to the next point.
Good.
Ah, but if we're unhappy--
if we're unhappy, then we have to do something.
And what we do is we take our theta
and we add to it the product of this x and its label, right?
So y is either plus 1 or minus 1.
So we take our x, the current input,
multiply it by plus or minus 1, and add it to the theta, OK?
And we take the y, and we add it to the theta naught.
Now this is totally not intuitive.
I don't expect you to sit in your chairs
and say, oh, of course, that's obviously the right thing
to do in this case.
It's really hard, actually, to understand that that's
the right thing to do.
Next time, we'll actually go through a bit
of a proof that argues why this is a good thing to do.
But it takes some thinking.
And it required a very clever human--
a very clever human--
to come up with this algorithm.
And the way this piece of history worked
was that the clever human, Rosenblatt, came up
with this algorithm.
And then there's been 30 or 40 years of scholarly papers
analyzing it and understanding why it works
and so on and so forth.
So, clever algorithm, lots of analysis.
Most of the things we look at, as I
said before, we'll actually sort of go the other way.
We'll write down an optimization criterion
and then derive the algorithm from the problem
we're trying to solve.
This one, very interestingly, states the algorithm.
And then it leaves it to us to actually prove
that it does good stuff.
OK.
So this is the perceptron algorithm.
What can we say about it?
Well, let me ask you guys a question.
What if I have a situation where--

like that?
Imagine that this is the separator that's
described by my current theta and theta naught.

What will the algorithm do from here on out?
Nothing, right?
It says, if you--
this hypothesis, this one, that one, any of those guys,
they're going to--
oh, let's see.
You shouldn't really let me draw a line without a normal.
Because I'm not promising which direction
is the positive direction.

So all those hypotheses get all those data points, right?
The positives are on the positive side,
the negatives are on the negative side.
If you happen to get one of those,
this thing will just buzz around.
So if you're implementing this, you
should write a little test, a little extra work that says,
if I've gone through this loop once
and I haven't made any updates, I can quit now.

And so here's the thing, there is a theorem,
which we'll look at next time, which
says that if your t is big enough
and your data can be perfectly classified
by a linear classifier, then this algorithm will find it,
which is kind of awesome.


### Lecture: The perceptron algorithm


LESLIE KAELBLING: Let's watch it work.
All right, so I hope you can see.
What you should be able to see is
that there are three positive points and three
negative points.
The positive points are plus and green.
The negative are minus and red.
And there's currently a point with the circle on it.
Also drawn on here right now is the current theta
and theta naught.
So I'm going to run the perceptron algorithm now.
When the perceptron algorithm wakes up, theta is equal to 0
and theta naught is equal to 0.
And that isn't even the thing that I can plot on the screen,
so it's not plotting.
And you'll also notice that the first point is always
considered a mistake.
No matter what the first point is,
no matter what the first training example in your file
is, if theta is 0 and theta naught is 0,
then this is going to be 0, and this is going to be 0,
and this is going to be less than or equal to 0.
So on the very first example you encounter,
you're going to do an update.
And then after that, it'll just run as shown.
So here, the first point in our file
is the one that's got the circle on it.
And so we've just gotten that wrong, because our input--
our current theta is all 0s.
So I do an update and I get that separator.
So I'm drawing the hyperplane that's
defined by the theta, and the theta naught, and the normal.
And it makes two mistakes, this one.
It gets two examples wrong.
You can see it's got one plus on the wrong side and a minus
on the wrong side.
So we going to come to another point.
That's the next one that we get wrong.
That one should be positive, but it's negative.
We do an update.
We get this.
Gets this wrong.
That one's negative.
It's currently classified as positive,
but it should be negative.
That rotates the guy around.
We're getting-- barely getting now one wrong.
And now, from here on out, we get it right.
So that worked out pretty nicely.
Let's do another one.

So here's another data set.
We get that one wrong.
Hypothesis-- wrong.
It's hard to predict exactly what's going to happen, but--
there we go.
It got it all right.
So it's a very simple algorithm--
easy to implement-- interestingly,
difficult to analyze.
But we can get a lot out of studying it.
A couple semesters ago, I was teaching this class,
and a student came into a lab and said, I talked to my friend
my friend says no one ever uses the perceptron algorithm.
So why are you teaching it to us?
So I was a tiny bit taken aback, but the answer
is that it's actually surprisingly
good in a bunch of cases, which we'll talk about.
And also, it gives us a lot of insight into what's going on.


### Lecture: The perceptron algorithm


LESLIE KAELBLING: So the last thing
that I wanted to talk about is how we decide--
just kind of, again, a little bit of high level
methodology, just to kind of close the loop here.
We've talked about two different machine learning algorithms,
now, right?
We've got Random Linear Classifier
which has a hyperparameter k.
And we have perceptron, which has a parameter t.
So you might imagine that if you have
a lot of data or it's in high dimensions
or you have kind of a difficult problem
that you might have to go through the data set
a lot of times.
And so, t, again, might govern how well
the answer that you get out is going to work, right?
So we have these two algorithms.
Each algorithm has a parameter.
And again, as we go forward in this class,
we'll have many more algorithms.
And those algorithms will have many more hyperparameters.
And it's our job.
If you look back over here, we said, OK, we're
going to pick a space of possible solutions.
We're going to characterize our objective.
And then we had to decide what algorithm to use.
And then we're going to run the algorithm.
And we have to validate the results.
So I'm going to talk about both of those pieces just briefly.
So what about validating the results?
So what did we just do?
We just trained an algorithm on a training set of data, right?
That's like the homework problems.
We said, OK, algorithm, learn these homework problems.
The algorithm ran.
And it gave out an answer, theta, theta naught,
in both of these cases.
But now we might want to characterize--
imagine, we're working now, we started our startup,
and we ran the perceptron, and it gave out this hypothesis.
And we would like to try to certify to the person who's
paying us to solve their problem how well we expect
that hypothesis to work in their actual application, right?
So let me ask you a question.
This is an important question to think about.
Imagine that I take my data set, I
put it into one of these algorithms,
I run the algorithm for a while, an answer pops out,
and I tell my customer, you know how well this is going to work,
here's my estimate.

What if I run my algorithm and I say, this
is my estimate of how well this algorithm is
going to work for you, OK?
So think about that.
And let me ask you a question.
Do you think that that estimate is
going to be too high or too low of the error, or perfect?
High, perfect, low.
Stick up a thumb.
Is it going to be too high, too low?
OK, good.
It's going to be too low.
That's right.
This is too low as a prediction, as an estimate, in general,
as an estimate of actual performance on new data.

Why?
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: It would have worked to just memorize
the training set, all right?
You can memorize the homework.
Won't work on the test, right?
Same thing.
So it would work.
This could be 0 if you were able to perfectly memorize
your training data.
But that's not the job that we were given.
Our job is to generalize.
Our job is to answer new questions.
So this is not a good estimate of how well this thing is
going to do in the world.
So what we generally do, as a matter of practice,
is to save out some validation data.

It should be drawn from the same distribution as your training
data.
You should take your input and shuffle it randomly
and save a piece out.

So then what you could do is you could take your data,
pour it into an algorithm, run the algorithm,
get out a hypothesis.
And then, see how well it works on your data.
Make it take the test.
AUDIENCE: [INAUDIBLE]

LESLIE KAELBLING: OK.
Well, so the methodology is how big--
so if you're given this much data, exactly how you
should manage this process is complicated.
I'll talk about it a little bit more.
But just fundamentally, you want to give some of your-- use
some of your data for training.
And then you can use some of it to validate the hypothesis that
came out and give you an estimate of how well it
will work on new data, OK?
So that's easy.
That's sort of a reasonable setup
if you know what algorithm you want to run, right?
So you said, over there, we're going to do that stuff.
We're going to pick an algorithm.
We're going to run the algorithm.
We're going to get a hypothesis.
And then we're going to validate it.
And we're going to validate it by looking at the E, right?
We're going to evaluate the error on the new data.
The plot thickens when we have multiple algorithms, right?
So here we already have two algorithms.
And in fact, we have more than two algorithms.
Because each of our algorithms has a hyperparameter.
And you could imagine running different algorithms
with different values of the hyperparameters.
So you might say, well, OK, what I'm
going to do is I'm going to run all
my algorithms on my training data
and use my validation data to pick the best algorithm.
OK.
And you could do that.
You could use your validation data to pick a good algorithm.
But again, after you've used the validation data
to help pick the best algorithm, that data
isn't any more impartial.
It got to help pick the algorithm.
And if you really want to, again, generate a hypothesis,
and you really want to generate a good estimate of how
well that hypothesis is going to work in the world,
then you need to hold out even more data
and test it on that data.
So, not this week's lab, but I think the next week's lab,
we'll explore a lot more in-detail strategies
for choosing among algorithms and using algorithms
to generate hypotheses and generating
estimates of how well those hypotheses will work out
in the world.
And actually, it's a tricky thing to think about.
And a lot of actual machine learning practitioners
kind of get it wrong.
So we're going to spend some time trying
to be sure that we're clear and careful in our machine learning


### Lecture: The perceptron algorithm


LESLIE KAELBLING: What we did at the end of last lecture
was finish up with this algorithm called perceptron.
And if you remember, it works to find a linear separator.
And I said that one of the things that
was kind of interesting and different about the perceptron
algorithm is that it's a case where rather than formulating
an objective and then kind of deriving an algorithm that
would do well with respect to that objective,
with the perceptron somebody came up with an algorithm.
But then once you come up with an algorithm,
you don't necessarily know what you
can say about how it will perform or what it will do.
And so what happened with perceptron
is that people came afterwards and proved things about it.
And so what we're going to do for the first part of today
is study the way in which the perceptron algorithm--
kind of a little bit more about how
it works and, in particular, prove
that it satisfies certain kinds of correctness conditions.
And doing the proof will give us some insight
into what's going on.
And what we're going to see is as the semester goes on
and we consider more and more complicated
kinds of hypothesis classes and learning methods,
we'll actually be able to make weaker and weaker guarantees
about what actually is going to happen.
So that part gets harder
So at the beginning the algorithms are easier
and the analysis is somewhat easier too.


### Lecture: The perceptron algorithm


LESLIE KAELBLING: OK.
Remember the perceptron algorithm--
and actually, we're going to consider a simpler version
of the perceptron algorithm.
You thought it was already as simple as it could be.
In particular, we'll talk about the perceptron through origin.
And for right now.
Don't worry too much about it.
But the thing that's different from the perceptron algorithm
we looked at before is that it has
no explicit offset parameter.
So we're going to start out with some vector theta of all zeros.
So it's just going to be zeros.

And we're going to loop.
So I just want to write the algorithm here
for t equals 1 to t.
This is an algorithm we looked at before for i equals 1 to n.
And then we have this question.
And the question is, if yi times theta transpose xi--
so this is our data points.
If this is less than or equal to zero,
then we're going to do something.
And what we're going to do is we're going to assign theta
is theta plus y times x.
So that's the basic perceptron algorithm without the offset.
So it should look like the algorithm we had before.
And the question-- so what we're going to do
is now we're going to kind of study this algorithm
and see what we can say about it.
Now, you might be sort of sad though,
because there is no offset parameter here.
And so this is really going to be-- this algorithm,
you could think about it.
This algorithm is about the problem
of finding a separator--
so some kind of good hypothesis.
So maybe you have some positive examples
and some negative examples.
And in this case, there's actually--
let me put some negative examples up here just
to make it a little trickier.
There's a separator that goes--
if you make the origin big enough,
there's the separator that goes through the origin.
So if I made this separator described
by some parameters theta.
So it's just a linear separator like we thought about before,
but there's no theta 0.
So if theta 0 is forced to be zero,
then it goes through the origin.
So let's just consider this class
of hypotheses for a little bit.
So here is a situation.
We have negative positive.
We have positive points.
We have a separator that goes through the origin,
and it's characterized by some vector of parameters theta.
So this is kind of the setup we're
going to be thinking about.
For right now, we're going to just
be thinking about being constrained for the separator
to go through the origin.
We'll come back to the idea of what
to do when it doesn't go through the origin.


### Lecture: The perceptron algorithm


LESLIE KAELBLING: So now we have a concept to introduce.
The concept is linear separability.
So linear separability.

So what is linear separability?
It's a property of a data set.
So it's a question we can ask about some data set, d.

And we'll say that this data set d
is linearly separable if there is some theta,
such that theta transpose xi times yi is greater than 0
for all i.

So that just says--
what does this say?
Let's practice what this says.
This is if I take the point x and I
take the dot product with theta, this gives me a value.
And normally, if we're thinking about the separator,
normally we would say, well, if the sign of this
is positive, we would predict plus 1,
if the sign is negative we would predict minus 1.
If we take that prediction--
not the sign, just the stuff inside-- and we multiply it
by the target value, we're making a correct prediction
when this product is positive because either this is negative
and this is negative or this is positive and this is positive.
Actually, this particular version
of linear separability that I wrote down here on the board,
because I didn't include theta 0, this
is actually linear separability through the origin.

The concept extends to not through origin,
and that's the definition that's written in the notes.
But this is linearly separable through the origin.
So we're interested-- we're going
to talk a lot about data sets that are linearly separable.
So that's one concept that we need.

All right.
The next concept that we need.
So this is just the question of whether the point is
on the right side of the separator or not.
Another concept that's actually-- it's
going to be important to us today
and it's going to come back over the next couple of lectures


### Lecture: The perceptron algorithm


LESLIE KAELBLING: So we're going to introduce
the idea of a margin.
So because there's an enormous amount of confusion--
I'm not completely sure why--
I'm going to just do it as simply as I can,
and hopefully we'll get the concept
and we can generalize it.
So the margin of a labeled data point--

OK.
So xy.
So a point x all by itself doesn't have a margin.
A point xy has a margin with respect to a hyperplane.
So with respect to a separator.

And we could do the case where we're
talking about the separator through the origin
or separator not through the origin.
For right now, I'll just write it this way.
But again, we could do it with theta 0.
It's done in the notes with a theta 0.
So what is the margin?
The margin is y times theta transpose x.

So what is this?
Does anybody recognize this expression?
Think back.
Something you typed into the computer a week ago maybe.
Yeah.
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: It's the distance--
OK, good.
So with this--
OK.
So imagine we're here.
Our theta is this, and here's the point x.

Then this is the distance from the hyperplane to the point.

And it's going to be positive if this point is on the same side
that the normal is pointing.
So this is the signed distance.

And this is the target label.

So I can indicate the target label
by putting a plus here and saying, oh,
if y is a positive guy--
so this is the signed distance from the hyperplane
to the point.
And then we're multiplying it by the desired classification
that we make.
So what's the sign of this if we're getting the point right?
Positive.
Sign is positive if we're getting it right.
And if we're getting it right, it's
the distance of the point to the hyperplane.
So that's the margin.
It's a kind of simple idea.
It's like roughly how right are we?
If it's negative, we're not very right.
If it's positive, we're right.
And if this point is like way over there,
then the margin is really big, and we're like way right.

OK?
How right are we?
That's the margin.

So that's the margin of a point--
a labeled point-- with respect to a separator.
What's the margin?
Then the other idea of a margin is the margin of a data set.
So margin of d--
some data set d--
with respect to theta.
We could ask that.
What's the margin of a data set d with respect to theta?
And that's just the min over data points
i of yi theta transpose xi over [INAUDIBLE] theta.

So that means that if we had positive--
imagine that we had this data set.

What's the margin of that data set?
So first of all, let me ask a question.
Here's a data set.
Here's a theta.
Is the margin of the data set with respect to theta
positive in that example that I drew?
Right.
It is positive.
How do you know it's positive without computing
numbers or anything?
Just by looking.
Yeah?
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: They're all correctly classified.
Awesome.
Right.
So this is positive, this thing is positive
if it's correctly classified.
Here, this correctly classifies everybody
so the margin's positive.
And now, who determines the margin?
Well, probably it's this guy.
That's the closest one.
So that would be the margin of that whole data set
with respect to that separator.
It doesn't matter if I draw a negative example over here.
That does not change the margin.
Or if I drew a positive one up there.


### Lecture: The perceptron algorithm


LESLIE KAELBLING: All right, this is all setup so
that I can tell you a fact, a theorem about the perceptron.
It's pretty cool.
So here we go.
So this is the perceptron convergence theorem.

OK.
Theorems almost always have the form-- if this, then that,
all right?
So if, so if--
I'm going to put the if up here.
Two conditions.
Condition A is there exists some theta star--
so some parameter vector theta star--
such that yi theta star transpose
xi over norm theta star is greater than 0 and gamma
for all i.

Another way to say that is the margin of theta
star with respect to the data--
or did I say data with respect to whatever?
OK, sure.
Whatever.
Yeah, OK.
Theta goes straight to data, data goes back to theta,
doesn't really matter.
The margin is less than or equal to gamma.

Greater than or equal to gamma.
Let's just say equal to gamma.
So if a margin is gamma, so if you say, OK,
so we have a data set and we say,
if there are some theta with margin at least
gamma, and the data points--
what is this?
The data points-- the norms of all your x's are less than some
constant.
So that just means if I take all my data points here, and here's
my origin, and I have all my data,
pluses or minuses, I don't really care,
there are some circle of radius r that contains all the data.
So is going to turn out that actually
the magnitude of your data points
could affect something about how--
not whether perceptrons will work or not,
but how long it might take to work.
You have a question?
Yeah.
AUDIENCE: Is gamma positive?
LESLIE KAELBLING: Gamma is positive.
Good.

A negative gamma is all sorts of trouble, right?
Negative gamma means we're not-- we're
getting some of them wrong and right.
So this theorem only applies in the case
that the data is linearly separable.
Good.
I should probably put that in the statement in my notes.
OK.
So here's the assumptions.
These are the assumptions.
We have a data set that's linearly separable, right?
So there's some gamma--
there's some theta star which gets
a positive gamma, a positive margin on every case.
And the data is contained in a ball.
So if these things are true, then this
is super cool and very surprising.
Then the perceptron will make at most r
over gamma squared mistakes.

OK.
So I left this algorithm over here,
this board down where it can reach it,
because I wanted to say, what is a mistake?
So intuitively, right?
What the perceptron algorithm is doing
is it's going through the data points and saying, hmm,
do I predict this one right?
If I do, I'm happy; if I don't predict this one right,
that counts as a mistake, and when I make a mistake,
I'm going to make an update, all right?
So this-- if I arrive here in my code, that's a mistake.

OK?
Is it clear what a mistake is, all right?
I'm just-- I'm going through my data so many times,
and every time, my current data doesn't predict correctly,
I'm going to do an update.
That's a mistake.

OK.
Yeah?
AUDIENCE: [INAUDIBLE] gamma [INAUDIBLE] use our data
[INAUDIBLE]
LESLIE KAELBLING: Ah, OK.
Good-- great question, and in fact, we
have all these exercises that will illustrate that.
So really good question.
So the question is, it feels like--
and this would be kind of really awful if it
were true-- it kind of feels like if we change
the units of our data, like we went from kilometers to inches,
that somehow that would change how the algorithm works.
But notice that r and gamma are measured in the same coin.
So they'll scale.
Yeah.
No, that's a really great kind of way
to be thinking about what's going on, a really
great question to ask, right?
But it's kind of like OK, right?
Because if I multiply my-- if I decide
to do it in inches instead of kilometers, then maybe
r and gamma, they both get really big,
but they'll scale with respect to each other in the same way.
So this is a relative scale between how big a ball does
the data live in and how big of a gap is there, right?
OK, so what is gamma?
Gamma, think of gamma, gamma stands for gap.
Our intuition is that maybe a data
set that looks like this with the minuses really
close to the pluses is hard, but a data
set that looks like this might be easier.
That's like the intuition, right?
That's the gamma, right?
This data set has a bigger gamma than that one.
OK, but this one lives in a smaller r,
so let me just put some pluses and minuses down here.
OK.
So r is the same for both of these guys,
but you could drive a truck between the pluses
and minuses on this one.
So we think maybe this would be easier for the algorithm,
right?
So that's the gamma.


### Lecture: The perceptron algorithm


LESLIE KAELBLING: , OK so this is our theorem,
this is our algorithm.
And I just want to dance us through the proof
because I think it's kind of good practice, a good way
to think about things.
And this is like the easiest proof
we could do in this class, so there's that, too.
OK.
Does the setup make sense?
Algorithm, statement.
Yep?
OK.

OK, so here's the idea.
The idea is that--

OK, we believe, we totally believe
that there is a theta star that separates our data.
We believe that because we assumed it
in our statement of the theorem, so there must be one.
So let's just draw this little picture,
the pluses are over here, the minus' is over here.
So we believe there is a theta star.
Now, of course, if your data is linearly separable
there is an infinity of theta stars
that will do the job for you.
We just have to believe in one, that will be enough
for us to make this proof.
So we just think of a particular one.
OK, and now, what is the perceptron going to do?
Well, it's going-- oh, no, that is not true.
If we're doing perceptron through origin,
it has to go through the origin, it turns out.
So we start out with this one we can't even draw, right?
We start out with theta 0 is just equal to the vector of 0s.
I can't even draw that one actually on this picture.
But as soon as we do anything, we'll get theta--
I'm going to write it theta 1.
And write this theta 0, sorry.
So I'm going to write theta k as the hypothesis
that we have after k mistakes.

Because when we're buzzing through that algorithm,
every time we make a mistake, we update our theta.
So we don't really have a superscript
on the theta and the algorithm.
But you could just think that every time we change theta,
every time we hit this line of code, we'll increment our k.
So we start out with theta 0 as all 0s, and then we go around--
and this theta is going to like dance around.
You saw it last time, the separator dancing
around when I did some demos.
So this theta is going to dance around,
it's going to go through the origin
all the time because we don't have a theta 0.
So theta's going to dance around.
And what we want to argue is that this angle, the angle
between the theta that is our current hypothesis
and this good theta, is going to get smaller on every iteration.
So it can bounce around, it doesn't always
stay on the same side, but that angle is going to get smaller.
That's the argument.
OK?
So to make that more concrete--

so we want to show that the distance between theta
and theta k get smaller.
We're going to show that the cosine between theta star
and theta k gets bigger.
All right, and cosine is big when the angle is 0.
All right, so we're going to show
that the cosine between these two things
is getting bigger and bigger over time.

Cosine gets bigger.
OK, so what's the cosine?
Well, we can remember that we can write this
as theta star dot theta k--
that's the definition of dot product--
norm theta star, norm theta k, right?
Usually we write this with these guys over there, but it's OK.

OK so it's enough to show that this thing gets
smaller and smaller, the angle gets smaller and smaller,
the cosine gets bigger and bigger,
and eventually it gets to 1.

OK, so that's what we're going to do.
and we're going to do it by taking this apart
into two pieces.
We're going to look first at this piece.

We're going to see how that changes as we increment k.
And we're going to see how this, 1 over theta to the k,
changes as we increment k.
And then that'll give us a way to think
about how this whole thing changes as we increment k.
Yep?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Let's see, in the bounds
that we write it's going to look monotonic,
but I actually haven't done the work
to show that it actually, absolutely goes monotonically.
I suspect it probably does, but I can't promise.
Yep?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: This is going to be a bound.
So, if there is a theta star, you'll
make, at most, this many mistakes.
We can't promise which theta you're going to hit exactly
and we can't say exactly how many mistakes
you're going to make, kind of in a general way.
So this is just a bound.
So we're just assuming the existence of this guy,
and we say, if such a guy exists, I promise you that.
But we're assuming about the theta star, right?
The theta star is the thing that picks the gamma.
So, if you say, oh, yeah, I could drive a truck
through my data, but the theta star I'm going to think about
is one that has a margin like this,
then you'll get a bad bound.
But really there is an awesome theta star,
which gives you a better bound.
So you kind of want to think about the theta
star as being the best one that you could get for your dataset.

OK, let's go.
Two parts, it's going to take two boards.
We'll just come over here.
OK.
So we're going to think about that first thing.
So we're going to think about theta to the k dot theta
star over norm theta star.

OK, what do we know about that?
Not too much.
What do we do?
Well, one thing that we do know is
that we know how theta to the k is related
to the previous theta k, right?
We know that from over there.
So what's theta k?
Well, theta k is theta k minus 1 plus--
and I'm going to write y i, x i, where we just are
going to make the assumption--
this just a naming assumption, it's not an assumption,
we're just going to let i be the point
that we made mistake k on.

All right, we're buzzing through our algorithm,
we make a mistake, at that moment, there's some i.
So whatever i it is, let that be the i over--

you believe that, right?
That's just one step of the algorithm.
This is going to be a little induction proof.
Induction proofs are nice because they're
like proofs for the lazy.
You only have to do one little bit of work
and then induction does the rest.
So we just doing a little bit of work,
let induction do the rest.
So what do we do?
Well, we can turn this into a sum, right?
So this is going to be theta k minus 1 dot theta
star over norm theta star.
This has the appealing property of looking a lot
like that guy, plus something.
Something is this, y i, x i dot theta star
over norm theta star.

Is this guy a friend of yours?

What friend?
What's his name?
Margin?
Yeah?
Yeah, margin?
OK, what's another name for the margin?
What can we assume about the margin in this case?
It's no bigger than--
no smaller than-- it's no smaller than gamma, right?

OK, let me just say that this guy is greater
than or equal to gamma, right?
So gamma, for any point-- we know
that the margin for all the points is at least gamma,
so the margin for this particular guy is gamma.
And so, then we can say, OK, so then what?
So then we just have--
this is a summation now, right?
We say this thing, at iteration k,
is the value that it was at iteration k minus 1,
let's say, plus gamma.
And it started out being 0, so after k iterations
it can't be bigger than k gamma.

Yay, favorite kind of recursion.
OK, so we're more than halfway done.
If this is not the kind of work that floats your boat,
we're almost done.
So I said, we were going to divide this into two parts.
And now what we know is that this part right here
is bigger than k gamma.
So that's promising, right?
We want this thing to be getting bigger as we iterate, right?
Because as cosine gets bigger, the angle gets smaller,
so we want this thing to be getting bigger.
So it seems good.
But we're nervous because we're dividing
by the norm of this thing, and that's probably getting bigger,
too.
So now it's like a race.
We have to see who's getting bigger faster.
OK, so let's do this other part.

So let's think about this guy, and we're
going to think about the squared norm.
OK, for those of you who are not used to doing proofs or seeing
this kind of derivation or something,
whenever anybody does it on the board for you,
it goes from the beginning to end in this very lovely way,
but nobody ever thinks about it in this order.
So if you, at this moment, are saying, why in the heck
is she squaring it?
The answer is because it's going to work out nicely later, not
because I can just immediately see that this is the right step
to take.
So don't worry that you think, oh my goodness, I
am inadequate.
I wouldn't know to write squared right now.
Most people wouldn't.
I mean, you have to get all the way to the end, work it out.
I got to prove this thing, how can I get there?
Some people reason backwards in proving things,
but they write them down going forward.
And that means that it's hard to see why you're doing things.
But I promise you, it will work out.
OK, so, here we go.
All right, so theta to the k squared, what is that?
Well, we don't know, but we, again,
know how to expand it out one, right?
So we can write theta k minus 1 plus y i x i.
We can write that.
Good.
And we know about quadratic stuff.

OK, so what can we do?
Again, we're going to be happy to find some kind
of an inductive maneuver.
So we have this quantity here and we have quantity minus 1,
so that's going to give us a kind of iterative lever
to do something.
Let's talk about this guy.
What do we know about this?
What assumption-- we made two assumptions,
we're going to use them both, otherwise,
we would not have made them.
So the theory of theorems tells us the other guy's
going to be relevant.
What is this bounded by?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: R squared, right?
So this guy is less than or equal to r
squared, this we know.

What's the sign of this?

STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Totally.
Typo.
Thank you.
So the sign of that, we're not sure.

What's the sign of that?

LESLIE KAELBLING: Negative.
Why?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Right.
Good.
It was the last--
at first, I thought you were talking about mistakes I made,
which I did just make a mistake.
No, it's not mistakes I made, it's
mistakes the algorithm makes.
Awesome.
This is a mistake, right?
The algorithm made a mistake on this data point.
And what it means to make a mistake on the data
point is to have this be negative sign
So this is negative.

So we can safely write that.

[INAUDIBLE] You're going to have to shout.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Oh, it is.
OK, good.
What's y i squared?
I'm being lazy.
y i squared is 1, no matter what I do.
Either y i is plus one or-- no, but I
should have written it down.
So, that's OK.
So this, that guy is equal to 1, that guy is squared.
Good.

Yep?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Oh, it's a dot product.
It's a dot product.
No, it's fine.
OK, so good.
So this is good.
So this seems to me then that we can
say that this is less than or equal to kR squared.

OK, so awesome.

OK, if-- then we know that 1 the 1 over this guy
is bigger than square root of 1 over k i.
OK, let's write it.

OK, cool.
So we had a bigger than for this term,
and now we have a bigger than for that term,
which means we can multiply them and we'll get a bigger than.
So this is bigger than k times gamma over root k R.
So this is 1 over root k gamma over R.
OK, so what about this?
Well, it says that this cosine tends to go up with k.
So that's good, it goes up with k.

But then we can ask the question,
what's the biggest cosine between two angles can be?
STUDENT: 1.
LESLIE KAELBLING: 1?
1.
So we could do it with some more reasoning.
We say, OK, well, if 1--

then we can solve that and end up with k being less than
or equal to R over gamma squared.

(WHISPERING) Then we get to draw the box.
OK.
OK?
I'm going to leave you to ponder this on your own
if you want to.
You should ponder it, it makes sense, none of the steps
here are too complicated, but it's kind of awesome.
I mean, it's actually really surprisingly awesome
because this theorem applies--
the easiest way to talk about the perceptron algorithm
is the case where we have a dataset
and we're just working through it over and over again,
and we can think about how many mistakes we
make on that dataset.
But the fact is that this theorem even
applies if you have an infinite stream of data.

As long as this fact holds, as long as there
is a separator with a certain margin, then
even if you have an infinite stream of data,
not just one file that you're buzzing through,
if there is a separator with a positive margin,
you will eventually stop making mistakes.
So that's kind of cool.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/8f4f9aca5581dde50291b0d0e29d0148/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_The_Perceptron.pdf
