# Feature representation

> Week 3: Features · MIT 6.036 courseware archive

## Notes – Chapter 4: Feature representation

Notes – Chapter 4: Feature representation
You can sequence through the Feature representation lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 4: Feature representation
notes as a PDF file.

## Lecture: Feature representation - transforming through-origin to not-through-origin

Lecture: Feature representation - transforming through-origin to not-through-origin
Lecture: Feature representation - transforming through-origin to not-through-origin
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Linear classifiers and the XOR dataset

Linear classifiers and the XOR dataset
Linear classifiers are easy to work with and analyze, but they are a very restricted class of hypotheses. If we have to make a complex distinction in low dimensions, then they are unhelpful.
Our favorite illustrative example is the “exclusive or" (
xor
) data set,
D. Melanogaster is a species of fruit fly, used as a simple system in which to study genetics, since 1910.
the drosophila
of machine-learning data sets:
There is no linear separator for this two-dimensional dataset! But, we have a trick available: take a low-dimensional data set and move it, using a non-linear transformation into a higher-dimensional space, and look for a linear separator there. Let's look at an example data set that starts in 1-D:
These points are not linearly separable,
What's a linear separator for data in 1D? A point!
note
but consider the transformation [mathjaxinline]\phi (x) = [x,x^2][/mathjaxinline]. Putting the data in [mathjaxinline]\phi[/mathjaxinline] space, we see that it is now separable. There are lots of possible separators; we have just shown one of them here.
A linear separator in [mathjaxinline]\phi[/mathjaxinline] space is a nonlinear separator in the original space! Let's see how this plays out in our simple example. Consider the separator [mathjaxinline]x^2 - 1 = 0[/mathjaxinline], which labels the half-plane [mathjaxinline]x^2 -1 > 0[/mathjaxinline] as positive. What separator does it correspond to in the original 1-D space? We have to ask the question: which [mathjaxinline]x[/mathjaxinline] values have the property that [mathjaxinline]x^2 - 1 = 0[/mathjaxinline]. The answer is [mathjaxinline]+1[/mathjaxinline] and [mathjaxinline]-1[/mathjaxinline], so those two points constitute our separator, back in the original space. And we can use the same reasoning to find the region of 1D space that is labeled positive by this separator.
This is a very general and widely useful strategy. It's the basis for
kernel methods
, a powerful technique that we unfortunately won't get to in this class, and can be seen as a motivation for multi-layer neural networks.
There are many different ways to construct [mathjaxinline]\phi[/mathjaxinline]. Some are relatively systematic and domain independent. We'll look at the
polynomial basis
in section
as an example of that. Others are directly related to the semantics (meaning) of the original features, and we construct them deliberately with our domain in mind. We'll explore that strategy in section
.
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:02 PM (revision f808f068e)

## Lecture: Feature representation - polynomial basis

Lecture: Feature representation - polynomial basis
Lecture: Feature representation - polynomial basis
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Polynomial basis

Polynomial basis
If the features in your problem are already naturally numerical, one systematic strategy for constructing a new feature space is to use a
polynomial basis
. The idea is that, if you are using the [mathjaxinline]k[/mathjaxinline]th-order basis (where [mathjaxinline]k[/mathjaxinline] is a positive integer), you include a feature for every possible product of [mathjaxinline]k[/mathjaxinline] different dimensions in your original input.
Here is a table illustrating the [mathjaxinline]k[/mathjaxinline]th order polynomial basis for different values of [mathjaxinline]k[/mathjaxinline].
Order
[mathjaxinline]d=1[/mathjaxinline]
in general
0
[mathjaxinline][1][/mathjaxinline]
[mathjaxinline][1][/mathjaxinline]
1
[mathjaxinline][1,x][/mathjaxinline]
[mathjaxinline][1,x_1, \ldots , x_ d][/mathjaxinline]
2
[mathjaxinline][1,x,x^2][/mathjaxinline]
[mathjaxinline][1,x_1, \ldots , x_ d, x_1^2, x_1x_2, \ldots ][/mathjaxinline]
3
[mathjaxinline][1,x,x^2,x^3][/mathjaxinline]
[mathjaxinline][1,x_1, \ldots , x_1^2, x_1x_2, \ldots , x_1x_2x_3, \ldots ][/mathjaxinline]
So, what if we try to solve the
xor
problem using a polynomial basis as the feature transformation? We can just take our two-dimensional data and transform it into a higher-dimensional data set, by applying [mathjaxinline]\phi[/mathjaxinline]. Now, we have a classification problem as usual, and we can use the perceptron algorithm to solve it.
Let's try it for [mathjaxinline]k = 2[/mathjaxinline] on our
xor
problem. The feature transformation is
[mathjax]\phi ((x_1, x_2)) = (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2)\; \; .[/mathjax]
Study Question:
If we use perceptron to train a classifier after performing this feature transformation, would we lose any expressive power if we let [mathjaxinline]\theta _0 = 0[/mathjaxinline] (i.e. trained without offset instead of with offset)?
After 4 iterations, perceptron finds a separator with coefficients [mathjaxinline]\theta = (0, 0, 0, 0, 4, 0)[/mathjaxinline] and [mathjaxinline]\theta _0 = 0[/mathjaxinline]. This corresponds to
[mathjax]0 + 0 x_1 + 0 x_2 + 0 x_1^2 + 4 x_1 x_2 + 0x_2^2 + 0 = 0[/mathjax]
and is plotted below, with the gray shaded region classified as negative and the white region classified as positive:
Study Question:
Be sure you understand why this high-dimensional hyperplane is a separator, and how it corresponds to the figure.
For fun, we show some more plots below. Here is the result of running perceptron on
xor
, but where the data are put in a different place on the plane. After 65 mistakes (!) it arrives at these coefficients: [mathjaxinline]\theta = ( 1, -1, -1, -5, 11, -5)[/mathjaxinline], [mathjaxinline]\theta _0 = 1[/mathjaxinline], which
The jaggedness in the plotting of the separator is an artifact of a lazy lpk strategy for making these plots–the true curves are smooth.
generates this separator:
Study Question:
It takes many more iterations to solve this version. Apply knowledge of the convergence properties of the perceptron to understand why.
Here is a harder data set. After 200 iterations, we could not separate it with a second or third-order basis representation. Shown below are the results after 200 iterations for bases of order 2, 3, 4, and 5.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:24 PM (revision 4f166135)

## Lecture: Example of perceptron algorithm with polynomial basis transformations

Lecture: Example of perceptron algorithm with polynomial basis transformations
Lecture: Example of perceptron algorithm with polynomial basis transformations
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Feature representation strategies for dealing with varied data

Lecture: Feature representation strategies for dealing with varied data
Lecture: Feature representation strategies for dealing with varied data
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Hand-constructing features for real domains

Hand-constructing features for real domains
In many machine-learning applications, we are given descriptions of the inputs with many different types of attributes, including numbers, words, and discrete features. An important factor in the success of an ML application is the way that the features are chosen to be encoded by the human who is framing the learning problem.
Discrete features
Getting a good encoding of discrete features is particularly important. You want to create “opportunities" for the ML system to find the underlying regularities. Although there are machine-learning methods that have special mechanisms for handling discrete inputs, all the methods we consider in this class will assume the input vectors [mathjaxinline]x[/mathjaxinline] are in [mathjaxinline]\mathbb {R}^ d[/mathjaxinline]. So, we have to figure out some reasonable strategies for turning discrete values into (vectors of) real numbers.
We'll start by listing some encoding strategies, and then work through some examples. Let's assume we have some feature in our raw data that can take on one of [mathjaxinline]k[/mathjaxinline] discrete values.
Numeric
Assign each of these values a number, say [mathjaxinline]1.0/k, 2.0/k, \ldots , 1.0[/mathjaxinline]. We might want to then do some further processing, as described in section
. This is a sensible strategy
only
when the discrete values really do signify some sort of numeric quantity, so that these numerical values are meaningful.
Thermometer code
If your discrete values have a natural ordering, from [mathjaxinline]1, \ldots , k[/mathjaxinline], but not a natural mapping into real numbers, a good strategy is to use a vector of length [mathjaxinline]k[/mathjaxinline] binary variables, where we convert discrete input value [mathjaxinline]0 < j \leq k[/mathjaxinline] into a vector in which the first [mathjaxinline]j[/mathjaxinline] values are [mathjaxinline]1.0[/mathjaxinline] and the rest are [mathjaxinline]0.0[/mathjaxinline]. This does not necessarily imply anything about the spacing or numerical quantities of the inputs, but does convey something about ordering.
Factored code
If your discrete values can sensibly be decomposed into two parts (say the “make" and “model" of a car), then it's best to treat those as two separate features, and choose an appropriate encoding of each one from this list.
One-hot code
If there is no obvious numeric, ordering, or factorial structure, then the best strategy is to use a vector of length [mathjaxinline]k[/mathjaxinline], where we convert discrete input value [mathjaxinline]0 < j \leq k[/mathjaxinline] into a vector in which all values are [mathjaxinline]0.0[/mathjaxinline], except for the [mathjaxinline]j[/mathjaxinline]th, which is [mathjaxinline]1.0[/mathjaxinline].
Binary code
It might be tempting for the computer scientists among us to use some binary code, which would let us represent [mathjaxinline]k[/mathjaxinline] values using a vector of length [mathjaxinline]\log k[/mathjaxinline].
This is a bad idea!
Decoding a binary code takes a lot of work, and by encoding your inputs this way, you'd be forcing your system to
learn
the decoding algorithm.
As an example, imagine that we want to encode blood types, which are drawn from the set [mathjaxinline]\{ A+, A-, B+, B-, AB+, AB-, O+, O-\}[/mathjaxinline]. There is no obvious linear numeric scaling or even ordering to this set. But there is a reasonable
factoring
, into two features: [mathjaxinline]\{ A, B, AB, O\}[/mathjaxinline] and [mathjaxinline]\{ +, -1\}[/mathjaxinline]. And, in fact, we can reasonably factor the first group into [mathjaxinline]\{ A, {\rm not}A\}[/mathjaxinline], [mathjaxinline]\{ B, {\rm not}B\}[/mathjaxinline]
It is sensible (according to Wikipedia!) to treat [mathjaxinline]O[/mathjaxinline] as having neither feature [mathjaxinline]A[/mathjaxinline] nor feature [mathjaxinline]B[/mathjaxinline].
note
So, here are two plausible encodings of the whole set:
Use a 6-D vector, with two dimensions to encode each of the factors using a one-hot encoding.
Use a 3-D vector, with one dimension for each factor, encoding its presence as [mathjaxinline]1.0[/mathjaxinline] and absence as [mathjaxinline]-1.0[/mathjaxinline] (this is sometimes better than [mathjaxinline]0.0[/mathjaxinline]). In this case, [mathjaxinline]AB+[/mathjaxinline] would be [mathjaxinline](1.0, 1.0, 1.0)[/mathjaxinline] and [mathjaxinline]O-[/mathjaxinline] would be [mathjaxinline](-1.0, -1.0, -1.0)[/mathjaxinline].
Study Question:
How would you encode [mathjaxinline]A+[/mathjaxinline] in both of these approaches?
Text
The problem of taking a text (such as a tweet or a product review, or even this document!) and encoding it as an input for a machine-learning algorithm is interesting and complicated. Much later in the class, we'll study sequential input models, where, rather than having to encode a text as a fixed-length feature vector, we feed it into a hypothesis word by word (or even character by character!).
There are some simpler encodings that work well for basic applications. One of them is the
bag of words
(
bow
) model. The idea is to let [mathjaxinline]d[/mathjaxinline] be the number of words in our vocabulary (either computed from the training set or some other body of text or dictionary). We will then make a binary vector (with values [mathjaxinline]1.0[/mathjaxinline] and [mathjaxinline]0.0[/mathjaxinline]) of length [mathjaxinline]d[/mathjaxinline], where element [mathjaxinline]j[/mathjaxinline] has value [mathjaxinline]1.0[/mathjaxinline] if word [mathjaxinline]j[/mathjaxinline] occurs in the document, and [mathjaxinline]0.0[/mathjaxinline] otherwise.
Numeric values
If some feature is already encoded as a numeric value (heart rate, stock price, distance, etc.) then you should generally keep it as a numeric value. An exception might be a situation in which you know there are natural “breakpoints" in the semantics: for example, encoding someone's age in the US, you might make an explicit distinction between under and over 18 (or 21), depending on what kind of thing you are trying to predict. It might make sense to divide into discrete bins (possibly spacing them closer together for the very young) and to use a one-hot encoding for some sorts of medical situations in which we don't expect a linear (or even monotonic) relationship between age and some physiological features.
If you choose to leave a feature as numeric, it is typically useful to
scale
it, so that it tends to be in the range [mathjaxinline][-1, +1][/mathjaxinline]. Without performing this transformation, if you have one feature with much larger values than another, it will take the learning algorithm a lot of work to find parameters that can put them on an equal basis. So, we might perform transformation [mathjaxinline]\phi (x) = \dfrac {x - \overline{x}}{\sigma }[/mathjaxinline], where [mathjaxinline]\overline{x}[/mathjaxinline] is the average of the [mathjaxinline]x^{(i)}[/mathjaxinline], and [mathjaxinline]\sigma[/mathjaxinline] is the standard deviation of the [mathjaxinline]x^{(i)}[/mathjaxinline]. The resulting feature values will have mean [mathjaxinline]0[/mathjaxinline] and standard deviation [mathjaxinline]1[/mathjaxinline]. This transformation is sometimes called
standardizing
Such standard variables are often known as “z-scores," for example, in the social sciences.
a variable
.
Then, of course, you might apply a higher-order polynomial-basis transformation to one or more groups of numeric features.
Study Question:
Percy Eptron has a domain with 4 numeric input features, [mathjaxinline](x_1, \ldots , x_4)[/mathjaxinline]. He decides to use a representation of the form
[mathjax]\phi (x) = {\rm PolyBasis}((x_1, x_2), 3) ^\frown {\rm PolyBasis}((x_3, x_4), 3)[/mathjax]
where [mathjaxinline]a^\frown b[/mathjaxinline] means the vector [mathjaxinline]a[/mathjaxinline] concatenated with the vector [mathjaxinline]b[/mathjaxinline]. What is the dimension of Percy's representation? Under what assumptions about the original features is this a reasonable choice?
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:28:24 PM (revision 4f166135)

## Video transcripts

### Lecture: Feature representation - transforming through-origin to not-through-origin


LESLIE KAELBLING: Some of you might feel slightly
disappointed, because--
like did I sell you a bill of goods?
I said last week, we talked about perceptron
not through the origin.
That seemed like a nice general algorithm.
And now here I am talking about the sort of limited case
where the line goes through the origin all the time.
So you might be kind of grumpy and say, well,
mmm, that doesn't seem like a very good algorithm anyway,
so why should I'd be happy about this proof?
And the answer is it's OK, because we
can reduce the problem of perceptron
not through the origin to the problem of perceptron
through the origin.
And that's actually going to be an example
of a more general kind of thing that we're
going to go on to do for the rest of the class.
So let's just talk about this--
first of all, just the simple case where I have some data--
you can just think of this as a--
so the general theme now is going to be transforming data.

And I talked a little bit about this last time about saying,
oh, you have a feature function phi,
and that takes a [INAUDIBLE] into a vector.
But sometimes we'll just want to turn vectors into vectors.
So you might imagine a situation where
you have your original data is in R to the little d,
but we're going to talk about some kind of feature
or function, feature transformation
that takes it to--
R to some other D.
Now often the place where we'll take it will be bigger.
Usually it will be, but it doesn't have to be.
And sometimes, for some choices of these feature
transformations, it makes your problem easier.
So let's first think about one very specific feature
transformation which lets us go from separators
through the origin to separators not through the origin, OK?
So imagine-- so originally our data was in theta to the d.

And imagine actually secretly over here, our original--
so imagine that we had theta--

it was theta 1 to theta d.
And we had theta 0, because we were doing that general purpose
perceptron not through the origin.
And we really like this theta and this theta 0,
they are awesome.
They would fit nicely in our data set.
And now trying to tell you that you could just do perceptron
through origin instead, and the answer
is you can just do perceptron through origin,
but you have to change your data around a little bit.
So what I'm going to suggest--
OK, so originally we had a hypothesis
that was the sine of theta transpose x plus theta 0,
right?
That was our original setup.
So now I'm going to suggest that we consider a feature
transformation.
Our feature transformation is going to take an x--
so a particular x which might have components x1 through xd--

and turn it into a new feature vector
that looks like x1, xd, 1.

OK?
So it's just going to be now in d plus 1 dimensions,
and this is going to be our old feature
vector with a 1 added on.
And now we're going to have a new theta.

I suggest that a really awesome new theta,
if we'd liked our old theta, would be this one.

So in this case, now if we limit ourselves to linear separators
through the origin, then our new hypothesis
will be the x1's times--
x1 through xd times theta 1 through theta
d, which would still be--
I'll write it our old way, theta transpose x,
and then this 1 plus the theta 0 plus theta 0.

OK, what did I just do?
What I just did is I said to you, if you have
a separator that doesn't go through the origin
in d dimensions, we can construct
a separator that does go through the origin in d
plus 1 dimensions.
So let me just do an example.
Here's my example.

First of all, I'm going to let d equals 1.

So here's my data set, here is 0, and I have a plus and a plus
and a minus and a minus--
OK, they're hard to see.
These are my data points.

They're in one dimension, d equals 1.
I can separate them, write a linear separator
in one dimension that's just a point,
but I can separate them with this separator right there,
a vector going that way.
So that separator is described by theta is equal to minus 1,
and theta 0 is 2.5.
In one dimension-- I can run perceptron in one dimension.
If I ran perceptron in one dimension,
I might get the solution or something like it, right?
So these guys are positive, those guys are negative.
d equals 1, I have a separator that doesn't
go through the origin, yeah?

OK.
Now let's do this maneuver.
What happens?
What happens is I take my data, it seems weird,
and all I do is I put them up one, right?
So here's my original x1, and then everybody else
is going to have 1.
So I'm going to have plus, plus, minus, minus.
And now my suggestion is that this separator, this theta
nu in the augmented space should be minus 1, 2.5.
And the offset is 0, there's no offset.
And that's a line that goes like that,
and that separates the nu.
So this is a general purpose trick,
you can do it in thousand dimensions, too.
I just can't draw it.
Yep?

What happens to your gap?
The margin does change.
Right.
And so this theorem, the way it's written,
and the proof and everything assumes
that we have to think about the margin in the augmented space,
in the space where we're going to go through the origin.
You can do a proof that involves theta 0 also,
but it's much more complicated.
This is meant to make us feel warm and happy
about the algorithm, not so much--
I mean, it's a tool of understanding anyway, so yeah.
Yeah?
AUDIENCE: Will this work on any data
set Or does the [INAUDIBLE] negative [INAUDIBLE]??
LESLIE KAELBLING: Great question.
OK.
So you've just totally led into my next section,
so I'm going to hold you for one second,
and then we're going to go right there.
Other questions about this maneuver?
This maneuver assumes that we have a linear separator
to start with in the lower dimensional space.
So the case that you suggested doesn't, and we'll
get there in a minute.
So the assumption is we have a linear separator, lower
dimensions, but it doesn't go through the origin.
By adding one dimension, we can make it go through the origin.
That's all.
Yep?
AUDIENCE: Why does the hypothesis [INAUDIBLE]??

LESLIE KAELBLING: Well, what I'm suggesting is that the new x--
OK, we write x nu here.
That theta nu transpose x nu is equal to that.

OK.
Oh, yep?
AUDIENCE: Why does the hypothesis
take the sign [INAUDIBLE]?

LESLIE KAELBLING: Why does it take the sign, I'm not--
AUDIENCE: --take the sign
LESLIE KAELBLING: Oh, oh, oh, oooooh.

Good.
That kind of sign.
One more, yep?
AUDIENCE: [INAUDIBLE]?

LESLIE KAELBLING: So I'm writing in two different ways.
So if I take a transformed x and my theta nu,
those are in d plus 1 space.
That's equal to this, which is equal to that.


### Lecture: Feature representation - transforming through-origin to not-through-origin


LESLIE KAELBLING: So this is a case
where the data was linearly separable in one dimension.
And by just doing this simple trick,
we made it linearly separable through the origin
in two dimensions.
Now let's think about this data set.
Actually, a slightly different setup
but not too much different.
I have a negative point here and a positive one here,
and a negative one here, and a positive one here.
OK.
Is that linearly separable in one dimension?
No.
Through origin, no, not through origin, no.
Just not linearly separable.
OK, so we're sad, right?
If I'm trying to sell you machine
learning, super cool awesome thing,
and I'm telling you about linear separability
and we can't even like, do this problem?
Then you can just leave, right?
I mean, this-- that would be sad.
Yup?
STUDENT: Can we use polar coordinates?
LESLIE KAELBLING: Can we use-- well, OK.
So good, so good question.
Can we use polar coordinates?
So the answer-- the theme here is transforming our data.
So what we're gonna think about is
ways of putting this data in a different space where
linear separability is still a nice idea.
And just-- lest you think that I'm totally out of touch
with reality, you can think of, in fact,
all the layers of the neural network, up to the last one,
as finding a good transformation of your data.
So this idea of transforming your data
is a kind of a fundamental and an important one.
OK.
So let's think about a transformation of this data.
So one transformation that's kind of nice--
so I could say, OK, this is my original space, this is x,
we'll call it x1.
But now I want to make a new feature space
and I'm gonna let phi of x be x--

xx squared, let's say.
So what if I do that?
So now I'm gonna have one axis that was my old x and one
axis that's x squared.
In my new world, this is x1 and this is x2, right?
This is the first dimension and the second dimension.
So now this guy is here and here but this guy, phi was 2,
he's up here.
And then we get somebody over here and somebody over here.

Now, is that linearly separable in two dimensions?
Yeah, right?
I can just go [HUMMING] so there's my separator.
Nicely linearly separable in two dimensions.
OK, good.
So this feels like, OK, if I'm a smart human
and I can take my data set and stare at it for a long time,
and maybe I can think of something to do
to it to make my data work out.
The answer to that is well, no, of course
we wouldn't expect you to do that
except for blackboard examples.
But there are systematic strategies
for transforming your data into, generally
speaking, higher dimensional spaces that do something good.
So I'm gonna talk about one particular one
because it just helps us think about things,
but there are others.
So let me talk about something called the polynomial basis.

And people might call it polynomial features
or polynomial transformation or some such thing.
And I am just gonna illustrate it for you by example
because trying to write it out in like, lovely algebraic
notation would make us all crazy.
So it has a parameter, K, which is the order
I'll say what that means.
So here's K, and here's the case where d=1,
and here's the general case.
OK, so, again, so order 0, our feature vector
just contains the feature 1.

What does that mean?
Well, it means you could predict the constant.
Right?
To get one feature you could multiply it by some data
and you could predict a constant.
Basis 1 will give us 1 and x, right, if d=1?
More generally it would give us 1 and x1 through xd.
So if actually you did this polynomial basis transformation
in one dimension-- with order 1, it
basically takes your normal feature set and adds a 1
to it, which is the thing that we just did.
So that means you could--
if you do this transformation to your data
you could just do through the origin and everything is cool.
You wouldn't have lost anything.
OK, Order 2.

We square-- we do all of the cross-terms.
So in this case, it just becomes that,
which is kind of like the one I did here
but we'd get one more dimension, which was a constant,
if we wanted it.
Here what we would get is 1, we'd get all the basic guys.
Then we'd get x1 squared, x1 x2, x1, x3,
and that would go on for a while.
And then x2 squared, and x2 times everybody.
And then xd squared, and xd times everybody.
So all the two-way products.

And third order, 1x, x squared, x cubed,
and I don't even want to try to write this
but it's all the three-way products of all the guys.

OK.
So this is kind of an interesting thing.
So what I'm gonna propose, then, right, so what you could do,
you could say, oh, I have a problem.
My data is kind of complicated and I can't separate it
with a linear separator.
But maybe I can transform it to a fancier space
and try to find a linear separator in that space.
So I wanna do some demos now.

OK, just for a warm-up, so what I'm doing
is I'm running the Perceptron algorithm still,
it's the only algorithm we really
know except for the silly one.
Running the Perceptron algorithm but I'm
gonna run it in a transformed feature space.

Actually, let's come back here for one second
because we need one more idea.

OK.
Let's actually-- so what did we do here?
We took our data set that was in one dimension
and we put it in two dimensions by taking x and x
squared as the dimensions.
And we found this separator.
Right?
So this particular separator that we found
is the separator that says x squared is bigger than 1.
That was the-- basically the separator we found, right?
And so x squared is bigger than 1, so what we can do
is we can turn this into a test on our x's, right?
No matter how complicated a complicated polynomial formula
this is, we can always go back to our original guys
and say, is this--
for any individual point here, we can say,
is this true of that guy?
So in this case, x squared bigger than 1,
so x squared bigger than 1 happens, you know, like here.
So it's kind of like saying, well,
for all the points here going that way and all the points
here going that way, right?
Because if x is less than -1 or x is bigger than +1, then--

well, actually the division happens over here, right?

I guess it depends on these data points.
1, 4, yeah, no.
So OK.
Let's assume that this is 1 and that these guys
were less than 1.
Then it works.
So but the point is that you can take
a separator in the complicated space
and you can go back and draw it in the original space.
It might not be easy to come up with a nice analytic
description of it, but you can always just--
for any given point you can ask, do I
satisfy this test on the more complicated feature space?


### Lecture: Feature representation - transforming through-origin to not-through-origin


LESLIE KAELBLING: So what I'm going to show you first
is with no transformation.
So the gray region here is the region where we say,
we're classifying this as negative,
and the white region is where we're
classifying it as positive.
It's all negative just because of the way it's drawn,
not because it should be jaggedy.
It's really just a straight line but I'm not
sampling it very finely.
So here I'm running the perceptron algorithm
on this data.

So is it going to find a separator?
Who thinks yes?
Who thinks no?
Good.
No is a good answer.
There is no linear separator of this data.
So we can run the perceptron for a long time.
It is never going to win.
So stop that.
But now let's do it with a polynomial basis of order 2.
So that means I'm going to take my data points,
two-dimensional data points, and I'm
going to put in all the x squared,
y squared, xy terms in there.
That's a new space.
It's a higher order feature space.
I can't draw the picture anymore.
It's six dimensions or something at this point.
But I'm just going to run perceptron there,
and I'm going to draw the answer back here.

I always have to make the windows big.
If I were smarter, I could figure out
how to do this automatically.
OK.
That's the first try.
Second try.
Third try.
We got it.
That's pretty sweet.

Let's try this just for fun.

0.
OK.
So this is also that same data basically, but it's--
before I had it centered around 0, and now I have it--
it's up above.
So the data points--
this corner right here is 1,1.
So this is 1, 1, 2, 2, 1, 2, 2, 1.
Prediction.
Is this going to be easier or harder than the one
that we just did?

Harder.
Why?

Yep?
AUDIENCE: [INAUDIBLE]

LESLIE KAELBLING: Right.
Right.
We can also look at this.
Gamma is the same, roughly.
Well, we have to think about gamma and the higher
dimensional space.
So maybe it's not exactly the same, but R got bigger.
And especially R got bigger when we square these guys and stuff.
So we used to have them right around zero
and we squared them.
Now this guy is at 2, 2 and we squared its features.
So R got a lot bigger.
So the R-gamma ratio actually gets worse.

And so this is harder, but we'll get there.
So perceptron is just doing it's plain old perceptron thing just
in the higher dimensional space.

And we're getting these funny shapes.
They're all quadratic.
These curves are all quadratic curves because a linear--
there we go.
Woo-hoo.
Yep?
AUDIENCE: [INAUDIBLE]

LESLIE KAELBLING: You don't have to make that many.
No.
That's right.
You absolutely don't have to make that many.
And some miraculous orderings of your data or something
might make it smaller.
But the tendency is actually pretty good.
Yeah.
OK.
So good.
So-- uh-huh?
AUDIENCE: [INAUDIBLE] simply translating the data
[INAUDIBLE]
LESLIE KAELBLING: It's because we're doing separators
through the origin.
You could do some algebra and kind
of see how that'll tend to go but it
tends to be that the bounds get worse if you just move it away.
Yep.
Audience: Can you explain how R and gamma change?
LESLIE KAELBLING: Right.
So the answer-- so the question is, can I
explain how R and gamma change between that case and this one.
The answer is, I haven't worked it all the way out
but you could, or I could later.
But it's going to be that--
in some sense, the distances--
if we transform these points in the high dimensional space,
the R gets big but the gamma--
the R gets bigger faster than the gamma gets big.
But I can't-- we'd have to do the work,
and I can't do it right here and now.
Yep.
One more.
Yep?
AUDIENCE: So is there any difference
between transforming data into a higher
dimension versus classifying into a higher dimension.
And at what point [INAUDIBLE].
LESLIE KAELBLING: OK.
Good.
So first question is, is there a difference
between transforming the data into higher dimensions
and transforming the classifier into higher dimensions.
And the answer is no.
As soon as we put the data in a higher dimensional space,
if we're doing a linear classifier,
it will have the same number of parameters
as the dimensionality of the new space.
So as we go to higher dimensional space,
we get more parameters.
And that more parameters is what gives us the freedom
to make funky shapes.
But you are foreshadowing a thing which
is as we let the shapes get funkier and funkier,
we can begin to overfit our data.
We can begin to kind of become too
attached to the particular placement of the data points.
And that's going to be a theme that we
hit more and more as we go.
Yeah.
I'm going to show you one more demo
and talk about one more idea, and then we should be done.
One more demo is this one.

So let's try this with second order.
So now I'm not drawing all the steps because there's too many.
So what did I do here?
So here's a data set.
Now it's sort of more complicated.
There's more pluses and minuses.
It's not as easy as the previous one.
What I did is I ran perceptron for a while.
It made 966 mistakes.
This is a second order polynomial basis.
It made 966 mistakes.
And this is the best it could do, so not so good.
You can see why I'm not stepping through it now.
So let's try third order basis.
So we could try third order--

syntax error.
OK.
Sure.
Fine.
We get this.
It's still not all the way right but it's charming.
So let's try again.
Fourth order.
So now it makes no mistakes.
I mean, no, it made a lot of mistakes,
but it got a good hypothesis.
It made 606 mistakes.
Same old perceptron algorithm, just
in this higher dimensional space.
And it gets it right now, but it's got, like, a crazy shape.
And this gets to the over fitting question.
Like, if one of those was really noise in some sense
and there was a simpler hypothesis,
we're not going to get it this way.


### Lecture: Feature representation - transforming through-origin to not-through-origin


LESLIE KAELBLING: What I want to do
now is just talk about some other strategies
for dealing with data.
What I talked about here with the polynomial basis was,
if you have some features and they're numbers
and what you'd like to do is give yourself
a richer way of talking about hypotheses,
that's a way to do it.
But the other thing that we should talk about a little bit
is how do we take data that comes from a real source
and encode it as input to some kind of a learning algorithm?
So in particular, imagine that you
have some data that's discrete.
So imagine you have some discrete input data.
So like it could be something like the number of cylinders
in a car, or it could be the phone--
the manufacturer of your phone, or it
could be the third digit of your social security number,
or it could be a bunch of things.
And the question is, if you have data like that
and you want to make some kind of a classifier,
maybe you have this and you have other features, too, right?
Generally speaking, you-- we'll just take one.
The question is, how could you make
that be the input to some kind of a linear classifier?
Like what would be a good way to do that?
And so let's just talk through some alternative strategies
for how you might deal with something like this.
So in one case, you could just treat it
as a numeric attribute.

Like you could treat it as being in--
basically in some real valued space.

That might make sense for something
like the number of cylinders of a car,
because you believe that the numeric value kind of has
some semantics, and that one is-- that 2 is bigger than 1
and that 3 is bigger than 2, and that the gap between 1 and 2
is the same as the gap between 2 and 3, all right?
So that kind of has a reasonable numeric interpretation.
On the other hand, if you were to just assign numbers to the--
Samsung is 1 and Apple is 2 and so on,
that would not be a thing that's sort of sensibly
encoded as a number, because those numerical values,
even though you can make a assignment of numerical values,
they don't have the right kind of properties of real numbers.
So the semantic notion that we have
that goes with a number scale doesn't apply.
So you can encode things as numbers,
but it's only sensible if those numbers are really
kind of like interpretable as numbers.

You could do something called one-hot.

So imagine you have-- your discrete value--
your discrete feature has m possible values.

Then one-hot encoding would say that I'm actually
going to encode this as a vector of m Booleans.

Either plus 1 or minus 1, or actually usually it's 1 and 0.
So we would encode the first value as this one,
and the second one would be--

and then-- does that make sense?
All right?
So phone manufacturers, this is how I would do it.
You've got four phone manufacturers.
I might say, OK, I'm going to have four different input
features.
Each one could be 1 or 0 or 1 or minus 1,
that's the thing you'd have to play with and see
what worked out better.
And then I'm just going to turn one of them
on at a time, all right?
This is a way of saying, look, these don't-- they don't make
sense numerically, this one is not bigger than that one,
they don't really even have a relationship to each other.
I just need to indicate to my classifier
which situation we're in.
So that's called a one-hot encoding.

You might be tempted--
the computer scientists, the few of you in the back are going,
grumble, grumble, that seems very inefficient,
I know about binary.
You could say, oh, I'm going to use binary numbers,
because then I only need log of m bits
to encode my thing into binary number.
This is a terrible idea.
Worst, bad, terrible, bad, don't do it idea.
Why?
If you take the trouble to code up
your values into binary number, and your machine learning
algorithm needs to do something special for Apple
phones versus Samsung phones, it has
to learn how to decode binary, right?
And why put that burden on your own algorithm?
It could.
Maybe you fancy enough big algorithm,
you could decode the binary, but why?
Don't do it.
OK, so you can compress--
generally speaking, compression makes things smaller but harder
to deal with, and one principle that we have learned finally
in machine learning is that smaller is not what we want.
Generally bigger and more spacious, more roomy.
Easy to interpret encodings are things that we like.
So we don't mind if it's big, but we do mind
if it's like gnarly.
OK, so that's not a good choice.
Let's do a little example of something--
do we have time for this?
We do.
Now I'm going to talk about factoring.

So one thing that can be nice--
so my example here, I'm just going
to illustrate this with an example, is blood types.

So I don't know if you know how blood types work, but roughly
there's A plus, A minus--
it seems like grades--
B plus, B minus, O plus, O minus.

So if you said there's six blood types,
I am going to-- you could.
You can say I'm going to assign them integers 1 through 6.
Bad idea, because they're not related in a scale.
You could say, I'm going to do one-hot encoding.
Pretty good idea.
Much better than thinking of them as 1 through 6.
But actually, if you know something--
which I don't, really, but apparently
if you know something about blood types,
there's some substructure.
Actually, that even the way they're
written indicates to you that there's
some substructure, right?
That there's at the very least, you might say,
well, I have A, B, and O, and I have plus and minus.
And you might imagine-- you, clever human,
who knows something about your domain,
you might imagine that it's useful to tell the algorithm,
yo, this might be something that you could consider,
and this feature might be something that you consider,
and they might have somewhat independent or systematic
relation to one another.
If that's true, you should encode these guys
as a group and these guys as a group.
So you could do a one-hot here with 3 bits
and one-hot here with 2, and that might be better.

There's also-- oh, OK.
But you know what?
There's also AB.
Oops, I forgot the AB's.
There's AB.
AB positive and AB negative.
OK, so this is A, B, O, and AB, so there's four there.
OK.
It turns out that like--
O is like not having anything.
So in fact, you might factor this even more.
You might say, do I have A or do I not have A?
Do I have B or do I not have B?
And then do I have the plus or do I have a minus?
And that might be an even better way
to describe what's going on.
So this is an example of the kind of situation
where if you know something about your problem,
not like exactly about the answer
or what the numbers should be, but like about what
the inputs mean, you can make a big difference
to the success of your learning algorithm
by taking what you already understand
like about the systematic or the structure
or the encoding of the information,
and putting that in the way-- in the encoding
and your choice of how you represent your problem
as features to the algorithm.
OK, I want to say one more thing, which is about numbers.

Just generally speaking, if you have a data set that
has numbers, numeric features, you may find that some of them
are in tiny units and some are in big units,
and so some features we like microns
or some are like really big.
And generally, that makes a problem
really hard for a machine learning algorithm.
And you can kind of almost intuitively
understand it in terms of this r and gamma,
that if you have some very small things,
you might have to actually-- you might
have kind of a small gamma.
But if you have some big ones, that might make your r big.
So you might need big slopes or something to get that right,
and that could take a long time for the perceptron to find.
Usually what we'd like to do is standardize numerical features.

And that means if you have some input feature, xi--
I'll call it j just because we've been using i's for cases.
So some particular dimension, right?
So this is like your temperature or the thickness of your hair
or something like that, all right?
So big numbers, small number.
Generally speaking, what we'd like to do
is replace it with some other value, which
is gotten by taking the actual xj,
subtracting the average value of the xj's in your data,
and dividing through by the standard deviation.
So this is the mean and this is the standard deviation.

That will put all your data so that it's
on average 0 and standard deviation 1.
So it puts all the numeric values
kind of on the same footing.
Moves it over to the origin, puts
a ball, a size 1-ish ball around it,
and that tends to make things better.
I don't know of cases where it makes it worse,
so people kind of just about do it by default.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/b5ca509c17bab346cc6252ca41a1aac7/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Feature_representation.pdf
