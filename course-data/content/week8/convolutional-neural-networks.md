# Convolutional Neural Networks

> Week 8: Convolutional Neural Networks · MIT 6.036 courseware archive

## Notes – Chapter 9: Convolutional Neural Networks

Notes – Chapter 9: Convolutional Neural Networks
You can sequence through the Convolutional Neural Networks lecture video and note segments (go to Next page).
F19 Lecture Slides
are also available.
You can also (or alternatively) download the
Chapter 9: Convolutional Neural Networks
notes as a PDF file.

## Lecture: CNNs - convolutional neural networks - intro

Lecture: CNNs - convolutional neural networks - intro
Lecture: CNNs - convolutional neural networks - intro
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Introduction to CNNs

Introduction to CNNs
So far, we have studied what are called
fully connected
neural networks, in which all of the units at one layer are connected to all of the units in the next layer. This is a good arrangement when we don't know anything about what kind of mapping from inputs to outputs we will be asking the network to learn to approximate. But if we
do
know something about our problem, it is better to build it into the structure of our neural network. Doing so can save computation time and significantly diminish the amount of training data required to arrive at a solution that generalizes robustly.
One very important application domain of neural networks, where the methods have achieved an enormous amount of success in recent years, is signal processing. Signals might be spatial (in two-dimensional camera images or three-dimensional depth or CAT scans) or temporal (speech or music). If we know that we are addressing a signal-processing problem, we can take advantage of
invariant
properties of that problem. In this chapter, we will focus on two-dimensional spatial problems (images) but use one-dimensional ones as a simple example. Later, we will address temporal problems.
Imagine that you are given the problem of designing and training a neural network that takes an image as input, and outputs a classification, which is positive if the image contains a cat and negative if it does not. An image is described as a two-dimensional array of
A
pixel
is a “picture element."
pixels
, each of which may be represented by three integer values, encoding intensity levels in red, green, and blue color channels.
There are two important pieces of prior structural knowledge we can bring to bear on this problem:
Spatial locality:
The set of pixels we will have to take into consideration to find a cat will be near one another
So, for example, we won't have to consider some combination of pixels in the four corners of the image, in order to see if they encode cat-ness.
in the image.
Translation invariance:
The pattern of pixels that characterizes a cat is the same no matter where in the image
Cats don't look different if they're on the left or the right side of the image.
the cat occurs.
We will design neural network structures that take advantage of these properties.
Download this chapter as a PDF file
This page was last updated on Thursday December 12, 2019; 09:33:44 PM (revision 4b592d7d7)

## Lecture: CNNs - one-dimensional filters

Lecture: CNNs - one-dimensional filters
Lecture: CNNs - one-dimensional filters
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: CNNs - two-dimensional filters

Lecture: CNNs - two-dimensional filters
Lecture: CNNs - two-dimensional filters
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: CNNs - a specific illustrative example filter

Lecture: CNNs - a specific illustrative example filter
Lecture: CNNs - a specific illustrative example filter
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: CNNs - convolutional neural network layers

Lecture: CNNs - convolutional neural network layers
Lecture: CNNs - convolutional neural network layers
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Filters

Filters
We begin by discussing
Unfortunately in AI/ML/CS/Math, the word “filter" gets used in many ways: in addition to the one we describe here, it can describe a temporal process (in fact, our moving averages are a kind of filter) and even a somewhat esoteric algebraic structure.
image filters
. An image filter is a function that takes in a local spatial neighborhood of pixel values and detects the presence of some pattern in that data.
Let's consider a very simple case to start, in which we have a 1-dimensional binary “image" and a filter [mathjaxinline]F[/mathjaxinline] of size two. The filter is a vector of two numbers, which we will move along the image, taking the dot product between the filter values and the image values at each step, and aggregating the outputs to produce a new image.
Let [mathjaxinline]X[/mathjaxinline] be the original image, of size [mathjaxinline]d[/mathjaxinline]; then pixel [mathjaxinline]i[/mathjaxinline] of the the output image is specified by
[mathjax]Y_ i = F \cdot (X_{i-1}, X_ i)\; \; .[/mathjax]
To ensure that the output image is also of dimension [mathjaxinline]d[/mathjaxinline], we will generally “pad" the input image with 0 values if we need to access pixels that are beyond the bounds of the input image. This process of applying the filter to the image to create a new image
And filters are also sometimes called
convolutional kernels
.
is called “convolution."
If you are already familiar with what a convolution is, you might notice that this definition corresponds to what is often called a correlation and not to a convolution. Indeed, correlation and convolution refer to different operations in signal processing. However, in the neural networks literature, most libraries implement the correlation (as described in this chapter) but call it convolution. The distinction is not significant; in principle, if convolution is required to solve the problem, the network could learn the necessary weights. For a discussion of the difference between convolution and correlation and the conventions used in the literature you can read section 9.1 in this excellent book:
https://www.deeplearningbook.org
.
Here is a concrete example. Let the filter [mathjaxinline]F_1 = (-1, +1)[/mathjaxinline]. Then given the first image below, we can convolve it with filter [mathjaxinline]F_1[/mathjaxinline] to obtain the second image. You can think of this filter as a detector for “left edges" in the original image—to see this, look at the places where there is a [mathjaxinline]1[/mathjaxinline] in the output image, and see what pattern exists at that position in the input image. Another interesting filter is [mathjaxinline]F_2 = (-1, +1, -1)[/mathjaxinline]. The third image below shows the result of convolving the first image with [mathjaxinline]F_2[/mathjaxinline].
Study Question:
Convince yourself that filter [mathjaxinline]F_2[/mathjaxinline] can be understood as a detector for isolated positive pixels in the binary image.
Two-dimensional versions of filters like these are thought to be found in the visual cortex of all mammalian brains. Similar patterns arise from statistical analysis of natural images. Computer vision people used to spend a lot of time hand-designing
filter banks
. A filter bank is a set of sets of filters, arranged as shown in the diagram below.
All of the filters in the first group are applied to the original image; if there are [mathjaxinline]k[/mathjaxinline] such filters, then the result is [mathjaxinline]k[/mathjaxinline] new images, which are called
channels
. Now imagine stacking all these new images up so that we have a cube of data, indexed by the original row and column indices of the image, as well as by the channel. The next set of filters in the filter bank will generally be
three-dimensional
: each one will be applied to a sub-range of the row and column indices of the image and to all of the channels.
These 3D chunks of data are called
We will use a popular piece of neural-network software called
Tensorflow
because it makes operations on tensors easy.
tensors
.
The algebra of tensors is fun, and a lot like matrix algebra, but we won't go into it in any detail.
Here is a more complex example of two-dimensional filtering. We have two [mathjaxinline]3 \times 3[/mathjaxinline] filters in the first layer, [mathjaxinline]f_1[/mathjaxinline] and [mathjaxinline]f_2[/mathjaxinline]. You can think of each one as “looking" for three pixels in a row, [mathjaxinline]f_1[/mathjaxinline] vertically and [mathjaxinline]f_2[/mathjaxinline] horizontally. Assuming our input image is [mathjaxinline]n \times n[/mathjaxinline], then the result of filtering with these two filters an [mathjaxinline]n \times n \times 2[/mathjaxinline] tensor. Now we apply a tensor filter (hard to draw!) that “looks for" a combination of two horizontal and two vertical bars (now represented by individual pixels in the two channels), resulting in a single final [mathjaxinline]n \times n[/mathjaxinline]
When we have a color image as input, we treat it as having 3 channels, and hence as an [mathjaxinline]n \times n \times 3[/mathjaxinline] tensor.
image.
We are going to design neural networks that have this structure. Each “bank" of the filter bank will correspond to a neural-network layer. The numbers in the individual filters will be the “weights" of the network, which we will train using gradient descent. What makes this interesting and powerful (and somewhat confusing at first) is that the same weights are used many many times in the computation of each layer. This
weight sharing
means that we can express a transformation on a large image with relatively few parameters; it also means we'll have to take care in figuring out exactly how to train it!
We will define a filter layer [mathjaxinline]l[/mathjaxinline]
For simplicity, we are assuming that all images and filters are square (having the same number of rows and columns). That is in no way necessary, but is usually fine and definitely simplifies our notation.
formally with:
number
of filters [mathjaxinline]m^ l[/mathjaxinline];
size
of filters [mathjaxinline]k^ l \times k^ l \times m^{l-1}[/mathjaxinline];
stride
[mathjaxinline]s^ l[/mathjaxinline] is the spacing at which we apply the filter to the image; in all of our examples so far, we have used a stride of 1, but if we were to “skip" and apply the filter only at odd-numbered indices of the image, then it would have a stride of two (and produce a resulting image of half the size);
input tensor size
[mathjaxinline]n^{l-1} \times n^{l-1} \times m^{l-1}[/mathjaxinline]
This layer will produces output tensor of size [mathjaxinline]n^ l \times n^ l \times m^ l[/mathjaxinline], where [mathjaxinline]n^ l = \lfloor n^{l-1} / s^ l \rfloor[/mathjaxinline]. The weights are the values defining the filter: there will be [mathjaxinline]m^ l[/mathjaxinline] different [mathjaxinline]k^ l \times k^ l \times m^{l-1}[/mathjaxinline] tensors of weight values.
This may seem complicated, but we get a rich class of mappings that exploit image structure and have many fewer weights than a fully connected layer would.
Study Question:
How many weights are in a convolutional layer specified as above?
Study Question:
If we used a fully-connected layer with the same size inputs and outputs, how many weights would it have?
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:14 PM (revision 4f166135)

## Lecture: CNNs - max pooling

Lecture: CNNs - max pooling
Lecture: CNNs - max pooling
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Max Pooling

Max Pooling
It
Both in engineering and in nature
is typical
to structure filter banks into a
pyramid
, in which the image sizes get smaller in successive layers of processing. The idea is that we find local patterns, like bits of edges in the early layers, and then look for patterns in those patterns, etc. This means that, effectively, we are looking for patterns in larger pieces of the image as we apply successive filters. Having a stride greater than one makes the images smaller, but does not necessarily aggregate information over that spatial range.
Another common layer type, which accomplishes this aggregation, is
max pooling
. A max pooling layer operates like a filter, but has no weights.
You can think of it as a pure functional layer, like a ReLU layer in a fully connected network.
It has a filter size, as in a filter layer, but simply returns the maximum value
We sometimes use the term
receptive field
or just
field
to mean the area of an input image that a filter is being applied to.
in its field.
Usually, we apply max pooling with the following traits:
[mathjaxinline]\text {stride} > 1[/mathjaxinline], so that the resulting image is smaller than the input image; and
[mathjaxinline]k \geq \text {stride}[/mathjaxinline], so that the whole image is covered.
As a result of applying a max pooling layer, we don't keep track of the precise location of a pattern. This helps our filters to learn to recognize patterns independent of their location.
Consider a max pooling layer of [mathjaxinline]\text {stride} = k = 2[/mathjaxinline]. This would map a [mathjaxinline]64 \times 64 \times 3[/mathjaxinline] image to a [mathjaxinline]32 \times 32 \times 3[/mathjaxinline] image.
Study Question:
Maximilian Poole thinks it would be a good idea to add two max pooling layers of size [mathjaxinline]k[/mathjaxinline], one right after the other, to their network. What single layer would be equivalent?
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:14 PM (revision 4f166135)

## Lecture: CNNs - typical architecture

Lecture: CNNs - typical architecture
Lecture: CNNs - typical architecture
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: CNNs - backprop and gradient descent

Lecture: CNNs - backprop and gradient descent
Lecture: CNNs - backprop and gradient descent
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Typical architecture

Typical architecture
Here is the form of a typical convolutional network:
Figure 1
:
*
Source: https://www.mathworks.com/solutions/deep-learning/convolutional-neural-network.html
After each filter layer there is generally a ReLU layer; there maybe be multiple filter/ReLU layers, then a max pooling layer, then some more filter/ReLU layers, then max pooling. Once the output is down to a relatively small size, there is typically a last fully-connected layer, leading into an activation function such as softmax that produces the final output. The exact design of these structures is an art—there is not currently any clear theoretical (or even systematic empirical) understanding of how these various design choices affect overall performance of the network.
The critical point for us is that this is all just a big neural network, which takes an input and computes an output. The mapping is
Well, the derivative is not continuous, both because of the ReLU and the max pooling operations, but we ignore that fact.
a differentiable function
of the weights, which means we can adjust the weights to decrease the loss by performing gradient descent, and we can compute the relevant gradients using back-propagation!
Let's work through a
very
simple example of how back-propagation can work on a convolutional network. The architecture is shown below. Assume we have a one-dimensional single-channel image, of size [mathjaxinline]n \times 1 \times 1[/mathjaxinline] and a single [mathjaxinline]k \times 1 \times 1[/mathjaxinline] filter in the first convolutional layer. Then we pass it through a ReLU layer and a fully-connected layer with no additional activation function on the output.
For simplicity assume [mathjaxinline]k[/mathjaxinline] is odd, let the input image [mathjaxinline]X = A^0[/mathjaxinline], and assume we are using squared loss. Then we can describe the forward pass as follows:
[mathjaxinline]\displaystyle  Z_ i^1[/mathjaxinline]
[mathjaxinline]\displaystyle = {W^1}^ T \cdot A^0_{[i-\lfloor k/2 \rfloor : i + \lfloor k/2 \rfloor ]}[/mathjaxinline]
[mathjaxinline]\displaystyle A^1[/mathjaxinline]
[mathjaxinline]\displaystyle = ReLU(Z^1)[/mathjaxinline]
[mathjaxinline]\displaystyle A^2[/mathjaxinline]
[mathjaxinline]\displaystyle = {W^2}^ T A^1[/mathjaxinline]
[mathjaxinline]\displaystyle L(A^2, y)[/mathjaxinline]
[mathjaxinline]\displaystyle = (A^2-y)^2[/mathjaxinline]
Study Question:
For a filter of size [mathjaxinline]k[/mathjaxinline], how much padding do we need to add to the top and bottom of the image?
How do we update the weights in filter [mathjaxinline]W^1[/mathjaxinline]?
[mathjax]\frac{\partial \text {loss}}{\partial W^1} = \frac{\partial Z^1}{\partial W^1} \cdot \frac{\partial A^1}{\partial Z^1} \cdot \frac{\partial \text {loss}}{\partial A^1}[/mathjax]
[mathjaxinline]\partial Z^1/\partial W^1[/mathjaxinline] is the [mathjaxinline]k \times n[/mathjaxinline] matrix such that [mathjaxinline]\partial Z_ i^1/\partial W_ j^1 = X_{i-\lfloor k/2 \rfloor +j-1}[/mathjaxinline]. So, for example, if [mathjaxinline]i = 10[/mathjaxinline], which corresponds to column 10 in this matrix, which illustrates the dependence of pixel 10 of the output image on the weights, and if [mathjaxinline]k = 5[/mathjaxinline], then the elements in column 10 will be [mathjaxinline]X_8, X_9, X_{10}, X_{11}, X_{12}[/mathjaxinline].
[mathjaxinline]\partial A^1/\partial Z^1[/mathjaxinline] is the [mathjaxinline]n \times n[/mathjaxinline] diagonal matrix such that
[mathjaxinline]\displaystyle \partial A_ i^1/\partial Z_ i^1= \begin{cases}  1 &  \text {if $Z_ i^1 > 0$} \\ 0 &  \text {otherwise} \end{cases}[/mathjaxinline]
[mathjaxinline]\partial \text {loss}/\partial A^1 = \partial \text {loss}/\partial A^2 \cdot \partial A^2/\partial A^1 = 2(A^2 - y)W^2[/mathjaxinline], an [mathjaxinline]n \times 1[/mathjaxinline] vector
Multiplying these components yields the desired gradient, of shape [mathjaxinline]k \times 1[/mathjaxinline].
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:14 PM (revision 4f166135)

## Video transcripts

### Lecture: CNNs - convolutional neural networks - intro


LESLIE KAELBLING: We talked the last two lectures
about the kind of standard fully connected
feed-forward neural networks.
And what they're appropriate for is a kind
of general-purpose transformation from inputs
in some space to an output like a regression
or a prediction of a classification
or something like that.
And the networks that we looked at are mostly generic.
They don't have much in the way of assumptions built into them.
And so if you don't know too much about your problem,
then you might just say, OK, I'm going
to set something up with these layers
that we looked at this last time.
But one of the things that we talked about a little bit
at the beginning of the class and that we understand
from kind of the theory of machine learning and so on
is that if you know something about the space
of possible answers to your problem,
you should try to build-- and you're
pretty sure it's true-- you should
try to build it into your machine learning
hypothesis space if you can.
And so at first if you look at neural networks
it looks like, well, it's kind of hard to build
in some knowledge that we might have
about our hypothesis space.
But actually what a lot of people are doing right now,
a huge amount of the work in research and practice
in neural networks at this moment,
is in this process of trying to understand
some structural underlying principles of the domains
that we're looking at and figuring out
how to build those structural principles into the structure
of the networks that we build in a way that makes learning more
computationally efficient, and in particular more data
efficient, so that it doesn't take so much data
to get a good estimate of the parameters in the network.
So there's been a ton of work on that
and it's a huge ongoing thing.
What we're going to do today is talk about, in particular,
networks that we're designed to do signal processing
kinds of problems.
So we'll think about a signal as some kind
of set of values that are related to one another
in some systematic way.
You can think of an image as a two-dimensional signal.
You can think of speech or language or music
as a kind of a one-dimensional or kind of a temporal signal.
So we'll today focus on thinking about things like images, where
we have the whole input in our hands
all at once, so it's not arriving piece by piece,
but where the inputs are related to each other
in a systematic way.
So normally, if we have a normal kind of neural network
that we've been looking at so far,
the inputs there might be 100 dimensions or 1,000 dimensions
of input, but we don't know anything
in particular about how dimension 1 is
related to dimension 2.
We don't make any assumptions about that.
There's nothing built into the neural network about that.
But if we take an image in as input,
we know something about how the different pixels in the image
are related to each other, and we're
going to take advantage of that.
So the canonical, the example everybody talks about.
So here's an image.
And as you probably know, in a computer
an image is divided up into pixels.
So this is a pixel.

In a black-and-white image it might
be just one value, typically an integer of maybe between 0
and 255 or something like that.
Could be binary.
We'll talk in just a minute about how we're going
to think about color images.
But so here's an image, and maybe what
we're interested in doing is building
some kind of a hypothesis that takes in an image
and gives out a plus 1 minus 1 a plus 1 0,
and it's trying to answer the question,
is there a cat in the image?
Let's say that's thing we want to do.
So all right, I look and I say, oh yes--
I can't draw-- whiskers, good, cat.
Yay, cat.
OK, so how are we going to think about--
What we're going to do is think about questions like,
is there a cat in this image?
Or did I just say a particular word in this speech signal?
That might be a question we would ask.
So what is it about images that makes
them need special treatment.
Or actually not need special treatment
but allow us to treat them specially?
So there's two really fundamental principles
about this question of trying to find a cat in an image.
It's not just for cats, it's just our example.
So one important thing is spatial locality.

And what we mean by this is that the set of pixels
in this image--
I mean, generally speaking this image might be really big,
and the cat might just take up a little piece of it.
We're not quite sure.
But wherever the cat is and however we're
going to think about the cat, it's
likely that the pixels that go into helping us answer
the question of whether a cat is here
are next to each other in the image.
It's wildly unlikely, if what I'm
doing is a kind of an object detection and image problem,
it's wildly unlikely that I would
need to look at this pixel and this one
down here and this one over here in order
to answer the question, is there a cat here?
Now if I'm answering a different question,
like what's the ambient light level in the room when
I took this image, maybe I will have
to look at all the pixels or a funny combination
of the pixels.
But if I'm answering sort of content-based questions,
because the objects in the world kind of cohere as clumps
of matter, then they tend to take up
pixels that are next to each other in the image.
That's just kind of a fundamental property of what
images and things are like.
So the pixels that are about a certain question
tend to be next to each other in the image.
And we know what "next to each other" means, right?
When we process an image, it will be important to us
to get these pixels in the right order.
Before when we were doing feed-forward neural networks,
planal, we got a bunch of inputs and then we multiply by weights
and then we get something else, you
could have re-ordered the input dimensions
and done the same learning algorithm
and gotten the same answer.
I mean, the weights would be permuted
to match the permutation that you made of your inputs,
but it wouldn't have changed anything about our ability
to represent stuff.
Now we're going to start treating images
in a way that takes seriously the arrangement of stuff
in the image.
So that if you were to permute the pixels
in the image, the networks that we're going to build now,
they wouldn't work anymore.
Because if you were to randomly rearrange the pixels
in the image, although the information would still
be there in some sense, we would have lost this principle
of spatial locality.
So that's going to be important to us.
The other important principle is translation invariance.

And the idea of translation invariance
is roughly a cat looks like a cat
no matter where it is in the image.
So if I have some ability to look at a group of pixels
and say, oh yeah, that looks like a cat,
I could apply it here and here and over here
and up there and all over, and it would be fine,
because cats tend not to look different depending on where
they appear in the image.
Now this is not an absolute certainty.
If cats could fly, possibly they would
look different in the tops of images than the bottom.
Or maybe if they're near the bottom they tend to be asleep.
So OK, so there might be some small variation,
but generally speaking, it doesn't
matter where the thing is in the image, it'll look the same.
So we're going to try to build these ideas
into the neural networks that we design to solve questions
about in particular what kinds of things
there are in an image.

So in order to get anywhere with this,
we have to introduce some kind of basic principles of image
processing because in fact, the way that--
Actually let me just introduce the name of this lecture.
So we're going to talk about convolutional neural networks,
known to their friends as CNNs, it's not Cable News Network,
however.


### Lecture: CNNs - convolutional neural networks - intro


LESLIE KAELBLING: So let's talk about the idea of a filter.

And I'm going to do a bunch of examples
today that involve one-dimensional images,
because they're easier to think about and work with,
but the ideas will generalize to two-dimensional images.
So I'm going to make an image here,
a particular example image in mind that I have in the notes.
Three 1's.

0.

There's an image.
It's a binary image.
It's got 1's and 0's in it.

Not super exciting, as images go,
but, anyway, that's an image, give it that way.
OK, so a filter is a local process
that-- it's a process you could do it to take one image
and turn it into another image.
And it involves doing some local process all over the image--
same process.
And in fact, actually, it used to be
sort of hard to explain this, but now
everybody's has applied a Snapchat filter, right?
So a filter is now in the common, popular parlance.
It used to not be, right?
So you guys are probably all familiar with the idea that you
could take an image that looks like a normal image
and then make it look old time or make it look space age
or make it look blurry or make it look purple or whatever.
So probably everybody has actually done image filtering.
I forget this now.
OK.
So you guys are all experts in what
it means to filter an image.
The question is, maybe, how would we implement that?
So let me show you a really simple filter.
We'll call this Filter 1.
And it's going to look like this--
minus 1, plus 1.
So a filter for us, a linear filter--
so we're just going to think about linear filters--
a filter is a--
you can think of it as a small chunk of image.
Generally speaking, if our image is one-dimensional,
our filter would be one-dimensional.
So it's a small chunk of image.
So this is a very small chunk of image.
It's a chunk of image.
And the way we're going to use a filter to compute a new image
is to run it along the image and take that product.
So if we take these two guys, these first two
elements in our image, and dot product
with the filter, that's going to give us,
let's say, the first value in our image.
And then we're going to slide it over by one
and take the dot product.
So minus 1 times 0, plus 1 times 1.
That gives me a 1.

And then I run it over here--
minus 1 times 1, plus 1 times 1.
That gives me a 0.
And I run it over here.
And we know that with 1, 1 that gives me a 0.
And we put it here, and we get to minus 1.
We put it here.
We get a plus 1 again.
Here we get a minus 1, then we're going to get 0's.

OK.
Is that cool?
Do you guys-- do you see what I did to go from this image
to this image using that filter?

OK.
So sometimes you can think of filters
as a detector for something.
So if you look at the image that resulted here,
we get two plus 1's.
What happened?
What was going on in the original image in the places
that plus 1's appeared in this resulting image?

STUDENT: [INAUDIBLE].

LESLIE KAELBLING: It's a place where we went from a 0 to a 1,
right?
Because the place where we go from 0 to a 1,
this minus 1 guy doesn't come on and the plus 1 does,
and we get a plus 1 down here.
So you can think of this filter, if you
want to, as a left edge detector, in a simple sense,
in the simple 1D image.
So it's a thing I can do to this image,
and it finds all the places in this image where
a certain thing was going on and puts a plus 1 there.
Now you could imagine if these are not just binary
and this is not just binary, you get, instead of, oh, yeah,
I definitely detected this, you get
maybe a degree to which this matches that,
but that's kind of the idea.
So another example might be the filter that looks like minus 1,
plus 1, minus 1.
So this is a filter now.
It's a bigger filter.
And I could apply it to this image.
And, in fact, I will let you work it out on your own.
But you might see that there is a place right over here
where we're going to get a plus 1.
And, in fact, that's the only place
in the image where we're going to get a plus 1.
So what's it a detector for?
Yeah?
STUDENT: 0, 1, 0.
LESLIE KAELBLING: It's a detector for 0, 1, 0.
Yeah, or you could think of it as a detector
for an isolated pixel or something like that.
OK.
So this is the idea of filtering.
One thing that you might notice is
that if I want this resulting image
to come out the same size as the one I started with--

which often we do.
Often if we start with an image of size n,
we'd like to end up with an image of size n.
Sometimes we have to--

like in this case, if I wanted--
generally if I have an odd-sized filter,
it's often the case that I would like
to start it out by lining this plus 1 up right here.
But if I line the middle of the filter
up with my first pixel to compute this value,
I need a secret hidden value over there, which I don't have.
So this is called padding.

And we'll most often secretly pad with a zero
or however many zeros we need.
So I would take this pattern, put it here,
see that I get minus 1, and write a minus 1
there in my image.

OK.
This process of taking a filter and running it along the image
and computing the new image is called convolution--
hence convolutional networks, convolution, or the verb
is to convolve.
I convolve this image with this filter, and I get a new image.
Uh-huh.
STUDENT: [INAUDIBLE].
You have the second though.
Do you run that on the original [INAUDIBLE]
or on the [INAUDIBLE]?
LESLIE KAELBLING: Oh, I see.
I just ran-- just now, I took this image and this filter
and was computing this.
We'll talk, though, about filtering sequentially,
but just to do this example good, I worked on this one
to get this one.


### Lecture: CNNs - convolutional neural networks - intro


LESLIE KAELBLING: We can do this-- generally speaking,
our images will be two-dimensional, right?
So here's our image, right?
Kind of like our cat image, something like that.
And we might say, OK, I want to use these pixels,
and I'm going to apply some filter,
and that's going to give me a new image where,
let's say, maybe these pixels went
into computing the value of that pixel in the new image.
Now, generally speaking, I'll have more than one filter.
So I might have-- imagine that I have four filters.
So I have filter one, filter two, filter three, filter four.
Each one of those--

I am not good at drawing, so you'll
have to just suffer with me here.
Each one of these filters, if I apply it to this image,
gives me a new image, all right?
And so, in keeping with our idea about doing multiple layers
in a neural network, we will indeed
do multiple layers of filters.
And again, like in engineering, people already did this.
There was a notion of a filter bank.
So a filter bank was like sets of filters.
So we would do some, maybe, simple filters first
and then do some further filtering.
One thing that's interesting is that when engineers set out
to design filter banks, they came up
with certain kinds of filters that were systematically
effective, Kind of like things that were, detect a left edge,
or, find a place where the signal goes up and then down
again.
And those same ideas of filtering, you can find,
if you look in mammalian visual cortex,
there are cells in there, neurons,
that seem to be computing filters of this kind, roughly.
In two dimensions, they might be filters
that look for a region that's dark
surrounded by a region that's light,
or an edge of dark on one side, light on the other side.
So there are cells in your brain and lots of other brains
that seem to respond to certain patterns
and images, that kind of create new images out of old images.
And the same basic sets of filters
also appear when you do image processing and look for, kind
of, a statistical underlying basis for representing
little pieces of images.
You tend to get the same set of underlying filters.
So there's a very good basic principled
idea of filters that find useful small local features
in an image.
OK, but there is a big step between finding
little bits of edge, or places where
there's something dark surrounded by something light,
and a cat.
So then the question is, well, how do you go from those things
to cat?
And the answer is layers.
So let's talk about the layers here for a minute,
because things get a little bit interestingly tricky.
So imagine that I start out with one image, and I compute four--
as I filter it, I can evolve it with four different features,
so now I have four different images.
So this thing that I have now, instead of thinking of it
as four different images, I'm going
to think of it as a tensor, which
is a three dimensional-- just think of it as-- it's
fancy name, makes you seem cool, but it's
like a three-dimensional array.
It's not that fancy.
So imagine that our original image was n by n.

And so now-- and we padded it when we filtered,
so we got n by n images.
But now we have four of them.
So no we have an n by n by 4 thing, right?
I just took those images and stacked them up.
I could have done 10, or 100.
It doesn't matter; however many.
And now, I would like to process it further, right?
I would like to do some more filtering on this image.
And so what that looks like now is, again, filtering,
but generally, at this point, I'm
going to filter it with a filter that's three-dimensional.
So my filter now is going to be a tensor.

It's going to look--
I mean, I don't know what it's going to look like.
It's going to be some box full of values.
But what I'm going to do is I'm going to look at--
imagine that this box is 4 by 4, right?
It takes 4 pixels from this image, and 4 from this one,
and 4 from this one, and 4 from this one, right?
So that gives us--
if we take this, all of the bottom right
corners of the image, that gives us a three-dimensional chunk
of stuff.
And then we can dot product that three-dimensional chunk
of stuff with this three-dimensional filter
and get one pixel value in a new image.

Does that-- does that make sense?

And we'll do it--
so let me talk about how we do this exactly, right?
So this filter, let's say that this filter is-- actually,
the way I drew it, this filter might be 2 by 2 by 4, right?
It's going to take a 2 by 2 region in each of the four
images, right?
So it's 2 by 2 by 4.
So I would dot product--
So when I can convolve this tensor with this filter,
the filter will always go all the way deep.
So if my image, if this kind of image tensor right
here is n by n by 4, my filters will all
be something by something by 4.
So these things will match, I guess, I mean,
let's see-- this is convention.
This is not necessary, but this is pretty much what
everybody does, right?
So what I'm going to do, then, is
I'm going to take my 2 by 2 by 4 filter,
and I'm going to run it around this tensor,
varying the row and column but always going all the way deep.
So I'm going to look at all the images I have,
but only in this local area, and I'm
going to use that to compute the next pixel over here.
OK, and obviously, I can do this 10 times
and get 10 images here, which would give me
a 10-deep tensor, which would require me,
if I'm going to filter again, to have my filters be 10 deep.
OK, so basically, I'm going from an image, to a set of images,
to a set of images, to a set of images.
All right.
Often, these things, this dimension, is called channels.

So if I filtered with four filters to get this tensor,
we'll often call these things four channels.
And the channel idea actually can be brought all the way back
to the input.
We talked about color.
Potentially, right, lots of images are color images.
It's interesting to operate on color images.
So if you have a color image, you
can think of it as, actually, three images, right?
One for red, one for green, and one
for blue, or whatever other three-dimensional color space
appeals to you.
But you can think of a color image as being,
now, a tensor, right?
So it would be--
have dimensions n by n by 3, if you had three color channels.
And then nothing would change.
it's just that your first set of filters
would also have to be deep.
They would have to be, like, k, by k,
by 3, so that they could reach through the r and the g.
It makes sense, right?
You probably don't want to--
I guess it depends on what you're doing,
but generally speaking, you probably
don't want to process the red channel separately
from the blue channel or separately
from the green channel, right?
So by making your filters go deep through the three images,
they kind of look at, collectively, what R, G,
and B values are all at once.


### Lecture: CNNs - convolutional neural networks - intro


LESLIE KAELBLING: So let's do an example, carefully constructed
example, six by six.

Is that six?
Yeah.
OK, good.
And imagine my image looks like this.
So these guys-- it's still one black and white image.

So there's my example, work of art.
That's my input image.
And I'm going to filter it with two filters.
I'm going to filter it with a filter that
looks like this, two 3 by 3 filters, 3 by 3 by 1.

So intuitively, this guy is, in some sense,
looking for a vertical stripe of size 3.
You could think of it that way.
And this guy's sort of like looking
for a horizontal stripe, a size 3.
And in fact, if you take this filter
and you run it around this image,
you'll get the most response--

so this is filter 1 that gives us a new image.

Just make my image a little bit bigger.

It's going to give me a new image.
And this new image is going to have a response here
and a response here and a response here.
So there were three places where--
I mean, it's not just going to be a binary response.

If I line this up right here, if these are ones and zeros,
I'll get a plus 3 for when it's right lined up here.
I'll get a plus 2 when it's of 1.
So I'm here just kind of drawing a cartoon version.
I'm showing you the places where it had the most response.

And similarly, if I take this guy,
I will get something that looks like this.
It gets a big response up here, and then some good responses
here and here.

Now let's say I was trying to detect this figure.
So now I'm just designing a detector.
I'm not being a neural network.
I'm designing it.
So if I want to design, if I want
to locate this thing, what I have to say is,
well, I need to find some horizontal bars that
have a certain spacing and some vertical ones that
have a certain spacing, and they have
to be related to the horizontal bars in some way.
So the way I detect that, again, is to say, well, at this point
now, I have whatever-- if this was 6 by 6, my original image,
I now have a 6 by 6 by 2 stack of images.
I didn't draw it as a stack because I
wanted to see them both.
So now I'm going to filter that tensor
with a filter that is 3 by 3 by 2,
and it's going to look like--
so this is my filter number 3.
It's going to look for some guys that
are spaced like this in one channel and some guys
in the other channel that are spaced like this.

So if this is the front part and this is the front part,
so imagine that this is the front, then that's the back.
So this guy's in front.
That's in the back.
This guy is in the front.
This guy in the front.
This guy is in the back, and we rub this filter around these
guys.
Then we'll end up, by filtering with F3,
we'll end up with one image that has
one pixel in the middle, which detected this figure.
Does that seem good?
Yeah?
SUBJECT: [INAUDIBLE]
LESLIE KAELBLING: So what I'm saying
is I started with 6 by 6 by 1 input and I filtered here--
yeah, that's right.
I filtered with 3 by 3 by 1 filters.
But because I had two different filters,
one produced this image, 6 by 6 by 1,
and the other guy produced an image 6 by 6 by 1,
and then I stacked them on top of each other.
So I'm just going to think about.
However many filters I have in this bank.
They're going to produce that many images
that I'm just going to stack them together into a cube.
Well, it won't be a cube, a three-dimensional thing.
Yeah?
SUBJECT: [INAUDIBLE]

LESLIE KAELBLING: Right.
OK, good.
No, no.
In fact, good.
In this case, it would be like over here.
So this is a 6 by 6 by 1, and that dot would correspond--
I mean, in this case, because of the way I've set everything up
and all the filters are odd sized
and so on, that dot would end up,
I'm pretty sure, in this space here.
Depending on the origin of the filters and so on,
it might not actually be like in the center of some figure
that you're looking for, but it'll
be related by some constant offset.
Yeah.
Uh-huh?
SUBJECT: [INAUDIBLE]
LESLIE KAELBLING: OK.
That's a great question.
For F3, does it matter which one is front and which one is back?
Yes, it does.
If I swapped front to back these guys
and ran them over these images looking
for that pattern of relationships,
I would never find it because it's
looking for two that are next to each other in a row,
in one channel, and two that are next to each other--
not quite next to each other, next to each other with a gap,
vertically in the other channel.
So in this particular example, the vertical ones
are in this channel and the horizontal ones
are in this channel.
And if they don't match when I run the filter around,
then I won't get a high response.

Yeah?
SUBJECT: [INAUDIBLE] first channel?

LESLIE KAELBLING: Yeah, OK.
So let's see.
Let's call this channel 1--
I drew them imperfectly-- and call this one channel 2.
Then this will be channel 1, channel 2.
So I probably should have--
yeah, thank you.
Was that your question too?
Yeah, OK.
Sorry.
Uh-huh?
SUBJECT: [INAUDIBLE] has exact same shape [INAUDIBLE]..
LESLIE KAELBLING: Good.
Right.
You say, OK, yes, you're fine.
You found this little figure in your image.

I could have done that with one filter,
and that's totally right.
I could have done it with one filter that looked like this.
And so why don't you do that?
And the answer is if I knew for sure
is that what I was doing in my life
was looking for this thing, that's what I would do.
But what's interesting is that when
you train neural networks to not just recognize
a cat or a particular cat, but look
for a large class of objects, potentially downstream,
it's, generally speaking, the structure that comes out
is one where you look for simple features
and then ways to combine the simple features into more
complicated things and then ways to combine
some features into more complicated things,
so that if you wanted to be able to detect
several different arrangements that were arrangements
of horizontal and vertical bars and size 3,
it would be more efficient to do the horizontal bars,
the vertical bars, and then the combinations, maybe,
than it would be to consider all the possible combinations
that you might want to make.


### Lecture: CNNs - convolutional neural networks - intro


LESLIE KAELBLING: Let's go to actually how
we would make a neural network.
So we'll define just a little bit of notation
and talk about one more kind of filter, and then we can go.
OK.
So let's talk about a filtering layer.

So we might have a filtering layer L. layer L.
Because this is going to be a layer that we
can combine with other kinds of neural network layers
eventually.
So to define a filter layer, we need the number of filters.

We'll call that ml.

We need the size of the filters.

And that will generally be kl.
[INAUDIBLE]

OK, let's just talk about this for a minute.
We're going to restrict our attention
to filters that are square in the way
that they look at the image.
You don't have to.
They don't have to be all 2 by 2 or 3 by 3 or 10 by 10.
They could be 2 by 10.
But it just makes everything more complicated
in terms of what we write on the board,
so we're just going to think about square filters.
So k by k.
And then what's this ml minus 1?
Well, remember that if I had two filters in my previous layer,
what I'm going to have is a tensor with a dimension 2
in the board direction.
And so my filters at this next level better be that deep.
So the depth of the filters at this layer
is to be equal to the number of filters at the previous layer.

OK.
Here's a new idea--
stride.

OK.
Stride idea is this.

Here's my big old image.

And here's my filter.
Let's say it's 3 by 3.
So I might apply it-- actually, I might apply it first
to a ghost.
That's not that big a ghost.
So imagine that I want to center it on this pixel
to compute the new value of the upper left corner.
So maybe I start it there.
So so far, what we've talked about
is then taking this filter and moving it over by 1
and computing the next pixel, and moving it over by another
and computing the next pixel.
And that's how we would compute the top row.
And then we would move it down 1 and compute the next row
in the next image.
The idea of a stride says we might want to not apply it
at absolutely every place.
So if I have stride of 2, I will apply it here.
And then I will skip to here and center my filter on this pixel
and compute a value.
And then center my filter here.
And then when I come down, instead of putting it here,
I'll put it here.
So if I use a stride of 2, I'm going
to get a new image which is like n over 2 by n over 2.
So a stride means that I'm just not going
to look absolutely everywhere.
I'm going to skip some.
So if I have stride as an integer, a positive integer.

Stride.

OK, stride.
Good.
So the input size.
So if my input tensor--
so I'm going to have an input tensor.

So assume it has size n l minus 1 by n l minus 1
by m l minus 1.
And then my output will be--
and I'm going to write it this way
and then we'll talk about it-- nl by nl by ml.
But nl is going to be nl minus 1 divided
by the stride with some floor or something
to make it into an integer size.

That seem cool?
Yep.
STUDENT: Why does the filter have
to be exactly the same size as the preview number of filters?
Like, why can't you have, say, half, and do the [INAUDIBLE]..
LESLIE KAELBLING: You could.
OK.
So let me repeat the question.
The question is, maybe these you're still
looking at up there.
So if I imagine that my first bank of filters is size 10.
So I have a tensor that's 10 deep.
I have 10 images stacked up on top of each other.
What I just said was that my filters at the next level
should also be 10 deep.
It is true that there is a conceptual notion
I could say, no.
I'm going to let my filters be only 2 deep.
But now if I have a filter that's
not as deep as this whole tensor, when I operate it,
when I convolve that tensor with that filter,
I'm going to get a tensor.
Yeah?
Right?
If I'm moving it in depth also.
But if I have 6 of those, now I have a four-dimensional thing.
I don't know what to do with that.
So roughly this keeps things three-dimensional.
There's no math tells us it has to be so.
So this is just a set of conventions.
But it keeps things kind of manageable.
It works out reasonably nicely.
That's really all I can say.
Yeah.
Uh-huh.
Oh, that's a stretch.
No problem.
OK.

So this is a neural network layer.
Where are the weights?
What are the weights?
How many weights are there?
OK.
So where are the weights?
The weights-- what are we going to learn?
What are we going to adjust?
We're going to adjust the entries in the filters.
Those numbers in the filters, those are our weights.
That's what we're going to change around.
And so we are going to have--
so what are our weights here?

Well, we're going to have ml filters.
So that's going to be ml chunks of weights.
And they are going to be of size kl by kl by ml minus 1.
So we're going to have ml chunks of weights.
And that's what's going to be adjustable
in our neural network layer.

So I'm going to define another layer type, talk
about the big network structure, and then we'll, at the very end
we'll do a simple example to show you that in the end,
backprop will do what we need for it to do.
Go backprop.

So there is another kind of layer.
I mean, and again, this is not--

sometimes when you teach, like, math or some fundamental stuff,
there's like, oh, it has to be like this.
This is not a it has to be.
This is a tool kit that people have
found useful kind of thing.
So I'm going to say there's two kinds of layers.
But really, you could invent six more
and they might be really awesome.
We don't have an obvious kind of way to understand that.
OK.
And so the other kind of layer that is popular is max pooling.

Oh, I should say, generally, you can do ReLUs.
You can do a standard nonlinearity
to any of these images.
And we will.
So ReLU is actually the standard nonlinearity that people use.
So if I have a tensor full of values,
what it means to run a tensor through a ReLU
is just the same thing as what it
means when we run a whole vector of values
and if you forward network through ReLU, which
means we just take each individual value
and say it's 0 or positive.
Remember that a ReLU looks like this.
So we can do that with these images also.


### Lecture: CNNs - convolutional neural networks - intro


LESLIE KAELBLING: Max pooling--
so this is a pretty simple idea.
Max pooling layer-- it has no weights.

So it's just a nonlinear transformation.
Really, if it was a nonlinear transformation,
if I apply it to a tensor, I just
get out a tensor of the same size,
because it applies to each guy individually.
Max pooling is a nonlinear thing,
which applies to something more than a single pixel at a time.
And it will have a value.
It will have a stride and a size.

And what it will do--
so imagine that our-- so the size is going to be k l by k l.

Generally speaking, it will maintain the--
so it's per channel.

So the size of the max pooling thing
will be k by k, but it will maintain the same number
of channels as--
so same number of channels in the input and output images.

So let's just think about it operating in one channel.
So if the size is k by k-- so let's say, it's 3 by 3--
it would look in this 3 by 3 window
and compute the max value.
That's all, just take the max.
So we're going to run it along like a filter.
But instead of taking the product
with a bunch of weights, we're just going to take the max.

So it would take the max of these values
and store it in a pixel.

And then, you play with a stride.
The most common-- the most?
I don't know.
A common way to apply it is to have the stride be
the same as the k.
If you apply it with a stride of k and a size of k,
then it takes these guys and gives you one value.
And then it would take these guys
and give you a value, and so on.
So it would just map an n by n image into an n over k
by n over k image.
It doesn't have to be that.
But generally speaking, you want your k--
so stride s-- generally speaking,
you want your k to be bigger than or equal to your stride.

Why is that?
If you have a really big stride and a small k,
you'd be taking the max here, and then kind
of leaping over and taking the max in this window,
and then leaping over and taking the max in this window.
It wouldn't kind of cover the whole image.
So you might lose your cat, which you wouldn't want to do.

OK, so max pooling has a stride and a size.
You apply it to each channel separately.
You get back a new chunk, which has
got the same number of channels as the previous one.
And so it's just-- it's another kind of transformation


### Lecture: CNNs - convolutional neural networks - intro


LESLIE KAELBLING: So how do we put all of this together?
A traditional kind of an architecture
starts with your image.
And maybe it's RGB.
So maybe it's three images, really.

Then we do some filters.

And that gives us some big chunk of stuff.
And then ReLU.

And then maybe we do some more filters and some more ReLU.
So maybe some more filters and some more ReLU.
And then maybe we do a max pooling.
Get us down to a smaller chunk.

And some more filters and some more ReLUs.
Maybe do another max pooling.
Get it down to a smaller chunk.

There is an operation which doesn't actually do anything
but people talk about it.
I have to always remember the name.
Flatten.
So imagine that we started out with, you know, like, I
don't know, 512 by 512 images.
And we've done a bunch of this stuff
and we're down to, like, 8 by 8.
8 by 8 by, I don't know--
8 by 8 by 2.
Imagine we're down to 8 by 8 by 2.
Then that, there is 164, 128 values in there.
We could flatten it into a vector.
So this is called flatten.

Now we have 128 values.
So this is 128.
And then we want to know is there a cat or not.
And so this might be fully connected,
which just means that we have one weight for every input.

Now more generally, let's say we don't just want
to know whether there's a cat.
We have 10 or 100 or 1,000 categories
that we're interested in and we'd
like to know what do we think is in this image.
Then we might have a vector here for all of our categories.
And this would now be a fully connected thing.
And then we might do softmax.

If we have a bunch of categories.
And what would come out of that was something
that you could interpret as a probability distribution
over the categories.
So image comes in over there.
What comes out here is a probability distribution
over the categories.
How likely I think it is to be a cat, or a rhino,
or a laptop, or whatever.

So this is a way that you might kind of construct
the overall architecture.
And exactly how many of these things to have and how deep
and how big and whatever is a matter of debate and contention
and experiment.
There are some standard architectures
that have worked out well And people now tend to adopt them.
I'm sure that they're not the best possible architecture.
It's just it's an enormous amount of work and time
to experiment with these things.
And so kind of what goes on is that somebody
does some experimentation, gets a really good result
on some data set, publishes their architecture,
and then everybody says, whew.
That was really great.
I'm just going to use that same architecture.
Yep.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Uh-huh.
STUDENT: After you get the filters, wasn't there
a certain dimension change, too?
Like, I know the third dimension for RGB--
LESLIE KAELBLING: These three dimensions are RGB.
These dimensions correspond to how many different filters
we apply.
Right.
So if we applied 10 filters, then that's that dimension.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Right.
The number of-- right.
This is always the number of filters
that we did to our previous object.
Yeah.
That's right.
Another thing that's interesting that's kind of going on
in the current neural network world,
especially for image detection and stuff,
is that some people have spent just an enormous number
of CPU hours training on an enormous amount of data.
And they get a representation somewhere in here,
kind of near the end-- maybe even here
at the next-to-the-last layer, but maybe some time before--
which seems to be a pretty good generic representation
for images.
Well, a pretty good generic representation for images
of the kinds that they trained on, because that's all it
could ever be.
Meaning that it's got filters that correspond
to finding useful little subpieces
and then putting the subpieces together, and so on.
So what a lot of people do now-- let's
say I wanted to train up this neural network
to recognize erasers.
but erasers weren't in the original training set,
so I can't use their thing to recognize erasers.
But what I could do is I could take their whole neural network
all the way up to here, maybe even all the way up to here,
and freeze the weights--
all those weights-- and say, OK.
That's good.
I think you made me an awesome representation of images.
But now really, I want to recognize erasers.
So I'm going to train just a last layer that
takes these features and tries to predict whether my image has
an eraser in it or not.
So there's lots of things that people are doing now
where, although this thing would be very difficult to train,
you can take a trained one as long as it was trained
on a big data set of images that are not
so different from the ones that you're interested in,
and then see if you can use it as preprocessing
and just train the last bits to do your job.
That's kind just a--
yep.
STUDENT: [INAUDIBLE] relationship
between filters and weights.
So do you define what the filters are before you enter it
in your neural network?
LESLIE KAELBLING: No.
So, good.
So when you design this neural network,
you design the sizes and shapes of the filters
but not the numbers that are in them.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: There is a different weight
for each pixel in the filter.
That's right.
But note that there are many fewer pixels in the filters
generally than there are in the image.
So the thing that you couldn't--
OK.
So, good.
So why is this a good structure?
Why is it interesting?
Well, one thing is that it has many fewer weights.
Imagine that you had a 512 by 512 image.
And you wanted to train some transformation that would give
you another 512 by 512 image.
Well, if you did a normal neural network layer,
you would need 512 by 512 times 512 by 512 weights.
And that's just like, too crazy.
Too big.
Right?
So we're doing this systematic transformation
from one image to another and we're
learning the transformation, but we're applying it
in a way that respects our basic principles, which
is that chunks of pixels that are near each other
are the ones that I should consider when I try
to do my transformation, and the transformation that I do here
to go from these pixels to this one
is exactly the same as the transformation I do here to go
from these pixels to this one.
Yeah.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: This example?
STUDENT: Yeah.
LESLIE KAELBLING: Yeah
STUDENT: [INAUDIBLE] So would it be a better option
to just train neural networks with 3 by 3 weights f
try to pick out that one computer?
LESLIE KAELBLING: Good.
So that was actually the question
we had earlier, but yeah.
So the answer is, if you knew for sure
that you were interested in finding one 3 by 3 pattern
and that's the only thing you ever cared about
in whole the world, then yes, it would be sensible to just do
a 3 by 3 thing.
The reason potentially for doing an architecture like this
is that it's kind of more flexible
and you can learn little pieces and put them together.
And so that gives you a kind of combinatorial power
for recognizing things.
Yeah.


### Lecture: CNNs - convolutional neural networks - intro


LESLIE KAELBLING: This seems like a big, complicated thing.
But in the end, your output, your y,
is going to be some function of your input x and some weights.
So x is your image.
Weights are the actual detailed, numerical values
that are in your filters.
And in this last layer.
Or wherever you want to put them.
I mean, really, it's up to you to design this things.
But as long as this is a smooth function
and you can compute the gradient, as long as you
can compute dLoss dW--
so you have to again, just as we were doing before,
you define the loss function on the outputs.
If you can compute dLoss dW, you can do gradient and descent.
And that's what we're going to do.
So it's just a function parameterized by the W's.
And it's linear.
It's the transformation of the way we convolve the values
in some image with a filter.
It's just a linear, it's just a dot product.
So we have dot products and maxes.
That's all we have.

OK.
So let's do just a simple example.
I am not going to try to write on the board
how you go all the way through one of these things.
It's not conceptually difficult, but it
becomes notationally kind of like a nightmare.
So let's consider the following example.
So we're going to go back to having a 1D image.
So here's our image.

x.
Also call it A0.
We're going to filter it with some weights.
So we'll call these the weights at layer 1.
It's a filter.

So we're going to have a filter layer.
And that's going to give us Z1.

So we filter with W1.

And because we only have one filter in my super simple
example because too many will just be too many
I's and J's and K's.
So we just get the one image.
So we get one 1 by n image--
n by 1.

And let's say we ReLU it.
So we'll get another one.
We'll call this A1.

And then we do, let's say, a fully connected layer
out to some value here.
And we're going to call this Z2.

I think I wrote something not--
oh, I see.
We'll call this Z2.
And we're going to have a no activation function.

on the output layer.

OK.
Unfortunately, this looks like Z squared.
That's why I had a moment of choke.
But it's just Z2.

OK.
So this is going to be our neural network.
Let's practice how many weights it has in it.

OK.
So let's let our filter be of size k.

So there's k weights in our filter.
And there's n weights in this fully connected layer, or n
plus 1 counting the offset.
So this is a function.
It has n plus k plus 1 parameters.
And we could do gradient descent to set the parameters.

So let's let the loss function-- let's
assume the loss function gets an actual squared error.

Right.
So this is a regression problem.
Image processing doesn't have to be recognition.
It can be regression.
How many cats are there?
How cat-like is this image?
Stuff like that.
OK.
So let's write down the forward path and then
the backward path.
So we're going to think about, again, we're
going to think about backprops.
So we're going to think about how do we
compute the value that comes out,
and then how do we go back and compute a gradient?
So first of all, the only part that's new and interesting
to us is, how do we compute the ith pixel of zy?
So that's the result of filtering.
So the ith pixel of Z1 is weight 1 transpose times A0.
And then we have to take a sub part of that image.
And it'll go from i minus k over 2 floor
to i plus k over 2 floor.

So that just says to compute--
I'm going to assume k is odd.
If k is even, then you just have to hassle more
about getting the indices right.
Let's just assume k is odd.
It's fine.
Just make k odd.
OK.
So then this says, if k is odd, if k were 5, k over 2 floor
would be 2.
And so we would go from I minus 2 to I plus 2.
That would extract a chunk out of the image, a size 5,
and we dot it with our weights.
And that's how we get the i'th value of Z.
OK.
And then after that, it's pretty straightforward, right?
A1 is ReLU.
of Z1.
And A2 is our fully connected weights transposed with a 1.
Oops, Z2.

With A1.
And Z2 is equal to A2.
Yeah.
So feedforward pass is pretty straightforward.
So now the question is, how do we
compute-- what is the gradient like?
So there's going to be one interesting piece
of the gradient.
So I'll write it down.
So we're interested.
If we wanted to do--
I'm assuming that you can just at this point
compute the gradient of the loss with respect
to the weights in the last layer.
That would be the same computation
as if it was one of our normal feedforward networks.

So what we're going to look at here
is the gradient of the loss with respect
to the weights in the first layer.
That is to say, those weights in that filter.

Because that's the new part.
That's the tricky bit.
And we're going to do chain rule fu.
And we'll get something that looks like this.
And again, only one piece of this is going to be new to us.

OK.
So I just unrolled it part way.
I didn't unroll it all the way.

I'm assuming that you could compute this again,
given the stuff that we did in the last couple of lectures.
And it's going to be an n by 1 vector that
just says for each of the values in A1,
how much do I blame them?

Oh.
This is supposed to be equals.
Thank you.
Good.
So this is just for each of the values in A1,
how much did it contribute to the loss?
So we made a prediction.
The prediction was not good in some way.
We measured the loss.
We did the derivative, and so on.
This is the n by n thing with the stuff on the diagonals
that we talked about last time.
So that's standard, too.
So the thing that's new and interesting for us is dZ1, dW1.
Because that's the filter layer transformation.
OK.
So the dimensions of this guy are going to be k by n.
Yeah, good.
So k by n.

So this whole guy is going to be k
by 1, which is good because that's
the dimension of the filter.
And then what I want to do is talk about this.
This dZ1 dW1
So it's a matrix.

We'll look at it like this.

So this is n and this is k.
And let's just imagine that k equals 5 for right now
because it's a good case to think about.
It makes it concrete.
So let's image k equals 5.

And let's consider a particular n.
Let's consider n equals 10.
OK.
So what we're asking here is what is--
so dZ10, dW.

We're asking, this column corresponds to the way
all the weights effect elements i in the output image.

So we have five weights, and we want to know
how do those five weights affect output 10 and the output image?

Well, if you look at the forward pass,
you say, well, how do the weights affect
a particular output?
If I take the derivative of this thing with respect
to the weights, I get the A. I get the input stuff.
I want to call it x.
This A0 is equal to x.
It's easier to write x in here.
So this is the input image.
And so this is going to be x8, x9.
x10, x11, x12.

Those are the input pixels that we dotted with these weights
to get the Z. And so those are the input values that
contributed to this particular output having the value that it
has.
And so that's dZdW.
So what's interesting is if I look at the previous one,
this is going to be x7, x8, x9, x10, x11.

Is that right?
Yeah, that's right.
And so what you're going to see is that the input,
this one input-- let's say x11--
it has an impact on five of the values in the resulting image.
So it's very simple one.
You would never want to make this whole matrix written out,
right?
You'd want to do something cleverer
to compute this, probably.
But does this matrix make sense for you?
It's a bunch of partial derivatives,
but it has a nice kind of structure.
It just reflects the idea that we were going through our image
applying the transformation as we went.
Applying the convolution.
Doing a dot product with the weights.
So normally when we compute the dependence of an output
layer on the weights, each input only
comes into the calculation one time.
Because each input is multiplied by one weight and comes out.
Here, each input is multiplied by several different weights.

So each input participates in the production
of several different output values.
And it involves several different weights
because it gets weighted by different weights
depending on how the filter is lining up
with that particular input.
This is probably not making a ton of sense.
Probably what you have to do is go home and ponder,
I'm guessing.

But so this is it.
So the tools and tricks-- you know them all.
There's two kinds of layers--
filtering and max pooling.
And you can work out the derivatives.
And you could implement this in our style
that you're going to do next week.
We'll do an implementation.
Simple Python implementation of all this stuff.
Just go layer by layer, take one piece at a time,
and it's pretty simple.
And it can do really kind of astonishing things.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/41c7c4a6141b76b324055d56387570c0/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Convolutional_Neural_Networks.pdf
- https://openlearninglibrary.mit.edu/assets/courseware/v1/cda92ed2c6672271916e8cb8974af568/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_conv_nets_slides.pdf
