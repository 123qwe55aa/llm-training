# Recurrent Neural Networks

> Week 11: Recurrent Neural Networks · MIT 6.036 courseware archive

## Notes – Chapter 12: Recurrent Neural Networks

Notes – Chapter 12: Recurrent Neural Networks
You can sequence through the Recurrent Neural Networks lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 12: Recurrent Neural Networks
notes as a PDF file.

## Lecture: Recurrent neural network model

Lecture: Recurrent neural network model
Lecture: Recurrent neural network model
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Introduction to RNNs

Introduction to RNNs
In chapter 8 we studied neural networks and how we can train the weights of a network, based on data, so that it will adapt into a function that approximates the relationship between the [mathjaxinline](x, y)[/mathjaxinline] pairs in a supervised-learning training set. In section 1 of chapter 10, we studied state-machine models and defined
recurrent neural networks
(
rnn
s) as a particular type of state machine, with a multidimensional vector of real values as the state. In this chapter, we'll see how to use gradient-descent methods to train the weights of an
rnn
so that it performs a
transduction
that matches as closely as possible a training set of input-output
sequences
.
Download this chapter as a PDF file

## RNN model

RNN model
Recall that the basic operation of the state machine is to start with some state [mathjaxinline]s_0[/mathjaxinline], then iteratively compute:
[mathjaxinline]\displaystyle  s_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = f(s_{t - 1}, x_ t)[/mathjaxinline]
[mathjaxinline]\displaystyle y_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = g(s_ t)[/mathjaxinline]
as illustrated in the diagram below (remembering that there needs to be a delay on the feedback loop):
block = [draw, fill=blue!20, rectangle, minimum height=3em, minimum width=3em] sum = [draw, fill=blue!20, circle, node distance=1cm] input = [coordinate] output = [coordinate] pinstyle = [pin edge=to-,thin,black]
So, given a sequence of inputs [mathjaxinline]x_1, x_2, \dots[/mathjaxinline] the machine generates a sequence of outputs
[mathjax]\underbrace{g(f(x_1, s_0))}_{y_1}, \underbrace{g(f(x_2, f(x_1, s_0)))}_{y_2}, \dots \; \; .[/mathjax]
A
recurrent neural network
is a state machine with neural networks constituting functions [mathjaxinline]f[/mathjaxinline] and [mathjaxinline]g[/mathjaxinline]:
[mathjaxinline]\displaystyle  f(s, x)[/mathjaxinline]
[mathjaxinline]\displaystyle  = f_1(W^{sx}x + W^{ss}s + W^{ss}_0)[/mathjaxinline]
[mathjaxinline]\displaystyle g(s)[/mathjaxinline]
[mathjaxinline]\displaystyle  = f_2(W^ Os + W^ O_0) \; \; .[/mathjaxinline]
We are very sorry! This course material has evolved from different sources, which used [mathjaxinline]W^ Tx[/mathjaxinline] in the forward pass for regular feedforward NNs and [mathjaxinline]Wx[/mathjaxinline] for the forward pass in
rnn
s. This inconsistency doesn't make any technical difference, but is a potential source of confusion.
note
The inputs, outputs, and states are all vector-valued:
[mathjaxinline]\displaystyle  x_ t[/mathjaxinline]
[mathjaxinline]\displaystyle : \ell \times 1[/mathjaxinline]
[mathjaxinline]\displaystyle s_ t[/mathjaxinline]
[mathjaxinline]\displaystyle : m \times 1[/mathjaxinline]
[mathjaxinline]\displaystyle y_ t[/mathjaxinline]
[mathjaxinline]\displaystyle : v \times 1 \; \; .[/mathjaxinline]
The weights in the network, then, are
[mathjaxinline]\displaystyle  W^{sx}[/mathjaxinline]
[mathjaxinline]\displaystyle : m \times \ell[/mathjaxinline]
[mathjaxinline]\displaystyle W^{ss}[/mathjaxinline]
[mathjaxinline]\displaystyle : m \times m[/mathjaxinline]
[mathjaxinline]\displaystyle W^{ss}_0[/mathjaxinline]
[mathjaxinline]\displaystyle : m \times 1[/mathjaxinline]
[mathjaxinline]\displaystyle W^{O}[/mathjaxinline]
[mathjaxinline]\displaystyle : v \times m[/mathjaxinline]
[mathjaxinline]\displaystyle W^{O}_0[/mathjaxinline]
[mathjaxinline]\displaystyle : v \times 1[/mathjaxinline]
with activation functions [mathjaxinline]f_1[/mathjaxinline] and [mathjaxinline]f_2[/mathjaxinline]. Finally, the operation of the
rnn
is described by
[mathjaxinline]\displaystyle  s_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = f_1\left(W^{sx}x_ t + W^{ss}s_{t - 1} + W_0\right)[/mathjaxinline]
[mathjaxinline]\displaystyle y_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = f_2\left(W^ Os_ t + W_0^ O\right) \; \; .[/mathjaxinline]
Study Question:
Check dimensions here to be sure it all works out. Remember that we apply [mathjaxinline]f_1[/mathjaxinline] and [mathjaxinline]f_2[/mathjaxinline] elementwise.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:42 PM (revision 4f166135)

## Lecture: Sequence-to-sequence RNN

Lecture: Sequence-to-sequence RNN
Lecture: Sequence-to-sequence RNN
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Sequence-to-sequence RNN

Sequence-to-sequence RNN
Now, how can we train an
rnn
to model a transduction on sequences? This problem is sometimes called
sequence-to-sequence
mapping. You can think of it as a kind of regression problem: given an input sequence, learn to generate the
One way to think of training a sequence
classifier
is to reduce it to a transduction problem, where [mathjaxinline]y_ t = 1[/mathjaxinline] if the sequence [mathjaxinline]x_1, \ldots , x_ t[/mathjaxinline] is a
positive
example of the class of sequences and [mathjaxinline]-1[/mathjaxinline] otherwise.
corresponding output sequence.
A training set has the form [mathjaxinline]\left[\left(x^{(1)}, y^{(1)}\right), \dots , \left(x^{(q)}, y^{(q)}\right)\right][/mathjaxinline], where
[mathjaxinline]x^{(i)}[/mathjaxinline] and [mathjaxinline]y^{(i)}[/mathjaxinline] are length [mathjaxinline]n^{(i)}[/mathjaxinline] sequences;
sequences in the
same pair
are the same length; and sequences in different pairs may have different lengths.
Next, we need a loss function. We start by defining a loss function on sequences. There are many possible choices, but usually it makes sense just to sum up a per-element loss function on each of the output values, where [mathjaxinline]p[/mathjaxinline] is the predicted sequence and [mathjaxinline]y[/mathjaxinline] is the actual one:
[mathjax]\text {Loss}_{\text {seq}}\left(p^{(i)}, y^{(i)}\right) = \sum _{t = 1}^{n^{(q)}}\text {Loss}_\text {elt}\left(p_ t^{(i)}, y_ t^{(i)}\right) \; \; .[/mathjax]
The per-element loss function [mathjaxinline]\text {Loss}_\text {elt}[/mathjaxinline] will depend on the type of [mathjaxinline]y_ t[/mathjaxinline] and what information it is encoding, in the same way as for
So it could be
nll
, hinge loss, squared loss, etc.
a supervised network.
. Then, letting [mathjaxinline]\theta =\left(W^{sx}, W^{ss}, W^ O, W_0, W_0^ O\right)[/mathjaxinline], our overall objective is to minimize
[mathjax]J(\theta ) = \sum _{i = 1}^ q\text {Loss}_{\text {seq}}\left( \text {RNN}(x^{(i)};\theta ), y^{(i)}\right) \; \; ,[/mathjax]
where [mathjaxinline]\text {RNN}(x; \theta )[/mathjaxinline] is the output sequence generated, given input sequence [mathjaxinline]x[/mathjaxinline].
It is typical to choose [mathjaxinline]f_1[/mathjaxinline] to be
tanh
Remember that it looks like a sigmoid but ranges from -1 to +1.
note
but any non-linear activation function is usable. We choose [mathjaxinline]f_2[/mathjaxinline] to align with the types of our outputs and the loss function, just as we would do in regular supervised learning.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:42 PM (revision 4f166135)

## Lecture: Back-propagation through time - forward pass

Lecture: Back-propagation through time - forward pass
Lecture: Back-propagation through time - forward pass
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Back-propagation through time - backwards pass

Lecture: Back-propagation through time - backwards pass
Lecture: Back-propagation through time - backwards pass
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Back-propagation through time - weight updates

Lecture: Back-propagation through time - weight updates
Lecture: Back-propagation through time - weight updates
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Back-propagation through time

Back-propagation through time
Now the fun begins! We can find [mathjaxinline]\theta[/mathjaxinline] to minimize [mathjaxinline]J[/mathjaxinline] using gradient descent. We will work through the simplest method,
back-propagation through time
(
bptt
), in detail. This is generally not the best method to use, but it's relatively easy to understand. In section
we will sketch alternative methods that are in much more common use.
Calculus reminder: total derivative
Most of us are not very careful about the difference between the
partial derivative
and the
total derivative
. We are going to use a nice example from the Wikipedia article on partial derivatives to illustrate the difference.
The volume of a cone depends on its height and radius:
[mathjax]V(r, h) = \frac{\pi r^2 h}{3}\; \; .[/mathjax]
The partial derivatives of volume with respect to height and radius are
[mathjax]\frac{\partial V}{\partial r} = \frac{2\pi r h}{3}\; \; \; \text {and}\; \; \;  \frac{\partial V}{\partial h} = \frac{\pi r^2}{3}\; \; .[/mathjax]
They measure the change in [mathjaxinline]V[/mathjaxinline] assuming everything is held constant except the single variable we are changing. But! in a cone, the radius and height are not independent, and so we can't really change one without changing the other. In this case, we really have to think about the
total derivative
, which sums the “paths" along which [mathjaxinline]r[/mathjaxinline] might influence [mathjaxinline]V[/mathjaxinline]:
[mathjaxinline]\displaystyle  \frac{dV}{dr}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{\partial V}{\partial r} + \frac{\partial V}{\partial h} \frac{dh}{dr}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{2 \pi r h}{3} + \frac{\pi r^2}{3} \frac{dh}{dr}[/mathjaxinline]
[mathjaxinline]\displaystyle \frac{dV}{dh}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{\partial V}{\partial h} + \frac{\partial V}{\partial r} \frac{dr}{dh}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{\pi r^2}{3} + \frac{2 \pi r h}{3} \frac{dr}{dh}[/mathjaxinline]
Just to be completely concrete, let's think of a right circular cone with a fixed angle [mathjaxinline]\alpha = \tan r / h[/mathjaxinline], so that if we change [mathjaxinline]r[/mathjaxinline] or [mathjaxinline]h[/mathjaxinline] then [mathjaxinline]\alpha[/mathjaxinline] remains constant. So we have [mathjaxinline]r = h \tan {^-1} \alpha[/mathjaxinline]; let constant [mathjaxinline]c = \tan ^{-1} \alpha[/mathjaxinline], so now [mathjaxinline]r = c h[/mathjaxinline]. Now, we know that
[mathjaxinline]\displaystyle  \frac{dV}{dr}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{2 \pi r h}{3} + \frac{\pi r^2}{3} \frac{1}{c}[/mathjaxinline]
[mathjaxinline]\displaystyle \frac{dV}{dh}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{\pi r^2}{3} + \frac{2 \pi r h}{3} c[/mathjaxinline]
The
bptt
process goes like this:
Sample a training pair of sequences [mathjaxinline](x, y)[/mathjaxinline]; let their length be [mathjaxinline]n[/mathjaxinline].
“Unroll" the RNN to be length [mathjaxinline]n[/mathjaxinline] (picture for [mathjaxinline]n = 3[/mathjaxinline] below), and initialize [mathjaxinline]s_0[/mathjaxinline]:
Now, we can see our problem as one of performing what is almost an ordinary back-propagation training procedure in a feed-forward neural network, but with the difference that the weight matrices are shared among the layers. In many ways, this is similar to what ends up happening in a convolutional network, except in the conv-net, the weights are re-used spatially, and here, they are re-used temporally.
Do the
forward pass
, to compute the predicted output sequence [mathjaxinline]p[/mathjaxinline]:
[mathjaxinline]\displaystyle  z_ t^1[/mathjaxinline]
[mathjaxinline]\displaystyle = W^{sx}x_ t + W^{ss}s_{t - 1} + W_0[/mathjaxinline]
[mathjaxinline]\displaystyle s_ t[/mathjaxinline]
[mathjaxinline]\displaystyle = f_1(z_ t^1)[/mathjaxinline]
[mathjaxinline]\displaystyle z_ t^2[/mathjaxinline]
[mathjaxinline]\displaystyle = W^ Os_ t + W_0^ O[/mathjaxinline]
[mathjaxinline]\displaystyle p^ t[/mathjaxinline]
[mathjaxinline]\displaystyle = f_2(z_ t^2)[/mathjaxinline]
Do
backward pass
to compute the gradients. For both [mathjaxinline]W^{ss}[/mathjaxinline] and [mathjaxinline]W^{sx}[/mathjaxinline] we need to find
[mathjaxinline]\displaystyle  \frac{d L_\text {seq}}{d W}[/mathjaxinline]
[mathjaxinline]\displaystyle = \sum _{u = 1}^ n\frac{d L_ u}{d W} ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  \nonumber[/mathjaxinline]
Letting [mathjaxinline]L_ u = L_\text {elt}(p_ u, y_ u)[/mathjaxinline] and using the
total derivative
, which is a sum over all the ways in which [mathjaxinline]W[/mathjaxinline] affects [mathjaxinline]L_ u[/mathjaxinline], we have
[mathjaxinline]\displaystyle  ~ ~ ~ ~[/mathjaxinline]
[mathjaxinline]\displaystyle = \sum _{u = 1}^ n\sum _{t = 1}^ n\frac{\partial L_ u}{\partial s_ t}\cdot \frac{\partial s_ t}{\partial W} \nonumber[/mathjaxinline]
Re-organizing, we have
[mathjaxinline]\displaystyle  ~ ~ ~ ~[/mathjaxinline]
[mathjaxinline]\displaystyle = \sum _{t = 1}^ n\frac{\partial s_ t}{\partial W} \cdot \sum _{u = 1}^ n\frac{\partial L_ u}{\partial s_ t} \nonumber[/mathjaxinline]
Because [mathjaxinline]s_ t\  \text {only affects}\  L_ t, L_{t + 1}, \dots , L_ n[/mathjaxinline],
[mathjaxinline]\displaystyle  ~ ~ ~ ~[/mathjaxinline]
[mathjaxinline]\displaystyle = \sum _{t = 1}^ n\frac{\partial s_ t}{\partial W} \cdot \sum _{u = t}^ n\frac{\partial L_ u}{\partial s_ t} \nonumber[/mathjaxinline]
[mathjaxinline]\displaystyle = \sum _{t = 1}^ n\frac{\partial s_ t}{\partial W} \cdot \left(\frac{\partial L_ t}{\partial s_ t} + \underbrace{\sum _{u = t + 1}^ n\frac{\partial L_ u}{\partial s_ t}}_{\delta ^{s_ t}}\right)[/mathjaxinline]
(1.1)
[mathjaxinline]\delta ^{s_ t}[/mathjaxinline] is the dependence of the loss on steps after [mathjaxinline]t[/mathjaxinline] on the state at time [mathjaxinline]t[/mathjaxinline].
That is, [mathjaxinline]\delta ^{s_ t}[/mathjaxinline] is how much we can blame state [mathjaxinline]s_ t[/mathjaxinline] for all the future element losses.
note
We can compute this backwards, with [mathjaxinline]t[/mathjaxinline] going from [mathjaxinline]n[/mathjaxinline] down to [mathjaxinline]1[/mathjaxinline]. The trickiest part is figuring out how early states contribute to later losses. We define
future loss
[mathjax]F_ t = \sum _{u = t + 1}^{n}\text {Loss}_\text {elt}(p_ u, y_ u) \; \; ,[/mathjax]
so
[mathjax]\delta ^{s_ t} = \frac{\partial F_ t}{\partial s_ t}\; \; .[/mathjax]
At the last stage, [mathjaxinline]F_ n = 0[/mathjaxinline] so [mathjaxinline]\delta ^{s_ n} = 0[/mathjaxinline].
Now, working backwards,
[mathjaxinline]\displaystyle  \delta ^{s_{t -1}}[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{\partial }{\partial s_{t - 1}}\sum _{u = t}^ n\text {Loss}_\text {elt}(p_ u, y_ u)[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{\partial s_ t}{\partial s_{t - 1}} \cdot \frac{\partial }{\partial s_ t}\sum _{u = t}^ n\text {Loss}_\text {elt}(p_ u, y_ u)[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{\partial s_ t}{\partial s_{t - 1}} \cdot \frac{\partial }{\partial s_ t}\left[\text {Loss}_\text {elt}(p_ t, y_ t) + \sum _{u = t + 1}^ n\text {Loss}_\text {elt}(p_ u, y_ u)\right][/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{\partial s_ t}{\partial s_{t - 1}} \cdot \left[\frac{\partial \text {Loss}_\text {elt}(p_ t, y_ t)}{\partial s_ t} + \delta ^{s_ t}\right][/mathjaxinline]
Now, we can use the chain rule again to find the dependence of the element loss at time [mathjaxinline]t[/mathjaxinline] on the state at that same time,
[mathjax]\underbrace{\frac{\partial \text {Loss}_\text {elt}(p_ t, y_ t)}{\partial s_ t}}_{(m \times 1)} = \underbrace{\frac{\partial z_ t^2}{\partial s_ t}}_{(m \times v)} \cdot \underbrace{\frac{\partial \text {Loss}_\text {elt}(p_ t, y_ t)}{\partial z_ t^2}}_{(v \times 1)}\; \; ,[/mathjax]
and the dependence of the state at time [mathjaxinline]t[/mathjaxinline] on the state at the previous time, noting that we are performing an
elementwise
multiplication between [mathjaxinline]W^ T_{ss}[/mathjaxinline] and the vector of [mathjaxinline]{f^1}'[/mathjaxinline] values, [mathjaxinline]\partial s_ t /\partial z^1_ t[/mathjaxinline]:
There are two ways to think about [mathjaxinline]\partial s_ t / \partial z_ t[/mathjaxinline]: here, we take the view that it is an [mathjaxinline]m \times 1[/mathjaxinline] vector and we multiply each column of [mathjaxinline]W^ T[/mathjaxinline] by it. Another, equally good, view, is that it is an [mathjaxinline]m \times m[/mathjaxinline] diagonal matrix, with the values along the diagonal, and then this operation is a matrix multiply. Our software implementation will take the first view.
note
[mathjax]\underbrace{\frac{\partial s_ t}{\partial s_{t - 1}}}_{(m \times m)} = \underbrace{\frac{\partial z_ t^1}{\partial s_{t - 1}}}_{(m \times m)} \cdot \underbrace{\frac{\partial s_ t}{\partial z_ t^1}}_{(m \times 1)} = \underbrace{{W^{ss}}^ T * f^{1'}(z_ t^1)}_{\text {not dot!}}\; \; .[/mathjax]
Putting this all together, we end up with
[mathjax]\delta ^{s_{t - 1}} = \underbrace{{W^{ss}}^ T * f^{1'}(z_ t^1)}_{\frac{\partial s_ t}{\partial s_{t - 1}}} \cdot \underbrace{\left({W^ O}^ T\frac{\partial {L_ t}}{\partial z_ t^2} + \delta ^{s_ t}\right)}_{\frac{\partial F_{t - 1}}{\partial s_ t}}[/mathjax]
We're almost there! Now, we can describe the actual weight updates. Using equation
and recalling the definition of [mathjaxinline]\delta ^{s_ t} = \partial F_ t / \partial s_ t[/mathjaxinline], as we iterate backwards, we can accumulate the terms in equation
to get the gradient for the whole loss:
[mathjaxinline]\displaystyle  \frac{ d L_\text {seq}}{d W^{ss}}[/mathjaxinline]
[mathjaxinline]\displaystyle += \frac{\partial F_{t-1}}{\partial W^{ss}} = \frac{\partial z^1_ t}{\partial W^{ss}} \frac{\partial s_ t}{\partial z^1_ t} \frac{\partial F_{t-1}}{\partial s_ t}[/mathjaxinline]
[mathjaxinline]\displaystyle \frac{ d L_\text {seq}}{d W^{sx}}[/mathjaxinline]
[mathjaxinline]\displaystyle += \frac{\partial F_{t-1}}{\partial W^{sx}} = \frac{\partial z^1_ t}{\partial W^{sx}} \frac{\partial s_ t}{\partial z^1_ t} \frac{\partial F_{t-1}}{\partial s_ t}[/mathjaxinline]
We can handle [mathjaxinline]W^ O[/mathjaxinline] separately; it's easier because it does not effect future losses in the way that the other weight matrices do:
[mathjax]\frac{\partial L_\text {seq}}{\partial W^ O} = \sum _{t = 1}^ n\frac{\partial L_ t}{\partial W^ O} = \sum _{t = 1}^ n\frac{\partial L_ t}{\partial z_ t^2} \cdot \frac{\partial z_ t^2}{\partial W^ O}[/mathjax]
Assuming we have [mathjaxinline]\frac{\partial L_ t}{\partial z_ t^2} = (p_ t - y_ t)[/mathjaxinline], (which ends up being true for squared loss, softmax-NLL, etc.), then on each iteration
[mathjax]\underbrace{\frac{\partial L_\text {seq}}{\partial W^ O}}_{v \times m} += \underbrace{(p_ t - y_ t)}_{v \times 1} \cdot \underbrace{s_ t^ T}_{1 \times m}[/mathjax]
Whew!
Study Question:
Derive the updates for the offsets [mathjaxinline]W_0[/mathjaxinline] and [mathjaxinline]W^ O_0[/mathjaxinline].
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:42 PM (revision 4f166135)

## Lecture: RNNs - training a language model

Lecture: RNNs - training a language model
Lecture: RNNs - training a language model
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Training a language model

Training a language model
A
language model
is just trained on a set of input sequences, [mathjaxinline](c_1^{(i)}, c_2^{(i)}, \ldots , c_{n^ i}^{(i)})[/mathjaxinline], and is used to predict the next character, given a sequence
A “token" is generally a character or a word.
of previous tokens:
[mathjax]c_ t = \text {RNN}(c_1, c_2, \dots , c_{t - 1})\; \;[/mathjax]
We can convert this to a sequence-to-sequence training problem by constructing a data set of [mathjaxinline](x, y)[/mathjaxinline] sequence pairs, where we make up new special tokens, [mathjaxinline]\text {start}[/mathjaxinline] and [mathjaxinline]\text {end}[/mathjaxinline], to signal the beginning and end of the sequence:
[mathjaxinline]\displaystyle  x[/mathjaxinline]
[mathjaxinline]\displaystyle  = (\langle \text {start}\rangle , c_1, c_2, \dot, c_ n)[/mathjaxinline]
[mathjaxinline]\displaystyle y[/mathjaxinline]
[mathjaxinline]\displaystyle  = (c_1, c_2, \dots , \langle \text {end}\rangle )[/mathjaxinline]
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:42 PM (revision 4f166135)

## Lecture: RNNs - gating mechanisms and LSTM

Lecture: RNNs - gating mechanisms and LSTM
Lecture: RNNs - gating mechanisms and LSTM
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Vanishing gradients and gating mechanisms

Vanishing gradients and gating mechanisms
Let's take a careful look at the backward propagation of the gradient along the sequence:
[mathjax]\delta ^{s_{t -1}} = \frac{\partial s_ t}{\partial s_{t - 1}} \cdot \left[\frac{\partial \text {Loss}_\text {elt}(p_ t, y_ t)}{\partial s_ t} + \delta ^{s_ t}\right]\; \; .[/mathjax]
Consider a case where only the output at the end of the sequence is incorrect, but it depends critically, via the weights, on the input at time 1. In this case, we will multiply the loss at step [mathjaxinline]n[/mathjaxinline] by
[mathjax]\frac{\partial s_2}{\partial s_1} \cdot \frac{\partial s_3}{\partial s_2} \cdots \frac{\partial s_ n}{\partial s_{n-1}}\; \; .[/mathjax]
In general, this quantity will either grow or shrink exponentially with the length of the sequence, and make it very difficult to train.
Study Question:
The last time we talked about exploding and vanishing gradients, it was to justify per-weight adaptive step sizes. Why is that not a solution to the problem this time?
An important insight that really made recurrent networks work well on long sequences was the idea of
gating
.
Simple gated recurrent networks
A computer only ever updates some parts of its memory on each computation cycle. We can take this idea and use it to make our networks more able to retain state values over time and to make the gradients better-behaved. We will add a new component to our network, called a
gating network
. Let [mathjaxinline]g_ t[/mathjaxinline] be a [mathjaxinline]m \times 1[/mathjaxinline] vector of values and let [mathjaxinline]W^{gx}[/mathjaxinline] and [mathjaxinline]W^{gs}[/mathjaxinline] be [mathjaxinline]m \times l[/mathjaxinline] and [mathjaxinline]m \times m[/mathjaxinline] weight matrices, respectively. We will compute [mathjaxinline]g_ t[/mathjaxinline]
It can have an offset, too, but we are omitting it for simplicity.
as
[mathjax]g_ t = \text {sigmoid}(W^{gx} x_ t + W^{gs} s_{t-1})[/mathjax]
and then change the computation of [mathjaxinline]s_ t[/mathjaxinline] to be
[mathjax]s_ t = (1 - g_ t) * s_{t-1} + g_ t * f_1(W^{sx}x_ t + W^{ss} s_{t-1} + W_0)\; \; ,[/mathjax]
where [mathjaxinline]*[/mathjaxinline] is component-wise multiplication. We can see, here, that the output of the gating network is deciding, for each dimension of the state, how much it should be updated now. This mechanism makes it much easier for the network to learn to, for example, “store" some information in some dimension of the state, and then not change it during future state updates, or change it only under certain conditions on the input or other aspects of the state.
Study Question:
Why is it important that the activation function for [mathjaxinline]g[/mathjaxinline] be a sigmoid?
Long short-term memory
The idea of gating networks can be applied to make a state-machine that is even more like a computer memory, resulting in a type of network called an
lstm
for “long short-term
Yet another awesome name for a neural network!
memory."
We won't go into the details here, but the basic idea is that there is a memory cell (really, our state vector) and three (!) gating networks. The
input
gate selects (using a “soft" selection as in the gated network above) which dimensions of the state will be updated with new values; the
forget
gate decides which dimensions of the state will have its old values moved toward 0, and the
output
gate decides which dimensions of the state will be used to compute the output value. These networks have been used in applications like language translation with really amazing results. A diagram of the architecture is shown below:
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:42 PM (revision 4f166135)

## Video transcripts

### Lecture: Recurrent neural network model


LESLIE KAELBLING: What we're going to do today
is talk about a different kind of sequential model.
So when we did the little kind of,
we just stepped back for a minute
and we looked at sequential models.
We looked at Markov Decision Processes
and saw how that was the basis for reinforcement learning.
Then, if you remember, just before that,
we also looked at state machines.
And I think people have been wrestling with some state
machine exercises lately.
And we've been asking you to come up
with state machines that do certain kind of jobs.
And so what we're going to do now
is look at recurrent neural networks, which
is a way to train up a state machine to basically do
a certain kind of job.
So again, this is another example
where we ask you to solve some simple problems
and see what it feels like, and then we'll
train up a network to do that.
OK so just a reminder of the recurrent neural network
model, and this is the state machine model,
we have a function f, and some other function g.
And the idea is that an x at input at time
t comes in here and this function
f computes a state, which is the s time t
and that comes out here and outcomes y at time t.
But, there is another thing that f
gets which is an input, which is the previous state.
So I'm going to actually draw a little delay block here.

So that this is s at t minus 1.
OK so this is the basic state machine formulation
that we saw before.
And we, you can have finite states,
we were, in some of the homework,
you've been looking at LTI systems
where f and g are just simple linear functions.
So in a recurrent neural network it
has this structure, recurrent neural network what's different
or what's additional, is a particular definition of f
and g.
So in recurrent neural network, f takes a state and an input
and we're going to let it be some activation function,
let me just write this out, and we're going to let g of s
be some other activation function output s
and that's like a zero.
OK so let me explain this a little bit more,
all right, so we're going to think of this
f now, it could potentially actually
be a really big and complicated neural network.
In the step that we look at it'll basically just be one
layer, you think of it as one layer of a nonlinear network
and so it's going to take the x, the input and multiply it
by a matrix of weights, take the state and multiply it
by a matrix of weights, and potentially have a vector
and offset vector, .
And together this will give us a vector
of pre-activation values.
We'll run it through a nonlinear function,
and that's what's going to give us our new state.

So let me just write some dimensions down
here so that we're sure we know what we're doing right.
So generally speaking now x, we'll call it l by one,
s, you can pick the dimension, let it be m by one,
and y we'll call it b by one.
OK, so then what are the dimensions of these things?
So w s x is m by l, w s s better m by m, and this is m by one.
So everybody ends up being an m vector,
an m by one column vector if you do all this,
we run it through f one, which still gives us
an m by one column vector, and that's
what comes out here, which is good because s
is supposed to be m by one.
OK?
Let me, at this moment, express a regret
that I only noticed this morning when I was preparing.
Which is when we did feed forward networks,
we had a weight matrix and an input
and we said we take the weight matrix times input,
and we have an offset.
And then we run it through nonlinear functions, that's
almost exactly the same as this, but when
we did the feed forward stuff, we said what we're going to do
is w transpose times x.
So we had W transpose times x going forward
and the w going backward.
In this case, for reasons of historical accident,
we're not transposing the w here,
which means that the transpose is going to come out
in the backward going part.
It totally doesn't matter, and we
should have done it all the same way,
but it is too late this semester for me to fix it.
So, just sorry, that's what I have to say about that.
Anyway this is self consistent and it's
just off by a transpose from what we did in the forward pass
and it really doesn't make a difference,
but it could be confusing, so I wanted
to let you know about that.

OK, so this is a recurring neural network,
does anybody have any questions about the form of this thing?

No, OK, so the way it works now, if you remember right,
is that you have to pick and s zero,
and then it acts like a transducer.
So you feed in x's in sequence, and for each x,
it takes the old state, computes a new state,
generates an output.
Then you feed it another x and takes the last state,
computes a new state, generates an output, and so on.
So if you feed in a sequence of x's it gives you out a sequence


### Lecture: Recurrent neural network model


LESLIE KAELBLING: And so now what we're going to do
is supervised learning, but in this supervised learning,
the training data is going to be a set of pairs of sequences.
So we're going to do supervised training,
and actually, people do different kinds of training,
so we'll call the training sequence
to sequence, which I think, the closest thing to supervised
learning here.
And so our data set now is going to be a set of pairs, xi, yi,
but now, xi is going to be a sequence of vectors of L
by 1 vectors, and yi is going to be a sequence of v
by 1 vectors, and they will have length ni.

OK, so just to be clear, in our training set,
our sequences can have different lengths,
but each pair of sequence has to have the same length.
So it's like, oh, I fed in this sequence of 3 of length 3
x's and I got a sequence of length 3y.
Or I feed in a sequence of length 1,000,
and I get out a sequence of length 1,000.
So our training set can have sequences of different length,
but these para sequences has to have the same length,
all right.
And this is a supervised training set in the sense
that we would like to say, well, I would like
you, neural network, to become a machine, such
that if I feed in sequences like x,
I'll get out sequences like y.
So in that sense, it's supervised training.
OK, so if we're going to do learning,
we have to specify data and then the next part
is loss function, all right.
So what we want to do is we have some data set,
we're going to try to learn to do a good job on this data
by adjusting the weights in that machine.
And so then the question is, what does it mean
to do a good job on that data?
So we're going to define loss function, so first of all,
define loss function and sequences,
so we have the sequence loss, and I'm
going to use P for predicted.

So throughout this segment, we'll
use P for the value of the thing that
comes out of the actual neural network we have, right?
So this is like the guess.
We often talk about the loss of guess actual,
so P is going to be the predicted sequence that
comes out of the neural network, and y is
going to be the target value.
What we wish we had.
And so the question is, well, if I
have a whole string of values, what's a loss
function on my prediction?
And you could define it to be something different than this,
but almost always, people just use a sum along the sequence.
So t equals 1 to ni.

I'll write ni.
So we're going to go along the length of the sequence
and add up the loss element-wise loss.
OK, this doesn't really get us too far, but we'll get there.

OK, so how mad are we about predicting
a particular sequence P when we should have
predicted the other sequence y?
Well, we're just going to say it's
how mad we are about each particular prediction.
OK, fine.
So now what is lost at sub [INAUDIBLE]??
And the answer is there, it depends
on what kind of sequence you're predicting,
just as it would in any other case.
So if your predictions are real numbers,
then it might make sense for this to be squared loss.
If your predictions are characters in an alphabet,
then it might make sense to do some kind of negative log
likelihood loss type, like a categorization loss.
So basically, depending on your problem,
depending on what these predictions are
that you're trying to make, what these particular elements
of the sequence are then you pick
a element y is loss function.
OK, so loss in the whole sequence
is the sum of losses on the particular outputs.
Loss on the particular outputs is something
that you get to pick.
Yeah?

OK, and remember that this should
be coordinated with the choice of one of those activation
functions.
Just to see if you're right, tell me
which activation function is the one that needs to line up
with the loss function nicely?
Who's making the predictions?
You guys all look so sleepy.
Yep.
G, good.
In particular, with the activation function, f2, right?
So G is making the predictions, G is in charge of actually
generating the y's.
And so for instance, if what you're predicting
is characters, like you're trying
to do some kind of a language model of some sort,
you're trying to take in the sequence of characters,
giving out a sequence of characters,
then you would, generally speaking, pick f2 to be softmax
and this lost to be a negative 1 likelihood.

OK, OK.
Good-o.
So finally then, our objective function
we would like to minimize over thetas some objective function
theta, and the way you want to think about this really, well,
I'll write it this way, and then I'll say what theta is.
So this is the sequence loss of the whole r and n.
So let's say, I have an r and n, and it takes in an xi,
so this is i equals 1 to, I don't know,
size of my database.
rn of xi of theta yi.

And here, theta is all our parameters.
And so all our parameters is Ws--

I wrote it upstairs?
Yes-- Wsx, Wo, W0, and Wo0.

All right, so this is our theta, in the sense,
these are the weight matrices that we have nothing.

And we want to find, this is just our standard problem,
we want to find values of those rate matrices
that minimize the loss on our training data.
And we're not doing regularization here.
We could talk about regularization later,
but right now, it's going to be enough to think about
how do we train this whole network to do this job.


### Lecture: Recurrent neural network model


LESLIE KAELBLING: It feels a little bit mysterious right now
how to do this, and there's different ways
to think about it.
But the easiest way to think about how
you train this thing is using a method
called backpropagation through time.
Sounds cool.
Sounds like time travel or something.
Backpropagation through time.

Also known as BPTT.

So back propagation through time.
So how do we do back propagation through time?
In order to understand that, we need a figure.
It's an awesome figure.
It's going to take me a little bit of time to draw,
but I'm going to draw it because it's so important,
and it's good to just kind of appreciate it in its grandeur.
But basically, the way we do BP through time,
so we're going to do--
it's stochastic gradient descent.
So stochastic gradient descent.
So the way that works is we're going
to pick a training example.
So we're going to initialize weights,
we're going to pick a training example,
compute a gradient update based on that one training example,
do the gradient update, and continue,
so we are familiar with that concept.
And so we start, and we'll pick a particular sequence.
So we're going to pick an xi, yi training example,
and let's just let its length--

I'm just going to call it n.
It should be ni because different training examples
will have different length.
Let's just let n be the length of the particular training
example we just picked out of the training there.
Because in fact, the picture we draw is going to depend on it.
OK, so step one is to pick training example,
and then step two is to unroll the neural network, OK.
So I'm going to unroll the neural networks for length n.
And the way that works is that we're
going to take this recursive computation that we have
up there, and instead of making any arrows that go backwards,
we're just going to make them all
go forward by making copies of that network, right?
So we'll say, all right, here we go.
So state 0 comes in, and we're going to multiply it by Wss,
and add that into something and put it through f1.

I can do this.
OK, good.
And then I will draw this part first.

Yes.
OK, good.
So now what goes on here?
So what else we add in here is Wsx, and here
times x at time 1.

OK, so here's one part of the network.
It's not the part that's computing the outputs yet,
but it's the part that's computing the states as we
go through, right?
So we start with some initial state
as a 0, multiplied by the weight, ss,
take the first input, multiply it by sx,
run it through the non-linearity,
and what we have right here is S1, right?
That's one step through this thing.
Yep?
STUDENT: What about the biases?
LESLIE KAELBLING: What about the biases?
You know what?
I'm going to leave them out of this picture
because we have enough going on, but you can figure out
where to put them in.
Yeah?
STUDENT: Are the matrices not changing from the subset?
LESLIE KAELBLING: They are not changing.
That's going to be a critical thing here.
So just like in a convolutional network,
we had one filter, right?
So one little array of weights and we
applied it all over in the matrix, all over
in the input image, here, we have this same weight matrix
playing this role all throughout the whole computation
of the sequence.
Yep?
STUDENT: Are we updating the weights
after each entire sequencing?
LESLIE KAELBLING: Right, so this is going to be the rub, right?
The reason that this is tricky, if these were all
different weight matrices, then I would just say,
you guys know how to do backprop, you should do that,
and we're done.
The reason that it's tricky is because these same weights
are implicated many, many times in the production
of the outputs, which I'll draw in a minute,
and therefore, in the loss on this whole sequence.
And the trickiest part of figuring out how to train this
is understanding, for instance, how
the weights, like the state-to-state weights,
or the input weights.

They can affect the state that I have here.
All right, so the weights effect what state I get here,
which affects what state I get here,
which affects what state I get here,
which affects my last output.
And so the ways in which any individual weight
can affect the output sequence are quite complicated.
Yeah, now that's actually that's really the critical point.

Right.
OK, I'll finish decorating this with the states,
and then we'll put the outputs in, right?
So this is the state of time 1 multiply,
add in the input of time 2, put it
through the non-linearity, state in time 2,
one more time, state and time 3.
So that's just the same computation as
done by that feedback circuit, but we've
unrolled it for length 3.
And you could, obviously, unroll it for length 1,000,
it would take longer, but that would just be the same process.
OK, let's do the outputs.
So the outputs go here, so we have the output
weights, and the non-linearity, and then here comes the y1.
Sorry about my shortness.
OK, and then here, output weights, non-linearity, y2,
and one more time.
Output weights, non-linearity, y3.
OK, so that's the actual whole computation
process unrolled for length 3.
Does that makes sense to everybody?
Yeah?

All right, so OK, here's the step.
Pick the data example from your training set,
unroll the neural networks to the length of the data sample
that you have.
And then the next step is to do the forward pass.

And the forward pass is just to repeatedly apply those
equations, and that lets us compute all the s's.
So given the sequence of x's of inputs,
we can compute the sequence of s's.
So s from 1 to n, and the sequence of y's, just
by doing a forward computational pass on that graph.
So that's pretty straightforward, right?



### Lecture: Recurrent neural network model


LESLIE KAELBLING: So now we have to take a deep breath
and figure out how to do the backward pass.
And there's a fork in my road here.
I could tell you you can figure it out,
or I could do some amount of it.
I'm going to do some amount of it,
I think, because it's good to just see
the structure of the computation a little bit,
because it's not really obvious how you would do this.
But, really, there are no new principles.
So now our principle is we need to compute--

so now what we need to compute is
d loss, the sequence loss dW for all of our different weight
matrices.
And we have five.
But it's important-- as somebody pointed out there,
it's not that we don't have a new weight
matrix at each layer.
So this WSS is the same.
But in this picture, we have three weight matrices.
We have the one from input to state,
and the one from state to state, and the one from state
to output.
And then we do also have biases.
I'm actually not going to talk about them.
You guys can work those through later,
because once you know how to do these, you can do those.
So this is going to be our essential problem.
If we can compute this for this one training example, this one
xy pair, we compute this, we do a gradient
update on the W's, pick a new training
example, unroll the network, compute
the forward pass, et cetera.
So that's the high-level structure of the computation.
So now the question is, how do we organize this?

And so, let's see.
Well, we observe that--
the first step is not so hard.
So what we know is that we're going to have to go--
I'll go-- I'm going to use u and t,
for reasons you'll see maybe eventually.
L, p.

So let's remember what p and y are.
So y, y is the actual target values that we got out
of the training set.
Ah.
So, sorry.
This y should be a p.
In the forward pass, we're computing--

sorry.
I should have--
I had p in my notes but I got carried away.
I want to keep straight the idea that y
is what was in our training set, and p is the prediction
that we make that comes out of the network.

So we have the sequence of y's that we
were supposed to generate.
We have the sequence of p's that we actually generated.
We have a per element loss function.
So this is L per element.
And so the loss on the whole sequence is the sum of the--
I mean, the gradient of the loss in the whole sequences
with respect to W is the sum of the gradients along a sequence.
OK, that's good.

So the question is, for some particular output--
so let's consider the out-- this one--
we have to think about, how did--
let's say the hardest part is like WSS.
How did WSS contribute to this loss?
And the trouble is that it could--
it had three opportunities to contribute to that loss.
So remember, this is the blame game.
So how is it that we might blame WSS for a bad prediction
at the end?
Well, it could be that it messed up right
in this last step in some way, that gave us a bad--
it just did a bad job of going from S2 to S3,
and that's why we made a bad prediction.
But maybe you did a lovely job of going from S2 to S3,
but really you did a terrible job of going from S1 to S2.
Or maybe yous did a bad job of going from S0 to S1.
So WSS has three different ways to cause
us trouble just with respect to even the last output.

So that's what makes this tricky.
The transition-- the transition function
is like it's the workhorse of this whole machine,
and if it ever does something dumb anywhere along the way,
it can cause trouble for the last output.

So what that's going to mean--
and I think what I'm going to do is-- the derivation
is worked out in detail in the notes.
I am going to cut to the chase here
a little bit, because it will take a long time
to do it all on the board and I'm not sure it will help you.
But we're going to end up being able to write this
in a form that looks like this.
And it's probably not so surprising
that it's going to go like this.

So let's just look at that.
It's a double summation.
We started with only one sum, which
is the sum over all the possible errors.
And then we have two sums.
Because, basically, as I said, for an error at some step,
for an error at some step t, the states all the way up
until that point could be somehow implicated
in generating that error, generating that loss.
So the way this works is we say, OK,
so how does the whole sequence's loss depend on W?
Say, OK, well, what we're going to do
is we're going to go from 1 to n and we're going to consider
how the state at time t--
so t from 1 to n.
So we're going the whole sequence.
We ask, how does the state at time t depend on W?
And then how do all the rest of the losses from t
forward depend on the state at time t?

So if we look at this case, we could say, well,
how does S at time 2 depend on WSS?
That's something we can ask.
And then we say, how do all the losses on the rest
of the sequence depend on S2?
Now, we know--
I mean, the reason that that second summation doesn't
go from 1 to n is because the loss at time 1
can't depend on the state at time 2.
So this is forward data flow.
So earlier losses can't depend on the state,
but later losses can depend on the state.
Yes.
SUBJECT 1: Are you saying the W in the denominator
is the same as W [INAUDIBLE]?
LESLIE KAELBLING: I'm trying to be a little bit
generic right now.
It's true for any of the W's, this general sort of form.
Yes.

Does it, kind of at least the intuitive level, make sense?
Exactly how you get from here to there
requires remembering a piece of calculus that I had forgotten
and probably everyone else has, which
is total derivatives instead of partial derivatives,
which you're welcome to go study on your own time.

OK, so good.
So let me then break this part down into two pieces.
So I'm going to take this and just
break it into two pieces in some sort of obvious way.
So there's the loss.
And now I'm going to write--

when I write a loss now without any extra subscripts,
just the loss at time t, what I mean is this.

I'll write that just so we're clear.

OK.
So this looks like it should be easy to figure
out, the loss at time t given the state at time t.
That's going to be pretty local.
That's going to just depend on the output apparatus.
Yep.
SUBJECT 2: [INAUDIBLE].

This, all I'm doing is picking out
the first term, the first element of the sum and give--
I'm sorry.
I lost eye contact with whoever just asked me a question.
OK, good, thank you.
It's weird.
This is a sum from t to n.
All I did was take the first guy and make him special, and then
the rest of the sum.

So that's the sum from t to n.
This is just the t case, and then this is t plus 1 to n.
And the reason is that this one's easy.
So my principle is, whenever I'm doing some math
and it looks kind of complicated, if there's
some part that I can just put my hands on and say,
oh, totally, I know how to compute that, then I
like to take it out.
So we know how to-- this is going to be easy.
So the loss at time t with respect to the state at time t.
So if I know this state, the question
is, how bad am I about that prediction?
And that's just a local question.
That just goes once right through this f2 and W0.
So we can write down that derivative.
That's not the hard part.
The hard part is this going backwards stuff.

So I'm just going to--
I'm going to take this thing and expand it
into these two pieces.
It's going to make some stuff easier.

And then what we're going to do is
we're going to name this guy--
because it turns out it's going to be useful recursively--

delta.
Just in case you didn't know what a small delta looks like,
that's what it looks like.
So the delta for a state at time t.
And what is that?
So how can we-- how can we think about in words what this means?
So what this delta is is really how the loss after t
depends on the state at time t.
So this is how the loss at t-- loss at t
depends on the state at time t.
And this is how the loss after t depends on the state at time t.
And what we're building up to here is a recursive--
the recursive.
We're building up to a recursive definition,
so that we're going to compute a delta like this at the end,
for t equals n, and then work our way backwards.
That's what we're building up to.


### Lecture: Recurrent neural network model


LESLIE KAELBLING: This is the thing
that we're going to care about.
And this is all just mildly hairy applications of a thing
you already know how to do, so it mostly
requires patience and bookkeeping to get it right,
OK.
So if we--
OK, now since we're after a recurrence,
we would like to know--

so we have this idea that--
well, actually, let's ask the question.
Oh, easy question.
Always good to ask an easy question.

So, all right.
Recurrence based case.
Easy question.
If this is how the loss after time, t,
depends on the state and time t, how does the loss after time n
depend on the state time, n?
What's that?
0.
Yeah, OK.
I get the base case is 0.

OK, so good.
So base case is 0, and now we have
to say, well, if we can't figure out a way
to write down delta at time t minus 1
in terms of delta at time t, then
we know how to compute all these delta.
And if we compute these deltas, then we
know how to compute that thing, and if we
know how to compute that, then we know the gradient,
and we know how to update our weight matrix.
OK, so now, let's do this.
So I'm just I'm going to write it down.

Times.

So this is what is going to come out to be,
and it's kind of beautiful and also not so surprising
if you think about it.
OK, so let's just see if we can get something interesting.
So this is how much the state of time t
affects our future losses, this is how much our stated time
t affects the current loss, we chain rulify
to get it to be, instead of about stated time t,
to be about stated time t minus 1,
and this is how much the state of t minus
affects the future losses.

You might or might not believe me,
you can go through work it out, OK.
OK, so that's the basic recurrence,
and so then we have to look at these pieces
and see if we know how to do them.

This is just the backdrop through one
normal neural network layer, right.
So it goes through--
so how do we get from the state at time t minus 1
to the state at time t?
We multiply it by the weight matrix, assess,
and we put it through the non-linearity F1, right?
So this is going to turn out to be--

and we have 2 transpose here because we
didn't transpose going forward.
And then this is F1 prime of z.
I should label the z's on our--
zt1, F1 prime ztm.

Let me label z.
This is z1, and this is z2, and this is the z3, right?
So the z's are the pre-activation values.
z1 because that's a pre-activation.
We have too many subscripts now-- and we have--
OK, but, yeah.
OK, so this thing turns into this, right?
And so this is just the neurologic forward pass,
the derivative going backwards is the weights,
and this is this vector of the first derivatives
of the activations, this is a square matrix,
and this is a column vector.
And we're just going to multiply that column vector
each time all the way along.
Another way to understand it--
I wrote it out both ways in the notes-- another way
to think about this is, instead of thinking about it
this is a column vector, you can think about this as an n
by a matrix with these values along the diagonal,
and then it becomes a matrix multiply.
But either way, it's the same operation.
OK, so this is a thing we kind of already know how to do.
And so then what's the rest of this?
OK, so let's see.

What is the loss of at time t, with respect
to the state at time t?
Well, that's the stuff that goes around that path,
and that's going to turn into dz2 time t,
so these are dz the z1's.
This is dz2 time t, ds at time t.
And then d loss, the element of y is loss of pt oit dct.

OK, and then-- we need the plus.
Plus delta s.

So these are bits and pieces that we know, right?
This is going to be another thing,
so the dz, ds, it's going to be something
that looks like this, right?
It's going to be a vector of the f2 primes.

This is the error of just the gradient of your loss function,
with respect to the activations that come in.
Sometimes we take these two things
together for many of our last functions.
It's easier to take these two things together
and that's fine too.
So these are pieces I understand,
this is a piece we understand, and this is just
the recurrence.
So in the end, what you can do is compute
these deltas going back.
Right?
So we unroll the error going forward,
we compute these delta is going back.

OK, and yep?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: So this is the z time t.
This is-- I'm super confusing and I
have to have three decorations, I
don't know how to get out of it.
OK, so the z1's are the z's--
let's see-- this is a z1, and this is a z2, right?
So the z1 and z2 is the stuff that goes into f1 and f2.

Then, in this picture, at time t, we have a z1 at time 1,
a z1 at time 2, and a z1 time 3.
Those guys are vectors.
So this is component 1 of z1 at time t, component 2 z1 at--
Oh, yeah.
Component 2 of z1 at time t, component m of the z1
at time t.
Yep?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: So the question is this.
So this is about all the layers.
And this is a way to recursively compute that
by seeing how it depends on the previous layers.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: This partial s, partial w
is only one layer, that's right.
STUDENT: OK.

LESLIE KAELBLING: Yeah.

That's right.
So in a minute now, I'm finally going
to get to computing the gradient.
Now that we computed the deltas, which is just this handy value,
we can get to the gradient.
Yep?
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: There's two ways to think about this.
Either you think about it as a column vector and then this
is a broadcast multiplication, where we multiply that column
vector by each column of Wss--
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: That's right.
Or, you can make this into a diagonal matrix with these guys
on the diagonal and 0's everywhere
and then make it into a matrix multiply.
It turns out to be the same.
Yeah.

OK, one more kind of step here and then
we'll leave this to you to think about.
OK, so what we figured out is we can
compute these deltas, right?
And again, the delta is, how does the future loss
depend on the state at time t.
That's what this is about.

That means that eventually, we can write down
the following thing, which is the d loss sequence, dWss, is
the sum for t equals 1 to n.

Let's see.

Delta st minus 1.

Yeah.
And we could decompose that into one more step if we wanted to--
I don't know if we can do it.

So let's see.
So this is just applying chain rule here,
because how do we get s of time t?
Well s of time t depends on z at time t,
and z at time t is a direct--
so the derivative of z with respect to Wss.
What's the derivative of z with respect to, let's say z,
with respect to Wss, it's just the state at the previous time,
right.
So this is going to be the state at t minus,
the derivative of s with respect to z1
is one of these f1 prime type objects,
and this is the thing that we get back recursively.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Excuse me?
STUDENT: But why is it t minus 1 and not t?
LESLIE KAELBLING: Why is it t minus 1 and not t?
OK, let's think that went through.

Maybe I got it wrong.
These things do sometimes happen.

OK, so let's see.
So for the whole sequence, we're going to goes from 1 to n,
so let's take the case where t is 1, or case where t is 1.
So we're asking the question, or did I
forget to add in the one step loss?
No, it gets add it in there.

So let's look at this equation for delta of s of t minus 1,
right?
It includes the loss at time t with respect to state t--

yeah, no.
I think it's OK, right?
It actually includes the loss of time t, plus the future losses.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: There is a difference.

OK, it's entirely possible I made a mistake.
I will not be to devalue it by staring at the board,
so I will go back and study my notes,
and let you know if it's wrong, OK?
OK.
The high level bit here is we can compute.
OK, what do we do?
We draw an example.
We do the forward pass.
You could think of it as doing a backward pass computing
all the deltas although, in fact,
we can do the backward pass and compute these terms as we
go back, so you really only need to do one backward pass.
So going back, you compute this delta and this component
of the update to dl dWss, and then now you're
ready to go back and compute, so you started n, do n minus 1,
n minus 2, all the way back to 1, add these guys up as you go,
you get this.
And this is the quantity that you
can use to take a gradient step with respect to W. OK,
there is a similar story, which I am not going to write out,
for Lseq, Wsx.
It looks kind of like this.

There is also a similar story for the loss
of Lseq with respect to Wo, but that's much easier
because the influence of Wo doesn't chain.
If I tell you which states happened,
then that's enough to just compute.
So if I say--

if I can compute the state sequence,
I know what the state sequence is,
and I know then given the state sequence, each of these guys
is completely independent the way in which the output
contributes to loss is completely independent of what
happens over here.
So those guys are a lot easier.
So that's also written out in the notes,
so I'm not going to do that in detail.
I want to talk a little bit more just
about how you would use this and in fact
how you make it be better and some other kinds
of circumstances.
The critical thing is this is just gradient descent,
and the only reason it's hairy is
that it's kind of hairy to compute the gradient.
But because we have these shared weight matrices everywhere,
in fact, I mean, this is first of all, what makes it hard.
But eventually, it gives us a compact solution
for writing down the way in which we
do compute the gradient.


### Lecture: Recurrent neural network model


LESLIE KAELBLING: So we've taken this sequence to sequence setup
as our main setting.
And sequence to sequence is a good idea,
let's say, if you're doing language translation.
Then you feed in words or even characters
of a sentence in French, and you get out characters in Chinese.
Or you feed in some other kinds of sequences of something
and you get out another kind of sequence.
So that's the sequence to sequence model, where we're
really doing a transduction.
But another really popular use for recurrent neural networks
is as a language model.

And when we use an RNN as a language model, we only--
we're interested now in kind of understanding
a distribution over strings, or really how to predict--
really, what this is about is predicting the output
at time t given--

and the reason--
I know that this caused some consternation in the homework.
So let me actually draw a picture
of what we're doing when we're thinking
about using a recurrent network as a language model.
The way to think about it, the way I think about it,
is that we still have our good old-- our f and our g.
And we had this setup where we took the state
and we ran it around and put it here.
So this is-- and we put it through a delay.

So this is the state at time t.
And this is y at time t.

But then what we're going to do is
we're going to take it and run it around, and let it be the x.

So this is y at t minus 1, which we're also secretly going
to call x at time t.
And this, after being delayed, is S t minus 1.

So this is kind of the language model idea.
The language model idea is I start out in some state.

That generates an output.
I take that output and feed it around as if it were my input.
Then I generate a new state, which generates a new output.
I feed it around, and so on.
So that's the kind of the setup for a language model.
And the way we'll use it typically is force--
well, there's several different ways you can use it.
But one way is you force in a sequence of characters.
So I write L-E-A-R. So I feed in L, I feed in E, I feed in A,
I feed in R. And now I look at the distribution.
Typically, we would use--
if this has characters, we would use as a softmax here.
So, really, what's going to come out here is a distribution.
Ah.

I should-- should--
so if what comes out here is a distribution,
then generally what we'll do is we'll
take the most likely character and feed it around,
if you want to think about this thing actually going--
actually kind of applying this.
But if I wanted to use this to make a prediction,
I would force L as x1, and force in E as x2, and force in R as--
A as x3 and R as x4.
And then I would ask, what's the probability distribution
over the y?
What do you think-- what's the most likely character
in your distribution and what you've
been hearing about lately?
N, maybe N, right?
So maybe you think of learning.
And maybe you think it's leary.
Right.
So we think it's an N with probability 0.8.
And I am leary of this, say 0.1.
And I don't know.
You could think of one other word.
So this is a super standard and popular use
of a recurrent neural network.
So it's often a component of a bigger natural language system.
There might be other things that are predicting, like, content.
Like what's an important piece of a content
that I might generate next.
But something like this can help decide, oh,
would it be-- like would it be linguistically
sensible to come next in the sequence as far as I'm going?
You could do it over words.
You could do it over letters.
So the question is, if we want to train-- so we just
wrote all this chalk all over the board of too
many derivatives to do the sequence to sequence thing.
Do we have to do that over again to do language models?
And the answer is no.
We just have to generate a data set in a mildly clever way.
So how would we train?
So, OK, so what's the setup here?
The setup here is that we just have
a bunch of our training data.
I'll write it C for character.
It goes C1 through C3.
That's a charac-- that's a word, let's say.
And then we have C1 through C10, different one.
So this is 1, 2.

So we have a bunch of words.
They're different lengths.
And that's all we have.
We don't have input-output pairs.
We just have these things.
But we can turn it into a training set of the kind
that we're kind of ready to consume.
And the way we do that is we take this guy,
let's say, and we turn it into a pair
where the x looks like start--

and the y looks like--

let me make these line up.
So C1 goes here, C2 goes here, C3, and end.

Yes.
AUDIENCE: So are we--
you're not indexing by [INAUDIBLE] characters
and words here.
You're indexing by--
LESLIE KAELBLING: Hah.
Thank you.

These-- right.
These are the characters, not the-- this is training example
1, let's say.
Training example 1 gives me x1 and y1.
Good.
I appreciate that.
So what I'm going to do is I'm going to say, oh, actually,
please be a transducer.
But be a transducer of the form, if you get the symbol start,
C1 would be a good thing to emit.
So, already, we're asking it to try
to kind of learn what a distribution is
over the first character.
If I just take my machine after I've changed it up and I put it
in the start symbol, it's going to give me a distribution over
possible symbols, these C's.
And that's like a distribution over our starting character.
And then we say, oh, well, please, if you've seen start
and C1, then a good thing to put out would be C2.
If you've seen start and the C1 and C2, a good thing to put
would be C3.
And if you've seen all these characters,
a good thing to put out would be end.
Because if I'm using this model to generate,
let's say, words from my language or sentences
or poems--
or in lab we'll use it to generate the names of rock
bands and recipes--
the terminal symbol is useful.
Yes, we get death, death, death, death.
It's kind of fun.
I trained it on metal bands.

If anyone hasn't-- and let me just say,
if anyone has a favorite data set of like--
of things that are kind of shortish phrases,
it turns out that band names and recipe names
are really good just for our simple notation.
If you have a database like that of something
that you find entertaining that people might like,
mail it to me and I'll train up an [INAUDIBLE] on it
and we can play with it.
Or you could train up an [INAUDIBLE] on it.
So, anyway, so this is how you could train a language model.
I think I saw a hand and I lost it.
Yes.
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: Start is a char--
I mean, start is a magic character
that we add as engineers to this problem.
So we just say, I'm going to make
a special symbol called start.
And the neural network is going to learn that start--
I mean, we're going to do one-hot encoding,
generally speaking.
So if we're doing something with characters,
then the way we code the x's that come in is using one-hot.
And the way we code the y's that come out
is as a-- like a softmax over that whole vector.
And so we can just add something to the list of characters.
That's no problem.
Yes, that's a good point.
Yes.
AUDIENCE: When you were talking about putting in L-E-A,
you said [INAUDIBLE].
But what if we don't get E and A as the output?
Then we're not following that model.
LESLIE KAELBLING: But it's OK.
I mean, you can still--
you can ask this model-- so this model, in some sense,
will give you a distribution over char-- over strings.
That's kind of what it wants to do if you don't force it.
But you can also, in some sense, ask
for a conditional distribution.
You can say, conditioned on the fact
that the first three-- four characters were this,
tell me what you think the distribution is
over the next character.
That's kind of what you're asking it.
Now, if you never trained it on strings that had L-E-A-R
in them, then this answer is going to be bad.
It's not going to be very helpful.
But it's a kind of an OK thing to ask it.
Because if you think about it, if you look at this,
this is like if you think of it as a transduction.
It's like you're saying, OK, I'm going to give you a string that
the x is going to be start L-E-A-R,
and then I want to know what you think the next character is.
I'm just ignoring your other outputs.
That's the way to think about it.
Yes.
AUDIENCE: If this were translation,
what if the output word is longer than the input word?
Then we would have more characters than
we have been getting the output--
LESLIE KAELBLING: Right.
So for sequence to sequence stuff where the--
you could say at the sentence level,
if we have different numbers of words,
or at the word level, what if they
have different numbers of characters,
and stuff like that.
You can add another symbol to your vocabulary,
which is like space.
It's not quite-- but you can say end, right?
So you-- this--
imagine I could feed in dog.
And I want [INAUDIBLE] to come out,
which is two letters longer.
So I'm just-- I'll keep running it
and it will keep generating characters,
and then it will say end.
So it's amazing in some sense.
And you can train one of these things to reverse stuff.
Also, it can-- it uses that state
to remember what has to come out even if what has to come out
is longer.
But it does that-- and that's partly
why we have this explicit symbol end.
So that we don't just stop after three characters.
We might-- if we're running it, we
might run it until the end character comes out.

But we don't do this sequence to sequence thing
if we're translating.
For translating, we would-- we would--
I mean, we don't do this language model thing.
We would do a sequence to sequence thing
if we're translating.

Let me say a word of caution and do this thing
that I always mildly hated as a student, which
is I've taught you the simplest version of this thing
but nobody uses it.

But we can play around with it, and it gives us all kinds
of good intuition and insight.
So now I'll explain why it is problematic
and what people do to fix it.
But we're not going to dig into that, because that's
like another whole story.
So where can we see the trouble that we get into?

I think we can see it here.
Where's our delta recursion?

Here's our delta recursion.
OK, good.
Here's our-- good, right.
OK.
So this delta we're computing going backwards,
it's a critical thing in all the gradient computations.
And if you just back up and ignore this term
for a minute, which won't actually
make too big of a difference, we're
taking the delta at the end and multiplying it
by this gradient and the weight matrix.
Now imagine for-- just ignore that-- even that gradient
for minute.
So weight matrix.
So if we wanted to compute the delta of S at time 1,
but our total n is like 10 or 20,
we will have multiplied this weight matrix together
10 or 20 times.
Now, I made this argument to you once before about exploding
and vanishing gradients.
So whatever this weight matrix is,
it's either going to make things get bigger
or get smaller, basically.
It's hard to-- it doesn't-- generally speaking,
unlikely that it's just going to keep everything the same.
The last time we ran into this, though,
these were many different weight matrices,
and we said, oh, it's OK.
We'll just use different step sizes,
and we can make up for the fact that the gradient is getting
bigger or smaller as we go.
But this is one and the same matrix,
and we have to raise it to the power n, basically.
And so what happens is if you're sequence is long,
the gradients either get-- these deltas either
get really big or really small.
The gradients either get really big or really small.
And, generally speaking, what that means
is also that your machine has trouble remembering stuff, that
if you--
imagine you wanted your machine to remember the x-- the very
first character you got, and then, 20 characters later,
emit that character.
It has to somehow put it into the state
and keep it unmolested in that state until you get to the end.
And then you can see that it's there
and do something about it.
And so it's-- just with this basic matrix multiply setup
that we have here, it's kind of pretty hard to arrange that.
So the thing that has made--
I mean, we can actually do some surprisingly cool examples
with the mechanism that we have right here,


### Lecture: Recurrent neural network model


LESLIE KAELBLING: The thing that has
made this really work and really transform translation
and speech and other kinds of things
is one more idea which is pretty important,
which is the idea of gating and there's a fancy thing called
LSTM, which most awesomely stands for long short term
memory.
Isn't that nice?
It's one of my favorite names.

The guy that came up with this, I know him.
He did it years ago, nobody paid any attention.
And now it's like the coolest thing,
and everybody's using it.
So this is how science works, so he's vindicated,
so that's good.
And it's basically a type of a gating network,
so I'm going to show you a simple gating network,
and then it is a more complicated gating system.
So we're going to add another variable called g.
This is not the function, g.
This is another name conflict, but I'm not
going to fix it right now.

With new rates.

It could have offsets too, but I'm
going to ignore that because our life is complicated enough.
So we're going to add another contraption to our network,
and we're going to change the way that we compute the state.

Am I good?

Sigmoid is just the nonlinearity that I'm going to use.
STUDENT: St or St minus 1?
LESLIE KAELBLING: Oh, thank you.
Got it.
Yep.

It is St minus 1, and for good reason, which is--
I'll show you.

OK, our old computation for S at time t
used to just be this, right?
It was like, oh, OK, we'll take our old S
and multiply it by the Wss, and take our new x,
and multiply it by this, and run it through the nonlinearity.
That's going to be my new state.
That used to be what we did.
What we're doing here is saying, OK, so this gt--
it's a column vector, right?
So it's m by 1.
It's a column vector.
It's the same dimension as our state,
and you can think of it as picking
which state dimensions we're actually
going to change this time.
So it's a sigmoid, which means it goes from 0 to 1, right?
So this is a convex combination.
It's a weighted combination of what our values use to be
and what we were updated to using these Ws.
But what effect this has is that if this g is
a 0, or very small for some element in the state,
we don't touch it.
It's like in a computer, right?
A computer, generally speaking, you
don't update all the memory cells in every cycle.
Somebody decides which ones you're going to update.
So this gating thing is deciding which components of our state
we're going to update on time t.
Now it's soft so that we can still take derivatives,
you can do gradients and compute--
figuring out how to train this is not
much worse than figuring out how to train that,
but it's much more able to remember a thing
and keep it in memory, and use it for later,
and so with this kind of a setup,
you can train a recurrent neural network that can remember
things for a longer time.
It doesn't have as much problem with the gradients exploding
or diminishing, and it's much better able to remember stuff
over the longer term.

OK, one last little piece of information.
OK, so LSTM, there's a picture of a LSTM in the notes,
but it basically, it has three gating networks.
One gating function is used to decide
how much we want to depend on the input,
relative to the state.
One gating function is used to decide how much should I--
this basic gating function.
And another one is used for so-called forgetting,
which kind of attenuates the old state.
So it's a much more complicated thing, but really, in the end,
it's a giant parametrized differentiable function.
And if you have enough patience, you
can compute all the gradients and train
it using gradient descent.
And so really, it's this LSTM that lots of people are using.
And of course, there's now a million variations on it.
But this basic idea of recurrence,
plus some way of shielding some important stuff
that you want to remember is what's


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/0de27572f5d771b35ad094df49a8e200/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Recurrent_Neural_Networks.pdf
