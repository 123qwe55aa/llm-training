# Making NN's Work

> Week 7: Neural Networks II · MIT 6.036 courseware archive

## Notes – Chapter 8: Neural Networks

Notes – Chapter 8: Neural Networks
You can sequence through the Neural Networks II lecture video and note segments (go to Next page).
This week's lecture notes are a continuation (later sections) of
Chapter 8: Neural Networks
which you can download as a PDF file.

## Lecture: Neural networks - brief review of layers and backprop

Lecture: Neural networks - brief review of layers and backprop
Lecture: Neural networks - brief review of layers and backprop
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - optimizing parameters - batch gradient descent training

Lecture: Neural networks - optimizing parameters - batch gradient descent training
Lecture: Neural networks - optimizing parameters - batch gradient descent training
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - optimizing parameters - adaptive step-size

Lecture: Neural networks - optimizing parameters - adaptive step-size
Lecture: Neural networks - optimizing parameters - adaptive step-size
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - optimizing parameters - running averages

Lecture: Neural networks - optimizing parameters - running averages
Lecture: Neural networks - optimizing parameters - running averages
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - optimizing parameters - momentum

Lecture: Neural networks - optimizing parameters - momentum
Lecture: Neural networks - optimizing parameters - momentum
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - optimizing parameters - adagrad and adadelta

Lecture: Neural networks - optimizing parameters - adagrad and adadelta
Lecture: Neural networks - optimizing parameters - adagrad and adadelta
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - optimizing parameters - adam step-size update strategy

Lecture: Neural networks - optimizing parameters - adam step-size update strategy
Lecture: Neural networks - optimizing parameters - adam step-size update strategy
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Optimizing neural network parameters

Optimizing neural network parameters
Because neural networks are just parametric functions, we can optimize loss with respect to the parameters using standard gradient-descent software, but we can take advantage of the structure of the loss function and the hypothesis class to improve optimization. As we have seen, the modular function-composition structure of a neural network hypothesis makes it easy to organize the computation of the gradient. As we have also seen earlier, the structure of the loss function as a sum over terms, one per training data point, allows us to consider stochastic gradient methods. In this section we'll consider some alternative strategies for organizing training, and also for making it easier to handle the step-size parameter.
Batches
Assume that we have an objective of the form
[mathjax]J(W) = \sum _{i = 1}^ n \mathcal{L}(h(x^{(i)}; W), y^{(i)})\; \; ,[/mathjax]
where [mathjaxinline]h[/mathjaxinline] is the function computed by a neural network, and [mathjaxinline]W[/mathjaxinline] stands for all the weight matrices and vectors in the network.
When we perform
batch
gradient descent, we use the update rule
[mathjax]W := W - \eta \nabla _ W J(W)\; \; ,[/mathjax]
which is equivalent to
[mathjax]W := W - \eta \sum _{i=1}^ n \nabla _ W \mathcal{L}(h(x^{(i)}; W), y^{(i)})\; \; .[/mathjax]
So, we sum up the gradient of loss at each training point, with respect to [mathjaxinline]W[/mathjaxinline], and then take a step in the negative direction of the gradient.
In
stochastic
gradient descent, we repeatedly pick a point [mathjaxinline](x^{(i)}, y^{(i)})[/mathjaxinline] at random from the data set, and execute a weight update on that point alone:
[mathjax]W := W - \eta \nabla _ W \mathcal{L}(h(x^{(i)}; W), y^{(i)})\; \; .[/mathjax]
As long as we pick points uniformly at random from the data set, and decrease [mathjaxinline]\eta[/mathjaxinline] at an appropriate rate, we are guaranteed, with high probability, to converge to at least a local optimum.
These two methods have offsetting virtues. The batch method takes steps in the exact gradient direction but requires a lot of computation before even a single step can be taken, especially if the data set is large. The stochastic method begins moving right away, and can sometimes make very good progress before looking at even a substantial fraction of the whole data set, but if there is a lot of variability in the data, it might require a very small [mathjaxinline]\eta[/mathjaxinline] to effectively average over the individual steps moving in “competing" directions.
An effective strategy is to “average" between batch and stochastic gradient descent by using
mini-batches
. For a mini-batch of size [mathjaxinline]k[/mathjaxinline], we select [mathjaxinline]k[/mathjaxinline] distinct data points uniformly at random from the data set and do the update based just on their contributions to the gradient
[mathjax]W := W - \eta \sum _{i=1}^ k \nabla _ W \mathcal{L}(h(x^{(i)}; W), y^{(i)})\; \; .[/mathjax]
Most neural network software packages are set up to do mini-batches.
Study Question:
For what value of [mathjaxinline]k[/mathjaxinline] is mini-batch gradient descent equivalent to stochastic gradient descent? To batch gradient descent?
Picking [mathjaxinline]k[/mathjaxinline] unique data points at random from a large data-set is potentially computationally difficult. An alternative strategy, if you have an efficient procedure for randomly shuffling the data set (or randomly shufffling a list of indices into the data set) is to operate in a loop, roughly as follows:
Adaptive step-size
Picking a value for [mathjaxinline]\eta[/mathjaxinline] is difficult and time-consuming. If it's too small, then convergence is slow and if it's too large, then we risk divergence or slow convergence due to oscillation. This problem is even more pronounced in stochastic or mini-batch mode, because we know we need to decrease the step size for the formal guarantees to hold.
It's also true that, within a single neural network, we may well want to have different step sizes. As our networks become
deep
(with increasing numbers of layers) we can find that magnitude of the gradient of the loss with respect the weights in the last layer, [mathjaxinline]\partial \text {loss} / \partial W_ L[/mathjaxinline], may be substantially different from the gradient of the loss with respect to the weights in the first layer [mathjaxinline]\partial \text {loss} / \partial W_ L[/mathjaxinline]. If you look carefully at equation
, you can see that the output gradient is multiplied by all the weight matrices of the network and is “fed back" through all the derivatives of all the activation functions. This can lead to a problem of
exploding
or
vanishing
gradients, in which the back-propagated gradient is much too big or small to be used in an update rule with the same step size.
So, we'll consider having an independent step-size parameter
for each weight
, and updating it based on a local view of how the gradient updates
This section is very strongly influenced by Sebastian Ruder's excellent blog posts on the topic:
ruder.io/ optimizing-gradient-descent
have been going.
Running averages
We'll start by looking at the notion of a
running average
. It's a computational strategy for estimating a possibly weighted average of a sequence of data. Let our data sequence be [mathjaxinline]a_1, a_2, \ldots[/mathjaxinline]; then we define a sequence of running average values, [mathjaxinline]A_0, A_1, A_2, \ldots[/mathjaxinline] using the equations
[mathjaxinline]\displaystyle  A_0[/mathjaxinline]
[mathjaxinline]\displaystyle = 0[/mathjaxinline]
[mathjaxinline]\displaystyle A_ t[/mathjaxinline]
[mathjaxinline]\displaystyle = \gamma _ t A_{t-1} + (1 - \gamma _ t) a_ t[/mathjaxinline]
where [mathjaxinline]\gamma _ t \in (0, 1)[/mathjaxinline]. If [mathjaxinline]\gamma _ t[/mathjaxinline] is a constant, then this is a
moving
average, in which
[mathjaxinline]\displaystyle  A_ T[/mathjaxinline]
[mathjaxinline]\displaystyle  = \gamma A_{T-1} + (1 - \gamma ) a_ T[/mathjaxinline]
[mathjaxinline]\displaystyle  = \gamma (A_{T-2} + (1 - \gamma ) a_{T-1}) + (1 - \gamma ) a_ T[/mathjaxinline]
[mathjaxinline]\displaystyle  = \sum _{t = 0}^ T \gamma ^{T-t}(1 - \gamma ) a_ t[/mathjaxinline]
So, you can see that inputs [mathjaxinline]a_ t[/mathjaxinline] closer to the end of the sequence have more effect on [mathjaxinline]A_ t[/mathjaxinline] than early inputs.
If, instead, we set [mathjaxinline]\gamma _ t = (t - 1) / t[/mathjaxinline], then we get the actual average.
Study Question:
Prove to yourself that the previous assertion holds.
Momentum
Now, we can use methods that are a bit like running averages to describe strategies for computing [mathjaxinline]\eta[/mathjaxinline]. The simplest method is
momentum
, in which we try to “average" recent gradient updates, so that if they have been bouncing back and forth in some direction, we take out that component of the motion. For momentum, we have
[mathjaxinline]\displaystyle  V_0[/mathjaxinline]
[mathjaxinline]\displaystyle  = 0[/mathjaxinline]
[mathjaxinline]\displaystyle V_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = \gamma V_{t-1} + \eta \nabla _ W J(W_{t-1})[/mathjaxinline]
[mathjaxinline]\displaystyle W_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = W_{t-1} - V_ t[/mathjaxinline]
This doesn't quite look like an adaptive step size. But what we can see is that, if we let [mathjaxinline]\eta = \eta '(1 - \gamma )[/mathjaxinline], then the rule looks exactly like doing an update with step size [mathjaxinline]\eta '[/mathjaxinline] on a moving average of the gradients with parameter [mathjaxinline]\gamma[/mathjaxinline]:
[mathjaxinline]\displaystyle  V_0[/mathjaxinline]
[mathjaxinline]\displaystyle  = 0[/mathjaxinline]
[mathjaxinline]\displaystyle M_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = \gamma M_{t-1} + (1 - \gamma ) \nabla _ W J(W_{t-1})[/mathjaxinline]
[mathjaxinline]\displaystyle W_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = W_{t-1} - \eta ' M_ t[/mathjaxinline]
Study Question:
Prove to yourself that these formulations are equivalent.
We will find that [mathjaxinline]V_ t[/mathjaxinline] will be bigger in dimensions that consistently have the same sign for [mathjaxinline]\nabla _{\theta }[/mathjaxinline] and smaller for those that don't. Of course we now have
two
parameters to set ([mathjaxinline]\eta[/mathjaxinline] and [mathjaxinline]\gamma[/mathjaxinline]), but the hope is that the algorithm will perform better overall, so it will be worth trying to find good values for them. Often [mathjaxinline]\gamma[/mathjaxinline] is set to be something like [mathjaxinline]0.9[/mathjaxinline].
The red arrows show the update after one step of mini-batch gradient descent with momentum. The blue points show the direction of the gradient with respect to the mini-batch at each step. Momentum smooths the path taken towards the local minimum and leads to faster convergence.
Study Question:
If you set [mathjaxinline]\gamma = 0.1[/mathjaxinline], would momentum have more of an effect or less of an effect than if you set it to [mathjaxinline]0.9[/mathjaxinline]?
Adadelta
Another useful idea is this: we would like to take larger steps in parts of the space where [mathjaxinline]J(W)[/mathjaxinline] is nearly flat (because there's no risk of taking too big a step due to the gradient being large) and smaller steps when it is steep. We'll apply this idea to each weight independently, and end up with a method called
adadelta
, which is a variant on
adagrad
(for adaptive gradient). Even though our weights are indexed by layer, input unit and output unit, for simplicity here, just let [mathjaxinline]W_ j[/mathjaxinline] be any weight in the network (we will do the same thing for all of them).
[mathjaxinline]\displaystyle  g_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \nabla _ W J(W_{t-1})_ j[/mathjaxinline]
[mathjaxinline]\displaystyle G_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \gamma G_{t - 1,j} + (1 - \gamma )g_{t,j}^2[/mathjaxinline]
[mathjaxinline]\displaystyle W_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle  = W_{t-1, j} - \frac{\eta }{\sqrt {G_{t,j} + \epsilon }}g_{t,j}[/mathjaxinline]
The sequence [mathjaxinline]G_{t,j}[/mathjaxinline] is a moving average of the square of the [mathjaxinline]j[/mathjaxinline]th component of the gradient. We square it in order to be insensitive to the sign—we want to know whether the magnitude is big or small. Then, we perform a gradient update to weight [mathjaxinline]j[/mathjaxinline], but divide the step size by [mathjaxinline]\sqrt {G_{t,j} + \epsilon }[/mathjaxinline], which is larger when the surface is steeper in direction [mathjaxinline]j[/mathjaxinline] at point [mathjaxinline]W_{t-1}[/mathjaxinline] in weight space; this means that the step size will be smaller when it's steep and larger when it's flat.
Adam
Adam has become the default method of managing step
Although, interestingly, it may actually violate the convergence conditions of
sgd
:
arxiv.org/abs/1705.08292
sizes neural networks
. It combines the ideas of momentum and and adadelta. We start by writing moving averages of the gradient and squared gradient, which reflect estimates of the mean and variance of the gradient for weight [mathjaxinline]j[/mathjaxinline]:
[mathjaxinline]\displaystyle  g_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \nabla _ W J(W_{t-1})_ j[/mathjaxinline]
[mathjaxinline]\displaystyle m_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle  = B_1m_{t - 1,j} + (1 - B_1)g_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle v_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle  = B_2v_{t - 1,j} + (1 - B_2)g_{t,j}^2 \; \; .[/mathjaxinline]
A problem with these estimates is that, if we initialize [mathjaxinline]m_0 = v_0 = 0[/mathjaxinline], they will always be biased (slightly too small). So we will correct for that bias by defining
[mathjaxinline]\displaystyle  \hat{m}_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{m_{t,j}}{1 - B^ t_1}[/mathjaxinline]
[mathjaxinline]\displaystyle \hat{v}_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{v_{t,j}}{1 - B^ t_2}[/mathjaxinline]
[mathjaxinline]\displaystyle W_{t,j}[/mathjaxinline]
[mathjaxinline]\displaystyle  = W_{t-1,j} - \frac{\eta }{\sqrt {\hat{v}_{t,j} + \epsilon }}\hat{m}_{t,j} \; \; .[/mathjaxinline]
Note that [mathjaxinline]B^ t_1[/mathjaxinline] is [mathjaxinline]B_1[/mathjaxinline] raised to the power [mathjaxinline]t[/mathjaxinline], and likewise for [mathjaxinline]B^ t_2[/mathjaxinline]. To justify these corrections, note that if we were to expand [mathjaxinline]m_{t,j}[/mathjaxinline] in terms of [mathjaxinline]m_{0,j}[/mathjaxinline] and [mathjaxinline]g_{0,j}, g_{1,j}, \dots , g_{t,j}[/mathjaxinline] the coefficients would sum to [mathjaxinline]1[/mathjaxinline]. However, the coefficient behind [mathjaxinline]m_{0,j}[/mathjaxinline] is [mathjaxinline]B_1^ t[/mathjaxinline] and since [mathjaxinline]m_{0,j} = 0[/mathjaxinline], the sum of coefficients of non-zero terms is [mathjaxinline]1 - B_1^ t[/mathjaxinline], hence the correction. The same justification holds for [mathjaxinline]v_{t,j}[/mathjaxinline].
Now, our update for weight [mathjaxinline]j[/mathjaxinline] has a step size that takes the steepness into account, as in adadelta, but also tends to move in the same direction, as in momentum. The authors of this method propose setting [mathjaxinline]B_1 = 0.9, B_2 = 0.999, \epsilon = 10^{-8}[/mathjaxinline]. Although we now have even more parameters, Adam is not highly sensitive to their values (small changes do not have a huge effect on the result).
Study Question:
Define [mathjaxinline]\hat{m_ j}[/mathjaxinline] directly as a moving average of [mathjaxinline]g_{t,j}[/mathjaxinline]. What is the decay ([mathjaxinline]\gamma[/mathjaxinline] parameter)?
Even though we now have a step-size for each weight, and we have to update various quantities on each iteration of gradient descent, it's relatively easy to implement by maintaining a matrix for each quantity ([mathjaxinline]m^{\ell }_ t, v^{\ell }_ t, g^{\ell }_ t, {g^{2}_ t}^{\ell }[/mathjaxinline]) in each layer of the network.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:06 PM (revision 4f166135)

## Lecture: Neural networks - regularization by weight decay

Lecture: Neural networks - regularization by weight decay
Lecture: Neural networks - regularization by weight decay
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - regularization by early stopping and dropout

Lecture: Neural networks - regularization by early stopping and dropout
Lecture: Neural networks - regularization by early stopping and dropout
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks - regularization by batch normalization

Lecture: Neural networks - regularization by batch normalization
Lecture: Neural networks - regularization by batch normalization
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Regularization

Regularization
So far, we have only considered optimizing loss on the training data as our objective for neural network training. But, as we have discussed before, there is a risk of overfitting if we do this. The pragmatic fact is that, in current deep neural networks, which tend to be very large and to be trained with a large amount of data, overfitting is not a huge problem. This runs counter to our current theoretical understanding and the study of this question is a hot area of research. Nonetheless, there are several strategies for regularizing a neural network, and they can sometimes be important.
Methods related to ridge regression
One group of strategies can, interestingly, be shown to have similar effects: early stopping, weight decay, and adding noise to
Result is due to Bishop, described in his textbook and here
doi.org/10.1162/ neco.1995.7.1.108
.
the training data.
Early stopping is the easiest to implement and is in fairly common use. The idea is to train on your training set, but at every
epoch
(pass through the whole training set, or possibly more frequently), evaluate the loss of the current [mathjaxinline]W[/mathjaxinline] on a
validation set
. It will generally be the case that the loss on the training set goes down fairly consistently with each iteration, the loss on the validation set will initially decrease, but then begin to increase again. Once you see that the validation loss is systematically increasing, you can stop training and return the weights that had the lowest validation error.
Another common strategy is to simply penalize the norm of all the weights, as we did in ridge regression. This method is known as
weight decay
, because when we take the gradient of the objective
[mathjax]J(W) = \sum _{i = 1}^{n}\text {Loss}(\text {NN}(x^{(i)}), y^{(i)}; W) + \lambda \| W\| ^2[/mathjax]
we end up with an update of the form
[mathjaxinline]\displaystyle  W_ t[/mathjaxinline]
[mathjaxinline]\displaystyle = W_{t-1} - \eta \left(\left(\nabla _{W}\text {Loss}(\text {NN}(x^{(i)}), y^{(i)}; W_{t-1})\right) + \lambda W_{t-1}\right)[/mathjaxinline]
[mathjaxinline]\displaystyle  = W_{t-1}(1 - \lambda \eta ) - \eta \left(\nabla _{W}\text {Loss}(\text {NN}(x^{(i)}), y^{(i)}; W_{t-1})\right) \; \; .[/mathjaxinline]
This rule has the form of first “decaying" [mathjaxinline]W_{t-1}[/mathjaxinline] by a factor of [mathjaxinline](1 - \lambda \eta )[/mathjaxinline] and then taking a gradient step.
Finally, the same effect can be achieved by perturbing the [mathjaxinline]x^{(i)}[/mathjaxinline] values of the training data by adding a small amount of zero-mean normally distributed noise before each gradient computation. It makes intuitive sense that it would be more difficult for the network to overfit to particular training data if they are changed slightly on each training step.
Dropout
Dropout is a regularization method that was designed to work with deep neural networks. The idea behind it is, rather than perturbing the data every time we train, we'll perturb the network! We'll do this by randomly, on each training step, selecting a set of units in each layer and prohibiting them from participating. Thus, all of the units will have to take a kind of “collective" responsibility for getting the answer right, and will not be able to rely on any small subset of the weights to do all the necessary computation. This tends also to make the network more robust to data perturbations.
During the training phase, for each training example, for each unit, randomly with probability [mathjaxinline]p[/mathjaxinline] temporarily set [mathjaxinline]a^{\ell }_ j := 0[/mathjaxinline]. There will be no contribution to the output and no gradient update for the associated unit.
Study Question:
Be sure you understand why, when using
sgd
, setting an activation value to 0 will cause that unit's weights not to be updated on that iteration.
When we are done training and want to use the network to make predictions, we multiply all weights by [mathjaxinline]p[/mathjaxinline] to achieve the same average activation levels.
Implementing dropout is easy! In the forward pass during training, we let
[mathjax]a^{\ell } = f(z^{\ell }) * d^{\ell }[/mathjax]
where [mathjaxinline]*[/mathjaxinline] denotes component-wise product and [mathjaxinline]d^{\ell }[/mathjaxinline] is a vector of [mathjaxinline]0[/mathjaxinline]'s and [mathjaxinline]1[/mathjaxinline]'s drawn randomly with probability [mathjaxinline]p[/mathjaxinline]. The backwards pass depends on [mathjaxinline]a^{\ell }[/mathjaxinline], so we do not need to make any further changes to the algorithm.
It is common to set [mathjaxinline]p[/mathjaxinline] to [mathjaxinline]0.5[/mathjaxinline], but this is something one might experiment with to get good results on your problem and data.
Batch Normalization
A more modern alternative to dropout, which tends to achieve better performance, is
For more details see
arxiv.org/abs/1502.03167
.
batch normalization
.
It was originally developed to address a problem of
covariate shift
: that is, if you consider the second layer of a two-layer neural network, the distribution of its input values is changing over time as the first layer's weights change. Learning when the input distribution is changing is extra difficult: you have to change your weights to improve your predictions, but also just to compensate for a change in your inputs (imagine, for instance, that the magnitude of the inputs to your layer is increasing over time—then your weights will have to decrease, just to keep your predictions the same).
So, when training with mini-batches, the idea is to
standardize
the input values for each mini-batch, just in the way that we did it in section of chapter 4, subtracting off the mean and dividing by the standard deviation of each input dimension. This means that the scale of the inputs to each layer remains the same, no matter how the weights in previous layers change. However, this somewhat complicates matters, because the computation of the weight updates will need to take into account that we are performing this transformation. In the modular view, batch normalization can be seen as a module that is applied to [mathjaxinline]z^ l[/mathjaxinline], interposed after the product with [mathjaxinline]W^ l[/mathjaxinline] and before input to [mathjaxinline]f^ l[/mathjaxinline].
Batch normalization ends up having a regularizing effect for similar reasons that adding noise and dropout do: each mini-batch of data ends up being mildly perturbed, which prevents the network from exploiting very particular values of the data points.
Let's think of the batch-norm layer as taking [mathjaxinline]z^ l[/mathjaxinline] as input and producing an output [mathjaxinline]\widehat{Z}[/mathjaxinline] as output. But now, instead of thinking of [mathjaxinline]Z^ l[/mathjaxinline] as an [mathjaxinline]n^ l \times 1[/mathjaxinline] vector, we have to explicitly think about handling a mini-batch of data of size [mathjaxinline]K[/mathjaxinline], all at once, so [mathjaxinline]Z^ l[/mathjaxinline] will be [mathjaxinline]n^ l \times K[/mathjaxinline], and so will the output [mathjaxinline]\widehat{Z}^ l[/mathjaxinline].
Our first step will be to compute the
batchwise
mean and standard deviation. Let [mathjaxinline]\mu ^ l[/mathjaxinline] be the [mathjaxinline]n^ l \times 1[/mathjaxinline] vector where
[mathjax]\mu ^ l_ i = \frac{1}{K} \sum _{j = 1}^ K Z^ l_{ij}\; \; ,[/mathjax]
and let [mathjaxinline]\sigma ^ l[/mathjaxinline] be the [mathjaxinline]n^ l \times 1[/mathjaxinline] vector where
[mathjax]\sigma ^ l_ i = \sqrt {\frac{1}{K} \sum _{j = 1}^ K (Z^ l_{ij} - \mu _ i)^2}\; \; .[/mathjax]
The basic normalized version of our data would be a matrix, element [mathjaxinline](i, j)[/mathjaxinline] of which is
[mathjax]\overline{Z}^ l_{ij} = \frac{Z^ l_{ij} - \mu ^ l_ i}{\sigma ^ l_ i + \epsilon }\; \; ,[/mathjax]
where [mathjaxinline]\epsilon[/mathjaxinline] is a very small constant to guard against division by zero. However, if we let these be our [mathjaxinline]\widehat{Z}[/mathjaxinline] values, we really are forcing something too strong on our data—our goal was to normalize across the data batch, but not necessarily force the output values to have exactly mean 0 and standard deviation 1. So, we will give the layer the “opportunity" to shift and scale the outputs by adding new weights to the layer. These weights are [mathjaxinline]G^ l[/mathjaxinline] and [mathjaxinline]B^ l[/mathjaxinline], each of which is an [mathjaxinline]n^ l \times 1[/mathjaxinline] vector. Using the weights, we define the final output to be
[mathjax]\widehat{Z}^ l_{ij} = G^ l_ i \overline{Z}^ l_{ij} + B^ l_ i\; \; .[/mathjax]
That's the forward pass. Whew!
Now, for the backward pass, we have to do two things: given [mathjaxinline]\partial L / \partial \widehat{Z}^ l[/mathjaxinline],
Compute [mathjaxinline]\partial L / \partial Z^ l[/mathjaxinline] for back-propagation, and
Compute [mathjaxinline]\partial L / \partial G^ l[/mathjaxinline] and [mathjaxinline]\partial L / \partial B^ l[/mathjaxinline] for gradient updates of the weights in this layer.
Schematically
[mathjax]\frac{\partial L}{\partial B} = \frac{\partial L}{\partial \widehat{Z}}\frac{\partial \widehat{Z}}{\partial B}\; \; .[/mathjax]
It's hard to think about these derivatives in matrix terms, so we'll see how it works for the components. [mathjaxinline]B_ i[/mathjaxinline] contributes to [mathjaxinline]\widehat{Z}_{ij}[/mathjaxinline] for all data points [mathjaxinline]j[/mathjaxinline] in the batch. So
[mathjaxinline]\displaystyle  \frac{\partial L}{\partial B_ i}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \sum _ j \frac{\partial L}{\partial \widehat{Z}_{ij}} \frac{\partial \widehat{Z}_{ij}}{\partial B_ i}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \sum _ j \frac{\partial L}{\partial \widehat{Z}_{ij}}\; \; ,[/mathjaxinline]
Similarly, [mathjaxinline]G_ i[/mathjaxinline] contributes to [mathjaxinline]\widehat{Z}_{ij}[/mathjaxinline] for all data points [mathjaxinline]j[/mathjaxinline] in the batch. So
[mathjaxinline]\displaystyle  \frac{\partial L}{\partial G_ i}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \sum _ j \frac{\partial L}{\partial \widehat{Z}_{ij}} \frac{\partial \widehat{Z}_{ij}}{\partial G_ i}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \sum _ j \frac{\partial L}{\partial \widehat{Z}_{ij}} \overline{Z}_{ij}\; \; .[/mathjaxinline]
Now, let's figure out how to do backprop. We can start schematically:
[mathjax]\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial \widehat{Z}} \frac{\partial \widehat{Z}}{\partial Z}\; \; .[/mathjax]
And because dependencies only exist across the batch, but not across the unit outputs,
[mathjax]\frac{\partial L}{\partial Z_{ij}} = \sum _{k=1}^ K\frac{\partial L}{\partial \widehat{Z}_{ik}} \frac{\partial \widehat{Z}_{ik}}{\partial Z_{ij}}\; \; .[/mathjax]
The next step is to note that
[mathjaxinline]\displaystyle  \frac{\partial \widehat{Z}_{ik}}{\partial Z_{ij}}[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{\partial \widehat{Z}_{ik}}{\partial \overline{Z}_{ik}} \frac{\partial \overline{Z}_{ik}}{\partial Z_{ij}}[/mathjaxinline]
[mathjaxinline]\displaystyle = G_ i \frac{\partial \overline{Z}_{ik}}{\partial Z_{ij}}[/mathjaxinline]
And now that
[mathjax]\frac{\partial \overline{Z}_{ik}}{\partial Z_{ij}} = \left(\delta _{jk} - \frac{\partial \mu _ i}{\partial Z_{ij}}\right) \frac{1}{\sigma _ i} - \frac{Z_{ik} - \mu _ i}{\sigma _ i^2} \frac{\partial \sigma _ i}{\partial Z_{ij}} \; \; ,[/mathjax]
where [mathjaxinline]\delta _{jk} = 1[/mathjaxinline] if [mathjaxinline]j = k[/mathjaxinline] and [mathjaxinline]\delta _{jk} = 0[/mathjaxinline] otherwise. Getting close! We need two more small parts:
[mathjaxinline]\displaystyle  \frac{\partial \mu _ i}{\partial Z_{ij}}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{1}{K}[/mathjaxinline]
[mathjaxinline]\displaystyle \frac{\partial \sigma _ i}{\partial Z_{ij}}[/mathjaxinline]
[mathjaxinline]\displaystyle  = \frac{Z_{ij} - \mu _ i}{K \sigma _ i}[/mathjaxinline]
Putting the whole crazy thing together, we get
[mathjax]\frac{\partial L}{\partial Z_{ij}} = \sum _{k=1}^ K\frac{\partial L}{\partial \widehat{Z}_{ik}} G_ i\frac{1}{K \sigma _ i}\left(\delta _{jk}K-1 - \frac{(Z_{ik} - \mu _ i)(Z_{ij} - \mu _ i)}{\sigma _ i^2} \right)[/mathjax]
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:06 PM (revision 4f166135)

## Video transcripts

### Lecture: Neural networks - brief review of layers and backprop


LESLIE KAELBLING: So last week in the lecture,
we talked about the basics of neural networks, the idea
that we have these kind of neurons which take a--
you can think of them as taking a dot product of their input
with the weights and then running it
through a nonlinear function.
And we looked at how you can make a layer out
of a bunch of these things, and how
you can take layers and put them together and make a network.
And we spent some time doing chain rule
to see that you could-- there's kind of a direct and efficient
way to compute the gradient of the loss function
with respect to any of the weights in the network.
Because generally speaking, the way
we'll train on a neural network is by doing gradient descent.
And so what we saw last time was just the basics of how do you
compute the gradient?
So once you know how to compute the gradient,
you're almost ready to go.
But there are some practical aspects
of actually making the gradient descent work out,
which we're going to talk about today.
And then we're going to talk about regularization some.
And then depending on time, there's
one more topic, potentially.
OK, so let's, though, first start out
talking about training.


### Lecture: Neural networks - brief review of layers and backprop



### Lecture: Neural networks - brief review of layers and backprop


LESLIE KAELBLING: Kind of what we're doing now
is working on taking the basic idea we got last time
and trying to make it work.
So here we've talked about how much
data to sum the gradient over.
The next big question is eta.

So if you've ever played around, you've already
played around with ADA and just simple gradient
descent for, let's say, regression,
and you've seen that it can make a big difference in what
happens.
In a neural network, eta is a much, much more troublesome
and difficult thing.
So let me just give you some hints about why eta is tricky.
So first of all, there's this idea
that when we're doing stochastic gradient descent,
and even if we're doing batch gradient descent,
you could think of it as a kind--
it's a kind of stochastic gradient, right?
Because each time, we're making a step with this batch.
But that batch doesn't represent the whole gradient.
It's related to the whole gradient,
but it's not exactly the right direction.
So still, if you're doing batch gradient descent,
this ADA has to decrease over time.
If you want to make promises about a convergence.
So one issue with ADA is just, well, how big or small
do you want it to be, and how do you
want that to depend on the epoch-- on the time
through the data?
So that's one thing.
The other thing is this troublesome property
of deep neural networks.
That's right.
We don't just have neural networks.
We have deep neural networks.
So in our picture we had said, oh, OK.
x comes in.
The input comes in.
We're going to multiply it by the weights.
And then it's going to go through an F.
And then we're going to multiply it by some more weights.
And then it's going to go through another F.
And it's going to do this for a while.
And then there's some more weights.
And then there's another F. And then out pops our answer.
And these are different weights.
Weight 1, weight, 2, weight L. And these
are potentially different Fs.
OK.

And what we saw also was that when we computed the gradient--
I'll just recap this because it's an important idea.
So in order to do gradient descent,
we have to know roughly the derivative of loss
with respect to the weights at each particular layer.
Right?
And that's a thing that looks like this.

So we have to be able to compute the gradient loss with respect
to the weights, let's say, at this layer.
So we would have to know this.
In order to change the weights at layer 2,
we have to know how did the weights at layer 2
affect the loss?
So to know that, the easiest thing
is to know the D loss DZL.
So the Z is what comes out here.
This is Z1, Z2.

And how do we compute D loss DZL?
Well, that's this ginormous product.
So D loss DZL is a big old product.
DA L. DZL times w L plus 1, times DA L plus 1, DZ L plus 1,
times W L plus 2, et cetera.
Some more Ws.
Some more DA DLs.
Some more Ws.
So it's this big old product.

And so what that means is that--
oh.
And the final thing that we have, the last term,
is D loss DA L.
So this is like the real loss.
This is like, OK.
We gave AL out as the answer.
And we were sitting about unhappy with it.
And this is how much we're blaming
AL for that unhappiness.
And then we take that, whatever it is--
so it's like a vector of unhappinesses--
and we multiply it by a matrix.
And then run it through a derivative,
and multiply it by another matrix,
and run it through another derivative--
all the way back to here.
So if these matrices, if these Ws--
let's ignore for right, just ignore even the function part.
If these Ws are big, if those values are big numbers,
then we have what's known as the exploding gradient problem.
You can see that it's going to grow exponentially
in the depth.
Because we're basically taking some product of weight matrices
that's D deep-- however deep our network is.
So if those magnitudes tend to be big in the weight matrices,
then we'll get a really big number out here.
It'll be not so big when we're computing
the gradient with respect to W L.
But it'll be really big when we're
computing the gradient with respect to W1.
Because if the W's are big, the gradients will expand.
If the Ws are not big, that is to say
they are less than 1-ish, then the gradient
will tend to vanish.

In fact, almost never will the magnitude of the gradient
stay the same as you go back through the layers.
Does that make sense?
These Ws-- it's just like we looked
at some little, simple dynamical systems things.
And basically either everything is going to get smaller
or it's going to get bigger.
It hardly ever just stays the same.
So what that means is that the step size we want to make
is actually going to be different.
If this gradient is really big numbers
and this gradient is really small numbers or vise versa,
they're going to want to have different step sizes.
If the step size is too big, we go bouncing all over the place.
If the step size is too small, we only make small moves.
So what we're going to do is actually use different step
sizes.

And so we talked about using different step sizes over time.
Now we're going to use different step sizes actually
in each layer.

And in fact, what we're going to do
is go to an extreme version of that,
which is have a different step size for every single weight
in the network at every single training time.
So not just different eta in each layer, but each weight
and at each time.
OK.
So now you're thinking, oh my goodness.
I already hated tuning eta when there was only one of them.
Now you told me there's one for every weight and at every time?
This seems like a terrible, terrible idea.
OK.
So it's not a terrible idea because we're not
going to do the tuning--
the computer is.


### Lecture: Neural networks - brief review of layers and backprop


LESLIE KAELBLING: I'm going to take a little short discursion
because it's an idea that we'll need to talk
about the adaptive steps sizes.
So we're going to do adaptive here.
Let me just introduce the topic.
The topic now is adaptive step sizes.

So we're going to write rules that
adjust w step size for each weight
depending on how things are going.
Basic idea.
OK.
So first, let's look at the idea of a running average.

So imagine that I have a sequence of data.

It keeps going.

And I'm going to define a sequence of running averages.

A, et cetera forever.
And it's going to go look like this.
I'm going to set, generally speaking, not always,
but usually we'll set A0 to B0.

And then A for all other little t's.

OK.
So gamma is a parameter.
Gamma t is a parameter.
So a sequence of parameters-- oh my goodness.
I keep turning problems with one parameter
into problems with more parameters.
That does not seem like the right thing to be doing,
but it'll work out OK.
Hang tight.
In the simplest case, gamma is the same for all the things.
But we'll look at that.
OK.
So this is a running average.
It says somehow, what I'm going to do
is I'm going to take my old big A, some fraction
of the old big A.
Generally speaking, gamma is in 0, 1.
Well, it's dumb if it's 0 and it's dumb if it's 1,
so it should really be inside the interval.
And generally speaking--
I don't know.
You can think of gamma, if you just kind of want
to get a feeling for this, think of gamma as, like, 0.9.
So you're saying, OK.
I'm trying to average my values over time.
Kind of smooth them out, right?
So I'm going to take most of my old idea of what
the average was, but I'm going to add in some of this new data
point that I have.
So that's the basic idea.
There's a couple of useful special cases.

So we'll say when gamma is constant,
we'll call this a moving average.
And this is the case that we'll use mostly,
but I want to illustrate another thing to you.
All right.
So if gamma is constant, this is a moving average.
And how does it work?
So imagine that we're at time t.

What's the value of A at time t?
Let's just look at that for a minute to see what happens.
So the value of A at time t is gamma a at t minus 1
plus 1 minus gamma little a at t.
And that is gamma times gamma times A at t minus 2
plus 1 minus gamma little a at t minus 1 plus 1 minus gamma
little a at t.
So you could imagine rolling this all the way back to 0.
And if you roll it back to 0, what
you get is the sum from little t equals 1 through-- actually,
let me make this big T so that I can use T as an index.
Sorry, I know.
Never change something that's on the board.
Students hate you when you do that.

OK.
Big T. So T is the end of our sequence.
We're interested in knowing what's
the value at some terminal time T. So from little
t from 1 to big T. And this is going to be gamma
to the big T minus little t.
1 minus gamma a sub t.
So what this is is it's an average
in a certain sense of all the little a's
that you've ever seen.
But the ones that are farther away-- so remember, gamma
is less than 1.
And so the farther, the older something
is, the less influence it has.
The exponentially less influence it has.
Because we're raising this gamma to a power
that's related to how old the thing is.
So if little t is equal to big T,
that gamma is raised to the power of 0
so it doesn't come to the party.

So it's a decaying average.
More sensitive to more recent values.
Old values still have some influence.
And by changing the gamma, you change
how much it's weighted toward recent things versus old.

OK.
So that's the basic thing.
Oh, a little entertaining project for you to think about
is if you do this with a gamma t--
you can prove this to yourself at home.
I'm not going to do it here on the board.
If you set gamma t to t minus 1 over t.
So here we were considering a constant gamma.
If you instead consider a gamma that goes like this--
so actually a gamma that changes--
what you'll get is the true average of all the a sub i's.

So that's kind of cool, too.
And it's fun to work out.
Yeah.
STUDENT: 1 minus 1 over t.

LESLIE KAELBLING: Excuse me?
STUDENT: Shouldn't it b 1 minus 1 over t?

LESLIE KAELBLING: It is 1 minus 1 over t.
Oh, I see.
We need parentheses, you're right.
Yes.
Right.
This is the same as 1 minus 1 over t.
So bad orthography on my part.
Yeah.

OK.
So these are fun things.
You can mess around and get them to do different things for you.
And these are the underlying basis
for the adaptive steps size stuff that we're
going to look at now.


### Lecture: Neural networks - brief review of layers and backprop


LESLIE KAELBLING: So adaptive stepsize--
the first strategy actually is not precisely
an adaptive stepsize.
It's actually kind of-- you could think of it
more like an adaptive direction.
But so the motivating example for momentum
is imagine that these are the contours
of your objective function, right,
so that we're going downhill.
But the optimum is here in the middle.
If you do gradient descent--
straight up gradient descent--
on this and you take small steps,
then you'll probably be OK.
But if you take big, biggish steps, you'll tend to--
the steepest direction is always down the canyon, right?
The steepest direction is crosswise.
But really, that's not where you would like to be going, right?
Really generally-- so you know, if you set your stepsize,
you could get there.
But really, you'd like to kind of
be going, like, down that way.

The question is, how can you go down that way?
So one way to think about it is that there are some directions.
There's one-- especially if we were aligned according
to the axes, there would be directions,
where things are kind of going back and forth all the time,
right?
So there's positive sign, negative sign, positive sign,
negative sign.
And there are other directions that you're actually
continually making forward progress in.
So it kind of feels like if you were
to average the gradient directions over time,
that would point you a better way, right?
So these guys are going like this.
But their average is, in fact, going this way.
And so the idea behind the momentum rule
is to average the gradient and take a step
in the direction of a recent--
recency weighted average, a moving average of the gradient.
So I'm going to write the momentum rule down
in two different ways--
the way that you would normally see it in the book and the way
that I like to think about it in these terms.
So the normal way that you would see momentum--
so we're going to define a new variable, V. And then we
say V at time t plus some gamma at times V at time t
and minus 1 plus some stepsize times the gradient.

I'm going to write these rules down
for the rest of this conversation,
as if we're taking the gradient with respect to the whole data,
with the whole J. But they apply perfectly well
in the mini-batch and in the stochastic gradient case.
OK.
So this is the default way of writing it down.
And then, the rule--
I'm just going to write this.
But I'm going to write it right away a different way.
And we'll study the different way.
W t minus 1 minus Vt.

OK.
I don't really like that way of writing it.
But I had to do it, because that's what people do.
OK.
I like this way of writing it.
And then, I'll show you it's the same as that.

Sorry.
I got ahead of myself.

And then Wt is old Wt minus eta--
we'll call it eta prime--
Vt.
And this is the same if eta prime is--
oh, if eta-- if we set eta up there to be eta prime times
1 minus gamma.
OK.
So you don't really have to worry about that.
I just wanted to make you not feel
that you had been lied to if I only
showed you my version of it, because I
like my version better.
I think it's clearer.
And it's basically the same.
OK.
So what's my version?
My version says Vt is a moving average of the gradient, right?
We're just adding up the gradient,
weighting the gradients that we have recently
more heavily than the gradients we had a long time ago.
And then, we're taking your gradient [INAUDIBLE]
in the direction of that average.
OK.
That's momentum.

Can I tell you how to set gamma and eta?
No.
[LAUGHS]
You still have to experiment with those.
But this can often make convergence
more efficient, right?
The reason is if you don't have momentum,
then you have a choice of having a small stepsize, which
will kind of creep down here relatively directly
or a big step size, which will tend to bounce back and forth.
But either way, you spend a lot iterations
getting to where you're going.
And this helps you kind of orient yourself
in a better direction often.
For any one of these rules, someone
can construct a miserable landscape
that will make things worse.
So this stuff is all kind of, you know,
a little bit heuristic.
You have to try stuff out.
Sometimes it works.
Sometimes it doesn't.
I wish I could say, oh, this-- do this.
And everything will be better.
But unfortunately, things don't work like that.
But this is often helpful.
OK.
That's momentum.
That's one idea.
I have another idea.
We'll look at it.
And then, we'll put them together.
And we'll have a good thing.


### Lecture: Neural networks - brief review of layers and backprop


LESLIE KAELBLING: Another idea.
We'll call it adagrad.
That's not an idea, that's a name.
But I'll show you the idea.
Here's the idea.

OK.
Let's imagine that this is our optimization landscape.
Imagine the slope is actually--
I mean, it's really going down here.

So the thing is--

actually, I want to make a more dramatic example.

OK, that's a better example.
All right.
So let's talk about step size.
We're trying to find the minimum.
We're just in one dimension.
In order to find this, we need to have a really small step
size.

But if we have a really small step size, over here,
the gradient is almost 0.
So if we multiply a very small gradient by a very small step
size, we'll be like taking tiny, tiny steps.
We have to take tiny steps if we want to fall down this hole.

So there is an idea, which is to say
that we would like to take big steps when
there's low curvature.

And small steps when there's high curvature.

That would be nice.
So that while you're over here, you can be like, oh, man.
This is really boring.
I'm taking really big steps.
But you have to have some warning.
If this is really a discontinuous little jiggity
jag, then you'll never get it.
But if you start to notice that the curvature is getting high
here, then you can make smaller steps.

That's the idea.
OK.
So how can we arrange that?
Turns out to not be too hard to arrange.
So let me write the rule.
This is a adadelta.
OK.
There was a thing called adagrad.
It was not so the best.
So they made a small change to it.
Someone got another important publication.
And now it's called adadelta.
Adaptive gradient, adaptive delta.
I don't know.
OK.
So let's look at it adagrad.

Adagrad looks like this.

Now I'm really going to just do this per weight.
So G-- So assume that my weights are just
indexed by J. I'm just telling you what to do for each weight
individually.
It's going to be the same thing no matter what layer it's in
or whether it's a W0 or what.
So I'm just now, when I write this rule and the next rule,
I'm going to do it just for one weight.

So just assume that our giant, big old--
all the weights in our network, for right now,
they're just indexed by J. So one weight.
And I'm going to define a series of values G. The first G is
just our gradient.
So I can write this two ways.
I'm going to write it two ways just to make a point, I think.
Yes, I am.
I could write it that I could take the gradient.
That's going to be a giant--
you could think of it as a giant vector if you want.
If we took it all the way to the network
and just made a giant vector out of them,
then this gradient is a giant vector.
And we could take the J'th component.
So we could be concerned with the weights at time t minus 1.
We take the J'th component of that.
Another way to think about this is
just the same as the partial derivative of J
at weights at time t minus 1 with respect to weight J.
So this is just, how much does weight J,
some particular weight in our network, affect the loss?
Affect the J. OK, so that's little g.
That's playing the role of our A.
Now we can make a moving average for weight J like this.

Little g. gJ.

Squared.

So this is going to be a moving average
of the square of the gradient.
OK.
Why would we square it?
Well, we square it because we're interested in understanding
the magnitude, which tells us something about the curvature.
We're right now not so much interested
in whether it's positive or negative.
We just want a notion of how curvy is it in this direction.
Remember that we're doing an optimization--
I draw this one-dimensional picture
or a two-dimensional picture up there.
That's as if we had one weight in our network
or two weights in our network.
When we have 100,000 weights in our network, then WJ is 100,000
and our optimization is a function of 100,000 variables.
So we have 100,000 directions that we could be going in.
But we're just going to consider each one,
and we're going to look at the square of the gradient in each
of these directions.
And we're going to average that.
So this is just for one direction J. One direction J,
which this recency weighted average of that.
And then we're going to make our update rule be this.

We still have an ada, but--
G, t, J, plus epsilon.
[INAUDIBLE]

OK.
So what is this?
If you ignore this bit, this is just our good old gradient
descent, right?
If we just ignore that divided by something,
we have weight J at time t is weight J at time t minus 1
minus the step size times the gradient.
We've been writing this as a whole vector update,
but I'm here I'm just writing one component.
But it's the same thing.
I'm just writing one component as a big vector, update.
So this is just our totally normal update rule.
But now I'm doing this.
So what is this about?
So when is G big?

G is big when the curvature is high.

And so we divide by something big
when the curvature is high, which
means we take small steps.
So we'll take small steps when the curvature is high,
big steps from the curvature is small.
What's epsilon?
Epsilon is there to keep this from going too crazy.
So you just make it a small number
so that you're not dividing by something near 0
and making an enormous step size.
You always have to watch out for things like that.

OK.
That's adadelta.
Does that kind of seem OK?
Yeah.
STUDENT: [INAUDIBLE]

--before that dip.
LESLIE KAELBLING: Yeah.
STUDENT: You're going to have an [INAUDIBLE] taking big steps.
You're just going to overshoot the dip entirely.
LESLIE KAELBLING: Well, OK, good.
So momentum, kind of, but not exactly the same thing
as momentum as described up there.
You're right.
So depending on how aggressively your things change,
you would need to tune gamma.
So if you make gamma close to 1, then you
have this problem that you suggest,
which is that these big steps over here
will be weighing on you a lot.
The flatness over here will persist in your estimate
of the curvature.
If, on the other hand, you let this gamma move away from 1,
you let it be, like, 0.8 or 0.5, it decays pretty quickly.
So it's, again, a thing that's often useful.
But absolutely, you can thwart it.
Yep.
STUDENT: [INAUDIBLE]

LESLIE KAELBLING: What's this big G?
This big G is an average of the square of the gradient.
So you can think of it as the average
of a kind of an estimate of the curvature.
So it's a running average.
So by changing gamma, you're changing
how much weight you put on the recent guys
versus the old guys.
And because you raise it to the power of how old they are,
they decay really pretty quickly.
But you can govern that by changing the gamma.


### Lecture: Neural networks - brief review of layers and backprop

LESLIE KAELBLING: OK, so we have two ideas.
They're separate ideas.
This one says you should average the actual gradient--
the directions.
And this says you should moderate your step
size by some kind of a magnitude of the gradients.
And so that brings us to this thing called Adam--

it also has a convenient name, Adam--
which is just the straight up combination
of these two things.
And the reason that we want to go all the way to Adam
here is because if you use any neural network
package at this moment, it will probably
have the default behavior set to be-- to do Adam.
OK.
So on the one hand, let me just say
that Adam is kind of like a practical default
at this moment that seems to make things mostly work better.

It's also true that there's some theory papers that say
that it can wreck convergence.
Let me just say this.

So on the one hand, it seems to be pretty useful in practice.
On the other hand, if you're theoretically inclined,
you can show that, at least in some circumstances,
the guarantee that we used to have about stochastic gradient
converging to its optimum--
even, I think, a regular gradient under this--
is now violated if you use this rule.
Right now deep learning is generating
a lot of tension between theory and practice,
which is interesting.
And it makes everybody think harder
about what they're doing.
So this is what the current situation is.
It works pretty well.
The theoretical foundations might be a little iffy.
OK.
So it looks like this.
I'm just going to write it down.
But it's a combination of these two ideas.
So we'll let g be what it was before--

the Jth component of the gradient.
And then we're going to define the mean.
We're going to use m now to really stand for mean.
And we're going to have two learning rate things--
two gammas.
But they call them B, and so I'm going to go along.

B1, 1 minus B1, gtj.

And this is going to be something
you can think of as a variance.

B2, Vt minus 1j, plus 1 minus B2, gtj squared.

So you can think of this as a kind of a running
computation of a mean of the gradient,
and this is kind of a running computation of the variance
of the gradient.
So that's just the second orders.
This is just the sum of squares.

Then there's a correction that they have to make,
to make it work nicely, which is well justified in the notes.
And I think I'm not going to go into it right now, because--

Minus-- 1 minus B1, yeah, raised to the t power.

They observe that if you initialize
these two guys at zero--
the mean and the variance--
then especially at the beginning, they're too small.
They're kind of underestimates of what they should be.
And these maneuvers here will correct for that.
This is a power.
This is not an index.
This is raised to power t.

OK.
So we're computing the mean, and we're computing
the variance of the gradients.
And then the update rule, finally,
is going to be a kind of combo job, so Wtj minus theta.
OK, and then we have this thing that we had before.
V hat tj plus an epsilon to keep us out
of trouble, times m hat tj.
So this is the momentum rule, but we
have this correction for the gradient
for kind of like the curvature.
So that's Adam.
So this paper gets about a bazillion citations,
because at this moment, basically anytime anybody
writes a paper that says, I trained a neural network,
they say, I trained a neural network using out
Adam sight, because this is what everybody does.
So those guys are famous--
Kingma and Ba.
OK.


### Lecture: Neural networks - brief review of layers and backprop

LESLIE KAELBLING: The other topic for today
is regularization.
So far, we've been talking about RJ
as just being a sum of lost terms.
I started it that way at the very beginning.
But certainly we spent a lot of time
before we got into neural networks
talking about the idea of regularization,
about using different basis functions,
or penalizing the magnitude of the weight vector,
or all these other things to try to keep
the thing from overfitting.

So what about this?
What about this regularization thing?

So, there is a huge conceptual, and theoretical, and practical
tension at the moment in the practice of neural networks
because people making huge neural networks
with hundreds of thousands of weights,
and by all of our intuition and theoretical understanding,
huge neural networks with hundreds of thousands
of weights ought to overfit badly.
That's like having a basis that's
much too big, way too many weights to do the job that you
need to do.
But in practice, it's not nearly as much of a problem
as it ought to be in some sense.
So it's a funny thing.
So it's still useful to do regularization
in neural networks, but it's not nearly as big
a deal as we would have expected theoretically,
or as we understood from messing around
with other kinds of machine learning setups.
One reason could be actually related
to this little dip, which is overfitting tends to be finding
yourself down in very small canyons that
give you a really good local optimum on your training data.
So, overfitting often feels like going down a valley like this.
That's why we couldn't get gradient descent actually
to hit all those points in the lab where we did regression,
and we tried to--
we saw if we use a ninth-order polynomial,
we computed the solution analytically,
it went right there.
But when we tried to do gradient descent,
it wouldn't get that horrible crazy solution.
That's because that horrible creative solution
was like this.
It's like it's down a very narrow channel.
It's very hard to get the optimizer to go in there.
So, it's interesting that because gradient descent
is sometimes hard to get to actually find
a tricky little optimal like this, that's often fine with us
because often those optima that are hard to find
are solutions we didn't want anyway because they're probably
overfitting our data.
So there's a little--
there's an interesting relationship between what
is gradient descent--
what kinds of answers does gradient descent give you,
and are they the ones--
the kind of answers that you'd like even if they aren't
actually optima of the objective that you're trying to
[INAUDIBLE]?
So that's a subtle point.
So that's all to say that regularization is still
important in neural networks, but possibly not as critical
as we had thought it might be.
Nonetheless, there are ways to do regularization.
And sometimes they matter.
So, there's a set of methods that are actually--
this is a lovely result from quite a while ago
in neural networks-- that are actually provably basically
the same thing.
And they're provably-- they're closely related
to the kind of regularization that we did
when we did ridge regression.
So we penalized the squared norm of the weight vector.
So one thing you can do is do that directly.
So we'll talk about something called weight decay.

This is a nice name.
And I think that what happened is a bunch of these methods
people cooked up, and then they turn out
to be the same sort of thing.
But so if we make our loss function be--
I mean, not our loss function, but our objective now
be the sum of the losses--

loss of h of x--

and again, imagine that I just take all the way
to the neural network and make it a giant vector,
this is just the sum of the squares of the weight values.
It doesn't really matter whether it's a big weight,
we think of it as a bunch of matrices or what.
It's just the sum of the squares of the values of the weights.
So we could say, well, I don't know,
that was a nice idea in ridge regression.
Maybe that same idea would be good for us here now.
So if you take the gradient now with respect to W of that,
and you write down the update rule,
and you reorganize terms--
the derivation's in the notes.
I'm not going to write the derivation in the board--
if you reorganize the terms, you get
a rule that looks like this.

So if you ignore this, that's totally our normal batch
gradient update rule.
And then if you look at this, this is kind of interesting.
So under the assumption that lambda and eta are small enough
so that this thing stays positive--
if it's not positive, all terribleness happens.
So it better stay positive--
then what this says is that every time I update my weights,
I'm just going to shrink them a little.
I would say, hey, wait, whatever you were,
just be a little smaller.
And then I'll add the gradient to it.
So that's weight decay.
So it has a charming name because what it says
is, OK, I'm going to decay the weights a little
and then apply the gradient update.
But this is just what you get if you take the--
if you differentiate that.

So that's [INAUDIBLE] weight decay,
very easy to implement, trivial thing.
And that's a way to just keep your weights
from getting enormous.
And that's sometimes the useful regularizer.


### Lecture: Neural networks - brief review of layers and backprop


LESLIE KAELBLING: Probably the most common thing
that people do right now in neural network practice--
see, you can-- actually, this is easy to do.
Another thing is early stopping, which
is also pretty easy to do.

And early stopping hinges on the idea
that almost always as you train--
so this is now epochs, so this is as you
train the neural network--
sometimes we make these pictures with a mound
of training data on the x-axis.
But here, we're not thinking about the amount
of training data.
We're thinking about iterations through the data set.
And here we plot some kind of a loss.
Then generally speaking, the loss--
I mean, it might wiggle around a little bit.
But mostly the loss on the training set goes down.
We're doing gradient descent on the loss on the training set.
So generally, that should go down
if we're managing our eta as well.

It might bounce because we're doing matches
or because the eta's not perfectly managed.
So that seems fine.
But that doesn't tell you too much.
Remember that in the end, what we're interested in doing
is optimizing our ability to make predictions on data
that we haven't seen yet.
So at the same time, we could keep out a validation set, so
some data that we're not training on.
And after every epoch, we could take the current set of weights
and see how well they do making predictions
on a validation set.
And we can plot that.
And generally, it's almost always, always worse.

But it gets smaller for a while.
And then-- and it wiggles, too.
Everything wiggles.
But then it'll often start to go back up again.

So this would be on validation data.

And so what this is telling us is, though,
even though we're keeping going here, it's a waste of time.
But it's really a waste of time.
Not only is our answer not really getting that much better
on our training data, which we don't really care about,
it's getting worse on non-training data.
So, if I had run this experiment,
if I had trained this thing, and I got some different weight
vector-- at each epoch, I had a different set
of weights, which weights would I deliver to my customer?

The ones that minimize the validation error.
So finally deliver some weights to somebody,
I would deliver those weights.

So early stopping basically says,
really, you're going to eventually use
a validation set to decide which weights to give somebody,
so you might as well be watching as you go.
And by the time you're pretty sure that it's going up,
you could just quit.

And there's a nice theoretical story
that relates this actually to that, which is cool.
Or I can tell you it's 6-8, 6-7 if you want
to study that in more detail.
So that's one method.
I'm going to talk now about another family of methods
which are also closely related.
I mean, it's kind of interesting.
So one more strategy here is perturbing the training data.

And that means really that instead
of using our x's from our file of training data as the inputs
when we do stochastic gradient descent
or when we sum up the gradient, we're
going to use a an x hat instead of our x.
And it's going to be x--
this is just a particular example, x^i.
Let me do this, x hat i.
It's going to be x^i plus some noise.
I'll call it noise.
And this noise variable is going to be distributed
like a Gaussian with mean 0 and very small covariance
or a small diagonal variance usually.
If you're not used to continuous probability,
don't worry, small random numbers.
So what we do we take our data, and we just--
every time we pluck a data point out of our data set,
instead of using the actual data point that's in there,
we mess with it a little bit.
The intuition here is if you're messing with your data
points a little bit every time, you
can't get a solution that's to attached to the data points
because you don't even see the same data
points over and over again.
You're messing with them a little bit.
And there's a lovely proof that doing this basically gives you
the same effect as doing that, which in some ways
gives you the same effect as doing this.
So messing with your data is another way
to do regularization.

So in modern neural network literature,
if you read about regularization,
people don't really talk about this perturbing the training
data in this way.
But they talk about two other perturbations strategies,
which I'll talk about here just so that you know what they are.
And one of them I had in the notes,
and we have a homework assignment about it.
And if you read neural network stuff,
you'll hear people talking about dropout.
But, of course, one of the TAs said, ah,
dropout is like so three years ago.
So now there's a newer thing.
So I'll talk about the nearer thing, too.
But this stuff, it's evolving quickly.
So dropout has to some degree dropped out of what's going on.
But it's an interesting idea.
And it has an interesting effect.

So let's think about a neural network--
I'm actually going to draw a neural network picture.
We have our inputs coming in here.
And we have another layer.
And these guys are connected.
Everybody is connected to everybody.
Let's say this is a fully connected layer.
Next time we'll talk about not fully connected layers.
And maybe these guys are connected again
to some more things, all these connections.
So there's your neural network.
Dropout says with probability p, for each unit--

so now again, I'm just going to let--
think of these activations as just being index by
one index j.
So there would be, in this case, nine activations--
for each unit j set activation j to be 0 on each--
on each forward/backward pass.
Well, you only have to do it on the forward pass.
So, sorry about this.
So remember, the way we do a weight
update in a neural network is that we put in the x over here,
we do a forward pass to compute the output here,
and then we do the backward pass to compute the gradients,
and then we update the weights.
That's how that goes.
If we take one of these a's, so the output of this guy,
and say, oh, yeah, you were supposed to be minus 3.4,
but too bad, I'm setting you to 0.

You can do that.
And that will mean that this unit doesn't affect anybody
in the next layer.
And because it doesn't affect anybody
in the next layer, when we do the gradient-- the back prop,
the weights that affected this guy's output
will also not get updated because they're
obviously not having any effect on the output.

So that's dropout.
It's really easy to implement.
And it's kind of interesting.
Again, one way to think about what's going on
is that all these units in this layer
are a team that have to take the input
and transform it into some other representation
that the later layers are going to use to do a job.
But they can't specialize.
By doing dropout, we're saying, OK, yeah, you guys are a team.
And you have to encode the input in a way
that the rest of the network can work on it.
But you're going to have to come up
with a way of encoding the input that's
redundant because I'm going to kill p percent of you
each time.
This is like-- the team--
I don't know a good metaphor.
They all get sick, or I don't know.
But anyway, we're always missing a few team members
at random every day.
You never know which ones.
So if you had to get 10 people to do a job,
but you were only going to get eight of them on any given day,
but you didn't know which eight, you
would train them in some interesting way
to be able to cover for each other.
And they wouldn't be able to over specialize.
And that notion of not being all over specialize
keeps this network from over-fitting.

So that's dropout.

Cool?
Again, it definitely improves performance
and was the coolest thing for a while.
It's not the coolest thing anymore.
The coolest thing I'll tell you now
about is cool at least in part because it solves
two problems, not just one.
So that's always nice.


### Lecture: Neural networks - brief review of layers and backprop


LESLIE KAELBLING: This last thing
is called batch normalization, and I
think it was originally developed
to address a problem which we haven't discussed yet.
And then as a side effect, it works as regularizing,
so that was kind of cool.
So the problem that it was meant to address is covariate shift.

And it doesn't address the problem completely,
but it addresses it somewhat.
OK, so good-- so what the heck is covariate shift?
So a neural network, or a layer of a neural network,
or a unit in a neural network--
its job is to learn a good h that maps x's.
We mess with the parameters so that we learn a good h that
takes x's and gives out y's.
That's what it's supposed to do.
And you could imagine, secretly, there is a true h under there,
and the h is the same.

And it's just a relationship between x's and y's.
So we're going to still assume that there is a static,
fixed relationship between the x's and the y's.
OK, that's cool.
But what happens in any given training set
is that the x's come from some distribution.

So maybe were measuring the relationship
between people's height and their shoe size,
and maybe we think that relationship is fixed.

But we get a training set from here,
and we get people of certain distribution of heights.
And then, we go somewhere else and we get a training set.
And we get people with a different distribution
of heights.

Maybe they have the same relationship
between height and shoe size, but we have
people with different heights.
And the distribution over the x's will
affect what your neural network does, because it wants to--
usually, the way we train it is that we
want to minimize the loss on the training data
with some regularization.
But it means it's working hard to get cases
like the cases you're trained it on right.
So if you only trained it on people
from a certain population, it will tend to just do well
on that population.
So the problem of covariate shift
is the problem of this distribution changing.

It's not the problem of the true function changing.
It's the problem of the distribution
of the x values changing.
So this can happen a lot.
If you hire a machine-learning company,
and they train something up for you or your company
to make predictions about what people will buy,
but then the next season something's
different about the people, than you might have this problem.
So there's kind of external reasons
why covariate shift might happen.
But in neural networks, there's internal reasons why
covariate shifts might happen.
And I don't know, I guess I'll just re-draw my picture.
It's my favorite picture, after all.
Oh, good, I'll draw it with a gap.
OK, so here's w and f.

I'll draw one more.

So here are my layers of my neural network.
I guess I usually make these f's small.

Here's the layers of my neural network.
And this guy is getting data from the data set.
And if we assume that we just have the one data set,
then this guy is always getting data from the same data set.
And he-- this little first layer--
never suffers the covariate shift problem,
because he's always getting training data from the input.
OK, good.
Now, imagine that this is just a little machine--
think of this layer as its own dot.
If we fix the weights in the first layer,
then he would be getting data from a constant distribution
of some sort because he's going to get his input, the A1.

So if we fix those w's, the A1's will be just a transformed
version of the x's.
They'll be the same all the time.
And this person-- this unit, this layer--
will face the same learning problem all the time.
But that's not how it works, right?
The way it works is that we're updating
these w's all the time.
And so these units-- not only do they
have to adjust their weights just
to do a good job of the job that we gave them.
The job they have is changing all of the time, right?
This guy's job-- even if he was the perfect f
for those weights, the best f he could be for those weights,
these weights are going to change.
And because these weights are going to change,
the distribution of inputs on this guy are going to change.
And so he has to be different.
So all of the other layers in the network,
except the first one, face the problem of shifting sands.
They're supposed to be solving a different problem every time.
So that's extra tricky.
That's another reason why these things
are kind of hard to manage.
[SIGHS] OK, so what is a way to handle that?
So one thing is that the magnitudes of these values
might change.
All sorts of stuff might change.
So we're going to interpose a box, here and here, which I'll
call BN for Batch Normalize.

And what the batch normalize box does--
let's imagine that we're training in mini batches.
It doesn't make sense if you're doing stochastic gradient.
Is that true?
It is true.

Based on our batch, we're going to take all of our data points.
We're going to make an x hat.

Yeah, we'll just call it x hat--
or a hat, I guess.
Maybe a hat is better, because it's the A's that
are coming into the next layer.
a hat li is going to be A li minus the mean
on the mini batch over the variance on the mini batch.
And I guess we have to look at component j--
[SIGHS] component j.
OK, what the heck is all of this notation?
Let me see if I can explain.
At some layer in the neural network, like this--

so at some layer in the neural network,
we path the mini batch through.
And we got these A's we got these outputs.
What we're going to do with those guys is do our standard--
this should look familiar.
We're going to compute the mean over the mini batch
of that coordinate, and the standard deviation
of the mini batch over that coordinate.
And we're going to standardize the inputs to this layer,
just in the mini batch.

So this is a way of at least managing the fact
that, because the weights of the layers that come before me are
changing, I might naturally start getting inputs that
are getting bigger and bigger and bigger over time,
or smaller and smaller and smaller over time.
Here, I'm trying to just hold that tide back and say, yeah,
sure, you can scale if you want to.
But I'm going to re-scale so everything
is in my comfortable range and the range
of the inputs I have to deal with doesn't change.
The range of the inputs I have to deal with
is still kind of centered at zero
and going mostly in plus to minus one,
or something like that.
So this is a transformation I do on a whole batch of data.
I do it at each layer.
And it keeps the scales of the inputs
at each layer roughly similar.
OK, so good.
So I introduced this in the section on regularization.
I said it's here to address covariate shift.
The reason it helps-- but it works as a regularizer, too--
yea, two for one.
The reason is, you can think of it
as making small perturbations to your data.
Depending on which cohort comes in your mini batch--
each mini batch is randomly selected each time--
we're going to subtract off a slightly different mean.
We're going to scale by a slightly different
standard deviation.
So that's making perturbations to your data,


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/9c36c444e5df10eef7ce4d052e4a2ed1/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Making_NN_s_Work.pdf
- https://openlearninglibrary.mit.edu/assets/courseware/v1/9c36c444e5df10eef7ce4d052e4a2ed1/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Neural_Networks.pdf
