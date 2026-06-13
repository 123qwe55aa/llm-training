In this homework, we will explore training methods for neural networks. A code and data folder that will be useful for doing this homework can be found here.

Note that problem 2 (OPTIONAL) has you extend your homework 7 framework for building neural networks. The code for this problem is included in the file code_for_hw8_oop.py and you have to copy in your implementation from homework 7 to complement it.

For the rest of the problems (3-5), we will start using "state of the art" tools (Keras and Tensorflow) for building neural network models, as they are computationally efficient and allow us to easily and rapidly build models of varying complexity. The code for these problems is in the file code_for_hw8_keras.py.

Alternatively, you can use the colab notebook which contains the code for all the problems in the same notebook. (If clicking on the link opens a grey window, try opening it in a new tab.)

1) Batch Normalization

Batch normalization is discussed in the lecture notes as a good way of doing regularization in large networks. The authors of the paper that proposed the batch normalization technique argued that it mitigates the effects of the internal covariate shift issue, which is also discussed in the notes. We encourage you to read that section of the notes before attempting the problem.

Let notation $W^l$ indicate the matrix of weights at the $l^{th}$ layer of our neural network. The $l^{th}$ layer takes in $n^{l-1}$ activations from the previous layer and produces $n^l$ outputs. In this problem, we will always use mini-batch gradient descent during training, with a mini-batch size of $K$.

We now investigate the internal covariate shift problem and how batch normalization generally helps train neural network models.

1A) Based on your reading of the lecture notes, choose the true statements about the internal covariate shift issue and the concept of batch normalization:

Mark all that are True.
(1) The internal covariate shift problem can arise at the $l^{th}$ layer of a neural network, due to the changing weights in the $(l+1)^{th}$ layer.
(2) The first layer of neurons experiences the the worst internal covariate shift out of all layers, because backpropagation occurs all the way from the final to the first layer.
(3) Batch normalization can help prevent outputs of neurons from becoming extremely large or extremely small
(4) Batch normalization can be particularly helpful when we are using sigmoid units in the hidden layers.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

1B) We now investigate techniques we might use to better train our neural networks, including potentially getting rid of the internal covariate shift issue.

Mark all that are True.
Technique 1. Internal covariate shift is no longer a problem if we optimize the weights of our neural network layer by layer, starting from the input layer, and going all the way to the output layer. When updating weights of a layer, all other weights are kept fixed.
Technique 2. If technique 1 did not take a lot of time, we would have always used it, because it eliminates internal covariate shift and therefore guarantees that our network will perform well.
Technique 3. It would help in training a neural network if we took all $n^{l-1}$ x $K$ inputs going to to the $l^{th}$ layer, find the mean and variance of this whole set of scalars and normalize them. We could do the same for all layers of the network.
Technique 4. It would help in training a neural network if we took all $K$ inputs going to the $l^{th}$ layer through one of the $n^{l-1}$ input connections, find the mean and variance of this set of scalars and normalize them. We could then do the same for all input connections of all layers of the network.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

For the following questions, we will present some results obtained using neural networks built from our modular framework that we implemented in homework 7. We will analyze those results to draw conclusions later. Any one of our models is trained and tested on the MNIST training and validation data sets. MNIST is a very well-studied data set of 28x28 images of isolated digits (0-9), each pixel value in the range 0-255. There are 60,000 training images and 10,000 validation images. Our networks will take an image as input and try to predict the corresponding digit. Our results are obtained using only images of digits 0 and 1. Later in problem 5, we will look at the complete MNIST problem and we will try to achieve close to state-of-the-art performance on the whole data set.

We use the following core neural network in this problem (using our implementation of modules from homework 7):

nn_core = Sequential([Linear(A, 32), ReLU(), Linear(32, 4), ReLU(), Linear(4,B), ReLU(), SoftMax()], NLL())

1C) Choose the values of A and B that yield a neural network model with an appropriate number of inputs and outputs for the task at hand, based on the description given above.

Enter a python list of 2 values: [A, B]:

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

In the table below, we summarize the results of some experiments which we already ran for you. We specify our models used, which are nn_core instances, or slight modifications of those, which include batch normalization layers BatchNorm:

nn_bn_1 = Sequential([Linear(A, 32), BatchNorm(32), ReLU(), Linear(32, 4), BatchNorm(4), ReLU(), Linear(4,B), ReLU(), SoftMax()], NLL())

nn_bn_2 = Sequential([Linear(A, 32), ReLU(), BatchNorm(32), Linear(32, 4), ReLU(), BatchNorm(4), Linear(4,B), ReLU(), SoftMax()], NLL())

Note that in the notation above, BatchNorm(k) indicates a batch normalization layer with $k$ input connections into the BatchNorm module, and $k$ output connections to the next module in the Sequential object. If you are interested in implementing batch normalization for the modular framework that we have been developing since homework 7, you can attempt problem 2 which is optional.

We also specify our chosen learning rates in the table. All experiments were done using mini-batch gradient descent during training and the number of number of iterations of backpropagation used was $400$. The last column gives the accuracies obtained by our models at test time, when evaluated on the MNIST validation set. Based on our discussion in 1A), overflow/undeflow are potential issues we may face if the weights or outputs of our neural networks become too big/small inside numpy arrays. This would prevent us from getting any test accuracy and in those cases, we simply report '--' as the obtained accuracy.

#
Model used
Learning rate
Test accuracy

1
nn_core
1e-8
0.593

2
nn_bn_1
1e-8
0.401

3
nn_bn_2
1e-8
0.423

4
nn_core
1e-7
0.623

5
nn_bn_1
1e-7
0.415

6
nn_bn_2
1e-7
0.434

7
nn_core
1e-6
0.420

8
nn_bn_1
1e-6
0.502

9
nn_bn_2
1e-6
0.531

10
nn_core
1e-5
--

11
nn_bn_1
1e-5
0.651

12
nn_bn_2
1e-5
0.784

13
nn_core
1e-4
--

14
nn_bn_1
1e-4
--

15
nn_bn_2
1e-4
0.997

16
nn_core
1e-3
--

17
nn_bn_1
1e-3
--

18
nn_bn_2
1e-3
--

1D) Choose conclusions that you are able to draw from the experiment results above. Note that these conclusions are specific to this problem instance are may not be generally true.

Mark all that are supported by your findings.
(1) BatchNorm layers placed after the ReLU non-linearities allow better accuracies to emerge, versus placing them before the ReLU non-linearities.
(2) Based on the results obtained when using a learning rate of 1e-8, we can affirm that BatchNorm layers are useless.
(3) BatchNorm layers always seem to improve test performance.
(4) BatchNorm allows these neural networks to perform much better at lower learning rates.
(5) BatchNorm stabilizes training at higher learning rates by preventing some of the overflow/underflow issues that we encounter when not using it.
(6) BatchNorm helps our models reach the highest achievable accuracies much faster than when not using BatchNorm.
(7) Thanks to its stabilizing properties, BatchNorm allows any learning rate to be used without any problem.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2) Implementing Mini-batch Gradient Descent and Batch Normalization (OPTIONAL)

Show optional content
Last week we implemented a framework for building neural networks from scratch. We trained our models using stochastic gradient descent. In this problem, we explore how we can implement batch normalization as a module BatchNorm in our framework. It is the same module which you analyzed in problem 1.

Key to the concept of batch normalization is the doing gradient descent on batches of data. So we instead of using last week's stochastic gradient descent, we will first implement the mini-batch gradient descent method mini_gd, which is a hybrid between stochastic gradient descent and batch gradient descent. The lecture notes on optimizing neural network parameters are helpful for this part.

In mini-batch gradient descent, for a mini-batch of size $K$, we select $K$ distinct data points uniformly at random from the data set and update the network weights based only on their contributions to the gradient:

$$
W := W - \eta\sum_{i=1}^K \nabla_W \mathcal{L}(h(x^{(i)}; W), y^{(i)})\;\;.
$$

Our mini-batch method mini_gd will be implemented within the Sequential python class (see homework 7 problem 2) and will take the following as inputs:

* X: a standard data array (d by n)

* y: a standard labels row vector (1 by n)

* iters: the number of updates to perform on weights $W$

* lrate: the learning rate used

* K: the mini-batch size to be used

One call of mini_gd should call Sequential.backward for back-propagation and Sequential.step for updating the weights, for a total of iters times, using lrate as the learning rate. As in our implementation of sgd from homework 7, we compute the predicted output for a mini-batch of data with the Sequential.forward method. We compute the loss between our predictions and the true labels using the assigned Sequential.loss method. (Note that in homework 7, Sequential.step was called Sequential.sgd_step. While the functionality of the step function is the same, it has been renamed for convenience. The same is true for the module.step function of each module we implemented, where applicable.)

For picking $K$ unique data points at random from our large data-set for each mini-batch, we will implement the following strategy: we will first shuffle our data points X (and associated labels y). Then, we get $\frac{n}{k}$ (rounded down to the nearest integer) different mini-batches by grouping each $K$ consecutive points from this shuffled array. If we end up iterating over all the points but need more mini-batches, we will repeat the shuffling and the batching process.

2A)You need to fill in the missing code below. We have implemented the shuffling of indices and have provided you with the outer and inner loops.

1234567891011121314151617181920212223242526272829import math as m class Sequential: def __init__(self, modules, loss): self.modules = modules self.loss = loss def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10): D, N = X.shape np.random.seed(0) num_updates = 0 indices = np.arange(N) while num_updates < iters: np.random.shuffle(indices) X = None # Your code Y = None # Your code for j in range(m.floor(N/K)): if num_updates >= iters: break # Implement the main part of mini_gd here # Your code num_updates += 1 def forward(self, Xt): for m in self.modules: Xt = m.forward(Xt)XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Run Code
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2B) We are now ready to implement batch normalization into our neural network framework! Our module BatchNorm will sit between consecutive layers of neurons, such as the $l^{th}$ and $(l+1)^{th}$ layers, acting as a "corrector" which allows $W^l$ to change freely, producing outputs $Z^l$, but then the module corrects the internal covariate shift induced in the signals before they reach the $(l+1)^{th}$ layer, converting $Z^l$ to $\widehat{Z}^l$.

The following is a summmary what is described in the lecture notes, and it should guide your implementation of the module.

Any normalization between the $l^{th}$ and $(l+1)^{th}$ layers is done separately for each of the $n^l$ input connections leading to the $(l+1)^{th}$ layer. We handle a mini-batch of data of size $K$, and $Z^l$ is $n^l \times K$, and the output $\widehat{Z}^l$ is of the same shape.

We first compute $n^l$ batchwise means and
standard deviations. Let $\mu^l$ be the $n^l \times 1$ vector (self.mus in the code) where

$$
\mu^l_i = \frac{1}{K} \sum_{j = 1}^K Z^l_{ij}\;\;,
$$

and let $\sigma^l$ be the $n^l \times 1$ vector of standard deviations where

$$
\sigma^l_i = \sqrt{\frac{1}{K} \sum_{j = 1}^K (Z^l_{ij} - \mu_i)^2}\;\;.
$$

Note that in the code self.vars is the variance, or element-wise square of $\sigma^l.$

The normalized data self.norm is now the matrix $\overline{Z}$, where

$$
\overline{Z}^l_{ij} = \frac{Z^l_{ij} - \mu^l_i}{\sigma^l_i + \epsilon}\;\;,
$$

and where $\epsilon$ is a very small constant to guard against division by
zero.

We define weights $G^l$ (self.G) and $B^l$ (self.B), each being an $n^l \times 1$ vector, which we use to to shift and scale the outputs:

$$
\widehat{Z}^l_{ij} = G^l_i \overline{Z}^l_{ij} + B^l_i\;\;.
$$

The outputs are finally ready to be passed to the $(l+1)^{th}$ layer.

A slight warning (that we will not worry about here) about BatchNorm is that during the test phase, if the test mini-batch size is too small (imagine we are deploying a neural network that deals with live video frames), then the lack of samples would cause the freshly-calculated $\mu^l$ and $\sigma^l$ to be far off from their true values that the module's parameters $G^l$ and $B^l$ were trained to be compatible with. To fix that, people usually compute a running average of $\mu^l$ and $\sigma^l$ during training, to be used at test time. We will assume our test mini-batches are large enough.

In this problem we only implement the BatchNorm.forward and BatchNorm.step methods. We provide you with the implementation for BatchNorm.backward and the lecture notes contain the details of the derivations. You will need to fill in the missing code below.

123456789101112131415161718192021222324252627282930313233343536373839class BatchNorm(Module): def __init__(self, m): np.random.seed(0) self.eps = 1e-20 self.m = m # number of input channels # Init learned shifts and scaling factors self.B = np.zeros([self.m, 1]) self.G = np.random.normal(0, 1.0 * self.m ** (-.5), [self.m, 1]) # Works on m x b matrices of m input channels and b different inputs def forward(self, A):# A is m x K: m input channels and mini-batch size K # Store last inputs and K for next backward() call self.A = A self.K = A.shape[1] self.mus = None # Your Code self.vars = None # Your Code # Normalize inputs using their mean and standard deviation self.norm = None # Your Code # Return scaled and shifted versions of self.norm return None # Your Code def backward(self, dLdZ): # Re-usable constants std_inv = 1/np.sqrt(self.vars+self.eps) A_min_mu = self.A-self.mus dLdnorm = dLdZ * self.G dLdVar = np.sum(dLdnorm * A_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True) dLdMu = np.sum(dLdnorm*(-std_inv), axis=1, keepdims=True) + dLdVar * (-2/self.K) * np.sum(A_min_mu, axis=1, keepdims=True) dLdX = (dLdnorm * std_inv) + (dLdVar * (2/self.K) * A_min_mu) + (dLdMu/self.K) self.dLdB = np.sum(dLdZ, axis=1, keepdims=True) self.dLdG = np.sum(dLdZ * self.norm, axis=1, keepdims=True) return dLdXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Run Code
Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

3) Neural Network Packages

In the following problems, we will get experience with using a well-known neural network package for classification.

Keras and Tensorflow

We will be using Keras to build neural network models. Keras uses Tensorflow as the backend (to actually train the models). You will need to either (a) install both of these on your machine or (b) use the colab notebook, which comes with pre-installed tensorflow and keras, or (c) use the versions we've installed on Athena. All of these are Python-based installations and we will be using Python3 (see instructions on the Resources/Software tab).

You do not need to understand Tensorflow in detail. You should however get used to Keras. You should read some of the documentation, especially the examples. Given the lectures, homeworks and labs so far, you should be able to understand most of what's in the basic documentation, although there may be some different names for some of the concepts.

Make sure that you read: Getting started: 30 seconds to Keras on the Keras Home page and Guide to the Sequential Model and pay careful attention to the examples.

You should read some details in the following pages:

* Core Layers, read about Dense, Dropout and Flatten layers

* Convolutional Layers, read about Conv1D and Conv2D

* Pooling Layers, read about MaxPooling2D

* Optimizers read about Adam; we'll use Adam throughout this assignment, this is an adaptive step-size method that is a bit more sophisticated than Adagrad.

* Losses, read about categorical_crossentropy, this is what we call NLL loss

* Datasets, we will be using the MNIST dataset.

Basics

Note that Keras assumes that each data point is a ROW vector and that a data set X with N points with D features is an NxD matrix. The code we have provided handles this, but you should be aware of this if you want to use the code in the future.

We will be working with two main data sets:

* A set of relatively simple datasets in 2D

* A standard dataset of images of hand-drawn digits (the MNIST data set).

These are all classification problems. Neural networks can be used for regression as well but we'll skip looking at that.

In this assignment we will focus on exploring the choice of network architecture, numbers and types of units and layers. In particular, we will explore three classes of network architectures:

* No hidden units - this is essentially a linear classifier.

* Fully connected (FC) layer(s) - the standard approach to problems involving heterogeneous data.

* Convolutional layer(s) followed by FC layer(s) - a common approach for image data.

If you look through the Keras documentation you'll see that there are an enormous number of choices that can be made to define a model. We're only going to look at a few, in particular:

* number and type of layers

* number and type of units per layer

* number of passes over the data (epochs)

Here are a few choices that we are not going to explore; we'll keep these choices fixed throughout:

* We'll use 'ReLU' hidden units

* We'll use softmax activations on the output and categorical_crossentropy as the loss.

* We'll use Adam as the optimizer. This is a version of SGD that adapts the step size (learning rate) somewhat like Adagrad but a bit more sophisticated. We will use Adam's default parameters.

* For (mini)batch_size, we'll use 32 on MNIST and 1 otherwise.

We have provided you a couple of basic functions to experiment with; you mostly have to define layers. We strongly recommend that you do your experiments using the google colab file to keep track of your results. Alternatively, write scripts that can be run to produce your results and save them to a file. Have your script write out the settings for the experiment so that you can remember what they are when you go back to look at them.

For the 2D datasets, we have provided the following function:

run_keras_2d(data_name, layers, epochs, split=0.25, display=True, trials=5)

where:

* data_name is a string, such as '1', '2', etc.

* layers is a list of Keras layer definitions for a Sequential model, e.g.

[Dense(input_dim=2, units=10, activation='relu'), Dense(units=2, activation='softmax')]

* epochs is an integer indicating how many times to go through the data in training

* split is a fraction of the training data to use for validation if a validation set is not defined

* display whether to display result plots

* verbose whether to print loss and accuracy (percent correctly labeled) each epoch

* trials is an integer indicating how many times to perform the training and testing

2D Data

The two-class datasets have data_names: '1','2','3','4'. Target accuracies (percent correct) on the validation set are (99%, 90.5%, 96%, 94%).

In this problem, try the following 5 architectures, specified by the number of units in the hidden layers:

1: (0), 2: (10), 3: (100), 4: (10, 10), 5: (100, 100))

You may find the archs function in the code file to be helpful here.
Some of these questions ask for the "simplest" architecture; the list above is ordered starting with the simplest.

Use 10 epochs of training for each architecture and re-start the training and testing 5 times (i.e., use trials=5), look at the average accuracy on the validation set (reported as "Avg. validation accuracy" at the end of the run). But, notice the variation in scores across each run.

3A) For each two-class 2D dataset, what is the simplest network architecture that can reach the target accuracy on the validation set?

Enter a list of four indices from 0 to 5, indicating the simplest architecture or 0 if none works. The architectures can be 1-5.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

3B) What is the simplest (single) network architecture that can be used to classify all the of the two-class 2D data sets within 1% of the target validation accuracy.

Enter one of the architecture indices or 0 if none works:

Submit
View Answer Ask for Help

<img src="data:image/gif;base64,R0lGODlhEAAQAPIGAMLCwkJCQgAAAGJiYoKCgpKSkv///wAAACH/C05FVFNDQVBFMi4wAwEAAAAh/hpDcmVhdGVkIHdpdGggYWpheGxvYWQuaW5mbwAh+QQJCgAGACwAAAAAEAAQAAADMmi63P4wyklrAyEAGoQInAdOmGYBw7AxwLoMGcG2rkHEQFHQLTsQOd2mB9ERCpTWzpEAACH5BAkKAAYALAAAAgAKAA4AAAMraAYRoNAEIUJUs97VHgTD4EVDQ2xEM2wgMV5AUbyKLKNEvoxA3P8sYNCQAAAh+QQJCgAGACwAAAAACgAOAAADLWi6EAFrBSGCAmQ0as1wROFABuEM0TUQ5FUU7fK+aRkWNYD[Truncated]