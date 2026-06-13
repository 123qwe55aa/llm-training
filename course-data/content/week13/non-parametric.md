# Decision Trees and Nearest Neighbors

> Week 13: Decision Trees and Nearest Neighbors · MIT 6.036 courseware archive

## Notes – Chapter 14: Non-parametric methods

Notes – Chapter 14: Non-parametric methods
You can sequence through the Non-parametric methods lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 14: Non-parametric methods
notes as a PDF file.

## Lecture: Introduction to non-parametric models

Lecture: Introduction to non-parametric models
Lecture: Introduction to non-parametric models
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Introduction to non-parametric methods

Introduction to non-parametric methods
We will continue to broaden the class of models that we can fit to our data. Neural networks have adaptable complexity, in the sense that we can try different structural models and use cross validation to find one that works well on our data.
We now turn to models that automatically adapt their complexity to the training data. The name
non-parametric methods
is misleading: it is really a class of methods that does not have a fixed parameterization in advance. Some non-parametric models, such as decision trees, which we might call
semi-parametric methods
, can be seen as dynamically constructing something that ends up looking like a more traditional parametric model, but where the actual training data affects exactly what the form of the model will be. Other non-parametric methods, such as nearest-neighbor, rely directly on the data to make predictions and do not compute a model that summarizes the data.
The semi-parametric methods tend to have the form of a composition of simple models. We'll look at:
Tree models
: partition the input space and use different simple predictions on different regions of the space; this increases the hypothesis space.
Additive models
: train several different classifiers on the whole space and average the answers; this decreases the estimation error.
Boosting
is a way to construct an additive model that decreases both estimation and structural error, but we won't address it in this class.
Why are we studying these methods, in the heyday of neural networks?
They are fast to implement and have few or no hyper-parameters to tune.
They often work as well or better than more complicated methods.
Both can be easier to explain to a human user: decision-trees are fairly directly human-interpretable and nearest neighbor methods can justify their decision to some extend by showing a few training examples that the prediction was based on.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:59 PM (revision 4f166135)

## Lecture: Decision trees

Lecture: Decision trees
Lecture: Decision trees
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Regression trees - problem statement

Lecture: Regression trees - problem statement
Lecture: Regression trees - problem statement
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Building a tree - greedy algorithm

Lecture: Building a tree - greedy algorithm
Lecture: Building a tree - greedy algorithm
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Building a tree - minimum error splits

Lecture: Building a tree - minimum error splits
Lecture: Building a tree - minimum error splits
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Building a tree - pruning

Lecture: Building a tree - pruning
Lecture: Building a tree - pruning
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Classification trees

Lecture: Classification trees
Lecture: Classification trees
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Classification trees - impurity measures - gini index

Lecture: Classification trees - impurity measures - gini index
Lecture: Classification trees - impurity measures - gini index
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Classification trees - impurity measures - entropy

Lecture: Classification trees - impurity measures - entropy
Lecture: Classification trees - impurity measures - entropy
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Decision trees - the good and the bad

Lecture: Decision trees - the good and the bad
Lecture: Decision trees - the good and the bad
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Trees

Trees
The idea here is that we would like to find a partition of the input space and then fit very simple models to predict the output in each piece. The partition is described using a (typically binary) “decision tree," which recursively splits the space.
These methods differ by:
The class of possible ways to split the space at each node; these are generally linear splits, either aligned with the axes of the space, or sometimes more general classifiers.
The class of predictors within the partitions; these are often simply constants, but may be more general classification or regression models.
The way in which we control the complexity of the hypothesis: it would be within the capacity of these methods to have a separate partition for each individual training example.
The algorithm for making the partitions and fitting the models.
The primary advantage of tree models is that they are easily interpretable by humans. This is important in application domains, such as medicine, where there are human experts who often ultimately make critical decisions and who need to feel confident in their understanding of recommendations made by an algorithm.
These methods are most appropriate for domains where the input space is not very high-dimensional and where the individual input features have some substantially useful information individually or in small groups. They would not be good for image input, but might be good in cases with, for example, a set of meaningful measurements of the condition of a patient in the hospital.
We'll concentrate on the CART/ID3 family of algorithms, which were invented independently in the statistics and the artificial intelligence communities. They work by greedily constructing a partition, where the splits are
axis aligned
and by fitting a
constant
model in the leaves. The interesting questions are how to select the splits and and how to control capacity. The regression and classification versions are very similar.
Regression
The predictor is made up of
a partition function, [mathjaxinline]\pi[/mathjaxinline], mapping elements of the input space into exactly one of [mathjaxinline]M[/mathjaxinline] regions, [mathjaxinline]R_1, \ldots , R_ M[/mathjaxinline], and
a collection of [mathjaxinline]M[/mathjaxinline] output values, [mathjaxinline]O_ m[/mathjaxinline], one for each region.
If we already knew a division of the space into regions, we would set [mathjaxinline]\hat{y}_ m[/mathjaxinline], the constant output for region [mathjaxinline]R_ m[/mathjaxinline], to be the average of the training output values in that region; that is:
[mathjax]O_ m = {\rm average}_{\{ i \mid x^{(i)} \in R_ m\} }y^{(i)}\; \; .[/mathjax]
Define the error in a region as
[mathjax]E_ m = \sum _{\{ i \mid x^{(i)} \in R_ m\} }(y^{(i)} - O_ m)^2\; \; .[/mathjax]
Ideally, we would select the partition to minimize
[mathjax]\lambda M + \sum _{m=1}^ M E_ m\; \; ,[/mathjax]
for some regularization constant [mathjaxinline]\lambda[/mathjaxinline]. It is enough to search over all partitions of the training data (not all partitions of the input space!) to optimize this, but the problem is NP-complete.
Study Question:
Be sure you understand why it's enough to consider all partitions of the training data, if this is your objective.
Building a tree
So, we'll be greedy. We establish a criterion, given a set of data, for finding the best single split of that data, and then apply it recursively to partition the space. We will select the partition of the data that
minimizes the sum of the mean squared errors of each partition.
Given a data set [mathjaxinline]D[/mathjaxinline], let
[mathjaxinline]R^+_{j,s}(D) = \{ x \in D \mid x_ j \geq s\}[/mathjaxinline] be the set of examples in data set [mathjaxinline]D[/mathjaxinline] whose value in dimension [mathjaxinline]j[/mathjaxinline] is greater than or equal to [mathjaxinline]s[/mathjaxinline];
[mathjaxinline]R^-_{j,s}(D) = \{ x \in D \mid x_ j < s\}[/mathjaxinline] by the set of examples in [mathjaxinline]D[/mathjaxinline] whose value in dimension [mathjaxinline]j[/mathjaxinline] is less than [mathjaxinline]s[/mathjaxinline];
[mathjaxinline]\hat{y}^+_{j,s} = {\rm average}_{\{ i \mid x^{(i)} \in R^+_{j,s}(D)\} }y^{(i)}[/mathjaxinline] be the average [mathjaxinline]y[/mathjaxinline] value of the data points in set [mathjaxinline]R^+_{j,s}(D)[/mathjaxinline]; and
[mathjaxinline]\hat{y}^-_{j,s} = {\rm average}_{\{ i \mid x^{(i)} \in R^-_{j,s}(D)\} }y^{(i)}[/mathjaxinline] be the average [mathjaxinline]y[/mathjaxinline] value of the data points in set [mathjaxinline]R^-_{j,s}(D)[/mathjaxinline].
Now, here is the pseudocode. [mathjaxinline]{\bf BuildTree}(D)[/mathjaxinline]:
If [mathjaxinline]|D| \leq k[/mathjaxinline]: return [mathjaxinline]{\rm Leaf}(D)[/mathjaxinline]
Find the variable [mathjaxinline]j[/mathjaxinline] and split point [mathjaxinline]s[/mathjaxinline] that minimizes:
[mathjax]E_{R^+_{j,s}(D)} + E_{R^+_{j,s}(D)}\; \; .[/mathjax]
Return [mathjaxinline]{\bf Node}(j, s, {\bf BuildTree}(R^+_{j,s}(D)), {\bf BuildTree}(R^-_{j,s}(D))[/mathjaxinline]
Each call to
BuildTree
considers [mathjaxinline]O(d n)[/mathjaxinline] splits (for [mathjaxinline]d[/mathjaxinline] dimensions, since we only need to split between each data point in each dimension); each requires [mathjaxinline]O(n)[/mathjaxinline] work.
Study Question:
Concretely, what would be a good set of split-points to consider for dimension [mathjaxinline]j[/mathjaxinline] of a dataset [mathjaxinline]D[/mathjaxinline]?
Pruning
It might be tempting to regularize by using a somewhat large value of [mathjaxinline]k[/mathjaxinline], or by stopping when splitting a node does not significantly decrease the error. One problem with short-sighted stopping criteria is that they might not see the value of a split that will require one more split before it seems useful.
Study Question:
Apply the decision-tree algorithm to the XOR problem in two dimensions. What is the training-set error of all possible hypothesis based on a single split?
So, we will tend to build a tree that is too large, and then prune it back.
Define
cost complexity
of a tree [mathjaxinline]T[/mathjaxinline], where [mathjaxinline]m[/mathjaxinline] ranges over its leaves as
[mathjax]C_\alpha (T) = \sum _{m = 1}^{|T|} E_ m(T) + \alpha |T|\; \; .[/mathjax]
For a fixed [mathjaxinline]\alpha[/mathjaxinline], we can find a [mathjaxinline]T[/mathjaxinline] that (approximately) minimizes [mathjaxinline]C_\alpha (T)[/mathjaxinline] by “weakest-link" pruning:
Create a sequence of trees by successively removing the bottom-level split that minimizes the increase in overall error, until the root is reached.
Return the [mathjaxinline]T[/mathjaxinline] in the sequence that minimizes the criterion.
We can choose an appropriate [mathjaxinline]\alpha[/mathjaxinline] using cross validation.
## Classification
The strategy for building and pruning classification trees is very similar to the strategy for regression trees.
Given a region [mathjaxinline]R_ m[/mathjaxinline] corresponding to a leaf of the tree, we would pick the output class [mathjaxinline]y[/mathjaxinline] to be the value that exists most frequently (the
majority value
) in the data points whose [mathjaxinline]x[/mathjaxinline] values are in that region:
[mathjax]O_ m = {\rm majority}_{\{ i \mid x^{(i)} \in R_ m\} }y^{(i)}\; \; .[/mathjax]
Define the error in a region as the number of data points that do not have the value [mathjaxinline]O_ m[/mathjaxinline]:
[mathjax]E_ m = \left|\{ i \mid x^{(i)} \in R_ m \; \text {and}\;  y^{(i)} \not= O_ m\} \right|\; \; .[/mathjax]
Define the
empirical probability
of an item from class [mathjaxinline]k[/mathjaxinline] occurring in region [mathjaxinline]m[/mathjaxinline] as:
[mathjax]\hat{P}_{mk} = \hat{P}(R_ m)(k) = \frac{\left|\{ i \mid x^{(i)} \in R_ m \; \text {and}\;  y^{(i)} = k\} \right|}{N_ m}\; \; ,[/mathjax]
where [mathjaxinline]N_ m[/mathjaxinline] is the number of training points in region [mathjaxinline]m[/mathjaxinline]. We'll define the empirical probabilities of split values, as well, for later use.
[mathjax]\hat{P}_{mjv} = \hat{P}(R_{mj})(v) = \frac{\left|\{ i \mid x^{(i)} \in R_ m \; \text {and}\;  x^{(i)}_ j \geq v\} \right|}{N_ m}[/mathjax]
Splitting criteria
In our greedy algorithm, we need a way to decide which split to make next. There are many criteria that express some measure of the “impurity" in child nodes. Some measures include:
Misclassification error
:
[mathjax]Q_ m(T) = \frac{E_ m}{N_ m} = 1 - \hat{P}_{mO_ m}[/mathjax]
Gini index
:
[mathjax]Q_ m(T) = \sum _ k \hat{P}_{mk}(1 - \hat{P}_{mk})[/mathjax]
Entropy
:
[mathjax]Q_ m(T) = H(R_ m) = - \sum _ k \hat{P}_{mk} \log _2 \hat{P}_{mk}[/mathjax]
So that this is well-defined when [mathjaxinline]\hat{P} = 0[/mathjaxinline], we will stipulate that [mathjaxinline]0 \log _2 0 = 0[/mathjaxinline].
They are very similar, and it's not entirely obvious which one is better. We will focus on entropy, just to be concrete.
Choosing the split that minimizes the entropy of the children is equivalent to maximize the
information gain
of the test [mathjaxinline]X_ j = v[/mathjaxinline], defined by
[mathjaxinline]\displaystyle  {\rm infoGain}(X_ j = v, R_ m)[/mathjaxinline]
[mathjaxinline]\displaystyle  =[/mathjaxinline]
[mathjaxinline]\displaystyle  H(R_ m) - \left(\hat{P}_{mjv}H(R^+_{j, v}) + (1 - \hat{P}_{mjv})H(R^-_{j, v})\right)[/mathjaxinline]
In the two-class case, all the criteria have the values
[mathjax]\begin{cases}  0.0 &  \text {when $\hat{P}_{m0} = 0.0$}\\ 0.0 &  \text {when $\hat{P}_{m0} = 1.0$} \end{cases}[/mathjax]
The respective impurity curves are shown below, where [mathjaxinline]p = \hat{p }_{m0}[/mathjaxinline]:
There used to be endless haggling about which one to use. It seems to be traditional to use:
Entropy to select which node to split while growing the tree
Misclassification error in the pruning criterion
As a concerete example, consider the following images:
The left image depicts a set of labeled data points and the right shows a partition into regions by a decision tree.
Points about trees
There are many variations on this theme:
Linear regression or other regression or classification method in each leaf
Non-axis-parallel splits: e.g., run a perceptron for a while to get a split.
What's good about trees:
Easily interpretable
Fast to train!
Easy to handle multi-class classification
Easy to handle different loss functions (just change predictor in the leaves)
What's bad about trees:
High estimation error: small changes in the data can result in very big changes in the hypothesis.
Often not the best predictions
Hierarchical mixture of experts
Make a “soft" version of trees, in which the splits are probabilistic (so every point has some degree of membership in every leaf). Can be trained with a form of gradient descent.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:59 PM (revision 4f166135)

## Lecture: Bagging - bootstrap aggregation of models

Lecture: Bagging - bootstrap aggregation of models
Lecture: Bagging - bootstrap aggregation of models
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Bagging

Bagging
Bootstrap aggregation
is a technique for reducing the estimation error of a non-linear predictor, or one that is adaptive to the data.
Construct [mathjaxinline]B[/mathjaxinline] new data sets of size [mathjaxinline]n[/mathjaxinline] by sampling them with replacement from [mathjaxinline]{\cal D}[/mathjaxinline]
Train a predictor on each one: [mathjaxinline]\hat{f}^ b[/mathjaxinline]
Regression case
: bagged predictor is
[mathjax]\hat{f}_{\rm bag}(x) = \frac{1}{B} \sum _{b=1}^ B \hat{f}^ b(x)[/mathjax]
Classification case
: majority bagged predictor: let [mathjaxinline]\hat{f}^ b(x)[/mathjaxinline] be a “one-hot" vector with a single 1 and [mathjaxinline]K-1[/mathjaxinline] zeros, so that [mathjaxinline]\hat{y}^ b(x) = {\rm arg}\max _{k} \hat{f}^ b(x)_ k[/mathjaxinline]. Then
[mathjax]\hat{f}_{\rm bag}(x) = \frac{1}{B} \sum _{b=1}^ B \hat{f}^ b(x),[/mathjax]
which is a vector containing the proportion of classifiers that predicted each class [mathjaxinline]k[/mathjaxinline] for input [mathjaxinline]x[/mathjaxinline]; and the predicted output is
[mathjax]\hat{y}_{\rm bag}(x) = {\rm arg}\max _{k} \hat{f}_{\rm bag}(x)_ k\; \; .[/mathjax]
There are theoretical arguments showing that bagging does, in fact, reduce estimation error.
However, when we bag a model, any simple predictability is lost.
Random Forests
Random forests are collections of trees that are constructed to be de-correlated, so that using them to vote gives maximal advantage. In competitions, they often have the best classification performance among large collections of much fancier methods.
For [mathjaxinline]b = 1 .. B[/mathjaxinline]
Draw a bootstrap sample [mathjaxinline]{\cal D}_ b[/mathjaxinline] of size [mathjaxinline]n[/mathjaxinline] from [mathjaxinline]{\cal D}[/mathjaxinline]
Grow a tree on data [mathjaxinline]{\cal D}_ b[/mathjaxinline] by recursively repeating these steps:
Select [mathjaxinline]m[/mathjaxinline] variables at random from the [mathjaxinline]d[/mathjaxinline] variables
Pick the best variable and split point among them
Split the node
return tree [mathjaxinline]T_ b[/mathjaxinline]
Given the ensemble of trees, vote to make a prediction on a new [mathjaxinline]x[/mathjaxinline].
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:59 PM (revision 4f166135)

## Lecture: Random forests models

Lecture: Random forests models
Lecture: Random forests models
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Nearest neighbor models

Lecture: Nearest neighbor models
Lecture: Nearest neighbor models
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Nearest Neighbor

Nearest Neighbor
In nearest-neighbor models, we don't do any processing of the data at training time–we just remember it! All the work is done at prediction time.
Input values [mathjaxinline]x[/mathjaxinline] can be from any domain [mathjaxinline]\mathcal X[/mathjaxinline] ([mathjaxinline]\mathbb {R}^ d[/mathjaxinline], documents, tree-structured objects, etc.). We just need a distance metric, [mathjaxinline]d: \mathcal X \times \mathcal X \rightarrow \mathbb {R}^+[/mathjaxinline], which satisfies the following, for all [mathjaxinline]x, x', x" \in \mathcal X[/mathjaxinline]:
[mathjaxinline]\displaystyle  d(x, x)[/mathjaxinline]
[mathjaxinline]\displaystyle = 0[/mathjaxinline]
[mathjaxinline]\displaystyle d(x, x')[/mathjaxinline]
[mathjaxinline]\displaystyle = d(x', x)[/mathjaxinline]
[mathjaxinline]\displaystyle d(x, x")[/mathjaxinline]
[mathjaxinline]\displaystyle \leq d(x, x') + d(x', x")[/mathjaxinline]
Given a data-set [mathjaxinline]\mathcal D = \{ (x^{(i)},y^{(i)})\} _{i=1}^ n[/mathjaxinline], our predictor for a new [mathjaxinline]x \in \mathcal X[/mathjaxinline] is
[mathjax]h(x) = y^{(i)} \; \; \; \text {where}\; \; \; i = {\rm arg}\min _{i} d(x, x^{(i)})\; \; ,[/mathjax]
that is, the predicted output associated with the training point that is closest to the query point [mathjaxinline]x[/mathjaxinline].
This same algorithm works for regression
and
It's a floor wax
and
a dessert topping!
classification!
The nearest neighbor prediction function can be described by a Voronoi partition (dividing the space up into regions whose closest point is each individual training point) as shown below:
In each region, we predict the associated [mathjaxinline]y[/mathjaxinline] value.
Study Question:
Convince yourself that these boundaries do represent the nearest-neighbor classifier derived from these 6 data points.
There are several useful variations on this method. In
[mathjaxinline]k[/mathjaxinline]-nearest-neighbors
, we find the [mathjaxinline]k[/mathjaxinline] training points nearest to the query point [mathjaxinline]x[/mathjaxinline] and output the majority [mathjaxinline]y[/mathjaxinline] value for classification or the average for regression. We can also do
locally weighted regression
in which we fit locally linear regression models to the [mathjaxinline]k[/mathjaxinline] nearest points, possibly giving less weight to those that are farther away. In large data-sets, it is important to use good data structures (e.g., ball trees) to perform the nearest-neighbor look-ups efficiently (without looking at all the data points each time).
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:59 PM (revision 4f166135)

## Video transcripts

### Lecture: Introduction to non-parametric models


LESLIE KAELBLING: Today, we're going
to do something that's really different than all
the other kinds of methods and strategies
that we've talked about so far.
When we started out, we were doing linear classifiers.
We did linear regression.
Then, we started looking at more complicated kinds of hypotheses
by representing them in deeper and deeper neural networks
or recurrent things.
We looked at different settings of the problem and so on.
What we're going to do today is go back
to regression and classification,
but look at a category of models which
has the worst name ever, which is non-parametric methods.

And the reason this is the worst name
ever is because you might think that they
don't have parameters.
But they do, so that's why it's kind of a horrible name.
But the idea here is roughly that the hypothesis space
or the complexity of the hypothesis space
is not fixed in advance.
So in some sense, the complexity of the hypotheses
that we consider kind of depends on the data.

So in all the other examples we've seen so far,
we've picked a feature space.
Or we've picked a number of layers in the neural network
or the number of units in a layer.
We've picked that stuff.
Then, we train a hypothesis from that class which
has a fixed parameterization.
Once you've fixed the neural network,
let's say there's a certain number of weights.
And then you're going to train up those weights.
In an outer loop, you can do some kind of cross-validation
or something to decide, oh, really,
I would rather have a bigger model or a smaller model.
But that's kind of considered not part of a learning
algorithm itself.
It's considered this kind of outer selection process.
So in these non-parametric methods,
you can kind of think of us as doing the model complexity
choices while we're fitting our hypothesis.
And in one class at the very end,
I'll talk about nearest neighbor methods,
where we don't even really make an explicit hypothesis.
So they're kind of really outliers in the space.
But mostly-- so this is the interesting point.
So we'll talk about decision trees, mostly.

We'll talk about a method called "bagging,"
which addresses some of the problems of decision trees.
And we'll finish with nearest neighbor, which is the easiest.
You might wonder why we save the easiest algorithm for last,
but who knows?

So that's our plan.
So let's talk about decision trees.
So what's interesting-- so decision trees
is a very early machine learning algorithm.
It's interesting because it was developed almost
simultaneously, I think, by statisticians
in the US and a machine learning person in Australia.
And it went along for a while in two separate communities.
And then suddenly, they discovered each other
and said, whoa.
We basically just invented the same thing.
So that was kind of interesting.
And it used to be--
20 years ago or 30 years ago or something--
if you read the proceedings of a conference on machine learning,
half the papers were about decision trees.
So now, decision trees are not so much of a thing.
But they're worth talking about because they're
interesting in several ways and still useful.


### Lecture: Introduction to non-parametric models


LESLIE KAELBLING: So in trees, we're
going to recursively partition the input space
and fit a simple hypothesis in each region.
So I'll do a couple simple examples,
and then we'll look at the algorithm.
So let's look at classification first.
So here's classification in 2D.
So imagine that I have some points.

There are some points.
And I need to come up with a hypothesis.
So since this is classification, I
want some kind of a separator.

And let's see.
Well, we can look at this data and say
it doesn't look like there's probably
not a linear separator.
Maybe we could do some kind of fancy feature space,
or maybe we could get a neural network to do this.
But another strategy would be to just kind of look at this data
and say, hmm.
Maybe I can divide it up in pieces.
And so I might say, well, look, if I kind of draw
a line right here, everything above me is positive.
That seems good.
And if I were to draw a line, I don't know, over here,
that guy's a positive.
And I could draw a line here, and those would all
be negative.
And then I can draw a line here.
So there is a hypothesis now.
I'm going to label this region "positive."
I'm going to label this region "positive,"
this region "negative," this region
"negative," this region "positive," and I
have made a classifier.

So fundamentally, the idea is going
to be we divide the space up into regions.
We fit something simple in the region.
There's going to be variability in what kinds
of splits, how we decide to divide the space up
into regions.
And there can be variability in how
do we fit a simple hypothesis inside each region.
So that's a kind of a hypothesis for a classifier.
For regression, let me just do a regression in 1D.

Imagine that I have some data points that look like this.

So here's x.
Here's x.
Here's y.
So I'm trying to fit now a function y as a function of x.
And I might look at this data and say, hmm.
Well, I could divide it up here.
maybe in here and here and there.
And then I could fit a hypothesis here.
Maybe I could fit a constant.
That might be my regression line.
So I could make, say, this guy and then this.

So this is a regression.
It doesn't look like the regressions we've done so far.
It's not continuous, but it's not such a terrible fit
of the data.
It's a hypothesis we can understand.
It feels very different.
If we wanted to, we could do something fancier
inside the regions, like maybe we
could do a little linear regression in each region.
That would be OK, too--

so something like that.
You could do a quadratic regression inside the regions.
You could do all kinds of things inside the regions.
But the point is that the kind of way we get complexity here
is by dividing things up.

So let's talk about algorithms.
We'll actually talk about the regression algorithm first.
It's, I think, a little more clear how to do it.
But the character of the regression algorithm
and the character of the classification algorithm
are not really all that different.


### Lecture: Introduction to non-parametric models


LESLIE KAELBLING: So let's talk about regression trees.
So we're going to-- let's see.
We'll say that in the end, we're going to have M regions.
So that's M regions.
We're going to partition the x, the input space,
into m regions.
Call them "R1" through Rm."

And we're going to have output values O1 through Om.
So here, we're just going to fit a constant.
So we're going to fit the simple case where our regression is
going to be piecewise constant.

And so given a new x, our hypothesis,
our prediction for an x is going to be Om
when x is in region little m.

So partition means we divide the space up into regions.
The regions are exhaustive.
They take up all the space.
And they're not overlapping so that they don't overlap.
So there's exactly one region that this point will be in,
and we're going to output Om.
Oh, you ask, what if it's on the boundary?
The answer is you can do whatever you want to.
It's a very low-probability event.

So that's going to be our setup, our hypothesis base.
So it's kind of a funny hypothesis base,
because we might not decide in advance the value of M. What
we're going to see is that the algorithm is actually going
to pick M as it's going along.
So we're not going to start out in advance and say,
you can have 10 regions.
We're going to kind of let the data tell us
what regions it should have.
So I'm going to define a couple more things.

So let's decide that in some sense, what
we'd like to do is if we have Om for each region,
we can define the error in a region.
And the error in a region--
since we're doing regression, actually, it's
also not very hard to change loss functions here.
But let's just do a typical squared error loss.
So we're going to look at all the data points.

So i is the set of indices for which
this x-value is in R to the m.
So you take all the training data
that's in R to the m, Rm, in region m,
and say, well, we'll look at the squared difference between what
we decided the output should be for that region
and what the yi for that region is.
So that would be like saying, oh, I'm
outputting a constant value here.
So never mind the diagonal anymore.
I'll put this constant value, and I'm
going to measure the squared differences
from that constant value.
That's my error in that region.
And in our dreams, what would we like to do?
We would like to minimize some objective,
like the sum of the errors in the partitions
plus some penalty on the number of partitions.
So here is a question to be sure you're awake.
If I let the number of partitions
be as big as it wants to be, as big as you want,
what can we make the error?
SUBJECT: 0.
LESLIE KAELBLING: 0.
We can make the error 0.
This is an awesome algorithm.
We just put each point in its own partition.
We set the output value to be its y-value.
We have 0 error.
So this is a case where we better do regularization.
Otherwise, we'll just get something
that follows our data exactly.
So ultimately, we're going to have to think
about how we should pressure m.
But we'll do that inside the algorithm.
Now, unfortunately, this is hard to optimize.
So while our whole game so far has
been generally we write down an objective function
and then we say, OK, we're basically done.
We're just going to do gradient descent or something
like that to optimize that.
But in these trees, we're making these kind of discrete choices.
And the error function is counting points that
are inside a region and so on.
And so it's not easy to differentiate.


### Lecture: Introduction to non-parametric models


LESLIE KAELBLING: We are not going
to solve this problem by gradient descent,
because it's not differentiable.
And it's a combinatorially hard optimization problem actually
to solve this, right?
So to really imagine the data in high dimensions,
there's a lot of different partitions, right?
So you could say, OK, what's the stupidest possible algorithm?
Stupidest possible algorithm is let
us consider all the partitions of the data
and see which one has the best score, OK.
But that is too crazy to think about.
So we don't know how to do gradient descent,
and enumerating all the solutions is too hard.
And so we are left with one of our favorite tricks,
which we've used several times already,
which is some kind of a greedy algorithm.
So we're going to use a greedy algorithm,
because we don't know a better way to try to optimize that.
And our greedy algorithm is going
to start with basically m equals 1.
So we're going to start saying, well, I
could consider having all my data be in the same one,
just one partition.
And then it's going to greedily divide it up,
but then recursively and greedily divide.

And we'll start out by thinking about how to build the tree,
how to build this partition.
So it's going to be in the form of a tree.
And then we'll talk about how we bring the regularization
into it, because we actually do this in kind
of two different phases.
OK, so I have to define some stuff.
And I don't know how to get out of it,
so I'm just going to do it.
Maybe it's not so bad.
OK, here we go.
So R plus js of D--
I'm just going to write it down, and then I'll explain.
It's not that big a deal.

OK, so D is the set of data.

So it's going to be XY pairs.
We've been having XY pairs all along.
This is still XY pairs.
The X's, let's assume for now that the
is the Xi's are in the D. It's actually
one of the things that's nice about this algorithm.
It very easily can handle cases where actually some
of your inputs are discrete.
So regular linear regression doesn't really
do very well if some of the inputs are discrete.
But here we'll see handling discrete inputs is easy.

OK, so we have a data set of XY pairs.
And we'll assume that the-- well, for right now,
assume that X is in R to the D. And so what is this?
This says, I am all the data points whose j-th coordinate
is bigger than s, right?
So that means if I were to take my space along dimension j
and divide it at s, then these are the ones who
would be on the bigger side.
OK.
And then we're going to have R minus js of D, which
is this same basic thing.

OK, so this says, if we were to put a split,
these are the ones who would be on one side,
these are the ones who would be on the other side.
OK, so that's talking about how we split up the data.
And then the other thing to talk about
is what value would we put in a bit.
OK, so we will let Y hat--
so many decorations-- plus js is the average over the i's such
that Xi is in--
oh, of D, I need a D. Yeah.

So this is just really the average of--
oh, no, no, no, no, no.
Oh, sorry.
I told you a notation that was horrible.
OK, I need R plus js, good.
R plus js of D Yi.
OK.
So this says this is the average Y value of the data points
whose j coordinate is bigger than s, right?
So that says, oh, if I took my data over here
and I said, oh, this is where X sub j and this is where s,
I would look at these guys and say, what's the average?
And that would be this Y plus.
OK.
And you can guess there's going to be an analogous thing for Y
minus js of D. OK, so now we defined
the stuff we needed to define.
And we can basically just write the algorithm down now.
The algorithm is really easy.

So I can just do it here.
We'll say we're going to build a tree.

This is going to be a recursive algorithm.
It takes in a data set D. And so I start out with all my data.
And my base case is going to be something like,
if the size of D is less than some constant,
then I'm going to return a leaf.

So a leaf is going to be a leaf in the tree.

So base case says, once I'm down to a small number
of data points, that's a parameter of the algorithm.
So really K should be a parameter.
Once I'm down to a small number of data points, I'm just done.
And then if I'm not done, otherwise, I'm
going to find the dimension j and split s that minimizes E--
oh, the notation is going to pay off--
R plus js of D plus ER minus js of D.


### Lecture: Introduction to non-parametric models


PROFESSOR: So the dimension j, our data is in R to the d.
So maybe our data is in a 10 dimensional space.
So we have to consider all the different dimensions.
When I did regression here, I only did it in one dimension,
so there was only one dimension to consider.
But more generally, you could imagine, oh,
I want to make a slice along this dimension.
I'm going to make a slice along that dimension.
I'm going to makes a slice along another dimension.
So we have to consider all the dimensions.
So at least that's easy.
We know that there is d dimensions--
probably not too many.
That seems OK.
But now, this is a little suspect what I just wrote down.
I say find the split s that minimizes something.
So what is a split?
Well, a split is a real number.
So our decision tree is going to end up looking something like--
I might say if x--
if the 2 dimension is bigger than 3.3,
then we'll say, well, if that's not true,
then we get the smaller data.
And if it is true, we get the bigger data.
So this data set here is data set r minus 2, 3.3.
And this data set over here is data set r plus 2, 3.3.
So this is j.
This 2 here would be our j, what dimension.
And this 3.3 here would be our s, where to split.

So this is a discrete parameter.
This is a continuous parameter.
Do I have to consider all the continuous parameters,
all the possible continuous values s?
What do you think?
Yeah.
STUDENT: Only the x values present in the data actually
matter.
PROFESSOR: Only the x values present in the data matter.
That is right.
So if you look at this example, it
is true that there is a continuum of places
that I could have decided to partition the data.
But the fact is that, in terms of that score that we have,
it doesn't matter if I-- once I put up a partition
line between two data points in my actual training set,
it doesn't really matter whether it's closer to here
or closer to there.
So I really only have to consider the splits
the go between my data points.
Generally speaking, we'll pick the splits
that are halfway between the data points,
just to kind of stay away from the data.
Probably doesn't matter all that much.

So that means that this optimization is not as terrible
as you think.
There are d possible values of j and n--
if n is the size of our data set-- possible values of s.
So we're just going to consider carving the data up
between each data point.
Good if you have 100 million data
points, but OK if you have 1,000.
Good.
So we're going to consider dimension and a split.
And then what we want to do is minimize this.
And this is just the error that we
get by making the average prediction of the guys
on the small side of the split and the average prediction
of the guys on the big side of the split.
So we're going to find the place where,
if we were to divide this data up and then
put a constant at the average of values on either side,
we would minimize the error.
Does that make sense?
This is it.
This is basically it.


### Lecture: Introduction to non-parametric models

LESLIE KAELBLING: What do we do?
We return a node, a node in our tree.
And a node is going to have in it a j, which
is which dimension to split on, an s, which is where to split.
And then it's going to be a binary tree.
So it's going to have two subtrees.
And the left subtree we're going to get by calling build tree
on the left data.

Oh, let's see.
Let me do it this way, on the little data,
just to keep n build tree over here.

OK.

It's fun to implement this algorithm,
because it's, like, easy.

A long time ago, I was taking the train I commute
on the train.
A long time ago, I was on the train working away
reading some paper.
The paper had a picture of some kind of a tree in it.
And we're crammed in three people in a seat.
And the person next to me is reading over my shoulder.
And he can't contain himself any longer.
And he says, you know what?
That's a decision tree.
It turns out that it actually wasn't a decision tree.
It was some other tree.
But he was like, oh yeah, this is--
he was teaching me about decision trees.
I said, well, not really.
And he worked his own company where
they were selling the software to people in marketing,
he said, for $1 million a seat.
I said, oh, my undergraduate class
implements that as an exercise.
[LAUGHTER]
Anyway, it was kind of entertaining.
But, yeah, so anyway, this is regression trees.
OK, so what remains somewhat troublesome here is picking k.
And in fact, generally speaking--
and more generally, the notion of regularization, right?
So we wrote a criterion down there with some lambda.
There's no lambda here.
We have a k.
So what is that about?
It happens that it's very tempting
to try to implement regularization.
You could try to just implement it by k.
You could just say, oh, when I get down
to a certain number of points, I'm quitting.
But that's not so good, because really you
might want to be more sensitive to how much variation there
is in the partitions or something like that.
So then people are tempted to say, OK,
what I'm going to do is, once I get to a case
where, like, the things aren't improving very well,
right-- so a very tempting thing to say,
very tempting notion of how to stop this algorithm
would be to say, well, if none of my splits
improves the error very much, then maybe I should just quit,
right?
We're used to that as a kind of a termination criterion
for various kinds of algorithms.
You might say, oh, if the next split doesn't improve things
over where I was before, I'm just not going to bother.
But it turns out that our nemesis XOR, the things that
have XOR type structure, have the property that it
might be that the next split doesn't improve things
very well.
And maybe even the next one doesn't.
But one more after that and everything will be clear.
So it's very hard to kind of just decide to quit.
So what people typically do is make the tree much too big.
So what are we going to do is we're
going to now do two phases, which is we're first going
to grow a tree that's too big.

And then because we like the tree metaphor,
we are going to prune it.

And I wrote the algorithm out in the notes.
And I think I'm not going to really go through it.
But the idea is, OK, so now we have a tree, right?
In each node, it's a binary tree.
But it doesn't have to be balanced or anything.
So I know, maybe it looks like this.
And in the leaves is some kind of prediction, right?
So this is going to be number, 2.7 minus 4.1,
3.9, 0.2, something like that.
And then in the nodes is going to be
x3 is greater 6.1 and x1 is greater than minus 2.2
and something like that.
So we're going to draw this tree.
So at every internal node is a test governed by a j and an s.
And every leaf node is a value.
That's what our tree looks like.
So we're going to grow a tree that's much too big.
And then what we'll do is actually do
another process that goes from the bottom up
and considers now this objective directly.

So now given this tree, given a particular tree,
we can evaluate that objective.

And what we can do then is, again greedily, work our way up
from leaves and say, if I were to do away with this split,
would it improve that objective or would it not, right?
So you could imagine that, if these leaves have
pretty similar values to one another,
then getting rid of the split might not matter too much.
And depending on our lambda, it would decrease out, right?
So getting rid of a split decreases out.
m is the number of leaf nodes.
So if I get rid of a split, I decrease m by 1.
I'll increase the error.
But maybe it'll be worth it.
So I grow out the tree.
And then I walk back up just asking,
if I were to remove this node, would I improve my quality?
And I do that until I get a tree.
It's not optimal, but it's not so bad.

OK, that's regression trees.

For variations on this theme, there's
a bunch of different kinds of things you could do, right?
We talked about instead of--

oh, that Y plus whatever.
That thing, instead of making it be the average value,
the prediction that we're going to make in that note,
instead of making it be the average,
we could fit a little local regression.
We're all really good at linear regression now.
So we can throw in a linear regression there.
That's easy.

You could also potentially try to make--
our splits right now are aligned with the axes, right?
They're all of the form is the value along this dimension
bigger or less than some value.
There's also strategies for making them not
be aligned with the axes.


### Lecture: Introduction to non-parametric models


LESLIE KAELBLING: Let's quickly talk about classification.
So what gets different in classification?
The basic idea is the same.
And really, what's different for classification
is, how do we characterize the value of a split, the utility
of a split?
So for classification--

I'm going to do this a little bit less
carefully on the board.
It's also written out in the notes.
So for classification, our Om, the output value that we're
going to pick in a particular node,
we'll generally just pick it to be the majority.
Average doesn't make sense.
So if we've got a partition and classification,
it contains some positive points and some negative points.
What prediction should you make when you're there?
Well, you should predict the value that's most prevalent.
It's also true that classification trees
apply very nicely in cases where you have potentially
multiple discrete categories.
And so maybe it's divided into the reds
and the blues and the greens.
And again, you want to predict the majority in the region.
The error in a region, again, we get by counting.
So it will be something like the set
of the size of the set of indices, such that x i is in Rm
and yi is not equal to Om.

So what's the error in a partition?
It's the number of elements whose y-values
aren't the majority value.

Again, I have to define one piece of notation,
and then we're good.
So we're going to define p hat "mk."

And that is going to be--

OK.
So this is the number of elements in this region whose
y-value is k.
So if we're doing binary, k is an element of the y domain.
So if y is plus 1, minus 1-- so if we're just
doing binary classification, then k is plus 1 or minus 1.
Actually, I think in the notes, they use--
let me use "plus 1" and "0."
It totally does not matter, because they're just labels
here.
And we don't treat them as numbers.
So if it's a binary classification problem,
then k is either, let's say, 1 or 0.
But if it's multiple classes, then it
could be any element of that set.
So p hat is a hat because it's an estimate.
It's a p because it's a probability.
And it's roughly, what's the probability
of one element being in class k, given that it's in this region?


### Lecture: Introduction to non-parametric models


LESLIE KAELBLING: Now, we're going to talk about impurity.

So what we had here in regression
is this error measure, which was really convenient.
We just computed the squared error,
because we said, oh, well, if our prediction is going
to be the average, then we just look at the squared difference.
And that gives us a really convenient error measure.
In decision trees, especially in the old literature,
there used to be these battles about the best way
to measure impurity for classification.
I think eventually, people decided
it didn't matter all that much.
But it was fierce there for a little while.
So anyway, I'll show you some choices just kind of for fun.
So we need impurity measures.

And let me just say what--
first, let me spell.

So what we're doing is we're doing classification.
And we're going to do a recursive partitioning
algorithm.
And we're interested-- imagine that this thing is plus, plus,
plus, minus, and this guy's minus,
minus, minus, minus, minus, minus, minus, minus, minus,
minus, plus, let's say.
And we want to ask ourselves, is this a good choice?
Dividing it here, was that a good choice?
And so what we're going to do is we're going to say,
oh, I want to pick the split that minimizes
the impurity of the children.
That's what we're going to do.
So we're going to pick the split.
So the reason that we care about measuring impurity
is because we're going to pick the split that minimizes
the weighted sum of impurities of the children, weighted
because if I have one node that has a whole bunch of points,
one partition that has a lot of points,
and one that doesn't have very many,
I'm going to weight it by the number of points
or the proportion of points that go in each partition.
So now, I want to measure impurity of the children.
So how do I do that?
We've got three choices.

So this one, I like to talk about it
mostly because it comes up in the news for other reasons.
So it's good to just discuss what it is.
So one is this thing called the "Gini index."

And so it measures the impurity of a node
as the sum over the classes of p hat mk 1 minus p hat mk.
And what it looks like--
I'm just going to use a piece of board over here.

So let this be the p hat in the binary case.
So imagine that k is 2.
So there are only two possible values.
And this is p hat of m0.
So this is the probability of class 0 in this region.
So if you look at that value over there, when p hat is 0,
then it definitely has value 0.
And when p hat is 1, it also definitely has value 0.

Its maximum is at 0.5, and it goes up to 0.5.
And it basically looks like this.

So it's just a convenient way to take
sort of the degree to which things are not equally
distributed and make a measure.
So the Gini index gets used in economics
to measure income inequality and disparity and so on.
So it's sort of the lack of uniformity of the distribution
of something.
So that's Gini index.


### Lecture: Introduction to non-parametric models


LESLIE KAELBLING: Another popular choice
is entropy, which is also a good thing just to know about,
which looks like minus P hat log P hat.

And as long as you're willing for--
so you might say, oh my goodness,
what happens when P hat is 0?
And we just by fiat say that, because log of 0
is not a thing to think about, we just
say, oh, when it's 0 we just assume that this guy is
kind of like taking over.
And so it has value 0.
And what you get is a picture that looks almost like this,
except for it goes up to 1.

So this is entropy.

This is Gini.
It's hard to imagine.
People used to write learned papers
with lots of experiments arguing about whether it was better
to use this or that as your measure of how complicated
the leaves in the tree were.
The fact is it doesn't really matter all that much.
OK, so we have some way of measuring.
So of these two, which one will have higher entropy?
Just practice the concept, does this one have higher entropy?
Or does this one have higher entropy?

This one has higher entropy, right?
Because the P is 3/4.
And here the p is, I don't know, something over closer to 1.

So now what we're interested in, though,
is if we're considering splits, right--
so we might consider that split.
So we can call this split number 1.
And then there's, like, this split.
We could consider the split, right?
So we say, oh, we have these guys and those guys.
That one's going to be worse, though, right?
Because actually the entropy of this bottom region
is the same as the entropy of that one,
but the entropy of this one is worse, right?
It's got pluses and minuses mixed in together.
So considering this split will be worse.
So S2 will be worse than S1.
I don't know the actual best split of this data.
We could calculate it.
There's going to be a lab.
Next week's lab will be about decision trees.

OK, so I think at this point you probably
have the concept, right?
So what we're going to do?
We take all our data.
We do a recursive algorithm just like that one.
But instead of using that particular criterion to split,
we're still going to search over all the dimensions
j and all the split points s.
But now we'll use this notion of weighted entropy,
or weighted Gini, or weighted something like that
to decide what's the best split.
And we'll also do this thing of growing the tree out
to be too big, and then pruning it to try to actually optimize
the criterion that we're after.
OK.
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: The reason that we
don't use that error function is because this error
function is--
oh, I see.
So this error function for regression is a squared error.
And in classifications, squared error is not a sensible loss
function.
There's one more choice here, one more choice.
There's one more Q, which would just be the loss function.
You could just say it's 1 minus P hat and the majority, right?
So you could say that's another choice of Q, which
is just a prediction error.

Generally speaking, these ones I think people
have found that they work better than prediction error.
But it would be sensible to use prediction error also,
completely sensible.
Yeah.
OK, so now that we've kind of looked at this a little bit,
what makes trees and why do people care?
Why were trees a good idea?
So you know, for a little while, they were the best game in town
even just in terms of making good predictions.
At this point, generally speaking,
if you want really good generalization
and predictive accuracy, just plain old decision trees
by themselves are not necessarily a good choice.
We'll talk about a way of making them better.
But people still use them.
And one of the reasons that people still
use them is that a human can look at a tree
and make sense of it at some level, right?
So if x3 is, you know, the weight of your car
or somebody's blood pressure, right,
that's a thing that you could run,
let's say, your data through an algorithm,
get out a tree like this, show it to a human expert.
They could look at it and say, oh
yeah, that makes sense, right?
But if we're trying to predict gas mileage
and the color of the car shows up,
then you might say, ah, that doesn't make so much sense.
But then when you realize that people who have red cars,
like me, maybe the red cars tend to be
ones that don't have such good gas mileage.
And so maybe the color of the car
does have something to do with the gas mileage.
I don't know, right?
But so you can at least inspect a decision tree
and see if it kind of makes sense
and maybe even get insight into your data or problem
by looking at the tree.
It is very hard to get insight into your data or problem
by looking at the weight matrices
of a deep neural network.
Although there is a ton of people
working exactly that problem.
So it's a big issue, right?
You pour data into a machine learning algorithm, out
comes an answer.
How do you know if it's sensible?
So one strategy is where you let expert humans look at it
and tell you if it's sensible.
And then if you want that strategy,
you need a thing that an expert human can actually look at.
So that's one reason, one kind of interesting thing.
The other one is that it's, like, reasonably easy to much
this algorithm around in different ways.
It handles multiple classes well.
You can change the loss function,
and it's not too hard.
So it's a kind of a template almost
for a large set of different algorithms
that you can play with pretty easily.


### Lecture: Introduction to non-parametric models

LESLIE KAELBLING: What's bad about decision trees?
What's bad about decision trees-- and it's bad--
is that if you were to read most textbooks,
they would tell you they are "high variance."
In our language, we would say that they
have "high estimation error."

So probably, you've forgotten.
We talked about estimation error early on.
So estimation error is the idea that oh, your hypothesis class
is kind of pretty roomy.
It's not too hard to find hypotheses
that fit your data well.
And so you're never quite sure that you've nailed things down.
And when you have high estimation error,
you risk not having very good generalization.
And one thing that's really easy to see--
and you could do experiments with it,
and it comes out quite clearly.
A way to understand this notion of high estimation error
is if you change the data just a little bit and fit a new tree,
you can sometimes get just a really different tree
that makes really different predictions.
So this algorithm can be kind of touchy in that way,
kind of sensitive to the data that you give it.
So that's the sense in which some people
would say it has high variance.
The possible kinds of solutions that you might get just
changing the data around a little bit can change a lot.
And so that tends to mean that you're
overfit or too sensitive to the training data that you have.
So that's kind of a problem with these methods.
So we have strategies for dealing
with that problem, which are interesting because they
actually deal more generally.
People use them to fix the high variance
problem in decision trees.
It can also be used to fix high variance
problems in other kinds of learning algorithms.
And it's kind of surprising that it works.


### Lecture: Introduction to non-parametric models


LESLIE KAELBLING: We're going to now talk about bagging.
So bagging is a very general strategy.

Why is it called "bagging"?
It's called "bagging."
"Bagging" stands for "bootstrap aggregation."
I don't know if that helps, but that's what it stands for.

So bagging is like an outer loop.
So you have a machine learning algorithm,
let's say decision trees.
And then you're going to try to improve
its performance by bagging.
And so let's just look at bagging.
So let's do regression, because it's
kind of more straightforward.
So we're going to construct B data sets of some size--
we'll just say size n--

by sampling with replacement from D. So we have data set D.
And what we're going to do is we're
going to take our data set.
And see, we're used to doing something
like this in cross-validation, where you divide it up
into pieces.
What we're going to do here is just randomly--
draw with replacement just means we're
going to n times randomly pick an element out of this data set
and throw it in.
So sometimes, we'll have duplicates.
But we'll get a bunch of different data sets, all
of the same size, which are just re-sampled
from our original data set.
But some will have some points left out,
and other ones will have other points left out, and so on.
So they're drawn from the same distribution
that our original D was drawn from.
But they have some variability in them
that's generated by this random process.

You can probably guess where I'm going now.
So now, we're going to train a predictor.
So we're going to train our algorithm
on each of these b's to get a predictor, f hat b.

So we have a hypothesis now.
Whatever our algorithm is, we now have b of these data sets.
We trained on each data set, and that
means we get one predictor for each data set.

And now, our predictor, f bag of some input x,
is just going to be the average prediction.

I remember reading about this and thinking,
this seems kind of stupid.
Why would this be a good idea?
I'm just taking my data and I'm making other data sets
out of it and so on.
But there's actually interesting statistical theory
that shows that this does reduce the variance.
It reduces the degree to which the hypothesis
that you generate kind of varies as a function of the data set.
And you can see it kind of--
I don't know.
We talked a little bit about dropout
for regularizing our arch.
That's like dropping out some pieces of the hypothesis.
You can think of this as dropping out pieces of the data
and saying, well, if I were missing some bits of data,
I would get this hypothesis.
Or missing some other ones, I would get that hypothesis.
We do this at random a bunch of times,
and we just average the results.
Yep?
Isn't that the sum over b to n?

Indeed, it is.
Thank you.
Good.
Yeah?
Do we require here that each of the bootstrap data
sets be the same size as the original one?
Or is n just a free parameter?
No.
No, that's a good point.
We don't really care.
Certainly, we don't require that this be the same size
as the original one.

Are
There reasons why that might be a good idea?

I guess it depends.
Some algorithms are kind of sensitive to the n.
So it could be that by using much smaller data sets,
something sort of different would happen.
I think that that's probably a thing
that you play with that we use in cross-validation.
Yeah.

So trees are very flexible, easy to work with,
easy to understand, but kind of high variance.
Bagging reduces variance.
And there's an analog for classification, which, again,
is written out in the notes.
But it's the conceptually same thing.
So I'm not going to write it on the board.


### Lecture: Introduction to non-parametric models


LESLIE KAELBLING: If you take trees and bagging together,
and instead of just bagging trees,
you do a kind of a mixture thing, which
I'm going to draw on the board, you get a machine learning
algorithm that, in fact, whenever I have
a learning, whenever I have a regression problem,
and especially my students who are talking
to me about some fancy neural network thing, I say,
OK, but first, try random trees, random forests.
So if you take trees, the idea of trees
and the idea of bagging, although this is not precisely
bagging trees--
so this is the idea of trees and the idea of bagging,
you sort of get something called random forests.
And this-- I love this algorithm,
because it looks like an algorithm a monkey made up.
But it's actually one of the most competitive machine
learning algorithms.
OK.
So here we go.

So we're going to make bootstrap data sets.

So just by resampling like, over there.

And then for each of these trees,
we're going to do something.
It looks like I'm about to just do bagging trees,
but I'm going to do something weirder.

OK.
So grow tree.

So we're going to grow a tree on this data set,
on the bootstrap data set.
Right?
So this set that we got by resampling by recursively--
and then we're going to change our tree growing algorithm.
So we're like, oh yeah, we can do bagging on trees,
but we're going to add even more restraints to keep the tree
from over fitting the data.
So we're going to select m variables.

Right?
And this is from the set, 1D to D,
right, where D is the dimension of our data, right?
So maybe our data is in 100 dimensions.
We're going to randomly pick a subset of the dimensions.
So the normal decision tree is when we go to make a split,
we pick any dimension we want to,
the one that gives us the best split.
In random forest, we say, no, you
don't get to pick any dimension you want to.
We're going to randomly pick a set of dimensions,
and you can pick one of those.

So we'll pick the best split among them.

OK.
So we make B data sets.
For each data set, we grow a tree,
but we also do this extra thing where we kind of prevent
ourselves from doing what seems like the best
thing on every step.
And then, finally, at the end, we
vote the predictions of the trees, as you would in bagging.

So scikit-learn-- it's a Python library.
We haven't used it very much, but it's handy.
Has an implementation of random forests.
I encourage you, if you ever face a regression
problem in particular, just try random forests.
Sometimes, it's great.
Now, we said one of the reasons we
love decision trees was because they were easily interpretable.
This random forest is-- so it's forest, right, because it's
a collection of trees.
It is no longer interpretable, because it might have
a hundreds or a thousand trees.
People do post-hoc kind of analysis, though,
of the random forest.
So you can, for instance, take your whole forest
and say, which features come up the most frequently,
or appear near the tops of the trees?
So you can still, you can analyze this random forest,
and try to get some understanding of maybe what
kind of decisions it's making.
But it's much, much harder.


### Lecture: Introduction to non-parametric models


LESLIE KAELBING: The last thing I wanted to talk about--
because nobody should finish a machine learning class
without nearest neighbor-- is nearest neighbor.
And it's fun because it's the simplest algorithm
and the laziest algorithm.
So nearest neighbor.

OK, so the best thing about nearest neighbor
is that the learning algorithm is there isn't one.
So the learning algorithm, what you have to do,
the learning algorithm is remember your data.

OK, so that's a really nice learning algorithm.
OK, but then the pestering part is
that you're going to have to make predictions
on new data points.
So that's where all the work is.
In linear regression, or neural networks, or whatever what do
we do?
We take the data, and then we chew
on it for a really long time, spend a lot of CPU cycles,
and so on.
And what comes out is a hypothesis.
And then it's kind of quick to apply the hypothesis.
You can make predictions on a new input pretty easily.
This is like the dual.
It's the opposite.
Here, the data, we didn't even do any work with.
But at prediction time we have a lot of work to do potentially.
So it's kind of opposite that way.
So before we can go somewhere though,
we need the notion of a distance metric.
So the critical ingredient to nearest neighbor,
since it's got the name "nearest,"
is the distance metric.
And it is on two--
well, I'll talk about three different points
in the input space.

What's interesting here is that the input space doesn't
have to be real numbers.
It doesn't have to be a vector space.
The way we've been thinking about input space is a lot.
It could be documents, or trees, or potatoes.
You know, I don't know.
It could be really anything as long
as you can define a distance metric.
So it has to have the following properties--
that the distance between x and itself
is zero, the distance between x and another x,
that it's symmetric-- right?
So it doesn't matter which way the arguments go.
And it has to satisfy the triangle inequality.
xx double prime is less than or equal to the distance
between x and this thing.

OK, so that just means that it's a distance measure.
The behavior of this algorithm depends
on your distance measure a lot.
But what's nice is if you define that, then you're
basically ready to go.
OK, so learning algorithm is easy.
Now I can tell you what the hypothesis is.
So if you ask me to predict a value--
oh, and by the way, it does classification and regression.
Nice, OK.
So I ask you to predict the value, h of x.

It's y i.
So it's going to be one of the y values in your training set
where i is arg min over your data points of d, x, xi.
That's it.
OK, what is that?
It says, you asked me for prediction x on new value x,
I go looking through my data set and try to find
the one that's closest to this.
Then I predict the y value that went with that x.

That's it.
So let me draw you some pictures for fun.
I kind of like giving this lecture because there's
a bunch of basic background ideas that come up in it,
and they're good to know.
Imagine that this is my training data.

OK, there's my training data.
I absorb it.
And I say, cool, this is training data.
So let's see-- what am I doing?
I'm doing a classification problem.
So this is dimension 1 and dimension 2.
So my inputs are in r2, and my outputs are pluses and minuses.

So good.
So then when someone asks me a question, I say, OK,
which of my training points am I closest to?
This one?
I must be negative.
That's all.
One thing that's kind of interesting
is to think about what the decision boundary is.
So I'm not very--

yeah.
OK, I'm not going to get this right
because it's a triangulation, and this is not
a triangulation.
But whatever, we're going to get the basic concept here.

If you divide the input space up into regions,
where the regions have the property that for everybody
in this region this is their closest point,
it turns out this is something called the Delaunay
triangulation--

no, this is the Voronoi partition,
which comes from the--
OK, this is the Voronoi.

And in some sense, you could say, OK, I see.
I have some decision boundaries.
Like, this whole region is negative,
and this region is negative, and this region is negative,
and this one is, and everything else is positive.
It's not like there's no hypothesis, right?
Given the set of data, there is a hypothesis.
But we never bother computing a more explicit representation
of it.
We just compute our predictions from the data.
OK, there's a couple of things to know about this.
There's a couple variations on the theme and ways
of making it more efficient.
So the one big problem with this is
that it is potentially highly inefficient
if you have a lot of data because at prediction time,
it looks like you have to take your one point
and compare it to absolutely everybody.
And so if you have a lot of training data--
the time it takes you to make a new prediction
is related to how much training--
It looks like it's linear in your training data.
You can make it logarithmic, sort of,
in the size of your training data
by using good data structures.

People use something called ball trees,
and there's a bunch of fancy data structures.
But basically, if your data is in sort of high dimensions
and you have a lot of data, you can
organize it using clever data structures
so that the queries are not as expensive as it looks.
So that's one thing that's important.
Another thing is that you can do k nearest neighbors instead.
So this is what people call 1 nearest neighbor.
An alternative is to say, oh, I have this query point here,
but what I'm going to do is I'm going to find not
just the one nearest point, I feel
like I would like to be a little more robust to noise than that.
So I'm going to take the 3 or 5 or 7 or 9 nearest points
and give out the majority, or the average
if their real values.
If you're doing classification, it's
not good to pick an even number generally.
You just have to decide what to do about ties.
It's no fun.
So you pick an odd number for k.
So you can do k, find the k nearest neighbors.
If it's a classification problem,
generally you would just predict the majority.
If it's a regression problem, you
can do even more interesting things.
So if it's a regression problem, and you
pick the k nearest neighbors, again, you
could predict the average.
That would be the most straightforward thing.
You could also fit a linear regression.

So the whole idea here is that we're not
committing to the parametric form of the global function
because we don't really know what it's going to be like.
And by picking the parametric form of a big function,
it means that the data point that you get over here
could actually change the predictions you're making
in this part of the space.
And sometimes that doesn't feel very good.
So one nice property of the strategy is that it's local.
The data that you use to make predictions
in this part of the space is the data that's near you.
But you can make more sophisticated predictions.
So you could do a linear regression.
Or actually, a very popular strategy
is a weighted linear regression.
In weighted locally linear regression,
you pick the k nearest neighbors,
let's say, and you do a linear regression,
but you change your linear regression algorithm
so that you give some points more importance
than other ones.
And in particular, you would give the points
that were nearer to you more influence
in the linear regression than ones that are farther away.
And so that works out pretty nicely.

The last point is that all of this
is very sensitive to the distance metric.
So it's common-- so again, we started this class
with that data set about cars, which had cylinders and weight
and stuff in it, and they were not very commensurable.
And we talked about, well, maybe we should at least
standardize them so that they go between 0 and 1 or something.

So the default distance metric is euclidean distance.
I mean, if you download some nearest neighbor package,
it's going to just try to do euclidean distance.
And euclidean distance is OK if the meaning of deviation
of a particular size in one dimension
is the same to you as the meaning
of a deviation of that same size in another dimension.
So really, when you pick the distance metric,
you're asserting something about what
you think is the underlying similarity of data
in your domain.
And so often you have to think hard about what d should be.
I've also, in various points, messed around
with algorithms that use cross validation to learn a d.
That's another thing you could do is you could say,
oh, if I predict using this distance metric,
it doesn't work as well as if I predict using that distance
metric, and maybe I could actually
optimize my distance metric so that I make good predictions.
So that's kind of potentially a sensible thing to do here.
So now, our predictor itself is non-parametric.
It doesn't have a compact form in the terms
of a certain number of parameters.
But I sneak some parameters into the distance metric,
and I train those instead, and then


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/2ea3ce073a6c2ff4d5ead87a661e7e41/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Non-parametric_methods.pdf
