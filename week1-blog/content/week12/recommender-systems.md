# Recommender systems

> Week 12: Recommender Systems · MIT 6.036 courseware archive

## Notes – Chapter 13: Recommender systems

Notes – Chapter 13: Recommender systems
You can sequence through the Recommender systems lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 13: Recommender systems
notes as a PDF file.
You may find the (optional)
appendix
on collaborative filtering and the singular value decomposition (SVD) to be of interest.

## Lecture: Recommender systems - introduction

Lecture: Recommender systems - introduction
Lecture: Recommender systems - introduction
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Introduction to recommender systems

Introduction to recommender systems
The problem of choosing items from a large set to recommend to a user comes up in many contexts, including music services, shopping, and online advertisements. As well as being an important application, it is interesting because it has several formulations, some of which take advantage of a particular interesting structure in the problem.
Concretely, we can think about a company like Netflix, which recommends movies to its users. Netflix knows the ratings given by many different people to many different movies, and knows your ratings on a small subset of all possible movies. How should it use this data to recommend a movie for you to watch tonight?
There are two prevailing approaches to this problem. The first,
content-based recommendation
, is formulated as a supervised learning problem. The second,
collaborative filtering
, introduces a new learning problem formulation.
Download this chapter as a PDF file
This page was last updated on Thursday December 12, 2019; 09:34:32 PM (revision 4b592d7d7)

## Content-based recommendations

Content-based recommendations
In content-based recommendation, we try to learn a predictor, [mathjaxinline]f[/mathjaxinline], that uses the movies that you have rated so far as training data, find a hypothesis that maps a movie into a prediction of what rating you would give it, and then return some movies with high predicted ratings.
The first step is designing representations for the input and output.
It's actually pretty difficult to design a good feature representation for movies. Reasonable approaches might construct features based on the movie's genre, length, main actors, director, location, or even ratings given by some standard critics or aggregation sources. This design process would yield
[mathjax]\phi : \text {movie} \rightarrow \text {vector}\; \; .[/mathjax]
Movie ratings are generally given in terms of some number of stars, so the output domain might be {1, 2, 3, 4, 5}. It's not appropriate for one-hot encoding on the output, and pretending that these are real values is also not entirely sensible. Nevertheless, we will treat the output as if it's in [mathjaxinline]\mathbb {R}[/mathjaxinline].
Thermometer coding might be reasonable, but it's hard to say without trying it. Some more advanced techniques try to predict rankings (would I prefer movie A over movie B) rather than raw ratings.
note
Study Question:
What is the disadvantage of using one-hot? What is the disadvantage of using [mathjaxinline]\mathbb {R}[/mathjaxinline]?
Now that we have an encoding, we can make a training set based on
your
previous ratings of movies
[mathjax]\left\{  \left(\phi (m^{(1)}), \text {rating}(m^{(1)})\right), \left(\phi (m^{(2)}), \text {rating}(m^{(2)})\right), \ldots \right\}[/mathjax]
The next step is to pick a loss function. This is closely related to the choice of output encoding. Since we decided to treat the output as a real, we can formulate the problem as a regression from [mathjaxinline]\phi \rightarrow \mathbb {R}[/mathjaxinline], with [mathjaxinline]\text {Loss}(p, y) = \frac{1}{2}(y-p)^2[/mathjaxinline] We will generally need to regularize because we typically have a very small amount of data (unless you really watch a lot of movies!).
Finally, we need to pick a hypothesis space. The simplest thing would be to make it linear, but you could definitely use something fancier, like a neural network.
If we put all this together, with a linear hypothesis space, we end up with the objective
[mathjax]J(\theta ) = \frac{1}{2}\sum _{i \in D_ a} (y^{(i)} - \theta ^ T x^{(i)} - \theta _0)^2 + \frac{\lambda }{2} \left\lVert \theta \right\rVert ^2\; \; .[/mathjax]
This is our old friend, ridge regression, and can be solved analytically or with gradient descent.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:50 PM (revision 4f166135)

## Lecture: Collaborative filtering - framework

Lecture: Collaborative filtering - framework
Lecture: Collaborative filtering - framework
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Collaborative filtering - strategy

Lecture: Collaborative filtering - strategy
Lecture: Collaborative filtering - strategy
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Collaborative filtering - hypothesis space

Lecture: Collaborative filtering - hypothesis space
Lecture: Collaborative filtering - hypothesis space
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Collaborative filtering - objective function

Lecture: Collaborative filtering - objective function
Lecture: Collaborative filtering - objective function
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Collaborative filtering - alternating least squares idea

Lecture: Collaborative filtering - alternating least squares idea
Lecture: Collaborative filtering - alternating least squares idea
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Collaborative filtering - alternating least squares algorithm

Lecture: Collaborative filtering - alternating least squares algorithm
Lecture: Collaborative filtering - alternating least squares algorithm
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Collaborative filtering - stochastic gradient descent

Lecture: Collaborative filtering - stochastic gradient descent
Lecture: Collaborative filtering - stochastic gradient descent
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Collaborative filtering

Collaborative filtering
There are two difficulties with content-based recommendation systems:
It's hard to design a good feature set to represent movies.
They only use your previous movie ratings, but don't have a way to use the vast majority of their data, which is ratings from other people.
In collaborative filtering, we'll try to use
all
the ratings that other people have made of movies to help make better predictions for you.
Intuitively, we can see this process as finding the kinds of people who like the kinds of movies I like, and then predicting that I will like other movies
In fact, there's a third strategy that is really directly based on this idea, in which we concretely try to find other users who are our “nearest neighbors" in movie preferences, and then predict movies they like. The approach we discuss here has similar motivations but is more robust.
that they like.
Formally, we will start
We will in fact not
actually
represent the whole data matrix explicitly—it would be too big. But it's useful to think about.
by constructing a
data matrix
[mathjaxinline]Y[/mathjaxinline], where [mathjaxinline]Y_{ai}[/mathjaxinline] represents the score given by user [mathjaxinline]a[/mathjaxinline] to movie [mathjaxinline]i[/mathjaxinline]. So, if we have [mathjaxinline]n[/mathjaxinline] users and [mathjaxinline]m[/mathjaxinline] movies, [mathjaxinline]Y[/mathjaxinline] has shape [mathjaxinline]n \times m[/mathjaxinline].
[mathjaxinline]Y[/mathjaxinline] is very sparse (most entries are empty).
In the Netflix challenge data set, there a re 400,000 users and 17,000 movies. Only 1% of the data matrix is filled.
note
So, we will think of our training data-set as [mathjaxinline]D = \left\{ (a,i,r)\right\}[/mathjaxinline], a set of tuples, where [mathjaxinline]a[/mathjaxinline] is the index assigned to a particular user, [mathjaxinline]i[/mathjaxinline] is the index assigned to a particular movie, and [mathjaxinline]r[/mathjaxinline] is user [mathjaxinline]a[/mathjaxinline]'s rating of movie [mathjaxinline]i[/mathjaxinline].
We are going to try to find a way to use [mathjaxinline]D[/mathjaxinline] to predict values for missing entries. Let [mathjaxinline]X[/mathjaxinline] be our predicted matrix of ratings. Now, we need to find a loss function that relates [mathjaxinline]X[/mathjaxinline] and [mathjaxinline]Y[/mathjaxinline], so that we can try to optimize it to find a good predictive model.
Idea #1
Following along with our previous approaches to designing loss functions, we might want to say that our predictions [mathjaxinline]X_{ai}[/mathjaxinline] should agree with our data [mathjaxinline]Y_{ai}[/mathjaxinline], and then add some regularization, yielding loss function
[mathjax]\text {Loss}\text {}(X, Y) = \frac{1}{2} \sum _{(a,i) \in D} (Y_{ai} - X_{ai})^2 + \sum _{\text {all } (a,i)}X_{ai}^2\; \; .[/mathjax]
This is a
bad
idea!
It will set [mathjaxinline]X_{ai} = 0[/mathjaxinline] for all [mathjaxinline](a, i) \not\in D[/mathjaxinline].
Study Question:
Convince yourself of that!
We need to find a different kind of regularization that will force some generalization to unseen entries.
Linear algebra idea:
The
rank
of a matrix is the maximum number of linearly independent rows in the matrix (which is equal to the maximum number of linearly independent columns in the matrix).
If an [mathjaxinline]n \times m[/mathjaxinline] matrix [mathjaxinline]X[/mathjaxinline] is rank 1, then there exist [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]V[/mathjaxinline] of shapes [mathjaxinline]n \times 1[/mathjaxinline] and [mathjaxinline]m \times 1[/mathjaxinline], respectively, such that
[mathjax]X = UV^ T\; \; .[/mathjax]
If [mathjaxinline]X[/mathjaxinline] is rank [mathjaxinline]k[/mathjaxinline], then there exist [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]V[/mathjaxinline] of shape [mathjaxinline]n \times k[/mathjaxinline] and [mathjaxinline]m \times k[/mathjaxinline], respectively, such that
[mathjax]X = UV^ T\; \; .[/mathjax]
Idea #2
Find the rank 1 matrix [mathjaxinline]X[/mathjaxinline] that fits the entries in [mathjaxinline]Y[/mathjaxinline] as well as possible. This is a much lower-dimensional representation (it has [mathjaxinline]m + n[/mathjaxinline] parameters rather than [mathjaxinline]m \cdot n[/mathjaxinline] parameters) and the same parameter is shared among many predictions, so it seems like it might have better generalization properties than our previous idea.
So, we would need to find vectors [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]V[/mathjaxinline] such that
[mathjax]UV^ T = \begin{bmatrix}  U^{(1)} \\ \vdots \\ U^{(n)} \end{bmatrix} \begin{bmatrix}  V^{(1)} &  \cdots &  V^{(m)} \end{bmatrix} = \begin{bmatrix}  U^{(1)}V^{(1)} &  \cdots &  U^{(1)}V^{(m)} \\ \vdots &  \ddots &  \vdots \\ U^{(n)}V^{(1)} &  \cdots &  U^{(n)}V^{(m)} \\ \end{bmatrix} = X \; \; .[/mathjax]
And, since we're using squared loss, our objective function would be
[mathjax]J(U,V) = \frac{1}{2} \sum _{(a,i)\in D}(U^{(a)}V^{(i)} - Y_{ai})^2 \; \; .[/mathjax]
Now, how can we find the optimal values of [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]V[/mathjaxinline]? We could take inspiration from our work on linear regression and see what the gradients of [mathjaxinline]J[/mathjaxinline] are with respect to the parameters in [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]V[/mathjaxinline]. For example,
[mathjax]\frac{\partial J}{\partial U^{(a)}} = \sum _{i,\text { for }(a,i)\in D} (U^{(a)}V^{(i)} - Y_{ai})V^{(i)} \; \; .[/mathjax]
We could get an equation like this for each parameter [mathjaxinline]U^{(a)}[/mathjaxinline] or [mathjaxinline]V^{(i)}[/mathjaxinline]. We don't know how to get an immediate analytic solution to this set of equations because the parameters [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]V[/mathjaxinline] are multiplied by one another in the predictions, so the model does not have a linear dependence on the parameters. We could approach this problem using gradient descent, though, and we'll do that with a related model in the next section.
But, before we talk about optimization, let's think about the expressiveness of this model. It has one parameter per user (the elements of [mathjaxinline]U[/mathjaxinline]) and one parameter per movie (the elements of [mathjaxinline]V[/mathjaxinline]), and the predicted rating is the product of these two. It can really represent only each user's general enthusiasm and each movie's general popularity, and predict the user's rating of the movie to be the product of these values.
Study Question:
What if we had two users, 1 and 2, and two movies, A and B. Can you find [mathjaxinline]U, V[/mathjaxinline] that represents the data set [mathjaxinline](1, A, 1), (1, B, 5), (2, A, 5), (2, B, 1)[/mathjaxinline] well?
Idea #3
If using a rank 1 decomposition of the matrix is not expressive enough, maybe we can try a
rank [mathjaxinline]k[/mathjaxinline]
decomposition! In this case, we would try to find an [mathjaxinline]n\times k[/mathjaxinline] matrix [mathjaxinline]U[/mathjaxinline] and an [mathjaxinline]m \times k[/mathjaxinline] matrix [mathjaxinline]V[/mathjaxinline] that minimize
[mathjax]J(U,V) = \frac{1}{2} \sum _{(a,i)\in D} (U^{(a)} \cdot V^{(i)} - Y_{ai})^2\; \; .[/mathjax]
Here, the length [mathjaxinline]k[/mathjaxinline] vector [mathjaxinline]U^{(a)}[/mathjaxinline] is the [mathjaxinline]a^{th}[/mathjaxinline] row of [mathjaxinline]U[/mathjaxinline], and represents the [mathjaxinline]k[/mathjaxinline] “features" of person [mathjaxinline]a[/mathjaxinline]. Likewise, the length [mathjaxinline]k[/mathjaxinline] vector [mathjaxinline]V^{(i)}[/mathjaxinline] is the [mathjaxinline]i^{th}[/mathjaxinline] row of [mathjaxinline]V[/mathjaxinline], and represents the [mathjaxinline]k[/mathjaxinline] “features" of movie [mathjaxinline]i[/mathjaxinline]. Performing the matrix multiplication [mathjaxinline]X = UV^ T[/mathjaxinline], we see what the prediction for person [mathjaxinline]a[/mathjaxinline] and movie [mathjaxinline]i[/mathjaxinline] is [mathjaxinline]X_{ai} = U^{(a)} \cdot V^{(i)}[/mathjaxinline].
The total number of parameters that we have is [mathjaxinline]nk + mk[/mathjaxinline]. But, it is a redundant representation. We have 1 extra scaling parameter when [mathjaxinline]k=1[/mathjaxinline], and [mathjaxinline]k^2[/mathjaxinline] extra parameters in general. So, we really effectively have [mathjaxinline]nk + mk - k^2[/mathjaxinline] “degrees of freedom."
Study Question:
Imagine [mathjaxinline]k = 3[/mathjaxinline]. If we were to take the matrix [mathjaxinline]U[/mathjaxinline] and multiply the first column by 2, the second column by 3 and the third column by 4, to make a new matrix [mathjaxinline]U'[/mathjaxinline], what would we have to do to [mathjaxinline]V[/mathjaxinline] to get a [mathjaxinline]V'[/mathjaxinline] so that [mathjaxinline]U'{V'}^ T = UV^ T[/mathjaxinline]? How does this question relate to the comments above about redundancy?
It is still useful to add offsets to our predictions, so we will include an [mathjaxinline]n \times 1[/mathjaxinline] vector [mathjaxinline]b_ U[/mathjaxinline] and an [mathjaxinline]m \times 1[/mathjaxinline] vector [mathjaxinline]b_ V[/mathjaxinline] of offset parameters, and perform regularization on the parameters in [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]V[/mathjaxinline]. So our final objective becomes
[mathjax]J(U,V) = \frac{1}{2} \sum _{(a,i)\in D} (U^{(a)} \cdot V^{(i)} + b_ U^{(a)} + b_ V^{(i)} - Y_{ai})^2 + \frac{\lambda }{2} \sum _{a=1}^ n \left\lVert U^{(a)}\right\rVert ^2 + \frac{\lambda }{2} \sum _{i=1}^ m \left\lVert V^{(i)}\right\rVert ^2 \; \; .[/mathjax]
Study Question:
What would be an informal interpretation of [mathjaxinline]b_ U^{(a)}[/mathjaxinline]? Of [mathjaxinline]b_ V^{(i)}[/mathjaxinline]?
Optimization
Now that we have an objective, it's time to optimize! There are two reasonable approaches to finding [mathjaxinline]U[/mathjaxinline], [mathjaxinline]V[/mathjaxinline], [mathjaxinline]b_ U[/mathjaxinline], and [mathjaxinline]b_ V[/mathjaxinline] that optimize this objective: alternating least squares (ALS), which builds on our analytical solution approach for linear regression, and stochastic gradient descent (SGD), which we have used in the context of neural networks and other models.
Alternating least squares
One interesting thing to notice is that, if we were to fix [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]b_ U[/mathjaxinline], then finding the minimizing [mathjaxinline]V[/mathjaxinline] and [mathjaxinline]b_ V[/mathjaxinline] is a linear regression problem that we already know how to solve. The same is true if we were to fix [mathjaxinline]V[/mathjaxinline] and [mathjaxinline]b_ V[/mathjaxinline], and seek [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]b_ U[/mathjaxinline]. So, we will consider an algorithm that takes alternating steps of this form: we fix [mathjaxinline]U, b_ U[/mathjaxinline], initially randomly, find the best [mathjaxinline]V, b_ V[/mathjaxinline]; then fix those and find the best [mathjaxinline]U, b_ U[/mathjaxinline], etc.
This is a kind of optimization sometimes called “coordinate descent," because we only improve the model in one (or, in this case, a set of) coordinates of the parameter space at a time. Generally, coordinate ascent has similar kinds of convergence properties as gradient descent, and it cannot guarantee that we find a global optimum. It is an appealing choice in this problem because we know how to directly move to the optimal values of one set of coordinates given that the other is fixed.
More concretely, we:
Initialize [mathjaxinline]V[/mathjaxinline] and [mathjaxinline]b_ V[/mathjaxinline] at random
For each [mathjaxinline]a[/mathjaxinline] in [mathjaxinline]1, 2, \ldots , n[/mathjaxinline]:
Construct a linear regression problem to find [mathjaxinline]U^{(a)}[/mathjaxinline] to minimize
[mathjax]\frac{1}{2}\sum _{\{ i \mid (a,i) \in D\} } \left(U^{(a)}\cdot V^{(i)} + b_ U^{(a)} + b_ V^{(i)} - Y_{ai}\right)^2 + \frac{\lambda }{2}\left\lVert U^{(a)}\right\rVert ^2 \; \; .[/mathjax]
Recall minimizing the least squares objective (we are ignoring the offset and regularizer in the following so you can see the basic idea):
[mathjax](W\theta - T)^ T(W\theta - T)\; \; .[/mathjax]
In this scenario,
[mathjaxinline]\theta = U^{(a)}[/mathjaxinline] is the [mathjaxinline]k \times 1[/mathjaxinline] parameter vector that we are trying to find,
[mathjaxinline]T[/mathjaxinline] is a [mathjaxinline]m_ a \times 1[/mathjaxinline] vector of target values (for the [mathjaxinline]m_ a[/mathjaxinline] movies [mathjaxinline]a[/mathjaxinline] has rated), and
[mathjaxinline]W[/mathjaxinline] is the [mathjaxinline]m_ a \times k[/mathjaxinline] matrix whose rows are the [mathjaxinline]V^{(i)}[/mathjaxinline] where [mathjaxinline]a[/mathjaxinline] has rated movie [mathjaxinline]i[/mathjaxinline].
The solution to the least squares problem using ridge regression is our new [mathjaxinline]U^{(a)}[/mathjaxinline] and [mathjaxinline]b_ U^{(a)}[/mathjaxinline].
For each [mathjaxinline]i[/mathjaxinline] in [mathjaxinline]1, 2, \ldots , m[/mathjaxinline]
Construct a linear regression problem to find [mathjaxinline]V^{(i)}[/mathjaxinline] and [mathjaxinline]b_ V^{(i)}[/mathjaxinline] to minimize
[mathjax]\frac{1}{2}\sum _{\{ i \mid (a,i) \in D\} } \left(U^{(a)}\cdot V^{(i)} + b_ U^{(a)} + b_ V^{(i)} - Y_{ai}\right)^2 + \frac{\lambda }{2}\left\lVert V^{(i)}\right\rVert ^2[/mathjax]
Now, [mathjaxinline]\theta = V^{(i)}[/mathjaxinline] is a [mathjaxinline]k \times 1[/mathjaxinline] parameter vector, [mathjaxinline]T[/mathjaxinline] is a [mathjaxinline]n_ i \times 1[/mathjaxinline] target vector (for the [mathjaxinline]n_ i[/mathjaxinline] users that have rated movie [mathjaxinline]i[/mathjaxinline]), and [mathjaxinline]W[/mathjaxinline] is the [mathjaxinline]n_ i \times k[/mathjaxinline] matrix whose rows are the [mathjaxinline]U^{(a)}[/mathjaxinline] where [mathjaxinline]i[/mathjaxinline] has been rated by user [mathjaxinline]a[/mathjaxinline].
Again, we solve using ridge regression for a new value of [mathjaxinline]V^{(i)}[/mathjaxinline] and [mathjaxinline]b_ V^{(i)}[/mathjaxinline].
Alternate between steps 2 and 3, optimizing [mathjaxinline]U[/mathjaxinline] and [mathjaxinline]V[/mathjaxinline], and stop after a fixed number of iterations or when the difference between successive parameter estimates is small.
Stochastic gradient descent
Finally, we can approach this problem using stochastic gradient descent. It's easier to think about if we reorganize the objective function to be
[mathjax]J(U,V) = \frac{1}{2} \sum _{(a,i)\in D} \left(\left( U^{(a)} \cdot V^{(i)} + b_ U^{(a)} + b_ V^{(i)} - Y_{ai}\right)^2 + \lambda _ U^{(a)} \left\lVert U^{(a)}\right\rVert ^2 + \lambda _ V^{(i)}\left\lVert V^{(i)}\right\rVert ^2 \right)[/mathjax]
where
[mathjaxinline]\displaystyle  \lambda _ U^{(a)}[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{\lambda }{\#  \text { times }(a, \_ ) \in D} = \frac{\lambda }{\sum _{\{ i \mid (a,i) \in D\} } 1}[/mathjaxinline]
[mathjaxinline]\displaystyle \lambda _ V^{(i)}[/mathjaxinline]
[mathjaxinline]\displaystyle = \frac{\lambda }{\#  \text { times } (\_ , i) \in D} = \frac{\lambda }{\sum _{\{ a \mid (a,i) \in D\} } 1}[/mathjaxinline]
Then,
[mathjaxinline]\displaystyle  \frac{\partial J(U,V)}{\partial U^{(a)}}[/mathjaxinline]
[mathjaxinline]\displaystyle = \sum _{\{ i \mid (a,i) \in D\} } \left[\left(U^{(a)} \cdot V^{(i)} + b_ U^{(a)} + b_ V^{(i)} - Y_{ai}\right) V^{(i)} + \lambda _ U^{(a)}U^{(a)}\right][/mathjaxinline]
[mathjaxinline]\displaystyle \frac{\partial J(U,V)}{\partial b_ U^{(a)}}[/mathjaxinline]
[mathjaxinline]\displaystyle = \sum _{\{ i \mid (a,i) \in D\} } \left( U^{(a)} \cdot V^{(i)} + b_ U^{(a)} + b_ V^{(i)} - Y_{ai}\right)[/mathjaxinline]
We can similarly obtain gradients with respect to [mathjaxinline]V^{(i)}[/mathjaxinline] and [mathjaxinline]b_ V^{(i)}[/mathjaxinline].
Then, to do gradient descent, we draw an example [mathjaxinline](a, i, Y_{ai})[/mathjaxinline] from [mathjaxinline]D[/mathjaxinline] at random, and do gradient updates on [mathjaxinline]U^{(a)}[/mathjaxinline], [mathjaxinline]b_ U^{(a)}[/mathjaxinline], [mathjaxinline]V^{(i)}[/mathjaxinline], and [mathjaxinline]b_ V^{(i)}[/mathjaxinline].
Study Question:
Why don't we update the other parameters, such as [mathjaxinline]U^{(a')}[/mathjaxinline] for some other user [mathjaxinline]a'[/mathjaxinline] or [mathjaxinline]V^{(i')}[/mathjaxinline] for some other movie [mathjaxinline]i'[/mathjaxinline]?
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:50 PM (revision 4f166135)

## Lecture: Demo example - RNNs

Lecture: Demo example - RNNs
Lecture: Demo example - RNNs
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Video transcripts

### Lecture: Recommender systems - introduction


LESLIE KAELBLING: Today we're going
to actually take on a new different kind of machine
learning problem.
It is not going to involve neural networks,
although it will involve a lot of the concepts we already
have.
And the problem is, the general area is making recommendations.
So at this point, everyone's used to this idea,
but like on Netflix or Amazon or even when
you're reading news or reading your Facebook
feed or something, there is an idea
that there is a system that's trying
to predict what kinds of things you'd like to see.
Would you like to read this news article?
Would you like to watch this movie?
Would you like to buy this, I don't know,
cookie jar shaped like a dog, whatever kind of crazy thing
it is.
The question is, how can I predict whether you would
be likely to want this thing.
So I'm going to give this lecture thinking about movies,
but the same basic idea applies in general.
So the most straightforward way to think about this problem,
so we're going to talk about recommender systems.

And we're actually really going to think about just
the prediction part.
So the question is, I want to predict
a good new example of some sort, think of it as a movie,
to show a user or to recommend to a user.
And there's actually another whole dimension
of this problem, which is kind of connected to reinforcement
learning, which we won't really talk about,
but it's a nice kind of more advanced connection
that you could make.
You might imagine-- actually, you'd be kind of annoyed
if Netflix did this to you, but you wouldn't mind it
if probably a newspaper did, and certainly advertisement people
do this all the time.
They show you things to see if you like it or not.
So I show you this news article, maybe not necessarily
because I think you really, really want to read it,
but because I want to gain information
about your preferences, and then I'll use that in later decision
making.
So you can treat some problems like this as reinforcement
learning problems where the algorithm is actually
going to do exploration.

It's interesting.
It just hadn't really occurred to me
until I was thinking about it now.
I don't think--
I don't know-- I don't think that people like Netflix
do a ton of exploration.
They might do exploration in the sense
of showing you things but still assuming
that you're going to look at those titles
and think about them and decide whether to watch them.
But you'd be annoyed if somebody spent two hours of your life
on a movie that you really didn't like just because they
were messing with you trying to find out
what you did or didn't like.
So anyway, we're just going to focus today
on predicting something that we think you'll like,
so that's the problem, and so how do we formulate that.
Well, the most straightforward way
is content-based recommendation.
Actually, we're not going to spend
a ton of time talking about that,
but let me just lay it out.

So in content-based recommendation,
we basically take the items that we
might want to recommend to somebody, like movies
or items on Amazon, and somehow this is the trickiest part,
find a feature representation.

So in the straightforward content-based recommendation,
what we're going to do is, we're going to take this movie
and think of some representation that we
could make of the movie.
And then we're going to think of taking that and mapping it
into a rating.

And so that would be the way we would set up this problem.
And then given your ratings on the movies you've already seen,
we could try to train some kind of a predictor, some kind
of a regression algorithm.
And then we could take all the movies that you haven't seen
and see which one we think is going
to have the highest rating.
So that would be a potential way to solve this problem.
So we could think of this as a regression problem,
and we could just use your previous ratings.

So there's a couple of interesting things
to think about if we look at this.
So the first thing is, what would be a good feature
encoding.
So let me ask you, this is like the first lab
we did it in this class was to talk
about how to encode features of cars or something like that.
So imagine that you want to design a prediction system that
would take movies and represent them
as some kind of fixed length vector.
How would you think about doing that?
Yep.
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: Right.
Good.
So you could definitely take the genre, say, oh, it's a comedy,
it's a drama, whatever.
You might take, I don't know, the 100 most famous actors
and see whether they're in it or not or who the director is,
or whether it's got a lot of blue in it,
or whether it takes a long time, all kinds of things
you could imagine.
It's kind of hard to think about.
It's easy to think about the first few features you would
try, but then to make it richer, it's
a difficult design problem for humans,
I think, to figure out what a good feature set is.
Pandora, the music recommendation system,
actually operates using a set of hand-designed features
that musicologists thought hard about and so on.
And they hire people to actually listen to new songs
and categorize them-- heavy drumbeat, African influence,
whatever.
So they have all these features, and they
get humans to actually featurize the descriptions of the songs.
So there's that problem.
It's like coming up with features.
So then let's think about ratings.
So typical kind of rating, maybe we have some number of stars.
Maybe we have 1 to 5 stars.

We could use a one hot encoding.
So the question is, now how do we think about that.
We could make that be a one hot encoding.
I think that's probably not a good idea.
Anybody supply a reason why?
Or maybe it is a good idea.
No.
Yeah.
AUDIENCE: [INAUDIBLE] given ordering,
like 5 is closer to 4 than to 1.
LESLIE KAELBLING: Right.
Good.
So the reason that that's probably
not a good one hot thing is that really we
understand that 4 is close to 5, but 1 is not close to 5,
but if we made it into a one hot, that wouldn't be true.
It's typical, as I understand it, to just treat this
as if it were a numerical regression problem,
although that's a little weird too
because what is a negative rating mean, or 1,000?
But one totally, I mean not totally unreasonable way anyway
to do it is to just treat it as regression.
So we can say, OK, we're going to treat this
as regression so we'll make features of the movies.
We'll just treat these as numbers.
So we've decided on a feature strategy, which
means now if we took your previous ratings,
we could make a data set.
Yeah.
AUDIENCE: Why are you concerned whether to use
the rating as one hot?
Isn't the rating a label of your true representation?
LESLIE KAELBLING: The rating is the label,
but we also have to think about how to encode the output.
So far, when we've done like a multi-class classification,
you could think of that as a one hot output.
And we use Softmax or something as a kind of a output layer,
and we use log loss as a loss function.
And that's a good choice when things are really categorical
and when you know the categories aren't related semantically
to each other, it's just one of these.
But in this case, we really know that these guys are ordering
and the numbers have some meaning,
and so it's probably better to keep the numbers.
That's the argument.
So now if we had your previous ratings,
we could make a totally standard supervised learning data
set with v of x and rating.
And we can put that into a linear regression
or a neural network regression or some kind of regression
and get out a predictor,
Now, there's a reason why this may be not a super good.
There's a couple reasons why this
isn't the super good thing.
Any ideas?
OK, so one we talked about one already,
which is that it's hard to come up
with a feature representation.
What's another potential problem here?
Yeah.
AUDIENCE: [INAUDIBLE] the ratings are supposed to be.
LESLIE KAELBLING: Well, we know what
ratings-- you've watched some movies in your life,
and you've rated them on Netflix.
But the problem may be that you might be worried about
and that problem we should be worried about
is how many previous ratings you've done.
So how many movies have you watched?
A hundred or something.
Or how many have you rated?
Oh, I don't know, 20 or 30.
When you're starting out, not very many.
And so in a sense, we have this enormous space.
The space of movies is very complicated,
and you've rated a few.
And it's going to be very hard for the system to go from
the very few things that you've-- the small amount
of data you have to good generalizations without
understanding something more about the underlying structure
of the space.
Yep.
AUDIENCE: [INAUDIBLE] the dates?
LESLIE KAELBLING: Yes.
So you would just do this for you.
You personally.
No.
Another thing you could imagine is dumping all the movie
ratings in here completely, and you
might get some aggregate model of what people tend to like.
But yeah, that wouldn't be so good.
But if we do just you, then we suffer from the problem
of not having very much data.


### Lecture: Recommender systems - introduction


LESLIE KAELBLING: I think this is actually much more popular.
It's something called collaborative filtering.

So in collaborative filtering, we're
going to kind of go at this problem
in a really different way.
We're not going to try to make features of the objects really
at all.
In some sense, what we're going to do
is think about the features of an object--
the features of a movie.
One way to think about it is, is who else has liked it
or not liked it?
So another way you could think about this problem
is to say, well, I don't really know whether I'm
going to like this movie or not, but I
know that I really tend to like the same movies as my friend.
And my friend like that movie.
So if my friend liked that movie, then maybe I'll like it,
too.
So we're going to somehow try to learn
a model of the relationships among the preferences of people
and movies, and use that underlying
kind of set of relationships between people's preferences
to make predictions.
So we're not going to encode the content at all.
We're not going to say whether this
is a Western or a long movie.
We're just going to think about our ratings.
But we're also going to use everybody else's ratings of all
the movies to try to make predictions.
So that's the set up.

OK.
So what are we going to do?
So the first idea here is that you can think that there are--
I don't know exactly why this notation,
but it seems to be standard, so we're
going to kind of go with it.
So y is going to be our data matrix.
When you think about this, we're never
going to actually create this thing.
But imagine this, like, really enormous matrix that's
indexed in this direction.
Let me get my users.
So we have n users.

And we'll use a as an index, right?
So this is me, right?
And this is--

[INAUDIBLE]
And then we have m movies indexed by i.

And this is the number of stars that I gave to that movie.
Like, I, user a, watched movie i,
and I gave it three stars, OK?
And so in this matrix, there will be some entries,
but it will be very sparse, right?
An incredibly sparse matrix.

A long time ago-- maybe it was 10 years ago now--
Netflix did this challenge.
They emitted a data set and challenged
people to make predictions on it.
Their data set had 400,000 users and 17,000 movies, OK?
So that's a lot.
All right.
So first of all, if this is 400,000
and that is 17,000, then of course,
you would never actually make this matrix, right?
But the fact is that only one 1% of it was filled.
So another way to think about this
is that our data set-- we're going
to think of our data set as a set of tuples that are a, i, r,
right?
So this is-- this user on this movie gave this rating.
So these are just two different views of the same data.
But that's the data that we're going to work from.
OK.
And we're going to call this matrix y.
We're going to talk about matrix y sometimes,
but we're never really, in our computer,
going to allocate that storage.


### Lecture: Recommender systems - introduction


LESLIE KAELBLING: So then what are we going to do?
We would like to find some new, some matrix X of predictions.

So you could say, OK, there's my data.
And the secret thing that I wish I knew
was, for all the rest of these squares, what's their value?
Right there, you could imagine there's some secret value.
If you were to see Jaws, how many stars would you give it?
We don't happen to know it, but you
can imagine that value exists.
And what we would like to do in some sense
is to try to extrapolate from this matrix of data
that we have into a matrix of predictions, same size.

OK, so that's how we're going to approach this problem.
And so the question then is, well, we
like the optimization and formulation
of machine learning.
So we can move a little loss function, so some loss
function, something that measures X and Y and tells us
whether we're happy about the relationship between them.
And then if we can get a good loss function,
we can use our various tricks of optimization
to try to optimize.
So we're going to be interested in the loss function
on X and Y.
And I'll try out a not very good one maybe to start with,
and then we'll come up with some better ones.
But let's see.
One thing is-- seems important--
that whatever matrix we come up with should agree with the data
that we have on the entries that we have data for.
That seems like a good idea.
So we might want to sum over a i,
r in our data set and say that Xai minus Ya--
don't you like how it's AI?

We would like that to be small.
That seems good.

Oh, I don't know.
In the notes, I put a--
you guys are always on my case.
Like, do I put in the 1/2?
Do I take out the 1/2?
The answer is it doesn't really matter.
But we'll put the 1/2 in sometimes, yeah.
OK, we'll put in half.
So, OK, that's good.
And I don't know, what other property
would we like this matrix to have?
So if we said, this is our loss function,
what would we get if we optimized it?

Any ideas?
Yeah.
AUDIENCE: [INAUDIBLE] [? Yai ?] for [INAUDIBLE]??
LESLIE KAELBLING: Yeah, right.
Good.
So, well, certainly it has the--
if it got to adjust all the parameters in the matrix,
like absolutely totally invent a whole new Xai,
then it could absolutely set all the Xai's equal to Yai.
It could do that.
So it could get zero training loss.
And it could set all the rest of the matrix
to anything it wanted to, right?
So this wouldn't give us any generalization
or any extrapolation or anything, right?
And you might say, oh, we need a regularizer.
So you could say, oh, OK, you could also
say I would like the norm of the X, some matrix norm
or something on X to be small.
But that won't help either.
That will just force everything to be sort of like 0.
So just saying please fill in this matrix
without saying something about the underlying structure,
it won't help us.
So this is, like, not a good idea.
So this basic just find a general purpose X is not going
to work out very well, right?
This would be like saying for one of our classification
problems or prediction problem, please make a giant table
with all the possible inputs.
And then you could fill it in so that it matches
exactly your training data.
And it wouldn't be forced to do anything
in particular to relate the training examples
you had to the cases where you didn't have any data yet.
So this is not going to be good, because there's
no structural constraint that gives us generalization.
So I mean, it's important to remember
that that's what we're about still is generalization.
So there's no structural constraint
that gives generalization, that forces generalization.


### Lecture: Recommender systems - introduction


LESLIE KAELBLING: Even here we have to come up
with a hypothesis space.
So let me try out an idea of a hypothesis space.
Again, this is funny.
This is not a normal kind of prediction hypothesis space
that we're used to, but here's a possible hypothesis space.

I could say, let's imagine that my X--
so up there, I kind of just said, well,
X could be anything.
But I don't want to let X just be anything.
I could say, I'm going to consider X equals UV transpose,
where U is n by 1 and V is m by 1.

OK, because Y was n by m, so X better be n by m.
So here's a model.
So notice that model up there had n times m parameters,
right?
It had, like, an enormous number of parameters.
This model has-- we'll come back to this in 1 minute,
but it looks like it has n plus m parameters.

So it's a much smaller model space in a sense.
So it's a much smaller hypothesis space.
It doesn't give as much room to mess around.
Yup.
AUDIENCE: But if you just removed the regularization
for X, would it still be really bad?
LESLIE KAELBLING: It would still be really bad.
If I didn't regularize on X it says, in the places
where I have data, it has to match.
And in all the other places--
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: Oh, no.
It could be anything.
I don't really care, absolutely anything.
So there's no constraint on the other entries.
Yeah.
Yeah.
OK, so now we're going to put a ton of constraint on.
It's going to turn out this one.
This is the Goldilocks sort of thing, right?
That's too big, this is too small.
We'll end up with something in the middle.
But it's good to see the too small.
So let's think about this.
OK, so what does this say?
This says that my predictions are
going to look like basically there's you here
and V transpose here.
And I'm just going to multiply them.
And I'll get my big old matrix back.
So let's think about me.
So here's Ua, right?
So this is per user.
And this is Vi.
And so it's going to predict here that Xai is Ua times Vi.

OK, so this is like, well, how much am I going to like Jaws?
I've seen Jaws, actually.
I don't really remember too much about it.
But anyway, how much am I going to like Jaws?
Well, I don't know.
It's going to be the product of a coefficient about me
and a coefficient about Jaws.
So that's a little bit of a weak model.
But at least it gives us kind of a constraint.
So let me just write down the objective function here,
so that we can understand exactly what we're asking for.
We could have an objective function now on U and V, right?
It's U and V with respect to Y really, right?
Because Y is our data.
U and V are now going to be the parameters of this model.
And the objective function would be just--

write it this way--

a, i, r, and d.
And it would be Ua times Vi minus r squared.

So that would be our objective function.
We would try to find two vectors, one per user,
one per movie, such that when we multiply them in this way--
right, because if we multiply them,
that fills this whole matrix in.
And then we could say, well, we want
to find the U's and V's such that
in the places where our data is actually defined,
it's as close as it can be.
Does that make sense?

Questions about this model?
Ideas about why it might not be the best thing?
Yup.
AUDIENCE: [? We're ?] [? currently ?] not weighing
the difference of the users, of all the users that have rated
the movie.
We have taken all those ratings, like, at face value.
And we could introduce a variable
to weigh them according to the similarity to us, for example.
LESLIE KAELBLING: I see.
OK.
No, that's right.
So in fact, that brings up yet another style
of doing this problem, which we're not
going to really go into.
But, right, one idea would be to explicitly,
like, set out to find users that are like me or actually, even
as you suggested, kind of compute weightings
for other users depending on the degree to which they agree
with my ratings and then use those weightings
to weight their ratings.
That is another strategy.

So yeah, no, that's good.
That becomes algorithmically a little bit more
complicated, but actually sometimes very useful.

So what does Ua encode here?
What would be a good interpretation
for this coefficient, right?
a is me.
U sub a, that's my coefficient in this whole thing.
What could that possibly encode?

Yeah?
AUDIENCE: [INAUDIBLE] how similar your preferences
are [INAUDIBLE].
LESLIE KAELBLING: How similar our preferences are?
Not exactly, right?

So this is going to be my ratings
for all the movies, this row.
And so my rating for Jaws is going
to be my U times this V. Yeah?
AUDIENCE: [INAUDIBLE] one, like, all the movie.
And V is how good a movie is.
LESLIE KAELBLING: Good.
That's right you is, like, generally how happy or grumpy
I am in my reviewing roughly, right?
And V is going to be like, well, on average
how well-rated is this movie?
Because all I can do to predict a rating
for a movie is to take my general happiness
and the movie's general well-likedness
and multiply those together.
And that gives my rating.
So it's not ridiculous, like that one was.
That was like vacuous.
It wouldn't do anything.
But this really is only getting at a very kind
of like first order understanding of what's
going on.
Does that make sense, one coefficient per user,
one coefficient for movie?
OK.
But we could take this idea and make it a little bit more
general.
And it actually turns into something pretty useful.
So the generality is, OK, we're going
to actually-- our hypothesis base
is going to be almost the same.

It's going to look the same.
So I'm still going to say that my X is UV transpose.
But now I'm going to let u be n by k and V be k--
oh, m by k, because I'm going to transpose it.
OK.
And k is a parameter of our class of models.
So generally speaking, it's a smallish integer,
positive definitely, so, you know, 3 or 4 or 5 or something
like that.
And so what that's like is it's going
to look almost exactly like the picture that we had before,
only now we have n by k here.
And here we have m by k.
And we have still here our big matrix X.
But now for me Ua is a vector of k coefficients.
And Vi is a vector of k coefficients.
And this entry here Xai is still going to be Ua dot product Vi.

Yup.
AUDIENCE: Would there by a bias?
LESLIE KAELBLING: There should be a bias.
I'll put it in in a minute.

So UaVi, this X I'm just going to take it out
of here, because it's a little distracting.
This whole guy is X.
OK, so what this lets us do, one way to think about it,
is that it lets us have features now, right?
It lets us say, oh, this is a description.
You could think of this as a description of the movie.
You could think of it as, like, how many car crashes it has
and who the director is or something, right?
So we're not going to hand design it, though.
We're going to acquire it from the data in a sense.
But you could think of it as a kind
of description of the movie.
And you can think of this as a kind of description
of my preferences for movies lined up, right?
So this has k features.
This is k dimensions.
This has k dimensions.
We're going to assume that my preference for a movie
now is the dot product of my k preference coefficients
and the movie's k feature values.
You can think about it like that.
So this gives us something similar to this kind
of decomposition, but a richer representation.


### Lecture: Recommender systems - introduction


LESLIE KAELBLING: What's our objective function?
a, i, r Ua dot Vi--
that's our prediction.
And I'm going to add, as suggested--
oh, and I have now on the board been inconsistent
about subscripts versus superscripts.
I'm going to keep going with the subscripts.
We're going to have bu sub a plus bv sub i minus r squared.

Oh, and we have to sum over the--

I'm trying to convince while I go, but it's not working--
so plus 1/2 the sum over the i's.

So what have I done?
I have said I've added two kind of ideas here.
Neither of them is super surprising.
One is the offset that got asked for there in the back row.
So I'm going to add some just offsets b, one per user,
which is kind of like the b that we had before, which is just
some additive offset for my general cheerfulness,
and another additive offset per movie,
which is just some offset for the movie's general good value.
So those things can float up and down.
And then the Ua Vi is like the prediction
that we're making that relates my feelings about movies
and the movie's properties.
These guys are each little length-k vectors.
And this is a dot product.
This is what the rating was that I gave to this movie.

So this term is still really like that first term
we had in the bad objective function.
This term says, this model that I am making
should predict the data as well as I can.
But what's giving us leverage here now
is that the model is squeezed.
The model can't be anything it wants to.
It has to have that regular structure up there.
Yep?
SUBJECT 1: What's the difference there between the--
what's exactly the input and what's exactly parameter space,
because when we come up with a hypothesis class, generally,
it's a function of an input.
And it does have some parameters that
have to modify in order to minimize loss.
But here, we don't have specification
what exactly is the input.
LESLIE KAELBLING: Good.
So what are the parameters?
I would argue that the parameters are the U and the V.
SUBJECT 1: So then what's the input?
LESLIE KAELBLING: The input is our data set.
So we have a set of ratings of all the people who have ever
rated any movie on Netflix.
This user gave this movie three stars.
This other user gave this other movie two stars.
So we have all that data.
So that's our input.
So good.
No, that's a very good question.
So what's coming into this thing?
What's coming in is this data set D. And what's coming out
is x, our matrix, in some sense, our complete matrix
of predictions of who's going to like which movie.
Who's going to give what rating to which movie?
We won't make the complete matrix.
Really, in this case, what's going to come out
is U and V. So what's going to go on?
We're going to pour all the data in.
We're going to do some work, which we'll
talk about how to do the work.
And what will come out is U and V.
Now, it's time to recommend a movie for you.
So what do we do?
So to recommend a movie for somebody,
after we've computed U and V, we'd look up that person
and get their U.
So we look up the person.
We get their U. And then we just compute the whole row.
We don't have to compute the whole matrix x,
but we compute the whole row.
And then we recommend the things that
come out with the highest ratings
that you haven't seen yet.
So that's how we would make a recommendation.

Good.
Other questions about the setup?
Mm-hm?
SUBJECT 2: What is the point of Vi
multiplied by lambda but the one with Ua is not?

LESLIE KAELBLING: Was that your question, too?
That was a typo.
Sorry.
It's not a typo in the notes.
It's just on the board.
Yep?
SUBJECT 3: Yeah.
That's what also I was going to ask.
Why do you use the same lambda with two of them?
LESLIE KAELBLING: Ah.
That's a good question.
Why do you use the same lambda for these two?
It's absolutely not necessary.
And if you were a Netflix engineer trying
to squeeze the last ounce of performance out of your thing,
you might actually try two different lambdas.
It's just-- oh, how do you pick lambda?
Let's practice that.
How do you pick lambda?
SUBJECT 3: Cross-validation.
LESLIE KAELBLING: Cross-validation.
And so it just means more searching around
in parameter space.
But it might be better.
Yep.
SUBJECT 2: So before, the minus Ua minus Vi
we can think of as the cheerfulness
of the rater or the goodness or badness of a movie.
Now, they're doing records?
Or are they still--
LESLIE KAELBLING: No.

Bu is a whole vector.
ba is your personal cheerfulness for person a.
SUBJECT 2: So there's no standard for bu and a?
LESLIE KAELBLING: Yeah.
So bu is n by 1.
And bv is m by 1.
So you can basically think of now we
have an additive offset for user per user,
one parameter per user additive offset and one parameter
per movie additive offset.
And then we have k parameters per user and k
per movie, which we take the dot product of,
that characterize the interaction
of the user and the movie.

Think of it that way.
And if we wanted to-- and some people
do-- you could imagine putting one more b in here,
just one more scalar b, which just kind of measures
the average number.
Basically, it would come out to be the average number of stars
across the whole thing.
And that might make the problem a little bit easier.
And it wouldn't change it too much.

Good?
If you are a person who has taken linear algebra recently
or ever, you might understand this
as a rank k approximation of y.
If that doesn't help you, then don't worry about it.
But if you understand what a rank k matrix is,
that's a rank k matrix.
It's assuming some kind of simple underlying
structure in the matrix.


### Lecture: Recommender systems - introduction


LESLIE KAELBING: So this is our problem now.
We have an optimization problem.
The optimization problem we have, sometimes it get gets
called matrix completion, right?
Because we have a matrix that's partially filled in.
We've made an assumption about the relationships
among the entries in the matrix.
And now we want to fill in the rest of it.
So we could think of it as matrix completion.
But the good news is, because as we understand
this is an optimization problem, we can just go to town on it
as an optimization problem.
There's kind of two ways to approach it.
And I'll talk about them both.

Actually, we'll play with both of them in the homework also.
OK, so approach number one is called
alternating least squares.

And it's actually an instance of a general idea that's
worth knowing about in optimization
called coordinate descent.

We were just doing optimization in two dimensions.
And I don't know.
I'm going to be really bad at drawing contours,
but I'll draw some contours anyway.
So we have some function we're trying
to optimize in two dimensions.

And we know about gradient descent, right?
In gradient descent, we just we pick a point,
and we try to go take a step in two dimensions, right?
We affect both coordinates at once.
And we try to find the gradient and take
a step in the direction of the gradient.
In coordinate ascent, we go one coordinate at a time.
So instead of trying to address both x1--
we'll call it x1 and x2--
we're going to take our point, and we're going to adjust it.
let's say in x1 first, and then in x2.
And then in x1.
And then in x2.
So the path of the optimization will go on the coordinates.
And usually the reason for doing this
is because if we give ourselves the problem of only changing
one coordinate, we pose ourselves an easier question.
And sometimes, as will be true in the problem that we look at,
we can actually find the optimum in that slice.
OK, so if we're going to do coordinate ascent,
it means that we're going to consider this.
Let's say we're going to fix x1, and we're going to adjust x2.

And if everything is nice, we might
be able to find the optimum along this slice, right?
So the optimum along the slice.
I don't know.
Let's say it's here, right?
Maybe that's the least value I can get if I
hold this coordinate constant.
So I say, cool, with x1 fixed, this is my best x2.
And now I go again in another dimension.
I fix x2, and I say, OK, along this line,
what's the best I can do?
And maybe this is it, right?
So now I fixed x2 to be that.
And then I say, oh, good.
Along x1, what's the best I can do?
So I come here and I say, oh, this is my best value.
And then I go this way.
And I don't know.
This is my best.
So I do something until hopefully I get
it end up in the [INAUDIBLE].
So that's coordinate ascent.
And you could do it in more than two dimensions.
And you can do it in groups of coordinates at a time.
But it's this idea that sometimes your optimization
problem isn't easy all at once.
So, for instance, this looks a little bit
like a linear regression problem.
And so we might be tempted-- we might say, oh, that looks
like linear regression problem.
Maybe I can just take the derivatives and set to 0.
It's going to turn out.
You can work at that with your pencil later.
But it won't work out nicely because we have
a product of parameters, right?
The user parameters and the visa parameters.
And so if you differentiate that,
you'll get a set of equations, but they
won't be convenient linear equations
that you can just solve, which is
what we did to get the analytical solution
to linear regression.
So we're not going to be able to do something that easy.
But it's going to turn out that if we organize our calculation
right, we can go in an alternating fashion.
And each of the subproblems is not so hard.
So we're going to find some nice substructure in this problem.
OK, does this coordinate this thing makes sense?
Fix one, hold the other one.
OK.
Good.
So then what we're going to do is observe that if we fix--
let me just be sure that I did it in the same way
I did [INAUDIBLE].
We're going to fix the V's.
OK, we're going to just start by fixing the V's.
If we fix V--
all the V's-- and BV--

so imagine that we just say, I'm going to fix all the V's.
I'm just going to randomly, let's say,
randomly initialize the V's and fix them.
If I fix those things, then finding the best
U and BU is a linear regression problem.

Actually, it's a lot of linear regressions.

Does that make sense?
So if you look at this equation, if I
were to treat the V's constants, which
is what I do during coordinate descent.
I treat the visa as if they were constants.
Then my only parameters are the U and the BU.
And they are only occurring-- well,
they're linear inside the quadratic.
And when I take the derivative, they'll
come out with only a linear dependence.
So this is like a linear regression problem, right?
If you think about it, that's like saying,
oh, imagine that I fix the space of features, right?
Again, you can think of V as kind of defining a feature
space for movies.
You could say, oh, I'm going to fix this feature
space for movies.
And now I just have to do a linear regression type
prediction to find each user's features.
So let's see how that goes.
OK.
So particular, I wrote that this is a linear regression.
In fact, there's going to be a separate problem for each user.


### Lecture: Recommender systems - introduction


LESLIE KAELBLING: So now, I want to find Ua to minimize.

So remember, I fixed the V's.
I'm trying to find the coefficients for the users.
And I'm actually-- so remember, the coefficients for the users
are up there.
There's one row for each user.
And I'm going to be able to actually formulate
this as m separate linear regression problems, because I
need to find your row.
And the way I'm going to find your row is
to look at the feature assignments
that you've given to movies and look at the movie features that
have been discovered and stored in the V.
So let's see how this goes.
So for a particular user a, we want to minimize.

And I'm going to do this--

I'm going to do this without the offset and the regular riser,
because it will just help us see the structure of what's
going on.

So let me write it, and then I'll talk about it.

This actually makes it really very clear.

So let's look at this.
This is just the problem of finding
your row of coefficients.
So all we're going to do now is look at, in particular,
the movies that you've rated.
So this is for you for each a.
Yeah.
For each a, we're going to pose a regression problem.
So for each user, we're going to pose a regression problem.
We're going to go and find the data that's about that user.
So that's going to be the ratings you have made.

We'll say, assume the V's are fixed.
Find the ratings of the movies you've made.
Maybe this is only four data points or six or 97,
depending on how many movies you watch.
And we're going to just make a little linear regression.
So the r's are the target values.
These are the values we would like to predict.
The V's are coefficients.
The V's here play the role of the x's
in our normal linear regression.
You can think of the problem here--
let me now actually write.
We're used to thinking about a linear regression
as looking like--

this is the way we wrote it.

Back in the day, when we did linear regression,
these were our targets.

These were the features of the inputs.

And these were the parameters that we were looking for.

And then we went ahead and solved it analytically.
So to map this problem onto that problem,
just to kind of be clear, what do we have?
So we have theta is equal to Ua.

And it's a k by 1 parameter vector.
That's what we're looking for.
Those are the free parameters in this problem.
T is the target.

And we'll say that they're m a by 1,
where m a is the number of movies actually rated by a.
That's how many tuples are in here.
How many entries are there in our data set
that involve user a actually making a rating?
So in here, we have the parameters we're looking for.
Here are the target values on our data.
And then W corresponds to--

and it's m a by k.

And so W, its rows are V-values.

And they're the V-values of the movies i
that were rated by the user.

So we're assuming these are all constant.
You can think of them as feature vectors, describing each movie.
Admittedly, at the beginning, these
will be terrible feature vectors.
At the beginning, they'll be random.
This is a little weird, but that's OK.
Iterative algorithms start out kind of odd and get better.
So feature vector for each movie,
dot product with these parameters,
which we're going to look for--
we want that to be close to the target values.

So that means for every single user,
we can just solve a little linear regression problem.
And that will move us to the best values of U,
given the V's that we have.

So we said if we started out with these crazy features V,
here's what the U's should be.
And then you can see by the symmetry of all this--
and I'm not going to write it on the board--
that if we fix these U's now--
so we solve these things.
And we get the best possible U's, one for each user.
And we fix it.
Then we can get the best products.
Then we can solve for the best V's.
And then we'll fix those and find the best U's and fix
these.

So that's the setup.
So this is a pretty good algorithm.
It works pretty well.
I will have you guys implement it and play with it.
And it's nice, because it uses a thing that we already
know how to do as a sub-step.

Questions about this basic setup?

It takes a little pondering, but you'll sort it out.
I want to talk about one more thing here and then
do something entertaining and then set you free.
So the one more thing--
we know a couple of tricks.
So analytically solving some sub-problems
is a trick we know.
And when you can do it, it can be very helpful.
Notice, too, that these problems are never very big, because--
well, certainly, the a problems are not very big.
The a problems are, how many movies have you rated?
And I don't know.
The most prolific movie raters probably only rate,
what, 1,000 movies.
I don't know.
Anyway, not a huge number--
the V problems involve how many people have rated this movie.
So those could get big.
And it may be that they actually sub-sample
the data in that case.
So they don't use it all.
I'm not exactly sure.


### Lecture: Recommender systems - introduction


LESLIE KAELBLING: Another trick is good old stochastic gradient
descent.
And so in stochastic gradient descent,
we take our objective function over there.

We compute the gradient, which is not too hard.
I'm not even going to bother writing it on the board time
because it's like some kind of gradient
we've been taking now a lot.

OK, remember the stochastic gradient descent works.
When you can think of your objective--
so your J of all your parameters-- in this case,
u and v--
as the sum over your data of something because
in stochastic gradient descent, you
say, oh, I'm going to compute the gradient.
And the gradient is going to have the form the gradients
with respect to u, the gradients with respect
to v. It's also going to have the form
that it's the sum over d of some gradient thing.
So that's what we need because then we
can just sample an element out of that
set that we're summing over.
As long as we sample it random, then we can sample,
and we can take a small step, according to this gradient,
the one that's inside this one.
So the only thing that we have to do to make this work
is mess a little bit with the regularizers
so that they show up inside the sum.

And it's not too hard to do.
I'll just write it down.
So to do this, this is just reorganizing.

So ua.vi plus b minus r.

OK, I'm going to put a big parentheses here.
So I'm still inside, and this lambda a
is going to be lambda over the number of movies
rated by a times ua vi.
.

So notice that this is going to be a summation,
and we're going to-- it's going to be a sum over that user A is
going to appear is going to occur inside here--
this number of times that movie was rated by user A.
So that's how many times this thing is going to user A
is going to occur in here.
That's the number of times that this particular regularizer
on that vector is going to show up.
And so if we just divide lambda by that,
that means it's the same as pulling it back
outside the sum.
So this is just a rewriting of that.

OK, so we just fiddle with the constants a little bit
to get the same thing.

And then we can do stochastic gradient descent on this.
You can repeat the gradient.
Now then you have to do is draw a rating out of your data set.
Take a little step, draw a rating out of the data sets,
take a step, and so on.
And that will also give you your v.


### Lecture: Recommender systems - introduction


LESLIE KAELBLING: So I wanted to do-- just
show you some RNN examples.
OK, so I went--
can you read that now?
Apple buttermilk waffles with apple bourbon sauce?
See that?
Ginger kombucha cocktail-- are you hungry now?
No, not really?
Bacon wrapped Gulf shrimp, a blue cheese with hot butter?
OK, so these are the names of recipes that I scraped off
of some website somewhere.
And then I trained a simple RNN using our RNN code
on these strings.
So the idea is that I want to learn a language model.
So remember, a language model is the case where
we wanted to feed in a string and basically predict
the next one.
So I trained a language model on recipes--
no, not recipes-- just on the titles
of dishes in this database.
And then in the little code I'm going to show you,
the way it works is it's going to try to--
so you can force in a prefix.
You can force in some characters-- so x1, x2, x3.
And then after that, we let it run.
So if I type a three character prefix,
it will force those three characters into the RNN.
Then it's going to see which character
is most likely prediction for the next one.
So I add that to the word and then
feed it back in as the next input,
take the most likely one, add it to the word,
feed it back in, and so on.
So this way, I generate completions of strings.
I'm not generating random ones.
I'm picking the most likely character after my sequence.
OK, so I did this a little bit.
Let's see, which is the one?
No.
[INAUDIBLE] too many tabs.
OK so I did a couple of examples already,
but I will take suggestions of a prefix.
These are some.
I put it in chalk and eraser.
But give me a word.
Give me a prefix.
SUBJECT 1: Pumpkin.
LESLIE KAELBLING: Pumpkin.
We'll try pum.
Pumpkin cheesecake chicken wings--
OK.
Wouldn't you like that?
But what's interesting is if you do pump-- oh no, we get that.
If we do P-U, we get--
oh, P, pasta with chicken wings.
If I do past, pasta with chicken wings.
Pesto, pest-- pesto salsa with mushroom sauce.
So how about that, right?
That's awesome.
This is character by character.
It's sort of locally consistent but also really stupid.
This is just-- so people do this all the time in [INAUDIBLE]..
OK, so there we go.
One more suggestion for recipes, and then we'll do one more.
And then we'll quit.
Prefix.
SUBJECT 2: Bacon.
LESLIE KAELBLING: Bacon-- but that's going to be in here.
Oh no, it's not.
Bacon, two [? faux ?] carrots.
Isn't that nice?
How about bacon chocolate?
See if we get-- or chocolate, chocolate bacon.
Chocolate-- just, yeah.

Cool-- chocolate bacon [INAUDIBLE] chicken wings.
Awesome.
OK, this one is--
I don't even remember.
OK, this is going to be--
so we're going to put bacon in here
and get out bacon [INAUDIBLE].
Choc-- "choc-a-pow."

Eraser?
Erasium-- oh, OK.
So I have two-- there are three other databases this could be.
I have heavy metal band names.
I have popular song names, and I have some [? pom. ?]
But you can probably guess.
Oh, TR-- Trap Them.
So this is the death metal bands.

Vile-- see, I type V-I and I get Vile, right?
If I type vili--
vili-- OK, I don't know.
OK, one more.
We're going to do this one.
This is-- this is popular songs, I think.
I don't know.
It didn't come out as well as I had hoped.
Maybe I should train it more.
Pop May-- we could try chocolate.
Chocolate?
[INAUDIBLE] the word.
I don't know.
See what love turns into.
Love Is-- word Is--
[INAUDIBLE] Anyway, I had fun with this.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/8b609c84fc452d77fd213069ca3504e7/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Appendices.pdf
- https://openlearninglibrary.mit.edu/assets/courseware/v1/d47d5e8f3dc7157f83382f72e103f8e4/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Recommender_systems.pdf
