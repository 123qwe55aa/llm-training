# Reinforcement learning

> Week 10: Reinforcement Learning · MIT 6.036 courseware archive

## Notes – Chapter 11: Reinforcement learning

Notes – Chapter 11: Reinforcement learning
You can sequence through the Reinforcement learning lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 11: Reinforcement learning
notes as a PDF file.

## Lecture: Introduction to reinforcement learning

Lecture: Introduction to reinforcement learning
Lecture: Introduction to reinforcement learning
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Introduction to reinforcement learning

Introduction to reinforcement learning
So far, all the learning problems we have looked at have been
supervised
: that is, for each training input [mathjaxinline]x^{(i)}[/mathjaxinline], we are told which value [mathjaxinline]y^{(i)}[/mathjaxinline] should be the output. A very different problem setting is
reinforcement learning
, in which the learning system is not directly told which outputs go with which inputs. Instead, there is an interaction of the form:  Learner observes
input
[mathjaxinline]s^{(i)}[/mathjaxinline] Learner generates
output
[mathjaxinline]a^{(i)}[/mathjaxinline] Learner observes
reward
[mathjaxinline]r^{(i)}[/mathjaxinline] Learner observes
input
[mathjaxinline]s^{(i+1)}[/mathjaxinline] Learner generates
output
[mathjaxinline]a^{(i+1)}[/mathjaxinline] Learner observes
reward
[mathjaxinline]r^{(i+1)}[/mathjaxinline] [mathjaxinline]\ldots[/mathjaxinline]  The learner is supposed to find a
policy
, mapping [mathjaxinline]s[/mathjaxinline] to [mathjaxinline]a[/mathjaxinline], that maximizes expected reward over time.
This problem setting is equivalent to an
online
supervised learning under the following assumptions:
The space of possible outputs is binary (e.g. [mathjaxinline]\{ +1, -1\}[/mathjaxinline]) and the space of possible rewards is binary (e.g. [mathjaxinline]\{ +1, -1\}[/mathjaxinline]);
[mathjaxinline]s^{(i)}[/mathjaxinline] is independent of all previous [mathjaxinline]s^{(j)}[/mathjaxinline] and [mathjaxinline]a^{(j)}[/mathjaxinline]; and
[mathjaxinline]r^{(i)}[/mathjaxinline] depends only on [mathjaxinline]s^{(i)}[/mathjaxinline] and [mathjaxinline]a^{(i)}[/mathjaxinline].
In this case, for any experience tuple [mathjaxinline](s^{(i)}, a^{(i)}, r^{(i)})[/mathjaxinline], we can generate a supervised training example, which is equal to [mathjaxinline](s^{(i)}, a^{(i)})[/mathjaxinline] if [mathjaxinline]r^{(i)} = +1[/mathjaxinline] and [mathjaxinline](s^{(i)}, -a^{(i)})[/mathjaxinline] otherwise.
Study Question:
What supervised-learning loss function would this objective correspond to?
Reinforcement learning is more interesting when these properties do not hold. When we relax assumption 1 above, we have the class of
bandit problems
, which we will discuss in section
. If we relax assumption 2, but assume that the environment that the agent is interacting with is an
mdp
, so that [mathjaxinline]s^{(i)}[/mathjaxinline] depends only on [mathjaxinline]s^{(i-1)}[/mathjaxinline] and [mathjaxinline]a^{(i-1)}[/mathjaxinline] then we are in the classical
reinforcement-learning
setting, which we discuss in section
. Weakening the assumptions further, for instance, not allowing the learner to observe the current state completely and correctly, makes the problem into a
partially observed MDP
(
pomdp
), which is substantially more difficult, and beyond the scope of this class.
Download this chapter as a PDF file
This page was last updated on Saturday November 16, 2019; 07:31:37 PM (revision f808f068e)

## Lecture: K-armed bandits

Lecture: K-armed bandits
Lecture: K-armed bandits
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Bandit problems

Bandit problems
A basic bandit problem is given by
A set of actions [mathjaxinline]\mathcal A[/mathjaxinline];
A set of reward values [mathjaxinline]\mathcal R[/mathjaxinline]; and
A probabilistic reward function [mathjaxinline]R: A \rightarrow \mathbb {R}[/mathjaxinline] where [mathjaxinline]R(a) = P(R \mid A = a)[/mathjaxinline] is a probability distribution over possible reward values in [mathjaxinline]\mathcal R[/mathjaxinline] conditioned on which action is selected.
The most typical bandit problem has [mathjaxinline]\mathcal R = \{ 0, 1\}[/mathjaxinline] and [mathjaxinline]\lvert \mathcal A \rvert = k[/mathjaxinline]. This is called a
[mathjaxinline]k[/mathjaxinline]-armed bandit problem
.
Why? Because in English slang, “one-armed bandit" is a name for a slot machine (an old-style gambling machine where you put a coin into a slot and then pull its arm to see if you get a payoff.) because it has one arm and takes your money! What we have here is a similar sort of machine, but with [mathjaxinline]k[/mathjaxinline] arms.
note
There is a lot of mathematical literature on optimal strategies for [mathjaxinline]k[/mathjaxinline]-armed bandit problems under various assumptions. The important question is usually one of
exploration versus exploitation
. Imagine that you have tried each action 10 times, and now you have an estimate [mathjaxinline]\hat{p}_ j[/mathjaxinline] for the expected value of [mathjaxinline]R(a_ j)[/mathjaxinline]. Which arm should you pick next? You could
The theory ultimately tells us that, the longer our horizon [mathjaxinline]H[/mathjaxinline] (or, similarly, closer to [mathjaxinline]1[/mathjaxinline] our discount factor), the more time we should spend exploring, so that we don't converge prematurely on a bad choice of action.
Study Question:
Why is it that “bad" luck during exploration is more dangerous than “good" luck? Imagine that there is an action that generates reward value 1 with probability 0.9, but the first three times you try it, it generates value 0. How might that cause difficulty? Why is this more dangerous than the situation when an action that generates reward value 1 with probability 0.1 actually generates reward 1 on the first three tries?
There is a setting of supervised learning, called
active learning
, where instead of being given a training set, the learner gets to select values of [mathjaxinline]x[/mathjaxinline] and the environment gives back a label [mathjaxinline]y[/mathjaxinline]; the problem of picking good [mathjaxinline]x[/mathjaxinline] values to query is interesting, but the problem of deriving a hypothesis from [mathjaxinline](x, y)[/mathjaxinline] pairs is the same as the supervised problem we have been studying.
note
Note that what makes this a very different kind of problem from the batch supervised learning setting is that:
The agent gets to influence what data it gets (selecting [mathjaxinline]a_ j[/mathjaxinline] gives it another sample from [mathjaxinline]r_ j[/mathjaxinline]), and
The agent is penalized for mistakes it makes while it is learning (if it is trying to maximize the expected sum of [mathjaxinline]r_ t[/mathjaxinline] it gets while behaving).
In a
contextual
bandit problem, you have multiple possible states, drawn from some set [mathjaxinline]\mathcal S[/mathjaxinline], and a separate bandit problem associated with each one.
Bandit problems will be an essential sub-component of reinforcement learning.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:32 PM (revision 4f166135)

## Lecture: Objectives of the reinforcement learning problem

Lecture: Objectives of the reinforcement learning problem
Lecture: Objectives of the reinforcement learning problem
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Model-based learning

Lecture: Model-based learning
Lecture: Model-based learning
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Policy search

Lecture: Policy search
Lecture: Policy search
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Q-learning

Lecture: Q-learning
Lecture: Q-learning
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Q-learning select-action strategies

Lecture: Q-learning select-action strategies
Lecture: Q-learning select-action strategies
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: Neural networks and Q-learning

Lecture: Neural networks and Q-learning
Lecture: Neural networks and Q-learning
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Sequential problems

Sequential problems
In the more typical (and difficult!) case, we can think of our learning agent interacting with an
mdp
, where it knows [mathjaxinline]\mathcal S[/mathjaxinline] and [mathjaxinline]\mathcal A[/mathjaxinline], but not [mathjaxinline]T(s,a,s')[/mathjaxinline] or [mathjaxinline]R(s,a)[/mathjaxinline]. The learner can interact with the environment by selecting actions. So, this is somewhat like a contextual bandit problem, but more complicated, because selecting an action influences not only what the immediate reward will be, but also what state the system ends up in at the next time step and, therefore, what additional rewards might be available in the future.
A
reinforcement-learning (
rl
) algorithm
is a kind of a policy that depends on the whole history of states, actions, and rewards and selects the next action to take. There are several different ways to measure the quality of an
rl
algorithm, including:
Ignoring the [mathjaxinline]r_ t[/mathjaxinline] values that it gets
while
learning, but consider how many interactions with the environment are required for it to learn a policy [mathjaxinline]\pi : \mathcal{S} \rightarrow \mathcal{A}[/mathjaxinline] that is nearly optimal.
Maximizing the expected discounted sum of total rewards while it is learning.
Most of the focus is on the first criterion, because the second one is very difficult. The first criterion is reasonable when the learning can take place somewhere safe (imagine a robot learning, inside the robot factory, where it can't hurt itself too badly) or in a simulated environment.
Approaches to reinforcement-learning differ significantly according to what kind of hypothesis or model they learn. In the following sections, we will consider several different approaches.
Model-based RL
The conceptually simplest approach to
rl
is to estimate [mathjaxinline]R[/mathjaxinline] and [mathjaxinline]T[/mathjaxinline] from the data we have gotten so far, and then use those estimates, together with an algorithm for solving
mdp
s (such as value iteration) to find a policy that is near-optimal given the current model estimates.
Assume that we have had some set of interactions with the environment, which can be characterized as a set of tuples of the form [mathjaxinline](s^{(t)}, a^{(t)}, r^{(t)}, s^{(t+1)}))[/mathjaxinline].
We can estimate [mathjaxinline]T(s,a,s')[/mathjaxinline] using a simple counting strategy,
[mathjax]\hat{T}(s,a,s') = \frac{\# (s,a,s') + 1}{\# (s,a) + \left| \mathcal{S}\right|}.[/mathjax]
Here, [mathjaxinline]\# (s, a, s')[/mathjaxinline] represents the number of times in our data set we have the situation where [mathjaxinline]s_ t = s, a_ t = a, s_{t+1} = s'[/mathjaxinline] and [mathjaxinline]\# (s, a)[/mathjaxinline] represents the number of times in our data set we have the situation where [mathjaxinline]s_ t = s, a_ t = a[/mathjaxinline].
Study Question:
Prove to yourself that [mathjaxinline]\# (s,a) = \sum _{s'} \# (s,a,s')[/mathjaxinline].
Adding 1 and [mathjaxinline]\left|\mathcal{S}\right|[/mathjaxinline] to the numerator and denominator, respectively, are a form of smoothing called the
Laplace correction
. It ensures that we never estimate that a probability is 0, and keeps us from dividing by 0. As the amount of data we gather increases, the influence of this correction fades away.
We also estimate the reward function [mathjaxinline]R(s,a)[/mathjaxinline]:
[mathjax]\hat{R}(s,a) = \frac{\sum r \mid s, a}{\# (s,a)}[/mathjax]
where
[mathjax]\sum r \mid s, a = \sum _{\{ t \mid s_ t = s, a_ t = a\} } r^{(t)}\; \; .[/mathjax]
This is just the average of the observed rewards for each [mathjaxinline]s, a[/mathjaxinline] pair.
We can now solve the
mdp
[mathjaxinline](\mathcal S, \mathcal A, \hat{T}, \hat{R})[/mathjaxinline] to find an optimal policy using value iteration, or use a finite-depth expecti-max search to find an action to take for a particular state.
This technique is effective for problems with small state and action spaces, where it is not too hard to get enough experience to estimate [mathjaxinline]T[/mathjaxinline] and [mathjaxinline]R[/mathjaxinline] well; but it is difficult to generalize this method to handle continuous (or very large discrete) state spaces, and is a topic of current research.
Policy search
A very different strategy is to search directly for a good policy, without first (or ever!) estimating the transition and reward models. The strategy here is to define a functional form [mathjaxinline]f(s;\theta ) = a[/mathjaxinline] for the policy, where [mathjaxinline]\theta[/mathjaxinline] represents the parameters we learn from experience. We choose [mathjaxinline]f[/mathjaxinline] to be differentiable, and often let [mathjaxinline]f(s;\theta ) = P(a)[/mathjaxinline], a probability distribution over our possible actions.
Now, we can train the policy parameters using gradient descent:
When [mathjaxinline]\theta[/mathjaxinline] has relatively low dimension, we can compute a numeric estimate of the gradient by running the policy multiple times for [mathjaxinline]\theta \pm \epsilon[/mathjaxinline], and computing the resulting rewards.
When [mathjaxinline]\theta[/mathjaxinline] has higher dimensions (e.g., it is a complicated neural network), there are more clever algorithms, e.g., one called
reinforce
, but they can often be difficult to get to work reliably.
Policy search is a good choice when the policy has a simple known form, but the model would be much more complicated to estimate.
Value function learning
The most popular class of algorithms learns neither explicit transition and reward models nor a direct policy, but instead concentrates on learning a value function. It is a topic of current research to describe exactly under what circumstances value-function-based approaches are best, and there are a growing number of methods that combine value functions, transition and reward models and policies into a complex learning algorithm in an attempt to combine the strengths of each approach.
We will study two variations on value-function learning, both of which estimate the [mathjaxinline]Q[/mathjaxinline] function.
Q-learning
This is the most typical way of performing reinforcement learning. Recall the value-iteration
The thing that most students seem to get confused about is when we do value iteration and when we do Q learning. Value iteration assumes you know [mathjaxinline]T[/mathjaxinline] and [mathjaxinline]R[/mathjaxinline] and just need to
compute
[mathjaxinline]Q[/mathjaxinline]. In [mathjaxinline]Q[/mathjaxinline] learning, we don't know or even directly estimate [mathjaxinline]T[/mathjaxinline] and [mathjaxinline]R[/mathjaxinline]: we estimate [mathjaxinline]Q[/mathjaxinline] directly from experience!
update:
[mathjax]Q(s,a) = R(s,a) + \gamma \sum _{s'} T(s,a,s')\max _{a'}Q(s',a')[/mathjax]
We will adapt this update to the
rl
scenario, where we do not know the transition function [mathjaxinline]T[/mathjaxinline] or reward function [mathjaxinline]R[/mathjaxinline].
Here, [mathjaxinline]\alpha[/mathjaxinline] represents the “learning rate," which needs to decay for convergence purposes, but in practice is often set to a constant.
Note that the update can be rewritten as
[mathjax]Q[s, a] \gets Q[s, a] - \alpha \left(Q[s,a] - (r + \gamma \max _{a'} Q[s',a'])\right)\, ,[/mathjax]
which looks something like
It is actually not a gradient update, but later, when we consider function approximation, we will treat it as if it were.
a gradient update!
This is often called
temporal difference
learning method, because we make an update based on the difference between the current estimated value of taking action [mathjaxinline]a[/mathjaxinline] in state [mathjaxinline]s[/mathjaxinline], which is [mathjaxinline]Q[s, a][/mathjaxinline], and the “one-step" sampled value of taking [mathjaxinline]a[/mathjaxinline] in [mathjaxinline]s[/mathjaxinline], which is [mathjaxinline]r + \gamma \max _{a'} Q[s',a'][/mathjaxinline].
You can see this method as a combination of two different iterative processes that we have already seen: the combination of an old estimate with a new sample using a running average with a learning rate [mathjaxinline]\alpha[/mathjaxinline], and the dynamic-programming update of a [mathjaxinline]Q[/mathjaxinline] value from value iteration.
Our algorithm above includes a procedure called
select_action
, which, given the current state [mathjaxinline]s[/mathjaxinline], has to decide which action to take. If the [mathjaxinline]Q[/mathjaxinline] value is estimated very accurately and the agent is behaving in the world, then generally we would want to choose the apparently optimal action [mathjaxinline]{\rm arg}\max _{a \in \mathcal A} Q(s,a)[/mathjaxinline]. But, during learning, the [mathjaxinline]Q[/mathjaxinline] value estimates won't be very good and exploration is important. However, exploring completely at random is also usually not the best strategy while learning, because it is good to focus your attention on the parts of the state space that are likely to be visited when executing a good policy (not a stupid one).
A typical action-selection strategy is the [mathjaxinline]\epsilon[/mathjaxinline]-greedy strategy:
with probability [mathjaxinline]1-\epsilon[/mathjaxinline], choose [mathjaxinline]{\rm arg}\max _{a \in \mathcal A} Q(s,a)[/mathjaxinline]
with probability [mathjaxinline]\epsilon[/mathjaxinline], choose the action [mathjaxinline]a \in \mathcal A[/mathjaxinline] uniformly at random
Q-learning has the surprising property that it is
guaranteed
to converge to the actual optimal [mathjaxinline]Q[/mathjaxinline] function under fairly weak conditions! Any exploration strategy is okay as long as it tries every action infinitely often on an infinite run (so that it doesn't converge prematurely to a bad action choice).
Q-learning can be very sample-inefficient: imagine a robot that has a choice between moving to the left and getting a reward of 1, then returning to its initial state, or moving to the right and walking down a 10-step hallway in order to get a reward of 1000, then returning to its initial state.
The first time the robot moves to the right and goes down the hallway, it will update the [mathjaxinline]Q[/mathjaxinline] value for the last state on the hallway to have a high value, but it won't yet understand that moving to the right was a good choice. The next time it moves down the hallway it updates the value of the state before the last one, and so on. After 10 trips down the hallway, it now can see that it is better to move to the right than to the left.
More concretely, consider the vector of Q values [mathjaxinline]Q(0:10, \text { right})[/mathjaxinline], representing the Q values for moving right at each of the positions [mathjaxinline]0, \ldots , 9[/mathjaxinline]. Then, for [mathjaxinline]\alpha =1[/mathjaxinline] and [mathjaxinline]\gamma = 0.9[/mathjaxinline],
[mathjax]Q(i, \text { right}) = R(i, \text { right}) + 0.9 \cdot \max _ a Q(i+1, a)[/mathjax]
Starting with Q values of 0,
[mathjax]Q^{(0)}(0:10, \text { right}) = \begin{bmatrix}  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0\end{bmatrix}[/mathjax]
We are violating our usual notational conventions here, and writing [mathjaxinline]Q^{(i)}[/mathjaxinline] to mean the Q value function that results after the robot runs all the way to the end of the hallway, when executing the policy that always moves to the right.
note
Since the only nonzero reward from moving right is [mathjaxinline]R(9, \text { right}) = 1000[/mathjaxinline], after our robot makes it down the hallway once, our new Q vector is
[mathjax]Q^{(1)}(0:10, \text { right}) = \begin{bmatrix}  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  1000 &  0\end{bmatrix}[/mathjax]
After making its way down the hallway again, [mathjaxinline]Q(8, \text { right}) = 0 + 0.9 \cdot Q(9, \text { right}) = 900[/mathjaxinline] updates:
[mathjax]Q^{(2)}(0:10, \text { right}) = \begin{bmatrix}  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 &  900 &  1000 &  0 \end{bmatrix}[/mathjax]
Similarly,
[mathjaxinline]\displaystyle  Q^{(3)}(0:10, \text { right})[/mathjaxinline]
[mathjaxinline]\displaystyle = \begin{bmatrix}  0 &  0 &  0 &  0 &  0 &  0 &  0 &  810 &  900 &  1000 &  0 \end{bmatrix}[/mathjaxinline]
[mathjaxinline]\displaystyle Q^{(4)}(0:10, \text { right})[/mathjaxinline]
[mathjaxinline]\displaystyle = \begin{bmatrix}  0 &  0 &  0 &  0 &  0 &  0 &  729 &  810 &  900 &  1000 &  0 \end{bmatrix}[/mathjaxinline]
[mathjaxinline]\displaystyle \vdotswithin {=}[/mathjaxinline]
[mathjaxinline]\displaystyle Q^{(10)}(0:10, \text { right})[/mathjaxinline]
[mathjaxinline]\displaystyle = \begin{bmatrix}  387.4 &  420.5 &  478.3 &  531.4 &  590.5 &  656.1 &  729 &  810 &  900 &  1000 &  0 \end{bmatrix},[/mathjaxinline]
and the robot finally sees the value of moving right from position 0.
We can see how this interacts with the exploration/exploitation dilemma: from the perspective of [mathjaxinline]s_0[/mathjaxinline], it will seem, for a long time, that getting the immediate reward of [mathjaxinline]1[/mathjaxinline] is a better idea, and it would be easy to converge on that as a strategy without exploring the long hallway sufficiently.
note
Study Question:
Determine the Q value functions that will result from updates due to the robot always executing the “move left" policy.
Function approximation
In our Q-learning algorithm above, we essentially keep track of each [mathjaxinline]Q[/mathjaxinline] value in a table, indexed by [mathjaxinline]s[/mathjaxinline] and [mathjaxinline]a[/mathjaxinline]. What do we do if [mathjaxinline]\mathcal{S}[/mathjaxinline] and/or [mathjaxinline]\mathcal{A}[/mathjaxinline] are large (or continuous)?
We can use a function approximator like a neural network to store Q values. For example, we could design a neural network that takes in inputs [mathjaxinline]s[/mathjaxinline] and [mathjaxinline]a[/mathjaxinline], and outputs [mathjaxinline]Q(s,a)[/mathjaxinline]. We can treat this as a regression problem, optimizing the squared Bellman error, with loss:
[mathjax]\left(Q(s,a) - (r + \gamma \max _{a'}Q(s',a'))\right)^2\; \; ,[/mathjax]
where [mathjaxinline]Q(s, a)[/mathjaxinline] is now the output of the neural network.
There are actually several different architectural choices for using a neural network to approximate [mathjaxinline]Q[/mathjaxinline] values:
One network for each action [mathjaxinline]a_ j[/mathjaxinline], that takes [mathjaxinline]s[/mathjaxinline] as input and produces [mathjaxinline]Q(s, a_ j)[/mathjaxinline] as output;
One single network that takes [mathjaxinline]s[/mathjaxinline] as input and produces a vector [mathjaxinline]Q(s, \cdot )[/mathjaxinline], consisting of the [mathjaxinline]Q[/mathjaxinline] values for each action; or
One single network that takes [mathjaxinline]s, a[/mathjaxinline] concatenated into a vector (if [mathjaxinline]a[/mathjaxinline] is discrete, we would probably use a one-hot encoding, unless it had some useful internal structure) and produces [mathjaxinline]Q(s, a)[/mathjaxinline] as output.
For continuous action spaces, it is increasingly popular to use a class of methods called
actor-critic
methods, which combine policy and value-function learning. We won't get into them in detail here, though.
note
The first two choices are only suitable for discrete (and not too big) action sets. The last choice can be applied for continuous actions, but then it is difficult to find [mathjaxinline]{\rm arg}\max _{\mathcal A} Q(s, a)[/mathjaxinline].
There are not many theoretical guarantees about Q-learning with function approximation and, indeed, it can sometimes be fairly unstable (learning to perform well for a while, and then getting suddenly worse, for example). But it has also had some significant successes.
One form of instability that we do know how to guard against is
catastrophic forgetting.
In standard supervised learning, we expect that the training [mathjaxinline]x[/mathjaxinline] values were drawn independently
And, in fact, we routinely shuffle their order in the data file, anyway.
from some distribution.
But when a learning agent, such as a robot, is moving through an environment, the sequence of states it encounters will
For example, it might spend 12 hours in a dark environment and then 12 in a light one.
be temporally correlated.
This can mean that while it is in the dark, the neural-network weight-updates will make the [mathjaxinline]Q[/mathjaxinline] function “forget" the value function for when it's light.
One way to handle this is to use [mathjaxinline]\emph{experience replay}[/mathjaxinline], where we save our [mathjaxinline](s,a,r,s')[/mathjaxinline] experiences in a
replay buffer
. Whenever we take a step in the world, we add the [mathjaxinline](s,a,r,s')[/mathjaxinline] to the replay buffer and use it to do a Q-learning update. Then we also randomly select some number of tuples from the replay buffer, and do Q-learning updates based on them, as well. In general it may help to keep a
sliding window
of just the 1000 most recent experiences in the replay buffer. (A larger buffer will be necessary for situations when the optimal policy might visit a large part of the state space, but we like to keep the buffer size small for memory reasons and also so that we don't focus on parts of the state space that are irrelevant for the optimal policy.) The idea is that it will help you propagate reward values through your state space more efficiently if you do these updates. You can see it as doing something like value iteration, but using samples of experience rather than a known model.
Fitted Q-learning
An alternative strategy for learning the [mathjaxinline]Q[/mathjaxinline] function that is somewhat more robust than the standard [mathjaxinline]Q[/mathjaxinline]-learning algorithm is a method called
fitted Q
.
Here, we alternate between using the policy induced by the current [mathjaxinline]Q[/mathjaxinline] function to gather a batch of data [mathjaxinline]\mathcal D_\text {new}[/mathjaxinline], adding it to our overall data set [mathjaxinline]\mathcal D[/mathjaxinline], and then using supervised neural-network training to learn a representation of the [mathjaxinline]Q[/mathjaxinline] value function on the whole data set. This method does not mix the dynamic-programming phase (computing new [mathjaxinline]Q[/mathjaxinline] values based on old ones) with the function approximation phase (training the neural network) and avoids catastrophic forgetting. The regression training in line 9 typically uses squared error as a loss function and would be trained until the fit is good (possibly measured on held-out data).
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:32 PM (revision 4f166135)

## Lecture: Reinforcement learning demos

Lecture: Reinforcement learning demos
Lecture: Reinforcement learning demos
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Video transcripts

### Lecture: Introduction to reinforcement learning


Today, we're going to talk about reinforcement learning.
Cool name already.
Seems good.
Reinforcement learning-- and the fundamental idea
here is that we're not going to tell
the learning algorithm which output goes with the input.
OK, so that seems like it might be kind of hard to do.
But let's see.
If I asked someone to play a game with me,
who wants to play a brief game with me?
There.
OK.
I have a number in my head between zero and five.
What do you think it is?
Three.
No.
Two.
No.
One.
Yes.
OK, that was reinforcement learning.
OK.
So that is to say, there is an answer to a question,
and I know what the answer is.
I'm the world.
I know the answer.
He made guesses, and when he made a bad guess, I said no.
And he made a good guess, I said yes.
OK.
Or you could think about training an animal, right?
You want them to do some behavior.
When they do the behavior you like, you give him a cookie
or pat him on the head.
When they do the behavior you don't like, you tell them no.
And the idea is that they should learn from that information
how to behave.
So the fundamental setup--
this is really a different setup,
and I get to draw my favorite little robot cartoon.
So I work with robots, and so I always
think of this as a robot interacting with the world.
Now, of course, it isn't necessarily for robots.
Reinforcement learning is-- people
use it to learn to play games.
They use it to learn to place ads, right?
So if you are surfing some website and they're showing ads
and they want to show you ads that you'll click on,
they try experiments.
So this is the thing that's deciding what ads to show you.
This is you.
So this system-- the robot or the thing that's showing you
ads--
can pick an action, and it can also observe.
Right now, we'll say it can observe the state
of the system, all right?
So in the case of my robot, for instance, it
might be the robot knows where it is in the map,
but it doesn't exactly know how its wheels work,
or it doesn't know exactly a transition model.
So it picks an action.
That action will change the state of the environment.
So it changes the state of the world,
meaning it moves the robot, or it changes
the state of the person that I'm showing ads to because they
get to see something different.
And also, the agent gets to see a special signal called
the reward--
oh, this chalk is not cooperating--
the reward or the reinforcement.
So the setup here is the agent picks an action that
changes the state of the world.
The agent gets to see the change in the state of the world,
and the agent also gets a reward signal,
which is going to be a number.
And the agent is supposed to choose actions
in a way that gets it lots of reward.
So that's the setup.

So we talked--
[INAUDIBLE] came and gave you guys a lecture last week.
And in particular, we talked about Markov decision
processes, right?
And so in a Markov decision process,
the idea is that there is a policy.
There's some kind of policy that maps states into actions.

And just as in MDPs, in reinforcement learning,
we're going to try to learn.
We're going to somehow try to find a policy--
so something that goes in the head of the robot.
So there's a little policy, pi, that
goes in the head of the robot, right, that takes in the states
and generates the actions.
And the robot is supposed to learn a policy that's good,
that gets it a lot of reward over time.
So that's the setup.
So we're going to work up to this by looking at a simpler
version of the problem first, and then we'll
tackle the whole thing.


### Lecture: Introduction to reinforcement learning


The simpler version of the problem
is one where there is really no state.
So it's all about an agent picking actions
and trying to learn about the world it interacts
with in order to get reward.
And it's funny, because this problem
goes under the name the K-armed bandit.
And there's a whole scholarly literature
in math and statistics on bandit problems.
And lots of people who work in this area even
don't the reason why they're called bandit problems
or what the heck is a K-armed bandit.
So first, I'm going to teach you, especially
if you're not a native English speaker, that in a casino,
there are gambling machines.
Well, I don't know, there are gambling machines.
They have a little handle.
You put in your dollar or something, you pull the thing,
things spin, and if they all come out to be the same,
money comes out the machine.
So the thing is, that most of the time, nothing
comes out of the machine.
Occasionally money comes out, usually not.
So they are called in slang, a one-armed bandit,
because there's an arm and it takes your money.
So English slang, one-armed bandit.
So now imagine that you had not one of these,
but you had K of them.
So we have machine, money comes out, we have K. Handle, OK.
And now let's also assume that this is not like a real casino.
We're going to assume the following thing.
We're going to assume first of all,
that you don't even have to put a dollar into play.
So just this is like a special room in the casino.
There's K of these machines.
And each one has a secret hidden probability
that it will payoff.
So what's going to happen when you
pull the arm is with probability PJ, if you pull arm J,
you get $1.
And otherwise nothing.
So that's how this is going to work.
OK, so now, this is a question, you have to think about it.
Imagine that I say OK, here, come into my casino.
They all look the same and the Ps are hidden.
You don't get to see the Ps.
If you could see the Ps, it wouldn't be too hard a problem.
I say, come into this casino, and you get to pull arms,
let's say, 100 times.

OK, so ponder for a minute what your strategy would be.
Operating under the assumption that you like money,
so you would rather have more dollars than less.
If not, then I don't know what you're doing in the casino
anyway.
So you like money.
You get to play 100 times.
What would you do?
Somebody want to say what would you do?

Nobody, somebody.
Yeah, what would you do?
Doesn't it depend on the number K?
If K's significantly greater than 100,
then I would like to pick.
And then wherever I got a $1 I'd keep trying with subset of $1.
OK, that's good.
OK, so that's a good observation.
If K is really big relative to 100,
then it seems like if you find anything that ever wins,
you should just stick to it.
Let's imagine K is 3.
Let's make K smaller than the horizon.
Yeah?
[INAUDIBLE] 33 times and [INAUDIBLE] probability?
OK, good, awesome.
So the thing is, OK, let's say, the suggestion was,
pull each arm 33 times.
So if I get to play for 100 times
and I have three machines, if I pull each one 33 times,
OK, you will have optimized an objective.
That strategy optimizes the objective of knowing the Ps.
But it doesn't really get at the objective
of getting a lot of money.
Right?
So how might you modify that strategy,
if what you really want to do is get a lot of dollars?
Yeah?
[INAUDIBLE] easier to like, basically pick machines
at random increase your probability of [INAUDIBLE]
a given machine if it paid off.
Right.
OK, so you could imagine a strategy that sort of randomly
picks, but where you biased your randomness according
to the payoffs that you've gotten so far.
So good.
So that's kind of, that's a strategy.
Another strategy which you can show is optimal, is roughly,
I think it is pull each one for square root of K times.
So kind of like that strategy.
And at that point, see which one looks like the best,
and then just do that one for the rest of the time.
If you're going to live an infinite life or a discounted
life, then there's more complicated strategies.
So anyway, so this is interesting.
And what it brings up, is this fundamental problem
of exploration versus exploitation.

So the idea here is, if you already
have pulled these a certain amount of time
and you've got some estimates of these probabilities,
then you could exploit your knowledge
by pulling the one that seems like the best.
But if you do that too soon, if you settle too soon on what
you think is the best option and you don't try the other ones,
then you don't explore.
And if you don't explore, then you
risk not discovering something that was actually really good.
OK.
So this is a dimension of learning
that we haven't come to at all.
Reinforcement learning is going to mix together three problems.
It's going to mix together normal machine learning
kind of problems, like how do I find
a hypothesis that fits my data.
It's going to mix in Markov decision process value
iteration type solution of things.
And it's going to mix in this exploration versus exploitation
problem.
Because the robot, the agent up there gets to pick the actions.
And when it's learning, it has an incentive
to pick actions to try to get a lot of reward,
and also to pick actions to try to learn
about how the world works.
So that's, we combine all these three things together.
So when I think about the robot interacting
with the environment now, the question is,
how does that relate to a bandit problem?
And so the way I think about it is, that we're really now
in the MDP casino.
So we're in the Markov decision process casino.
In the MDP casino, we have a bunch of rooms.
We have many rooms.

And each room looks like this.
So each room, let's say has K arms you can pull.
That's like K actions you can take.
But when I pull an arm, not only do I get potentially
a reward, a payoff, I get teleported
according to some distribution, to a different room
in the casino.
That's, a Markov decision process is like that.
There's a transition distribution
that says if I'm in this state and I take this action,
here's a distribution over where I end up.
And there's a reward distribution
that says how likely am I to get this reward in this state
if I take this action.
In the model that we looked at, the rewards are deterministic.
That's fine.
So we live in this MDP casino, but unlike an MDP,
where we talked about last week where we knew the rewards
and we knew the transition model, in reinforcement
learning, we assume that we don't know the rewards
and we don't know the transition.
That's the fundamental difference.
There will be seven Piazza posts later about,
should I use value iteration on this problem or Q-learning?
And the answer is, value iteration
is an algorithm you can use when you know the models.
But if you don't know the models,
then you have to do learning.


### Lecture: Introduction to reinforcement learning


Let's talk about the learning problem.
There's actually two different settings, two different kinds
of objectives, that people-- well, people think of many.
But there's two different ways to think
about the reinforcement learning problem.
And it's important to keep them straight.
So in one version of the problem, we ignore--

ignore is not quite the right word.
But I'll explain.

So we're going to ignore our values during learning.
And we're going to try to find the optimal policy as
quickly as possible.

We can call this the offline case.

So you might imagine-- for instance,
people do this for things like car driving or game playing.
You might imagine that you get free practice
time with this domain that you don't know very well.
So maybe you have a simulator for driving.
This is not a good position to take
if you're driving an actual car because actual r's,
like running off the road or crashing into things,
do matter to you.
But imagine that you were learning
to drive in a simulator or you're
learning to play games where the downside consequences are not
so bad.
Then one way to think about reinforcement learning
is I'm going to interact with the simulator
or with this other person that I'm playing with or whatever.
I'm going to practice in a padded room.
And while I do that, I'm going to try all kinds of things
and fail and succeed and whatever.
And at the end of that period of offline experience,
I'm going to hope to have found an optimal strategy.
So that's one way to think about the problem.
The other way to think about the problem is, while learning,
try to maximize some sum of r's.
So this is the online case.

And this second case is actually-- in some sense,
it's more sensible if you imagine that this is really
a robot or really an ad-serving server that's right now trying
to make money or not run into things--
and so that you care about its performance
while it's learning.

This is a much harder setting.
The math exists for understanding what it would
mean to do this optimally.
But computationally, it's really hard.
So we will mostly not worry about that.
Yep?
[INAUDIBLE]

Good.
So this is why "ignore" was a poor word choice.
By ignore, I mean don't penalize the learner.

So by ignore, I mean we're going to use them in our algorithm
to do estimation and to help think about what
would be the optimal policy.
But I'm assuming that I'm in a simulator or a padded room.
So it doesn't really hurt if something goes badly.
Yep?
[INAUDIBLE]

Yeah.
[INAUDIBLE] the values for the statement.
I'll talk about that.
In fact, we have--
those are three different strategies
for approaching the problem.
In the end, what matters is the policy.
But on the way to learning a policy,
we might learn some other things.
But what we want to have come out--
really, what we want to have come out is good behavior.

Other questions about the setup?

Good.
So we're going to just think about part one,
just about algorithms that you could run and would
find a good policy for you, but not worry
too much about bad behavior while we're learning.

So three strategies, three strategies--

so we'll-- first, we could learn-- so the question is,
what are we going to learn?
So that was a perfect setup, that question.
What are we going to learn?
The answer is we could learn models,
we could learn a policy, or we can learn a value function.

And people do-- these are really three different approaches
to the problem.
And people do all three of these.
And there's some, like, fistfights
at conferences about what's the best way to do it.
So the theory is not all the way worked out
about which is the better way to do it.
But it's good to understand all three.
And they're not too hard to understand.
Value functions is probably the most popular thing.
And it's what we'll spend most of our time on.
But I want to introduce these other things.


### Lecture: Introduction to reinforcement learning


So let's talk about model-based.
Our fundamental premise here in, really, all of this
is we're going to assume we're in an MDP.

We don't know it.
But we're in it.
And usually, we have to assume that we at least know
the action space because if you don't know what you can do,
then it's hard to explore and figure out how things work.
So we're going to assume we're in an MDP.
We can observe the state.
So the simplest form of model-based reinforcement
learning is imagine that we're-- we have discrete state space--

this piece of chalk is done--

discrete state and action spaces.

So in this case, we can estimate the transition model.
So we're going to estimate transition model.
We'll call it T hat because it's not the real T.
And we have the transition model--
took a state and an action and a next state.

So imagine that we have--
for right now, let's not worry about how
we got this experience.
But imagine that we've interacted
with this environment a fair amount.
As we interact with the environment,
we can gather up tuples that look like the state at time
T, the action at time T, the reward that we got at time T,
and the state that happened at time T plus 1.

Just as in supervised learning, our fundamental chunk of data
is an x-y pair.
In reinforcement learning, the fundamental chunk of data
is this.
I was in this state.
I chose this action.
I received this reward.
And the environment transitioned to this new state.
So that's like a chunk of experience.
So imagine that I have a whole set of these for--
because I've been interacting with the world.
And now I can use this data to estimate a model.
So how am I going to do that?
So this is like machine learning,
super easy machine learning-- so such easy machine learning,
we never really talked about it.
But now we'll just write something
down here, something-- and then we're going to make it better.

So remember that the s's and the a's are
drawn from a discrete set.
So you can ask, what is the-- and this
is a transition probability.
So this is supposed to be the probability that I end up
in state s prime given that I started in state s
and took action a.
So I can-- you can think of this as a big, old table indexed
by states and actions and next states.
And I'm just going to have to estimate a probability.
So the most naive probability estimate is this one.
I count in my data set the number of times that I have
seen s, a, s prime-- that is to say,
the number of tuples that look like--
that match where sT is this s, where aT is this a,
and where sT plus 1 is this s prime.
I just count how many times I was in state 9.
I went to the left.
And I ended up in state 12.
And then I count how many times I was in state 9
and went to the left.
And I divide.
And that's my probability estimate.
Yep?
Does s [INAUDIBLE]?

The s-- the input or the state from the tuple?
We're not estimating a--
we're not working on a policy right now.
We're estimating a probability.
So this is really meant to be the probability
that random variable sT plus 1 is equal to s prime given
that sT was s and aT was a.
That's the thing we're trying to estimate.
And so does s stand for state or is it for [INAUDIBLE]??
Good.
So the question is, does s stand for state or input?
We're assuming here that we get to observe
the state completely.
So by saying it's an MDP, what we're saying is
the input is the state.
Good.
There's a whole interesting variation
on this where you don't get to see the whole state.
But it could become super hard then.

Is this cool?
Does this estimator make you nervous at all?
Is there a situation in which you think
you might get bad answers?
Yeah?
[INAUDIBLE] one data point given [INAUDIBLE] 100% [INAUDIBLE]..
Right.
Good.
Imagine that you've only ever tried this once.
Well, worse, imagine that you've never tried it.
So you've never tried it.
You can't divide by 0.
We all know that's not so good.
So that we-- so not good to divide by 0.
So we should fix that.
And then there's the question of, well,
what if we only tried it once or just a couple times?
Imagine I go to the left.
And I get hit on the head.
Does that mean that I've just never, ever going to--
my estimate of that is just terrible, absolutely terrible,
or not possible or--

So there's a way to just very simply fix this up,
which has a glorious justification in theory.
And if you take 68 and 67, you'll learn the justification.
But for right now, I'm just going
to tell you, which is to do this.

So this magical bit here--
it's worth understanding a little bit.
It's called the Laplace correction.
There's one name for it.
Some people call it Laplace smoothing or Laplace
correction.
So imagine that there were two possible states, just
to understand a simple case.
So imagine there's two possible states that I could end up in.

So the size of the state space is 2.
Imagine that.
Now imagine I have no data.
What's my estimate of the probability
of going to state 1?

It's 1/2.
If I have no-- if I've never tried this, ever,
my probability distribution is going to be a half a half.
That seems nice.

And then if I try--
if I've tried it a whole bunch of times,
this is going to go away.
So it's a way-- it's a kind of regularization.
Totally think of it like regularization, where it says,
you know what?
If I don't have very much data, I'm
just going to assume that my distribution is uniform.
And as I get more and more data, it
will pull my distribution away from uniform.

So that's a nice trick to know even not
in reinforcement learning.
If you ever need to do a counting
estimate of a probability, you should do this.

So we know we estimated T hat.
We can estimate R so that the reward of taking--
so if the reward is deterministic, then it's easy.
Otherwise, you can let it be the sum of the rewards
that you've gotten when you were in state s
and took action a over the number of times that you've
taken state s and action a.
When you're estimating a numerical value, it's not--
there's not such an obviously beautiful little trick
like that.
But you could just let it be 0 if you haven't seen anything.

So now that means if we have a bunch of experience,
we could estimate T hat and R hat.
And now we can solve the MDP that's
made of the state space, the action space, R hat, and T hat.

And you guys know how to solve MDPs.
So you could run the value iteration algorithm, say.
And that would give you a value function.
And you know from the value function
that you could compute the policy.
So this is slightly roundabout, but totally
respectable and easy to understand--
says what we're going to do is estimate the model.
From the model, we can compute a value function,
the optimal value function.
And from that, we can compute the optimal policy.
Now, it won't really be the optimal policy
if you don't have too much data.
We can compute the optimal policy
for this estimated R and T. But you also then
would have to argue that I-- well, as I
get more and more data, this R and this T
will be closer to the true R and T.
And therefore, the optimal solution
to this approximate MDP will get to be
close to the optimal solution to the actual underlying MDP.
Yep?
So is this under [INAUDIBLE]?
[INAUDIBLE] used for what?
Offline learning.
Offline learning.
Well, actually, you can use it for online learning, too.
So this is a great question.
So I said, oh, there's the offline and the online case.
What I didn't discuss here at all
is how to gather this data set.
I said if you have this much-- this experience,
then this is how you could compute a value function.
And this is how you can compute a policy.
Really, the offline/online part is most about
how you decide which actions to take.

And that's the hard part.
So the question is, how would you
choose actions to take here?
You could act uniformly at random if you're offline.
If you're offline, you could act uniformly at random
because what the heck?
You're in a padded room-- doesn't hurt too much.
The fact is that even if you're in the offline case,
it's better not to act uniformly at random.
It's generally better to do some mixture
of random and the currently best-seeming policy.
So I'll talk about that in a little bit more detail
later on.
But that's a great question.
So in fact, whether you pursue one of these methods
is independent of whether you're in the offline
or the online case.
The offline or the online case really
is about how you're going to pick the actions that you use
to experience the world and to gather data
and whether you care too much about
whether the actions you pick give you the reward.
Yep?
Why [INAUDIBLE] correction for the R?
It's less obvious what to do because it's not a probability
distribution.
It's a real number.
R might be a million or minus 2.
[INAUDIBLE]
Right.
So I-- what I suggested is to just
say, well, if I-- you could just let it
be 0 if you've never observed it.
You have to pick something to do.
But there's not quite such a clear strategy.

And I should say this method, exactly
like this, only works in the discrete state and action case.
There are ways to extend it to the continuous state and action
case, where you, obviously, can't do counting anymore
if it's continuous.
But there are, for instance, variations on neural networks
that will try to estimate probability distributions
for you or you could use a--
some kind of a softmax type thing to learn a distribution.
But it becomes trickier.
But it's conceivable that you could
apply this in the continuous case, and people do.
So policy search-- turns out that it usually
gets applied when things are continuous, certainly when
the states are continuous, and sometimes when the actions are.


### Lecture: Introduction to reinforcement learning


Let's talk briefly about policy learning.
So some people call this policy search.
So in policy search, what we're going to do
is pick a parametric form.
We're not going to talk about models.
We're not going to talk about value functions.
We're just going to talk about policies.
So we're going to pick a parametric form of a policy.
So we might say, oh, my policy is going to be of a form
that the action is going to be some functional form that
takes in the state.
And it has some parameters.
So I might say, oh, this functional form--

maybe it's a PID controller.
If you don't know what that is, it doesn't really matter.
It's a kind of very simple form of a policy
that control theory people like.
And it's got a few-- couple parameters in it.
So in that case, the thetas would be the game parameters
in your controller.
So that's like a--
something used to--
I don't know-- manage your thermostat or the elevator
or something.
And if you had a problem where you were pretty sure that
was a good class of controller, that would be totally cool.
You weren't sure what the gains were.
You could learn the gains or it could
be some giant neural network.

And then the theta would be the weights of the neural network
or it could be something in between.

So we pick a parametric form.
And we have some parameters.
And now we want to adjust the parameters
so that this thing is good.
So let's think about doing gradient descent with respect
to some objective.
So we would need-- if we want to think about this thing, what
we're going to do is some form of gradient descent.

And our objective will be something
like the sum may be discounted, if we
want it to be discounted, of the rewards that we get given
that we're currently executing the policy that we get
with these parameters theta.
This dot is funny.
This is like the mathematician's lambda.
It's like given that we're executing this policy,
we're using the-- this f with the thetas fixed
to be these thetas.
So let's just see-- let's just think about a controller.
So the gains in the controller, couple simple
parameters-- we set them currently
to have some certain values.

And then if we executed that controller and the elevator
was making people throw up or whatever our reward is, we--
or it may be not getting to the floor--
and so we have some reward about whether we like
what the elevator is doing.
So we would sum up the rewards given that we're
executing that policy.
And that would be a measure of how good the thetas are.
Does that make sense?
Yeah?
[INAUDIBLE]
Summing over time.

And maybe you would also sum over multiple episodes.
You might say, oh, I'm going to run the elevator 100 times
using this policy, or you could think
of that as a long period of time or you could think about it
as several finite episodes.
Either way is fine.
But so yeah.
I'm going to run this--
I'm going to run the elevator for a while,
see how it performs.
And that's going to be a measure of how good it is.

So now the simplest method and one that's not stupid
if you don't have too many thetas
is just to do a numerical gradient estimate.
You guys already wrote the code for this.
Imagine theta is just a couple of game parameters.
You could try increasing this one a little bit,
decreasing a little bit, increasing that one
a little bit, decreasing that one a little bit,
see how they work out.
That gives you an estimate of the gradient.
You move theta to reduce the error.
Oh, J is good.
J is good.
Do we use J is good?
Yes.
So this is good.
So you want to go--
you want to move in the direction that will increase J.
Now, we were using J is bad.
J was cost.
Do you guys remember?
Yeah?
If J is bad, if we want to minimize this,
then we would put a negative sign out here.

So do you get the idea, policy search?
Discount factor-- you don't have to have one.
But if you do, it can be there.

Yep?
[INAUDIBLE]
The dot is like a computer scientist would
write lambda s, f of s theta.
I just mean to say the function that you get if you
pick the thetas to be theta--
so this is computer science notation for this.
And this is math person's notation for this.
I prefer this one.
But not everybody knows it.
I don't know.
Anyway--

So the idea of policy search--
when is policy search a good idea?
The intuition is that it's a good idea when
you have a pretty good idea what would
make a good form of a policy.
Maybe you know some parts of this f already pretty well.
But you don't know other parts.
It's also a good idea-- sometimes
you have an intuition in some problems
that the policy will be simple even though the value
function is complicated.
And a really nice example of that
is finding your way out of an arbitrary maze.
So you may know this secret trick,
which only works if your maze doesn't
have any internal islands--
that you could just keep your right hand on the wall.
And eventually, you'll find your way out of the maze.
So a policy that says, I am going to move you in such a way
that I can keep my right hand on the wall--
that's pretty compact.
That's a pretty simple policy, whereas the value function
might be really big and complicated.
So it's hard to know in advance whether your problem is such
that this is a good idea or not.
The other thing about policy search
is it can work when your problem is not an MDP.

So we're not really going to talk about this.
But it's OK even when not an MDP.
So for instance, the--
finding your way out of the maze is a good example.
If you can only see the local area around you in a maze,
that's not an MVP because you don't
get to see the whole state.
You don't get to see where you are.
So that-- if that's true, this method won't work.
And the value function methods that I'm about to talk about
won't work.
But a policy search might still work.

That's about what I'm going to say about policy search.
We're not going to really pursue it any further.
But it's good to just know that it's an idea.
And it's a thing you could do.

Value functions-- so we did models.
We did policies.
Now we're going to do value functions.
And again, this is by far the most popular strategy,
although now it's turning out that
some interesting combinations of these three approaches
are working even better.
But--


### Lecture: Introduction to reinforcement learning


We will talk about Q-learning.
So you already last week studied Q-functions.
And let me just write down the definition of the Q-function.

That is the math.
It's the definition of a Q-function for an MDP.
Now, to figure out the Q-function, first of all,
you have to know R. And you have to know T.
And so as I said over here, if we could estimate R and T,
then we could compute the Q-function.
And we would compute it using something like value iteration.
And you remember that the value iteration algorithm is awesome
because all you have to do is turn this equal sign
into an assignment statement.
And you get the value iteration algorithm.

The value iteration algorithm computes this Q-function.
And the Q-function has the property just
to remind us that the policy that you should take--
so the-- if you've learned the optimal Q-function,
the best action to take in state s
is arg max over the actions of Q star s, a.
So the good thing about the Q-function
is once you have it in your hands,
it's really easy to figure out what the optimal behavior is.

But what are we going to do?
We don't know R. And we don't know T. That's the premise.
We don't know R. We don't know T.
But we don't want to estimate them, either, because--
I don't know-- because we're lazy.
So what can we do that would be better?

And the basic idea--
I guess I should write the algorithm down.
Sure.
So we're going to initialize.
We're going to estimate the Q-function directly.
That's what we're going to do, estimate Q directly.
So we're going to initialize Q of s, a for all s's and a's.
Again, for right now, assume that things are discrete.
So we could just make a table.
We'll initialize it to 0.
As in value iteration, the fact is that you can initialize it
to something else.
And that would be OK, too.

And then we're going to loop.
We're going to let s--
we'll start out letting s be some initial state
or some state that we draw at random
or the state that's actually the state of the world.
And then we're going to loop.

We're going to say the action is going to--
I'm going to write down here a function which
we'll come back to.
So we're going to have some function that
picks an action given state s and our current estimates
of the Q-values.
So we'll come back to that one.
And then we're going to get--
we're going to observe R and s prime by executing.

We're going to assume-- so this execute means we're
that little robot up there.
We just-- the Q is in our head.
We observe the s.
The Q is in our head.
We're going to pick an a.
We're going to send it to the environment.
That's what execute means.
And then we get to observe the reward value in the next state.

And now it's time to do a learning step.
And the learning step will be this.
We're going to update our estimate of Q s,
a to be 1 minus alpha times Q of s, a.
This is-- looks like-- it's going
to look like a moving average.

That's Q-learning right there.
That's the whole algorithm.

So what is it?
Well, at the outermost level-- so this alpha
is a learning rate.
So alpha is some kind of learning rate usually
in the 0.1 or 0.01-ish kind of range-- so smallish number.
And you say, what did I just do?
I just found myself in state s.
And I chose to do action a.
And something happened.
And now I want to update my estimate of how good is it.
How good is it in the long run?
That's what this means.
How good is it in the long run for me
to take action a when I'm in state s
and to behave well after that?
So the outermost level-- this looks like a running average.
So I take 1 minus alpha times my old estimate plus alpha times
a new estimate.
So that's an averaging.
And then what is this new estimate?
Well, if you look at this expression
here and compare it to that expression up there,
you can think of it as a sample in a certain sense, a sample
from the distribution of values of what
might happen if you were to take this action in this state.
If R is deterministic, then this R is exactly equal to that R.
If R is not deterministic, then it's
a sample of what it's like to do R.
And then here, we were taking an expectation
over possible next states with respect
to the transition distribution.
Here, the world drew from us--
drew for us a sample s prime from this distribution.
This s prime is governed by T s, a, s prime.
We don't know it.
The world knows it.
It's built into the world.
And when we take action a in state s, we get a sample of s
prime.
So we say, oh, we got a sample of s prime.
So s prime is what my future is like.
It's a sample of what my future is going to be.
And then I go and I look and see, well,
for s prime, what would be the best action I could take?
And I discount it and add it into the R.
Does that make at least some kind of intuitive sense?
Yep?
[INAUDIBLE]
It is.
Ah, good point.
I didn't put this in the notes.
I should put it in there.
Yeah, good.

Yep?
[INAUDIBLE]

Right.
This seems like craziness.
I just initialized Q of s, a to be junk.
And now I'm using Q of s prime.
I'm using this, and it's junk, to update my value here.
So, well, we will be computing it
over and over and over again.
It's going to start out stupid, completely stupid and wrong.
Just actually, if that worries you here,
it should worry you just as much.
It's an important worry.
It should worry you.
It should worry you in value iteration,
also, really, because at least initially, these estimates
are not very good.
And then as we do more and more iterations,
they come closer and closer to the right value.
This is super-- the fact that it works is kind of a miracle.
So let me tell you something, a theorem.
A theorem is if you do select actions right,
and I'll talk about that in a minute--
if you do selection actions right,
this is guaranteed to converge to the optimal Q-value
function, and therefore give you the optimal policy.
So it's actually, I think, really
surprising because you have two kinds of iterations going on
at the same time.
There is one iteration, which is the running average iteration
that--
I'm averaging over these values.
So that's one thing happening.
And then there's this other iteration,
which is like the dynamic programming
iteration from value iteration.
But we're only getting samples.
So we're averaging over the samples at the same time
as we're doing a dynamic programming.
But these estimates you can prove get better and better
in expectation over time.
And these guys get better.
And because they get better, then these guys get better.
And we all get closer to the true value.
So it's super cool, super cool, Q-learning.
Yeah?
[INAUDIBLE]

Yeah.
You do-- in order for this to really be proved to converge,
you have to tune alpha.
And it has to decrease in the same way
that the stochastic gradient descent alpha has to decrease.
So it has to go down kind of fast, but not too fast.
Yeah.
Mm-hmm?
[INAUDIBLE]

Well, in a minute, we'll talk about how
to make it-- how to do it in a neural network context.
But just for right now, this is an assignment statement.
And so I have to be able to store a table.

So we will be able to put in an approximator.
Once we put in an approximator, all the theorems pretty much
go out the window, though.
So what we can prove is if we have discrete states
and actions and this Q-function is
stored in a table that's indexed by states and actions,
then it will coverage.
So we don't get into any trouble of messing states nearby
together in a way that we shouldn't or anything
like that.
You've got states.
We've got actions.
They have values.
We're going to learn them.
Yep.
Good.


### Lecture: Introduction to reinforcement learning


So how to pick actions.
You have to pick actions in a way
that they say is persistently exciting.
OK, we have to be persistently exciting.
What does that mean?
That means roughly, you have to visit all the state action
pairs infinitely often, if you're going
to have an infinite lifetime.
So that you can't like just decide,
as we did in the finite horizon bandit case,
you can't just decide, OK, I found the best way to do this.
I'm not going to try the other options anymore.
You have to keep trying.

A typical select action strategy.
So how do we select actions?

The one that like lots of people use, and we will use it
in class, even though it's not really the very best thing,
is something called the epsilon greedy strategy.

And it just says, so given some state and some parameter
epsilon, what we're going to do is with probability 1 minus
epsilon, we're going to return arg
max over the a's of my current q function, of s and a.
Right?
So this is the greedy part, because it says,
given what I believe right now, I'll do the thing that I think,
I'll take the action that I think is
going to be best in the state.
And with probability otherwise, we'll
just choose an action completely random.

So that's epsilon greedy.
It is persistently exciting, in the sense that it will,
if you don't make the epsilon get smaller,
then it will always keep trying.
You are allowed to make epsilon get smaller at some rate,
but you have to be careful about what that rate would be.
Now, this might or might not be the best thing
to do if you're trying to optimize your reward over time.
So it's complicated.
And how you set epsilon, if you're
in a place where exploration doesn't hurt too much,
you can set epsilon kind of high.

But if you set epsilon, to be 1, you could set epsilon to be 1.
If you set epsilon to be 1, this algorithm will still converge.
Because totally if you set epsilon to be 1,
you'll just always act at random,
and you'll see the sights.
But usually that doesn't make you
learn as quickly as possible.
And the reason is, there's some part of the state space
where you're going to hopefully spend your time.
Right?
Like imagine that you're a robot trying to learn to walk.
So if you do some joint torques at random,
you'll spend most of your time like not upright.
And that's a part of the space which
you could try to learn about.
But you would really probably not want
to learn about that part of the space a lot.
So it's better to try to spend some time doing things
that you think are going to work out.

OK, that's Q-learning.
I'm going to talk about something that's
not so good about it, and talk about how to connect it up
to neural networks, and then do demos.
OK, what is not so good about it?
So we have an example in the notes.
So imagine the robot, well, OK, let's see.
I'm going to make a world.
I'll actually make the one.

This is state minus 1, 0, 1, 2, 3, 4, 5, 6,
I'll keep going, 7, 8, 9, 10, oh, 8, 9, 10.
OK, here we go.
We're a robot.
And this is our home location.
So we start out here.
And we can move to the left or the right.
That's all we know, we can move to the left or the right.
And if we take any action from, well
let's see, if we move to the left from this state,
we'll come back to state 0.
And if we move to the right from the state,
we'll come back to state 0.
And our actions are deterministic.

And let's imagine that taking any action in,
let's see, how did we set it up?
Any action in this state gives us value plus 1?
Plus 1.
So when we take, when we go from here to here, we get plus 1.
And when do we go from here to here, we get plus 1,000.
Imagine that.

So the robot has to decide from its home location,
should go left or should it go right, basically.
But it has to decide between left and right and all
these states.

So now, imagine that we have gamma equal 0.9.
And imagine that we have alpha equal 1.
So the learning rate is going to be just
like 1, so we simplify matters.
OK.
So now, robot, let's imagine that it manages,
like if it's acting at random, until you've seen a random walk
work, it's really, you're amazed at how horrible it is.
So if the robot's randomly picking left and right,
it's just going to flail around a lot.
But eventually, eventually, eventually it
will walk from here to here and it will get 1,000,
and it will say, this is the awesomest thing ever,
this state.
If we initialize everything to 0.
All this random walking is just going to make things be 0.
Finally we do this, and we say, awesome.
That state is worth 1,000.
OK, good.
And then we end up back here.
And in the meantime, we mess around a little bit,
and we'll figure out this state is worth 1,
they're going left here is worth 1.

So then, I'm sitting here and I'm
saying, well should I go left?
I can get 1.
Should I go right?
Oh, it doesn't look too good.
So unless we explore, if we converge
too early, if we say oh, left looks good, we'll just go left.
But we really, if we explore more, if we explore more,
we're going to walk down this hallway again.
We walk down the hallway again and we're going to get here.

And we're going to take a transition.
And although the r is 0, the Q of s prime will be pretty good.
Like 1,000.
We'd be like, oh, awesome.
And this state's value will get updated to 900.
OK.
And then, and then, down here, and then we spend a lot of time
getting the 1s, because we're kind of short- sighted.
Then we embark on this journey and then
we go down the hallway again, all the way down to here.
And then this is going to get what?
810 and so on.
So plain old raw Q-learning is highly sample inefficient.

Just that algorithm, just straight up that algorithm,
we have to walk down this hallway 10 times.
And there's not an incentive to walk, right?
I mean, it's going to flail around and do random walky
stuff for a long time.
You'll have to get all the way through here 10 times before it
could realize that it would be better
to go this way than that way.
So that's like not a good thing.
So how do people counteract this problem?
They do something called experience replay.

So I should, OK.
Right.

This update is like really kind of cool and romantic.
There are dopamine neurons in the brain that
seem like they might do some kind of computation that
looks kind of like this.
So this is like, this is beautiful and simple and cool,
and for a long time, nobody was willing to kind of like add
warts onto this algorithm.
But it turns out that to make it actually work,
you need to add the warts.
You need to add some extra mechanisms
so that it's sort of useful.
And one thing to add is this experience replay.
Experience replay just means I should remember.

Where did I write?
SRS pairs.
I should remember whatever I'm doing.
I should remember these pieces of experience, which
is how I update this rule.
Right?
This rule is based on an s an a, and r and an s prime.
I should remember those in a buffer.
And you can imagine, people talk about dreaming
and hallucinating and whatever, but basically replaying
in your mind experience that you've had before.
And if you do that instead of actually physically
having to walk down the stupid hall 10 times,
you could at least think in your mind.
Replay the experience and propagate the value back.
So this makes a huge difference in
the actual practical effectiveness of Q-learning.

It's not as beautiful anymore, but it works way better.
And maybe brains do it.
Who knows?


### Lecture: Introduction to reinforcement learning


We're talking about neural networks.
So note that really, I would argue
we have three flavors of learning going on here,
in a sense.
And we're going to-- about to add a fourth.
So the three flavors of learning-- one
is averaging over samples.
One is this MDP kind of dynamic programming, where
you learn what the future values-- how
they affect your current value.
And one is this bandit problem thing,
which is not maybe learning exactly, but managing
your experience.
So to manage our experience, average over samples
and do dynamic programming.
That's what we're already doing.
Now we can add in one more thing, which is actually
the thing that we've spent all of our time
so far in this class on, which reinforcement learning people
call function approximation.
So by function approximation, what
I mean is to say that I could--
there's going to be three different strategies
for doing this.
I could put the state in here.
Let's say I'm going to do-- this is the simplest
one to think about.

Imagine that I have two actions.
You could say I am going to--
I don't want to enumerate my states.
I'm going to enumerate my actions.
It turns out it's a lot easier-- most
problem-- many problems you can more easily model
with discrete action set and continuous states,
like am I going to show you this data or am I not?
Am I going to turn left or right?
Sometimes, you have to worry about continuous actions, too.
But let's imagine we have discrete set of actions
and continuous set of states.
Then we could try to use something like a neural network
to approximate the Q-function.
So take in a state.
Give out the Q-value for a2.
Take in a state.
Give out the Q-value for a1.

So we could think about this as a function approximation.
So that would be a way then to not
have to store the Q-function in the table.
We could store the Q-function in a neural network.
So in order to see why this might be a good idea,
let me rewrite this update.
I'm just going to gather the terms
in a slightly different way.
So this is just algebra.
I can write that same update like this.

Does that seem familiar to you in any way?
What does it look like?
Just back up and defocus your eyes a little bit.
What does it look like?

It looks like that.
But it looks like something from a few weeks ago, too.
Yeah?
Gradient step.
It looks like a gradient step, kind of,
like I'm going to update my Q-function
to be the old value of the Q-function minus some step
size times some kind of difference between what
I generated and what I wished I had generated.
So for regression with squared error,
it looks a lot like this.

So basically, what people do to do
Q-learning with a neural network is to do this update.
Instead of thinking about it as an update here with a table,
we can say, I'm going to take a gradient step to adjust
the weights in my network.

And where the loss is the square difference between--

have to get all the parentheses.

So if my Q-function is really a neural network,
then what I want to do after every piece of experience
is update the weights in the neural network.
And I can treat the problem as a regression problem.
And I can say, you know what?
I would really like the value to be like this.

This is the value you're giving me.
This is the value I want.
This is a loss function.
If I differentiate it, I'll end up with a--
So now the thing is that although it
looks like a gradient step, for subtle mathematical reasons,
it's not exactly a gradient step.
So if you're a math person and you want to dig into this,
there's a bunch of literature.
And you'll find that this, for complicated reasons,
because these things are not really independent of one
another in the right way--
doing this is not precisely right.
But it is what everybody does.

So what this means is we can--
instead of storing the Q-function in the table,
we can store it in a neural network or some other kind
of supervised learning method because this kind
of looks like a supervised learning update now.
But you have to think of it as a stochastic gradient step
because this is just a sample of what that Q-value should be.
So just as when we do stochastic gradients
and we have the big objective function,
but then we just take updates based on single points,
here we have a big objective, which
is to get this really right in expectation.
But we're going to make updates based on sample points.

There is more detail in the notes, which I'm not
going to go into right now, about some slightly different
ways to organize the Q's and about something
called fitted Q which is a little more reliable.

But you'll get to implement that and play around with it.
And this is the fundamental idea.


### Lecture: Introduction to reinforcement learning


So first of all, I'm going to quickly show you
some examples of MDPs, just solving MDPs,
because it's kind of fun.
Imagine that you have a robot that's
wandering around in the world.
So this is like a grid, discrete states, discrete actions.
The robot can go north, south, east, west.
If it tries to go north, with probability of 0.5,
it goes exactly where it's supposed to.
And with the rest of the probability,
it goes somewhere nearby.
And in this domain, there are two states
that have very high reward, and everything else
has reward zero.
So I'm going to do value iteration.
And I'm going to show-- so this is a picture of the value
function.
So right now, it's initialized to be zero.
And if I do one iteration of value iteration,
I find that there are some states that
have really high value.
And the rest of them I haven't computed the value for yet.
If I do another iteration of value iteration--
so again, this is the case where we know the model--
then I am immediately able-- in some sense,
you can think of it as propagating the reward out
from the places where I get it.
So I know that the value for these other states
is pretty good, because I can get
to the tasty state with some reasonably
high probability in one step.
So as I go, [INAUDIBLE].
And the arrows show the optimal action.
So as the value function gets better,
the optimal policy starts to converge to something
reasonable.
And basically, at this point, it's done.
And it says I'm going to go toward the best place.
So that's value iteration.
Let me show you another one.
Control-P, Control-P, test grid walls.

I want to-- because I'm going to show
you learning in a minute on these same domains.

So here's one.
Looks the same, but it turns out that this domain
has obstacles in it.
You can't see them yet.
Same thing, one iteration.
Two iterations, looks a little funny.

So in fact, what's going on here is that there
are some walls in this world.
It's more like a maze.
And so the robot can't walk through the walls.
But the value kind of propagates backward,
and it computes the optimal policy as it goes.
So there's that.
One more.
OK, looks the same to start with.
Now, the thing is, this one has one place where you get reward.
And it has a vortex, which means if you ever fall in there,
you can never get out again.
And I set the noise to be-- oops.
I set the noise to be really high.
So now it only goes where you tell it to
with probability 0.2.
And there's a lot more uncertainty about where
it's going to end up.
OK.
So so far, it likes the good state.
And that seems good.
Now, where's the vortex?
Right, it's that purple state, because if you fall in there,
you're never going to get any more reward ever again.
So that's an unhappy state.
So everything is kind of cool.
But watch the neighborhood of the purple state.
So what we're figuring out is not only
do we not like the vortex, we don't
like the states near the vortex, because we're not
very good at controlling our own motion.
And if we're near there, we might fall in.
So now it's decided that instead of going straight up,
it would rather go out, to avoid it.
OK.
So value iteration's awesome.
So there's that.
Now let's do Q-learning.
So, test.
OK, so here is a super small example.
And now I'm doing Q-learning.
It's got the same kind of underlying dynamics.
And there's one attractive location, here.
But the robot doesn't know anything.
It doesn't know the transition model.
It doesn't know the reward model.
So I'm going to just hit Return for each step.
Now, what's going on here?
It's pretty exciting, right?
What's going on is that the robot is bumbling around,
and is getting zero reward, and is not
updating the value function.
And I hope, for the sake of the demo,
that pretty soon it will bumble onto the--
I should have set the random seed.
OK, poor guy.
He's wandering in the dark.
Oh, man.
Don't we feel bad for this robot?

Come on!
We're going to run out of time before--
this tells you something about what Q-learning is like.
Right, like you can't learn--
oh, yay!
OK, so we happened on the tasty spot, once.

But we're going to have to get there again.
Oh, we got there again!
Yay!

It's kind of painful.
It's learning some stuff, slowly.
OK, that was painful.
I'm going to do this again.
I knew it was going to be like that.
I mean, not that bad.
I'm going to do it again.
But now I'm only going to show you
what happens every, let's say, 20 iterations,
so we don't have to carriage return so much.
Oh, wait, go away.

OK, so here we go.
Now, this is now I'm showing you every 20 iterations.
Oh, this one got lucky, too, earlier.
So every 20 iterations, I'm drawing the value function.
So it's slowly estimating the Q function, slowly figuring out
what it should do in life.
So that's pretty good.
I'll show you one more, and then we'll be done.
Test, grid, walls, learn.

So here's one more big example.
It's that same domain with the walls,
but the robot doesn't know the transition model,
and it doesn't know where the rewards are.
And now I'm drawing every 10,000 iterations.
After 10,000 iterations, it hasn't found the thing.
Another 10,000, nope.
Yay, we found the goal!
OK, and so there's Q-learning in action.
If somebody wants to help me with demos,
they can add experience replay to this,
and it will be much more attractive.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/153f87d9a11295896ffa8215253bf354/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Reinforcement_learning.pdf
