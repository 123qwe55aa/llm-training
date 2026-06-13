# Sequential models

> Week 9: State Machines and Markov Decision Processes · MIT 6.036 courseware archive

## Notes – Chapter 10: Sequential models

Notes – Chapter 10: Sequential models
You can sequence through the Sequential Models lecture video and note segments (go to Next page).
You can also (or alternatively) download the
Chapter 10: Sequential models
notes as a PDF file.

## Lecture: Sequential models - state machines

Lecture: Sequential models - state machines
Lecture: Sequential models - state machines
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Introduction to sequential models

Introduction to sequential models
So far, we have limited our attention to domains in which each output [mathjaxinline]y[/mathjaxinline] is assumed to have been generated as a function of an associated input [mathjaxinline]x[/mathjaxinline], and our hypotheses have been “pure" functions, in which the output depends only on the input (and the parameters we have learned that govern the function's behavior). In the next few weeks, we are going to consider cases in which our models need to go beyond functions.
In
recurrent neural networks
, the hypothesis that we learn is not a function of a single input, but of the whole sequence of inputs that the predictor has received.
In
reinforcement learning
, the hypothesis is either a
model
of a domain (such as a game) as a recurrent system or a
policy
which is a pure function, but whose loss is determined by the ways in which the policy interacts with the domain over time.
Before we engage with those forms of learning, we will study models of sequential or recurrent systems that underlie the learning methods.
Download this chapter as a PDF file
This page was last updated on Thursday December 12, 2019; 09:33:57 PM (revision 4b592d7d7)

## Lecture: SM - State machine as a transducer

Lecture: SM - State machine as a transducer
Lecture: SM - State machine as a transducer
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: SM - Towards recurrent neural networks

Lecture: SM - Towards recurrent neural networks
Lecture: SM - Towards recurrent neural networks
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## State machines

State machines
A
This is such a pervasive idea that it has been given many names in many subareas of computer science, control theory, physics, etc., including:
automaton
,
transducer
,
dynamical system
,
system
, etc.
state machine
is a description of a process (computational, physical, economic) in terms of its potential sequences of
states
.
The
state
of a system is defined to be all you would need to know about the system to predict its future trajectories as well as possible. It could be the position and velocity of an object or the locations of your pieces on a game board, or the current traffic densities on a highway network.
Formally, we define a
state machine
There are a huge number of major and minor variations on the idea of a state machine. We'll just work with one specific one in this section and another one in the next, but don't worry if you see other variations out in the world!
as
[mathjaxinline](\mathcal{S}, \mathcal{X}, \mathcal{Y}, s_0, f, g)[/mathjaxinline] where
[mathjaxinline]\mathcal{S}[/mathjaxinline] is a finite or infinite set of possible states;
[mathjaxinline]\mathcal{X}[/mathjaxinline] is a finite or infinite set of possible inputs;
[mathjaxinline]\mathcal{Y}[/mathjaxinline] is a finite or infinite set of possible outputs;
[mathjaxinline]s_0 \in \mathcal{S}[/mathjaxinline] is the initial state of the machine;
[mathjaxinline]f: \mathcal{S} \times \mathcal{X} \rightarrow \mathcal{S}[/mathjaxinline] is a
transition function
, which takes an input and a previous state and produces a next state;
[mathjaxinline]g: \mathcal{S} \rightarrow \mathcal{Y}[/mathjaxinline] is an
output function
, which takes a state and produces an output.
The basic operation of the state
In some cases, we will pick a starting state from a set or distribution.
machine is to
start with state [mathjaxinline]s_0[/mathjaxinline], then iteratively compute:
[mathjaxinline]\displaystyle  s_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = f(s_{t - 1}, x_ t)[/mathjaxinline]
[mathjaxinline]\displaystyle y_ t[/mathjaxinline]
[mathjaxinline]\displaystyle  = g(s_ t)[/mathjaxinline]
The diagram below illustrates this process. Note that the “feedback" connection of [mathjaxinline]s_ t[/mathjaxinline] back into [mathjaxinline]f[/mathjaxinline] has to be buffered or delayed by one time step—-otherwise what it is computing would not generally be well defined.
block = [draw, fill=blue!20, rectangle, minimum height=3em, minimum width=3em] sum = [draw, fill=blue!20, circle, node distance=1cm] input = [coordinate] output = [coordinate] pinstyle = [pin edge=to-,thin,black]
So, given a sequence of inputs [mathjaxinline]x_1, x_2, \dots[/mathjaxinline] the machine generates a sequence of outputs
[mathjax]\underbrace{g(f(x_1, s_0))}_{y_1}, \underbrace{g(f(x_2, f(x_1, s_0)))}_{y_2}, \dots \; \; .[/mathjax]
We sometimes say that the machine
transduces
sequence [mathjaxinline]x[/mathjaxinline] into sequence [mathjaxinline]y[/mathjaxinline]. The output at time [mathjaxinline]t[/mathjaxinline] can have dependence on inputs from steps [mathjaxinline]1[/mathjaxinline] to [mathjaxinline]t[/mathjaxinline].
One common form is
finite state machines
, in which [mathjaxinline]\mathcal S[/mathjaxinline], [mathjaxinline]\mathcal X[/mathjaxinline], and [mathjaxinline]\mathcal Y[/mathjaxinline] are all finite sets. They are often described using
state transition diagrams
such as the one below, in which nodes stand for states and arcs indicate transitions. Nodes are labeled by which output they generate and arcs are labeled by which input
All computers can be described, at the digital level, as finite state machines. Big, but finite!
causes the transition.
One can verify that the state machine below reads binary strings and determines the parity of the number of zeros in the given string. Check for yourself that all inputted binary strings end in state [mathjaxinline]S_1[/mathjaxinline] if and only if they contain an even number of zeros.
Another common structure that is simple but powerful and used in signal processing and control is
linear time-invariant (LTI) systems
. In this case, [mathjaxinline]\mathcal S = \mathbb {R}^ m[/mathjaxinline], [mathjaxinline]\mathcal X = \mathbb {R}^ l[/mathjaxinline] and [mathjaxinline]\mathcal Y = \mathbb {R}^ n[/mathjaxinline], and [mathjaxinline]f[/mathjaxinline] and [mathjaxinline]g[/mathjaxinline] are linear functions of their inputs. In discrete time, they can be defined by a linear difference equation, like
[mathjax]y[t] = 3y[t - 1] + 6y[t - 2] + 5x[t] + 3x[t - 2] \; \; ,[/mathjax]
(where [mathjaxinline]y[t][/mathjaxinline] is [mathjaxinline]y[/mathjaxinline] at time [mathjaxinline]t[/mathjaxinline]) and can be implemented using state to store relevant previous input and output information.
We will study
recurrent neural networks
which are a lot like a non-linear version of an LTI system, with transition and output functions
[mathjaxinline]\displaystyle  f(s, x)[/mathjaxinline]
[mathjaxinline]\displaystyle  = f_1(W^{sx}x + W^{ss}s + W^{ss}_0)[/mathjaxinline]
[mathjaxinline]\displaystyle g(s)[/mathjaxinline]
[mathjaxinline]\displaystyle  = f_2(W^0s + W^0_0)[/mathjaxinline]
defined by weight matrices
[mathjaxinline]\displaystyle  W^{sx}[/mathjaxinline]
[mathjaxinline]\displaystyle : m \times \ell[/mathjaxinline]
[mathjaxinline]\displaystyle W^{ss}[/mathjaxinline]
[mathjaxinline]\displaystyle : m \times m[/mathjaxinline]
[mathjaxinline]\displaystyle W^{ss}_0[/mathjaxinline]
[mathjaxinline]\displaystyle : m \times 1[/mathjaxinline]
[mathjaxinline]\displaystyle W^{0}[/mathjaxinline]
[mathjaxinline]\displaystyle : n \times m[/mathjaxinline]
[mathjaxinline]\displaystyle W^{0}_0[/mathjaxinline]
[mathjaxinline]\displaystyle : n \times 1[/mathjaxinline]
and activation functions [mathjaxinline]f_1[/mathjaxinline] and [mathjaxinline]f_2[/mathjaxinline]. We will see that it's actually possible to learn weight values for a recurrent neural network using gradient descent.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:23 PM (revision 4f166135)

## Lecture: SM - Markov decision processes - states and actions

Lecture: SM - Markov decision processes - states and actions
Lecture: SM - Markov decision processes - states and actions
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: SM - MDP - the transition function

Lecture: SM - MDP - the transition function
Lecture: SM - MDP - the transition function
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: SM - MDP - the reward and policy functions

Lecture: SM - MDP - the reward and policy functions
Lecture: SM - MDP - the reward and policy functions
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: SM - MDP - finite horizon and the value function

Lecture: SM - MDP - finite horizon and the value function
Lecture: SM - MDP - finite horizon and the value function
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: SM - MDP - computing the value function

Lecture: SM - MDP - computing the value function
Lecture: SM - MDP - computing the value function
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: SM - MDP - finding an optimal policy

Lecture: SM - MDP - finding an optimal policy
Lecture: SM - MDP - finding an optimal policy
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Lecture: SM - MDP - infinite-horizons and the value iteration algorithm

Lecture: SM - MDP - infinite-horizons and the value iteration algorithm
Lecture: SM - MDP - infinite-horizons and the value iteration algorithm
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Markov decision processes

Markov decision processes
A
Markov decision process
(
mdp
) is a variation on a state machine in which:
The transition function is
stochastic
,
Recall that stochastic is another word for
probabilistic
; we don't say “random" because that can be interpreted in two ways, both of which are incorrect. We don't pick the transition function itself at random from a distribution. The transition function doesn't pick its output
uniformly
at random.
note
meaning that it defines a probability distribution over the next state given the previous state and input, but each time it is evaluated it draws a new state from that distribution.
The output is equal to the state (that is [mathjaxinline]g[/mathjaxinline] is the identity function).
There is an interesting variation on
mdp
s, called a
partially observable
mdp
, in which the output is also drawn from a distribution depending on the state.
note
Some states (or state-action pairs) are more desirable than others.
An
mdp
can be used to model interaction with an outside “world," such as a single-player
And there is an interesting, direct extension to two-player zero-sum games, such as Chess and Go.
game.
We will focus on the case in which [mathjaxinline]\mathcal S[/mathjaxinline] and [mathjaxinline]\mathcal X[/mathjaxinline] are finite, and will call the input set [mathjaxinline]\mathcal A[/mathjaxinline] for
actions
(rather than [mathjaxinline]\mathcal X[/mathjaxinline]). The idea is that an agent (a robot or a game-player) can model its environment as an
mdp
and try to choose actions that will drive the process into states that have high scores.
Formally, an MDP is [mathjaxinline]\langle \mathcal S, \mathcal A, T, R, \gamma \rangle[/mathjaxinline] where:
[mathjaxinline]T : \mathcal S \times \mathcal A \times \mathcal S \rightarrow \mathbb {R}[/mathjaxinline] is a
transition model
, where
[mathjax]T(s, a, s') = P(S_ t = s'|S_{t - 1} = s, A_{t - 1} = a)\; \; ,[/mathjax]
specifying a conditional probability distribution;
The notation here uses capital letters, like [mathjaxinline]S[/mathjaxinline], to stand for random variables and small letters to stand for concrete values. So [mathjaxinline]S_ t[/mathjaxinline] here is a random variable that can take on elements of [mathjaxinline]\mathcal S[/mathjaxinline] as values.
note
[mathjaxinline]R: \mathcal S \times \mathcal A \rightarrow \mathbb {R}[/mathjaxinline] is a reward function, where [mathjaxinline]R(s, a)[/mathjaxinline] specifies how desirable it is to be in state [mathjaxinline]s[/mathjaxinline] and take action [mathjaxinline]a[/mathjaxinline]; and
[mathjaxinline]\gamma \in [0, 1][/mathjaxinline] is a
discount factor
, which we'll discuss in section
.
A
policy
is a function [mathjaxinline]\pi : \mathcal S \rightarrow \mathcal A[/mathjaxinline] that specifies what action to take in each state.
Finite-horizon solutions
Given an
mdp
, our goal is typically to find a policy that is optimal in the sense that it gets as much total reward as possible, in expectation over the stochastic transitions that the domain makes. In this section, we will consider the case where there is a finite
horizon
[mathjaxinline]H[/mathjaxinline], indicating the total number of steps of interaction that the agent will have with the
mdp
.
Evaluating a given policy
Before we can talk about how to find a good policy, we have to specify a measure of the goodness of a policy. We will do so by defining for a given
mdp
policy [mathjaxinline]\pi[/mathjaxinline] and horizon [mathjaxinline]h[/mathjaxinline], the “horizon [mathjaxinline]h[/mathjaxinline]
value
" of a state, [mathjaxinline]V^{h}_\pi (s)[/mathjaxinline]. We do this by induction on the horizon, which is the
number of steps left to go
.
The base case is when there are no steps remaining, in which case, no matter what state we're in, the value is 0, so
[mathjax]V^0_{\pi }(s) = 0\; \; .[/mathjax]
Then, the value of a policy in state [mathjaxinline]s[/mathjaxinline] at horizon [mathjaxinline]h + 1[/mathjaxinline] is equal to the reward it will get in state [mathjaxinline]s[/mathjaxinline] plus the next state's expected horizon [mathjaxinline]h[/mathjaxinline] value. So, starting with horizons 1 and 2, and then moving to the general case, we have:
[mathjaxinline]\displaystyle  V^1_{\pi }(s)[/mathjaxinline]
[mathjaxinline]\displaystyle = R(s, \pi (s)) + 0[/mathjaxinline]
[mathjaxinline]\displaystyle V^2_{\pi }(s)[/mathjaxinline]
[mathjaxinline]\displaystyle = R(s, \pi (s)) + \sum _{s'}T(s, \pi (s), s') \cdot R(s', \pi (s'))[/mathjaxinline]
[mathjaxinline]\displaystyle \vdots[/mathjaxinline]
[mathjaxinline]\displaystyle V^ h_{\pi }(s)[/mathjaxinline]
[mathjaxinline]\displaystyle = R(s, \pi (s)) + \sum _{s'}T(s, \pi (s), s') \cdot V^{h - 1}_{\pi }(s')[/mathjaxinline]
The sum over [mathjaxinline]s'[/mathjaxinline] is an
expected value
: it considers all possible next states [mathjaxinline]s'[/mathjaxinline], and computes an average of their [mathjaxinline](h-1)[/mathjaxinline]-horizon values, weighted by the probability that the transition function from state [mathjaxinline]s[/mathjaxinline] with the action chosen by the policy, [mathjaxinline]\pi (s)[/mathjaxinline], assigns to arriving in state [mathjaxinline]s'[/mathjaxinline].
Study Question:
What is [mathjaxinline]\sum _{s'} T(s, a, s')[/mathjaxinline] for any particular [mathjaxinline]s[/mathjaxinline] and [mathjaxinline]a[/mathjaxinline]?
Then we can say that a policy [mathjaxinline]\pi _1[/mathjaxinline] is better than policy [mathjaxinline]\pi _2[/mathjaxinline] for horizon [mathjaxinline]h[/mathjaxinline], i.e. [mathjaxinline]\pi _1 >_ h \pi _2[/mathjaxinline], if and only if for all [mathjaxinline]s \in \mathcal S[/mathjaxinline], [mathjaxinline]V_{\pi _1}^ h(s) \geq V_{\pi _2}^ h(s)[/mathjaxinline] and there exists at least one [mathjaxinline]s \in \mathcal S[/mathjaxinline] such that [mathjaxinline]V_{\pi _1}^ h(s) > V_{\pi _2}^ h(s)[/mathjaxinline].
Finding an optimal policy
How can we go about finding an optimal policy for an
mdp
? We could imagine enumerating all possible policies and calculating their value functions as in the previous section and picking the best one...but that's too much work!
The first observation to make is that, in a finite-horizon problem, the best action to take depends on the current state, but also on the horizon: imagine that you are in a situation where you could reach a state with reward 5 in one step or a state with reward 10 in two steps. If you have at least two steps to go, then you'd move toward the reward 10 state, but if you only have step left to go, you should go in the direction that will allow you to gain 5!
One way to find an optimal policy is to compute an
optimal action-value function
, [mathjaxinline]Q[/mathjaxinline]. We define [mathjaxinline]Q^ h(s, a)[/mathjaxinline] to be the expected value of
starting in state [mathjaxinline]s[/mathjaxinline],
executing action [mathjaxinline]a[/mathjaxinline], and
continuing for [mathjaxinline]h - 1[/mathjaxinline] more steps executing an optimal policy for the appropriate horizon on each step.
Similar to our definition of [mathjaxinline]V[/mathjaxinline] for evaluating a policy, we define the [mathjaxinline]Q[/mathjaxinline] function recursively according to the horizon. The only difference is that, on each step with horizon [mathjaxinline]h[/mathjaxinline], rather than selecting an action specified by a given policy, we select the value of [mathjaxinline]a[/mathjaxinline] that will maximize the expected [mathjaxinline]Q^ h[/mathjaxinline] value of the next state.
[mathjaxinline]\displaystyle  Q^0(s, a)[/mathjaxinline]
[mathjaxinline]\displaystyle = 0[/mathjaxinline]
[mathjaxinline]\displaystyle Q^1(s, a)[/mathjaxinline]
[mathjaxinline]\displaystyle = R(s, a) + 0[/mathjaxinline]
[mathjaxinline]\displaystyle Q^2(s, a)[/mathjaxinline]
[mathjaxinline]\displaystyle = R(s, a) + \sum _{s'}T(s, a, s') \max _{a'} R(s', a')[/mathjaxinline]
[mathjaxinline]\displaystyle \vdots[/mathjaxinline]
[mathjaxinline]\displaystyle Q^ h(s, a)[/mathjaxinline]
[mathjaxinline]\displaystyle = R(s, a) + \sum _{s'}T(s, a, s') \max _{a'} Q^{h - 1}(s', a')[/mathjaxinline]
We can solve for the values of [mathjaxinline]Q[/mathjaxinline] with a simple recursive algorithm called
value iteration
which just computes [mathjaxinline]Q^ h[/mathjaxinline] starting from horizon 0 and working backward to the desired horizon [mathjaxinline]H[/mathjaxinline]. Given [mathjaxinline]Q[/mathjaxinline], an optimal policy is easy to find:
[mathjax]\pi _ h^*(s) = \text {arg}\max _{a}Q^ h(s, a) \; \; .[/mathjax]
There may be multiple possible optimal policies.
Dynamic programming
(somewhat counter-intuitively, dynamic programming is neither really “dynamic" nor a type of “programming" as we typically understand it.) is a technique for designing efficient algorithms. Most methods for solving MDPs or computing value functions rely on dynamic programming to be efficient.
The
principle of dynamic programming
is to compute and store the solutions to simple sub-problems that can be re-used later in the computation. It is a very important tool in our algorithmic toolbox.
Let's consider what would happen if we tried to compute [mathjaxinline]Q^4(s, a)[/mathjaxinline] for all [mathjaxinline](s, a)[/mathjaxinline] by directly using the definition:
To compute [mathjaxinline]Q^4(s_ i, a_ j)[/mathjaxinline] for any one [mathjaxinline](s_ i, a_ j)[/mathjaxinline], we would need to compute [mathjaxinline]Q^3(s, a)[/mathjaxinline] for all [mathjaxinline](s, a)[/mathjaxinline] pairs.
To compute [mathjaxinline]Q^3(s_ i, a_ j)[/mathjaxinline] for any one [mathjaxinline](s_ i, a_ j)[/mathjaxinline], we'd need to compute [mathjaxinline]Q^2(s, a)[/mathjaxinline] for all [mathjaxinline](s, a)[/mathjaxinline] pairs.
To compute [mathjaxinline]Q^2(s_ i, a_ j)[/mathjaxinline] for any one [mathjaxinline](s_ i, a_ j)[/mathjaxinline], we'd need to compute [mathjaxinline]Q^1(s, a)[/mathjaxinline] for all [mathjaxinline](s, a)[/mathjaxinline] pairs.
Luckily, those are just our [mathjaxinline]R(s, a)[/mathjaxinline] values.
So, if we have [mathjaxinline]n[/mathjaxinline] states and [mathjaxinline]m[/mathjaxinline] actions, this is [mathjaxinline]O((mn)^3)[/mathjaxinline] work—that seems like way too much, especially as the horizon increases! But observe that we really only have [mathjaxinline]mnh[/mathjaxinline] values that need to be computed, [mathjaxinline]Q^ h(s, a)[/mathjaxinline] for all [mathjaxinline]h, s, a[/mathjaxinline]. If we start with [mathjaxinline]h=1[/mathjaxinline], compute and store those values, then using and reusing the [mathjaxinline]Q^{h-1}(s, a)[/mathjaxinline] values to compute the [mathjaxinline]Q^ h(s, a)[/mathjaxinline] values, we can do all this computation in time [mathjaxinline]O(mnh)[/mathjaxinline], which is much better!
Infinite-horizon solutions
It is actually more typical to work in a regime where the actual finite horizon is not known. This is called the
infinite horizon
version of the problem, when you don't know when the game will be over! However, if we tried to simply take our definition of [mathjaxinline]Q^ h[/mathjaxinline] above and set [mathjaxinline]h = \infty[/mathjaxinline], we would be in trouble, because it could well be that the [mathjaxinline]Q^\infty[/mathjaxinline] values for all actions would be infinite, and there would be no way to select one over the other.
There are two standard ways to deal with this problem. One is to take a kind of
average
over all time steps, but this can be a little bit tricky to think about. We'll take a different approach, which is to consider the
discounted
infinite horizon. We select a discount factor [mathjaxinline]0 < \gamma < 1[/mathjaxinline]. Instead of trying to find a policy that maximizes expected finite-horizon undiscounted value,
[mathjax]\mathbb {E}\left[\sum _{t = 0}^{h}R_ t \mid \pi , s_0\right]\; \; ,[/mathjax]
we will try to find one that maximizes the expected
infinite horizon discounted value
, which is
[mathjax]\mathbb {E}\left[\sum _{t = 0}^{\infty }\gamma ^ tR_ t \mid \pi , S_0\right] = \mathbb {E}\left[R_0 + \gamma R_1 + \gamma ^2 R_2 + \ldots \mid \pi , s_0\right] \; \; .[/mathjax]
Note that the [mathjaxinline]t[/mathjaxinline] indices here are not the number of steps to go, but actually the number of steps forward from the starting state (there is no sensible notion of “steps to go" in the infinite horizon case).
There are two good intuitive motivations for discounting. One is related to economic theory and the present value of money: you'd generally rather have some money today than that same amount of money next week (because you could use it now or invest it). The other is to think of the whole process terminating, with probability [mathjaxinline]1-\gamma[/mathjaxinline] on each step of the interaction. This value is the expected amount of reward the agent would gain under this terminating model.
Evaluating a policy
We will start, again, by evaluating a policy, but now in terms of the expected discounted infinite-horizon value that the agent will get in the
mdp
if it executes that policy. We define the value of a state [mathjaxinline]s[/mathjaxinline] under policy [mathjaxinline]\pi[/mathjaxinline] as
[mathjax]V_{\pi }(s) = \mathbb {E}[R_0 + \gamma R_1 + \gamma ^2 R_2 + \dots \mid \pi , S_0 = s] = \mathbb {E}[R_0 + \gamma (R_1 + \gamma (R_2 + \gamma \dots ))) \mid \pi , S_0 = s] \; \; .[/mathjax]
Because the expectation of a linear combination of random variables is the linear combination of the expectations, we have
[mathjaxinline]\displaystyle  V_{\pi }(s)[/mathjaxinline]
[mathjaxinline]\displaystyle  = \mathbb {E}[R_0 \mid \pi , S_0 = s] + \gamma \mathbb {E}[ R_1 + \gamma (R_2 + \gamma \dots ))) \mid \pi , S_0 = s][/mathjaxinline]
[mathjaxinline]\displaystyle  = R(s, \pi (s)) + \gamma \sum _{s'}T(s, \pi (s), s')V_{\pi }(s')[/mathjaxinline]
This is
so
cool! In a discounted model, if you find that you survived this round and landed in some state [mathjaxinline]s'[/mathjaxinline], then you have the same expected future lifetime as you did before. So the value function that is relevant in that state is exactly the same one as in state [mathjaxinline]s[/mathjaxinline].
note
You could write down one of these equations for each of the [mathjaxinline]n = |\mathcal S|[/mathjaxinline] states. There are [mathjaxinline]n[/mathjaxinline] unknowns [mathjaxinline]V_{\pi }(s)[/mathjaxinline]. These are linear equations, and so it's easy to solve them using Gaussian elimination to find the value of each state under this policy.
Finding an optimal policy
The best way of behaving in an infinite-horizon discounted
mdp
is not time-dependent: at every step, your expected future lifetime, given that you have survived until now, is [mathjaxinline]1 / (1 - \gamma )[/mathjaxinline].
Study Question:
Verify this fact: if, on every day you wake up, there is a probability of [mathjaxinline]1 - \gamma[/mathjaxinline] that today will be your last day, then your expected lifetime is [mathjaxinline]1 / (1 - \gamma )[/mathjaxinline] days.
An important theorem about
mdp
s is: there
Stationary means that it doesn't change over time; the optimal policy in a finite-horizon
mdp
is
non-stationary.
exists a stationary
optimal policy [mathjaxinline]\pi ^*[/mathjaxinline] (there may be more than one) such that for all [mathjaxinline]s \in \mathcal S[/mathjaxinline] and all other policies [mathjaxinline]\pi[/mathjaxinline], we have
[mathjax]V_{\pi ^*}(s) \ge V_{\pi }(s) \; \; .[/mathjax]
There are many methods for finding an optimal policy for an
mdp
. We will study a very popular and useful method called
value iteration
. It is also important to us, because it is the basis of many
reinforcement-learning
methods.
Define [mathjaxinline]Q^*(s, a)[/mathjaxinline] to be the expected infinite-horizon discounted value of being in state [mathjaxinline]s[/mathjaxinline], executing action [mathjaxinline]a[/mathjaxinline], and executing an optimal policy [mathjaxinline]\pi ^*[/mathjaxinline] thereafter. Using similar reasoning to the recursive definition of [mathjaxinline]V_\pi[/mathjaxinline], we can express this value recursively as
[mathjax]Q^*(s, a) = R(s, a) + \gamma \sum _{s'}T(s, a, s')\max _{a'}Q^*(s', a') \; \; .[/mathjax]
This is also a set of equations, one for each [mathjaxinline](s, a)[/mathjaxinline] pair. This time, though, they are not linear, and so they are not easy to solve. But there is a theorem that says they have a unique solution!
If we knew the optimal action-value function, then we could derive an optimal policy [mathjaxinline]\pi ^*[/mathjaxinline] as
[mathjax]\pi ^*(s) = \text {arg}\max _{a}Q^*(s, a) \; \; .[/mathjax]
Study Question:
The optimal value function is unique, but the optimal policy is not. Think of a situation in which there is more than one optimal policy.
We can iteratively solve for the [mathjaxinline]Q^*[/mathjaxinline] values with the value iteration algorithm, shown below:
Theory
There are a lot of nice theoretical results about value iteration. For some given (not necessarily optimal) [mathjaxinline]Q[/mathjaxinline] function, define [mathjaxinline]\pi _{Q}(s) = \text {arg}\max _{a}Q(s, a)[/mathjaxinline].
After executing value iteration with parameter [mathjaxinline]\epsilon[/mathjaxinline], [mathjaxinline]\lVert V_{\pi _{Q_{\text {new}}}} - V_{\pi ^*} \rVert _{\text {max}} < \epsilon[/mathjaxinline].
This is new notation! Given two functions [mathjaxinline]f[/mathjaxinline] and [mathjaxinline]f'[/mathjaxinline], we write [mathjaxinline]\lVert f - f' \rVert _\text {max}[/mathjaxinline] to mean [mathjaxinline]\max _ x \lvert f(x) - f'(x)\rvert[/mathjaxinline]. It measures the maximum absolute disagreement between the two functions at any input [mathjaxinline]x[/mathjaxinline].
note
There is a value of [mathjaxinline]\epsilon[/mathjaxinline] such that
[mathjax]\Vert Q_{\text {old}} - Q_{\text {new}} \rVert _{\text {max}} < \epsilon \Longrightarrow \pi _{Q_{\text {new}}} = \pi ^*[/mathjax]
As the algorithm executes, [mathjaxinline]\lVert V_{\pi _{Q_{\text {new}}}} - V_{\pi ^*} \Vert _{\text {max}}[/mathjaxinline] decreases monotonically on each iteration.
The algorithm can be executed asynchronously, in parallel: as long as all [mathjaxinline](s, a)[/mathjaxinline] pairs are updated infinitely often in an infinite run, it still converges
This is very important for reinforcement learning.
to optimal value.
Download this chapter as a PDF file
This page was last updated on Friday May 24, 2019; 02:29:23 PM (revision 4f166135)

## Lecture: SM - MDP - Grid world example demos

Lecture: SM - MDP - Grid world example demos
Lecture: SM - MDP - Grid world example demos
No playable video sources found.
Your browser does not support this video format. Try using a different browser.
0:00 / 0:00

## Video transcripts

### Lecture: Sequential models - state machines


LESLIE KAEBLING: So far throughout the class since
we started we've been looking at machine learning.
And we've been looking at in particular
trying to learn a function.
And we've been doing supervised learning, which
means we've been trying to learn to function where we're
given training examples, the training examples of pairs
of an input and an output, and we
have to try to find some function that
does a good job of mapping those inputs and outputs.
And we've looked at all different kinds of ways
of doing that, and the complicated feed-forward neural
networks, and even convolutional networks
are still examples of trying to learn the mapping from inputs
to outputs given these examples.
So now we're about to start on different setups for machine
learning.
Not just the simple kind of straightforward
mappings from inputs to outputs, but in particular kinds
of problems where we're worried about things
that can now happen over time.
So these are sequential models.
And so what we're going to do today
and this week is study some background
material that will support learning in sequential models.
And so eventually we're going to look at reinforcement learning,
and we're going to look at recurrent neural networks,
which are two kinds of learning that involve sequential models.
But today we're just going to focus
on the underlying sequential models themselves.
And I recognize that you come from a big collection
of backgrounds, so I can't rely on any particular background
from any particular person, so we're just
going to try to set it up.
So we're going to start with the idea of a state machine.

And if you've taken computer science discrete math
class or something, maybe you've seen finite state machines.
If you come from an electrical engineering background
or something like that, you may have seen LTI systems
or something like that.
So this is a kind of a generalization of those ideas.
So for us, a state machine is going
to be defined as some big set S of states x, y, s0, f and g.
So that's a state machine.
The S is a set of states.

And a state for right now we'll understand
what it is as we go forward and elaborate on it,
but it could be a finite set of states or it
could actually be an infinite set of states.
So for us, it's just some kind of a set of states.
We have a set of inputs, also finite or infinite,
set of outputs, same deal.

And we have s0, which is an element of our big set S,
and that's an initial state.

So we might later on make this a little bit more complicated,
but for right now let's assume that we have an initial state.
So then we got two more things, f and g,
and really f and g are where the action is.
So f is a function from S cross x into S.
And it's a transition function.

And what it says is that if our state right now is--
So we'll say that if our state right now
is some particular S at time t, and our input
x is some particular little x, then f of x
is going to be the new state.

So the idea is we have some system.
The description of the system is a state.
That's kind of like what we know about it right now.
And if we give it an input, then it's
going to make a transition into a new state.
And f tells us how that's going to happen.
So f is just some function from an S and an x into an S. So
that's f.
And then there's g, which is a mapping from S's--
I always have to remember exactly how we've done this--
into Y's we'll call this the output function.

And it says if S is our state right now, then g of S
is our output, the output at some time t.
Now you may have seen variations on this theme.
Sometimes people make the output depend on the input.
There's all kinds of minor things
you can do to change it around, but they're
all kind of roughly the same.
So this is some particular setup for state machines.
It might not be exactly the one you saw,
but it should be close enough if you've seen one before.


### Lecture: Sequential models - state machines


LESLIE KAELBLING: So let's now take this and understand
in block diagram terms what we get.
So one way to think about it is that the state comes in here.
It comes into f.
What else comes into f?
Well, actually let me make the picture this way. x comes in.
The input comes into f.
And the state comes into f.
And out comes the next state, which goes
into g, which generates a y.
And also this s come around like this.
Now, if you're a singles person or something,
you kind of have a minor freak out at this moment because I
have this arrow just going around like this,
and you don't know what happens, and maybe it's not well-formed.
So if you want to think about it, what you really need
is something like a delay here.
So you're going to say, oh, yeah.
An x at time t comes in.
And the s at time t is here.
And what we get here is s at t plus 1 in some sense.
And you could think of this as the y at t plus 1.

So you have to do something to be sure that this is not
a runaway feedback loop.
But what's interesting is if you just draw a box like this,
you sort of put all this f and g and delay
and loop and stuff inside a box, then what
you have is this bigger box.
And this bigger box does what I would call a transduction.
I would call this whole thing a transducer.
It's kind of fancy name.
Sounds pretty cool.
Not as good as perceptron, but almost.
So a transducer.
And what does a transducer do?
It takes a history, a string.
You could think of it as mapping strings of inputs or sequences
of inputs, a sequence of inputs, into a sequence of outputs.

Because if I feed--
think of these x's as characters or integers or real numbers
or whatever.
If I feed them in one at a time, each time I feed one in,
an output is going to come out.
But notice the output that comes out, let's say, at time step
10 could depend on the inputs all the way up until time step
10.
So that's the important feature, right?
It might look, if you back up, it might look like a function.
It might look like something that takes an x
and gives you y.
But it's not a pure function.
People might say it's not a pure function
because the y that comes out doesn't only
depend on the x it's coming in right now.
The y that comes out could depend on the whole history
of x's that we've seen so far.
So like the critical thing Does this kind of make sense?
Yep.
STUDENT: [INAUDIBLE].
You take the Laplace transform of this thing
and do stuff to it.
Right.
So it looks like an--
to you it might look like an LTI system.
right.
So OK, good.
So when x and s and y are all real numbers, or more
generally, even vectors of reals, and f and g are linear,
then this is an LTI system.

If you haven't heard of an LTI system, don't worry.
This is for people who have heard of an LTI system.
How does that relate to this?
LTI stands for Linear Time-Invariant.
It's a little confusing.
Why would you call this thing a time-invariant system?
It doesn't seem very time-invariant to me.
The thing that's time-invariant if you call this a linear
time-invariant system--
what's time-invariant are the f and the g.
f and g are plain old fixed functions.
They don't change over time.
The state changes, but the f and the g don't change.
So in that special case, then there's
a bunch of extra cool stuff, special analysis you can do,
and so on.
But in the general class where f and g could be anything,
then ideas of transforms and stuff
don't necessarily make sense.

Oh, OK.
So the question is, did I goof my indices?
st plus 1 is f of s.
Ah, OK.
Good, good, good, good.
All right.
Let me do this.

Good.


### Lecture: Sequential models - state machines


LESLIE KAELBLING: We can talk about exactly what the sequence
y is, right?
So imagine that we have input sequence x1, x2, and so on.
Our output-- so what's our first output going to be?
If our first input is x1, our first output,
it's going to be g of something.

g takes a state as an argument.
It's going to be g of a state, the first state, not a 0.
It's going to be g of f of s0 and x1.

That make sense?
That's going to be the first output.

And then the next output--
so if this is going to be the output sequence,
what's the next output?
Well, it's going to be g of something.
It's going to be g of the state that happens.
So this is-- right here, this is state 1, right?
That's the first state.
We start in state 0.
We take a transition according to f.
We get to state 1.
Say, OK, the next output is going to be g of state 2.
What is state 2?
State 2 is f of state 1 and x2.

But if you wanted to write that out in more detail,
you could say, this is g of f of state 1 is f of state
0 and x1, x2, like that.

So no mystery, really.
It gets kind of complicated after awhile.
But each output is just very clearly computable
as a function of a sequence of inputs
that has come up until that.
So for state machines, I just want to say one more thing,
and then we're going to move into another kind of section
of stuff.
So for state machines, part of the reason
that we're introducing this particular kind of state
machine in a transduction and something that
takes x's into y's is that it's going
to be the basis of recurrent neural networks.
So we're working our way up to, in some sense,
recurrent neural networks.
And recurrent neural networks are exactly
an instance of this class.

And basically, what makes a recurrent neural network
is it's a state machine, so a state machine.

And I should say-- well, where f and g are
feed-forward neural networks.

I should say there are a whole bunch of different kinds
of recurrent neural networks.
We're going to study one particular form.
So we're going to make this more specific
and pick a particular form of recurrent neural network.
But the fundamental idea that makes
a recurrent neural network a recurrent neural network
is that there is some internal feedback.
There's some stuff that we're generating,
and it comes back around.
And this is what lets us remember something
about the history of observations,
the history of inputs that we've gotten so far.
We can remember something about that.
And then our next output can depend on the whole history.
So that's the idea of recurrent neural networks.
So we're going to put a neural network in here,
a neural network in here, and in two weeks,
we'll look in detail at kind of like how we're
going to do that-- in particular, how we can train up


### Lecture: Sequential models - state machines

LESLIE KAELBLING: So one use of the idea of a state machine
is to think about this transduction,
about mapping a sequence of inputs
through a sequence of outputs.
OK, another thing that we can do with state machines
is use them-- now, this is-- it's really the same kind
of fundamental idea.
But we're going to use it in a different way.
So here, we're using a state machine
to learn a kind of hypothesis.
Our hypotheses are going to be state machines.
Another thing that we can do with a state machine
is model an environment that we might be interacting with.
So now, we're going to think about state machines
as environment models.

And I have to draw my favorite picture in the whole world,
which I will do right now.
So since I work on robots, I always
think about this little robot right here.
He's on my computer, that little robot.
And the robot is interacting with some kind of environment.

And the idea is that the robot can take actions.

And it can make--
in the simplest version of this problem,
we'll say that the robot can actually observe
the state of the environment.
So the environment for a robot, it could be--
if it's a mobile robot that's running around in this room,
its environment might be the way the room works,
the way the robot's particular situation changes over time.
If the robot moves forward, certainly its position
is going to change.
Maybe some other stuff will change,
like more people would come in or come out of the room.
So this is the world that the robot interacts with.
And our view with this robot interacting with the world
is the robot can choose some actions of its own.
It could decide to move, or wave its arms around,
or pick something up, or empty out my water bottle,
or do some kind of thing like that.
So the robot can pick an action.
When it takes an action, that action
is going to change the state of the world in some way, right?
And then, it gets to see what new state happened
and then potentially pick another action
to change the state of the world.
So also, you assume that the environment--
remember state, right?
The state of the world that the robot's interacting with
is persistent.
The state of the world that the robot's interacting with
depends on the history of actions
that the robot has taken.
So in that sense, we're going to model
the environment of the world--
the environment-- as a state machine also.

So state machines as environment models.
And then, what we're going to work up to is figuring out--
eventually learning, when we study reinforcement learning--
learning good ways for the robot to interact
with its environment.
And this model gets used for--
I draw my little mobile robot.
But you can use models like this to talk
about controlling a chemical plant, or playing backgammon,
or basically any kind of a system that's outside the robot
that you would like to interact with.
Control systems, you can think of as operating this way too.
In fact, standard control setup is to model the environment
as an LTI system.
We're going to look at different kinds of models
of the environment.
But that's completely reasonable, right?
So you can maybe--
there's all different ways of thinking about this.
OK, so state machines as environment models--
one thing is, especially from the perspective
of a robot interacting with the world, is that an idealization
that we made over here--
over here we said, oh, when we're
thinking of this as a computational model
that we're building, then it can be deterministic, because we're
in control of the f.
F is a function.
And f is going to take the current state
and give you a new one.
It's something we're designing and building.
So we might as well make it be a deterministic function.
On the other hand, if we think about the interactions
that the robot has with an environment,
it might be that when the robot tries
to take a particular action, it doesn't always
come out the same way.
It could be--
I'm not going to flip this chalk,
because I know it will come out into many pieces--
not precisely how many pieces or in which way.
But if I were to just throw the chalk up into the air
and drop it on the floor, something would happen.
But I can't say exactly what.
And so instead of modeling the transitions here
as being a deterministic function,
we're going to model them as being probabilistic,
or stochastic--
randomized.
So we're going to look at a particular special kind
of state machine called a Markov Decision
Process, known as an MDP.
OK, what does a Markov decision process have?
So it still has a set of states, same as before, so S--

set of states.

We're going to say that it has a set of actions.
The actions correspond over there to the inputs.

We could call them inputs.
If we were control people, we would call them u.
And we would call them controls, because control starts with u.

But we're going to call them actions.
So we have states and actions, right?
So here's the robot, states, and actions.
We are not going to explicitly think about the outputs.
So for right now, we're going to assume
that the robot can actually see the state of the world.
So one way to think about it is that the state of--
that the outputs and the states are the same and, in fact,
that the output at time t is just
equal to the state at time t.
So we're not going to worry for right now about this g
function.
So we're not going to worry about output.
We're just going to assume that the state flies right out.
OK, so we have a set of states, a set of actions.
We're not going to worry about outputs.
There is a fancier version of Markov decision processes
called partially observable Markov decision processes,
where you worry about the outputs.


### Lecture: Sequential models - state machines


LESLIE KAELBLING: Now we have a transition model.
We're going to call it something other than F
because it's a fancier thing.
So we have transition model T. And T
takes a state and an action and another state
and gives you a value between 0 and 1.

And really what you want to think of it is
that the T of S, a, S prime--
so imagine T is a function.
T is a function.
It takes a state, an action, and a state.
So if I give T a state and an action
and another state, what it's supposed to mean is--
and I will decode this for you--
is the probability that the state at time t plus 1
is equal to S prime given that the state at time t is s
and that action at time t was a.
OK.
So let me talk through this thing I just wrote down.
I think everyone has been exposed in some way or another
to discrete probabilities.
So for right now, we'll for now think of these
as being discrete.

When we get to reinforcement learning in neural networks,
we'll let them be continuous.
But for now let's think of states and actions
as being discrete.
So I have some finite discrete set of states
and some finite discrete set of actions.
And I-- OK.
So now let's decode all the symbols in here.
When I write a capital letter like this,
it's a random variable.

So it's some quantity whose value we don't necessarily
know.
We can make assumptions about it, ask questions about it,
consider what its distribution might be.
But we don't really know its value.
So all these capital letters are random variables.
And so this random variable big S little t
is the state I was in at time t.
Or the state the world was in at time t.
Let's say that.
The state the world was in at time t plus 1.
The action I took at time t.
So those are random variables.
This conditioning, this bar means given.
So conditional probability-- if you
don't remember basic discrete conditional probability,
you should go read the Wikipedia article.
It will do it for you.
Practice a little bit.
So conditional probability is a given.
So we say, OK.
The stuff on the right-hand side is under the assumption that--
another way to think about it--
so under the assumption that the state at time t
was some particular state little s.
So this little s could be 7, or 93, or raining,
or something like that.
Under the assumption that S sub t
was this value and that the action I took
was this particular value.
Now I'm interested in knowing, what can I
say about the new distribution?
What can I say about the next state?
And what I'm going to be able to say
about the next state is exactly--
I'm not going to be able to say it's exactly
this particular state.
Instead I'm going to say, oh.
I don't know this value.
But I can tell you a probability distribution
over what it will be.
So I can tell you the probability
that it's some particular next state S prime,
but I can't tell you for sure what the next state will be.
So this is the probability that my next state
would be S prime under the assumption
that my previous state was S and I took action A. Is
this sufficiently decoded?

OK.
So that's what this T tells us.
Now what makes this--
so this is a name, Markov.
So Markov was a Soviet probability theorist.
I'm not sure if he's in the past tense or not, actually.
So is or was a Soviet probability theorist.
And so what makes a Markov decision process a Markov
decision process?
That's a question.
And actually, what makes a Markov decision process
a Markov decision process is related to the question, what
makes a state a state?
So they go together.
So what makes something Markov, makes the Markov
the Markov assumption, is that the probability distribution
over the state at time t plus 1, given all the previous states
and actions.
So S0, A1, S1, all the way up to S t and A t.
So this says, imagine that you knew the entire history so far
of states and actions and you wanted
to figure out a probability distribution on the next state.
But that's the same as this.
So that says, everything that matters to me
about the history for predicting the future is summed up in S t.

That's what makes a state a state.
It captures all the information you
need to make the predictions you need to make.
It doesn't mean you can make perfect predictions given St,
because maybe you just can't.
Maybe your domain has just got some fundamental noise
built into it.
But it does mean that if you were
to look farther back into the history, it wouldn't matter.
So that's like saying, oh, to predict
the price of some stock--
this isn't going to be true, but you
might conjecture that to predict the price of a stock tomorrow
it's important only to consider its value today.

That's probably not true.
It probably depends on the history of values, and also
a bunch of other stuff about the world, and who knows what.
On the other hand--
let's see.
So we could talk about physics.
So imagine that we're trying to predict-- imagine
that we have a particle, and it's got some position,
and we're trying to predict where it's going to go next
on the next step.

If I just try to assume that the position was the state,
would that be a Markov model?
Someone who took physics sometime in recent history.
Can I predict the next position of an object
given its current position?
No.
What do I need?
STUDENT: Velocity.
LESLIE KAELBLING: Velocity.
Good.
So the position wouldn't be a state
if I wanted to predict the next position.
Position and velocity might be enough to let
me predict the next position.
Or maybe I need the acceleration, too.
So now it depends on what assumptions
you're making about your system.
But if you want to say I'm modeling my system as a Markov
process, then you need to do your best
to capture everything that matters to make predictions
in the state.
We'll never get it exactly right, but we try.
OK.
So that's the Markov assumption.
That's what makes this a Markov decision process.
This is what means that in order to describe the dynamics, how
the states change over time, we only specify this T.
OK.
So now, things are complicated, though, right?
We might know that we started in some state.
But then when we take an action, there's
a distribution over states we might be in.
And if we take another action, there's
a different distribution over states
we might be in, and so on.


### Lecture: Sequential models - state machines


LESLIE KAELBLING: One more component to an MDP--
so what made it a Markov process was this Markov assumption.
What makes it a decision process and, again,
what kind of distinguishes it from just a plain old state
machine model is that we have some choices to make.
So we're going to get to pick--
we.
So I'm a robot.
You know, I'm a robot, and I'm trying maybe
to do something in the world, like clean the floor
or keep the chemical plant from exploding
or whatever it is that my job is.
So there's an environment here, and there's now
I have an opinion that some states are better than others.
I like states where the floor is cleaned.
I don't like states where the chemical plant has exploded.
And so I'm going to pick actions.
I'm going to make decisions.
I'm going to pick actions based on the state
so that the environment goes into states that I
like more than the states that I don't like.
That's kind of our plan, what we're going
to do with this MDP model.
So we have to have a way of articulating
what states we like and what states we don't like.
So we have one more component.
So we have S, A, T. And we'll have a function R.
And again, people will do this differently.
We're going to map states and actions into real numbers.

And R stands for reward.
It's funny.

If you read about a computer scientist or a psychologist
talking about MDPs, then they use R for reward,
and they're going to try to maximize the reward.
Control theorists seem to be kind of pessimistic
or something, and they have costs,
and they try to minimize the cost.
But we're going to take the optimistic view.
We're going to try to maximize reward, not minimize cost.
It doesn't make any difference.
It's just a minus sign here and there.
But anyway, we're going to work on maximizing reward.
So reward is good, generally speaking.
So we say some states are good to be in.
That's why we depend on the state, maybe.
But we might also depend on the action.
Sometimes it makes things easier.
You might say, oh, if you take an action that
uses a lot of fuel or something, maybe
that would have a lower reward than if you
take a very efficient action or something like that.
So there's a way to say I have an opinion
about the state and the action that I take.

So S, A, T, and R together define a Markov decision
process.

Questions about that?

OK.
OK, good.
So now we're going to talk about what
we would do with the Markov decision process if we had one.
And so we're going to think about what the robot does.
So the robot here is taking states and generating actions.
And for our purposes, we're going
to assume that the robot doesn't have internal state.
It turns out that if Y is not equal to S,
then the robot might have to have memory,
but we're going to look at memory-less robots
for this class.
So memory-less robots are fine in an MDP.
So we're going to think about the robot
as executing a policy pi.

And pi is a mapping from states to actions.
So the robot looks at the state of the environment,
says, oh, there's some dirt over there--
I'm supposed to clean the floor--
and maybe tries to take a step in that direction.
Now maybe we have a wobbly robot or a slippery floor.
So when the robot tries to take a step in this direction,
the robot might not quite go where it's intending to.
But after it tried to move in this direction,
it gets to observe where it actually ended up
and then pick another action.
So the robot is going to go in this interactive loop,
dancing round with the environment,
observing the state, picking an action.
That's its loop.

And so our job, given an MDP, we're
going to think about finding a good policy to put
in the head of the robot.
And now the job for the rest of this lecture and the stuff that
we're going to spend a lot of time on this week is thinking
about how given a reward function, which is very local--
a reward function just says how much do I like each individual
state-action pair--
given a award function and given this transition model,
how can we derive a policy that is in some sense optimal?
It gets as much reward as it can.


### Lecture: Sequential models - state machines

LESLIE KAELBLING: So first of all,
before we can think about finding a good policy,
we have to talk about how to evaluate a policy.
What makes a policy good?
So I'm going to define something.
Well, OK, and we're going to think
about two different regimes.
So we're going to think about, first of all, horizon.
So horizon is-- you can think of it as how many steps--
how many time steps does the robot have to go?
So imagine that we're going to put
the robot in an environment.
And we might say, hey, robot.
You get to live for 10 steps.
Or you get to live for a million steps.
But you get to interact with the environment
for some amount of time.
That's idea of horizon.
And we're going to talk about first finite horizon problems
and then infinite horizon problems.

But we'll start with finite.
So in a finite horizon problem, we
say, OK, we know the transition model.
We know the reward function.
And we know that the robot has a certain fixed number of steps
to execute in the world.
And the question we want to ask is, what is the best policy?
We'll start with asking, how good is some specific policy?
So imagine that we are given a policy and a horizon.

We can define a quantity, which is very decorated,
called V for value.
And what it is, is the expected sum of rewards, right?
So the robot is going to now execute actions for h steps.
And each time it executes an action,
it's going to get a reward, R, that
depends on the state of the environment and the action
it executed--
so the expected sum of rewards, given that we start in state
s and execute policy pi for h steps.

OK, so that's a value function.
It's not too hard to write down a recursive definition.
So I love recursion, because I'm a lazy person.
And it means I only have to think
through one little, little bit of stuff at a time.
So what's the simple case?
So V pi s 0.
So if I am not able to take any actions--
so 0 means I have no actions left to take.
I'm in state s.
I can't behave anymore--
what do I earn?
Well, if I can't behave, I don't earn anything.

So there is a base case for a recursion.
Good.
I love it.
That was so easy.
We're going to write the next case on the same line.
V1-- what if I have only one step left to go?

How much reward am I going to get if I'm in state s,
I'm executing policy pi, and I have one action I can take?
OK, so it's going to be R of s.
What goes next?
What action will I pick under these assumptions?
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: Yeah, OK, someone said it over there.
So let's see.
How do I pick actions?
I pick actions.
I'm executing pi.
Pi is a policy.
Pi is a policy that takes in a state, the current state,
and gives out an action.
So if I'm sitting here, and the world is in state s,
the action I take is going to be pi of s, right?
Pi of s is an action.

Maybe my policy is to always turn left when I see a light
and otherwise go straight.
That could be a policy.
So I say, OK.
Well, if in state s, I see a light, then I should turn left.
This would be turn left.
But if state s is not one where I see a light,
then I would go straight.
Whatever pi is, pi of s tells me what action I would pick.
And so, now, I say, OK.
Well, I'm assuming I'm executing pi.
So this tells me what action to take.
This is the state.
R of s a is how good it is to take that action in that state.
And I only had one action to take.
So that's it.
Does that make sense?

OK, now just the recursion--
but we can do two.
Let's do two.
So if I have two steps to take--
I'm in state s.
I have two steps to take.

What's going to happen?
What reward am I going to get right now if I'm in state s,
and I have two steps to take?

Yeah?

AUDIENCE: The one you already had.
LESLIE KAELBLING: The one I already had, right.
Why am I asking such a question like this,
because we already know?
It looks like the reward I'm going to get
is this one, R of s, pi of s.
Good, that's good.
So I'm going to get that one.
Now, now, things get a little bit more tricky,
because I have two steps left to behave.
So this is the first step.
This is easy.
That was the one we did already, there.
Now I'm going to end up in another state, right?
If I take this action, in this state,
I'm going to end up in another state.
What state am I going to end up in?

Do we know?
No, we don't know.
AUDIENCE: Probability distribution.
LESLIE KAELBLING: But we know a probability distribution, good.
So here, I have to say, man, I do not
know where I'm going to end up.
So I'm going to take an expectation.
I'll explain.
I'm going to write this down and then talk about it a little bit
more.
I'm not sure where I'm going to end up.
There's all these possible next states I could end up in.
I really don't know which I'm going to end up in.
For each possible state I might end up
in, I do know this is the probability that I'll
end up in that state, right?
This is the probability that I'll end up in s prime.
OK, now, if I end up in s prime--
imagine I end up in s prime.
And I have-- how many steps will I
have left to go once I end up in s prime?
One.
And then how good is that?
What's the expected sum of rewards,
given that I start in s prime and live for one step?
Good, V1.
OK, so V1 pi s prime.

So this says, I get this reward, totally.
I get it.
I'm not sure what state I'm going to end up
in after I take this action.
Oh, I made a slight error.
What did I do?
I wrote down a symbol here that's not defined.
AUDIENCE: [INAUDIBLE]
LESLIE KAELBLING: t? t is defined over there.
t is my model of the MDP.

What symbol is not defined?
a.
a is not defined.
What should a be? a should be pi of s, right?

Pi of s-- I'm promise--
I'm executing policy pi.
In state s, if I'm executing policy pi--
pi of s.
That's it.
OK, so we say, I'm not sure.
This is the action I'm going to take.
I'm in this state.
This is the action I'm going to take.
I'm not sure where I'm going to end up.
So I consider a whole probability distribution
over states that I might end up in.
And you could think of it as-- this expectation
as it's like a weighted sum.
And then what's the weighted sum over?
Well, it's how much value I'm going to get in the future
if I end up in that state.

Is that good?

Then, we can leap to t.
So the cool thing is, well, after you do this,
you can just say--

oh, so short.
OK, I'm just going to write V h.
So if we write an h, here, of pi of s, right-- so now,
for general horizon, it's going to be,
OK, I get this reward right now.
That's how this always goes.
And then, I'm not sure where I'm going to end up.
But I consider the transition probability.
I'll write it better this time.

And then it's whatever value I get there.
So that's how to evaluate a policy.
If I were in this state, and I were
to execute this policy for this many number of steps,
that's the expected value of the reward


### Lecture: Sequential models - state machines


LESLIE KAELBLING: I want to discuss how you would
compute this just briefly.
So let's consider a naive strategy for computation.
Imagine that I'm interested in V,
I don't know, horizon 3 with respect
to some policy of some state.

Imagine that.
I could be interested in that.
Now to compute that--
I'm just going to make a little kind of informal computation
tree, all right?
To compute that value--
actually, let's make it 4 just to be--
to compute that value, I have to actually compute
V pi 3 of s1 at worst all the way to V pi 3 of s n.
Imagine that I have n states.
Right?
Because if this transition probability could be non-zero
for all the states, then I might have to know V pi of s prime
for all possible s's.

So I do need to know these values in order
to compute this one.
And then if I'm, like, a naive implementer of recursive
programs-- which is really easy to do because you just write
the equation in your computer program--
then to compute this one, it seems like you,
OK-- you would need V pi 2 of s1 through V pi 2 of s n.

But you would need all these guys to compute this one.
But you would also need all these guys to compute this one.
And you'd also need all these guys to compute this one.
Each V 3 pi of s value depends potentially
on all the V 2 pi of s values.
And so do they depend on all the v 1s.

And so if you just implement this as a computer program
by writing down the recursive function,
it's going to be a kind of a sad day.
Because you're going to, to compute this guy,
you're going to compute all of these ones.
And then to compute this guy, you're
going to compute all of these ones.
And then to do this one over here,
you're going to compute all these again.
So you'll get something like n to the h
if you just do the simple recursive thing.
Now those of you who've taken algorithms
know of a trick called dynamic programming.
Which I have to say, when I learned
what dynamic programming was, it confused me completely
because it's not a kind of programming in any sense
that anyone understands really at all.
So just think of it as a name.
Like, it might as well be Fred or foobar or something.
Like those words dynamic programming
don't help you understand the technique.
But the technique is easy.
Just don't think about the words.
The technique is if these results are
going to be used a whole bunch of times
by a whole bunch of guys, I should just compute them once
and remember them.

So compute shared subresults and remember.

Yeah, OK.

That's the dynamic programming idea.
So if you wanted to compute a value of some state with
horizon 4, the way to approach this problem is not
by implementing the recursion directly but first by computing
v 1 of s for all the s's.

And then v 2 of s.
And then v 3 of s, and so on.
So we're going to make use of this technique
over and over again in MDPs.
And it's actually the underlying idea
behind reinforcement learning.
So it's a really important idea.
So if you're going to reuse a result,
you should just compute it and remember it.
And often, we end up then solving a problem
by computing answers to simple questions,
and then more complicated ones and then more complicated ones
until we get the answer that we need.


### Lecture: Sequential models - state machines


LESLIE KAELBLING: What we have now is a way,
given a policy, to compute the value of a state.
How good would it be for me to execute
this policy in this state?
Now, what we would like to do instead is to find,
instead of computing the value of a policy,
we would like to find pi star.
So pi star is an optimal policy.
Pi star h.
Pi h star.
So for some fixed horizon--
so, OK.
So we're going to be given S, A, T, and R.
We would like to find the best policy for some horizon h.
And this best policy is going to have the property that V pi
star h of S is greater than or equal to V pi h of S for all
S's and pi's.

So what does that say?
That says for all the states, there's
going to be one policy, one policy
that for all the states and all possible other policies,
this guy's value is at least as good.

So pi star is, this is going to be an optimal policy.

And there's two interesting things.
One is that there is a theorem that says,
at least one pi star satisfying those requirements exists.
So we're not looking for a unicorn.
There might be more than one.
It might be that I'm trying to find the optimal way to get
to the exit door.
So like, I could go this way or that way.
And maybe they're equally good.
So there could be multiple policies that are equally good.
But there is at least one that's good in all the states.
That's the important part.
Yep.
STUDENT: [INAUDIBLE]

LESLIE KAELBLING: Yep.
Good.
Yep.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Awesome question.
That's a beautiful question.
Would pi star change depending on the h?
I don't know.
What do you think?

Yes.
I see some nods.
So if I have one day left to live,
should I behave the same way as if I have 100 years?
Maybe not, right?
Maybe I should eat all the cookies.
[LAUGHTER]

Yeah.
So, no.
So horizon totally matters.
And so what that means is the reason that I carefully wrote--
ah.
Yeah, good.

Let me see, actually, how I wrote it in my notes.

Yeah, OK.
Technically-- OK, good.
So the pi depends on the h.
So there's the right pi--
in some sense, there's the right pi for h steps.
But the right pi for h steps-- that is to say,
the particular mapping pi for h steps--
is not going to be the same as the mapping for h
minus 1 steps.
So technically speaking, to find pi star--
actually, I can leave it like that.
But to find pi star-- like, the real pi star--
I have to find pi star for h, pi star for h minus 1,
pi star for 2, and pi star for 1.
And this would be like, eat all the cookies.
And this would be like, plant some wheat in your garden.
So depending on where you find yourself,
you should behave in different ways.
Good.
Good, good, good.
OK.
So to do this, I'm going to introduce a new something--
a minor variation on the value function
which is going to be super awesomely
useful in infinite horizon and in reinforcement learning.
So I kind of apologize because there's
kind of a lot of letters in this lecture.
But it's the best way I can give you
a foundation to understand the stuff we're
going to do next time.
So here we are.

OK.
So, Q. I'm going to define a new thing.
But it needs an h.

And Q is kind of like V. It's almost exactly like V,
except, well, it's going to be the expected sum
of future awards.

Given what?
Given that I start and stayed s.

That I take action a.

And then I execute pi star h minus 1 for h minus 1 steps.
This is a weird definition because it's
a definition in terms of a thing we don't know.
But it's kind of like, OK.
So what does it say?
It's the expected value of starting and staying
S, taking action a, and then behaving optimally after that.

OK?
We're not sure what that is exactly, but we'll compute it.
It'll be OK.
But that's what Q is.
Starting and staying s, take action a,
behave optimally after that.

OK.
But we can write down a recursive definition.
So let's do it.
So Q-- ooh.
Start with Q0.

If I can't take any steps, my value is 0.
That one was easy.
And the next one's going to be easy, too.
Q1 of S, a.
So if I only get to execute one action,
it's going to be action a in state S.
And what's our expected sum of future rewards
if we only get to live one step and take a in state S?
Yeah.
R, right.
So R s, a.
So far, easy.
Now let's just cut to the chase.

Imagine that I wanted to think about how good it
would be to start in state h.
Take action-- start in state s.
Sorry.
Start in state s, take action a, and go for h steps optimally.
Well, on my first step, it's going to be this.
It's always going to be that.

And now I'm going to behave optimally in the future.
OK.
So I have to think about how that's going to go.
First of all though, I'm not sure where I'm going to end up.
So this is the same as before.
I'm not sure where I'm going to end up.

And now I want a recursive definition.
And so I would like to write it in terms of Q.
So I'm going to write it in terms of Q.
So first of all, let's see.
How many more steps do I have?
Good. h minus 1.

What state am I going to be in after I've taken this one step?
I started in s, I took action a.
Right-- s prime.
Good.
So I'm going to be in s prime.
And then the question is, what action goes there?
I'm not sure.
What action goes here?
I'm going to call it a prime.
a prime is the best action I could take.
It needs to be my optimal way of behaving, right?
So it's the best action I could take.
So what goes in here--
I should've left a gap when I was writing this--
is I'm going to write in here, max over a prime.

So I say, I find myself in state s.
I take action a.
I'm not sure where I'm going to go.
But wherever I go, I'm going to pick the action,
I get to pick the action a prime that works out
the best for the future given that I'm sitting in state
s prime.

Does that seem OK to everybody?
Yeah.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Yeah, OK.
My typography is lacking.
Let me just fix it.
I meant to leave a gap but I got carried away.

STUDENT: [INAUDIBLE]
LESLIE KAELBLING: So the arg max is the action you should take.
But the max is a value.
So this isn't expected.
This is a number.
This is a real number.
These are real numbers.
These are all real numbers.
Scalars.
Yep.
STUDENT: [INAUDIBLE]
LESLIE KAELBLING: Yes, we absolutely could.
Good observation.
Right.
We could define here this a prime--
ooh, and I lost my h minus 1.
That a prime is exactly pi star h minus 1 of s prime.
That's right.
Or another way to-- exactly right.
So if you compute this Q function,
this is just true in general, right?
That no matter what, if you computed this Q function--
which now you should see how to compute, right?
It's just a recursive definition using bits and pieces we know.
We know R. We know T. It's going to count down to 0.
We have the base cases.
We would like to compute it using dynamic programming.
Oh, that's good.
But once you know this, then by definition-- right-- pi
star of h of s is arg max over a of Q h s, a.
Righto.


### Lecture: Sequential models - state machines

LESLIE KAELBLING: It's time to go to the infinite horizon.
This is discrete.
It's counting.
We're counting the horizons.
We started at horizon 0.
We worked our way back to horizon whatever.
That is all OK.
Now, let's think about a case where we're not sure, actually,
how long we're going to live.
So this is an infinite horizon case.

And there's different ways to model it.
But we're going to just talk about one particular way.
And so we're going to talk about an idea of discounting.

So we're going to have this number, gamma,
which is bigger than 0 and less than 1.
And it's called the discount factor.

And the idea, here, is going to be--
there's two ways to think about it.
The way I like to think about it is
that we're playing this game.
And on every step, on every step, our interaction
with the world, our life, our game,
ends with probability 1 minus gamma, right?
So every day, a cosmic coin is flipped.
And maybe you keep playing.
And maybe you don't, OK?
So your life is long if gamma is close to 1.
So the expected lifetime is this.

You can figure that out later if you want to.
OK, so what do we do with this?
So what's interesting about this model
is that now, when you do decision making,
you actually don't need a different policy
on different days, because under this model,
if I survive to today, my expected future lifetime
is just as long as my expected future lifetime was yesterday.

If you survive to today, your expected future lifetime
is 1 over 1 minus gamma.
And so that means your decision making, your policy,
your value, is the same all the time, so
that instead of needing a value function for every horizon,
or a policy for every horizon, you just need one--
one value function, one policy.
So it simplifies things.

And it's pretty much the standard way
of thinking about this stuff.

So now, our value--
we can think about the value of being in some state or just,
really, the value of getting some sequence of rewards.
Imagine we get a reward at time 1, and a reward at time 2,
and a reward at time 3, and so on.
The value to us of those rewards is
going to be r1 plus gamma r2 plus gamma r3 plus--
which is equal to r1 plus gamma r2 plus gamma squared r3 plus--
so this is the expected--

so imagine someone said, OK, this is the rewards
that you would get.
Imagine you just-- somebody rolled the dice for you.
And these are the rewards that you would
get as you lived your life.
But there is this process that might
stop your life at any point.
The rewards that happen soon are worth more to you
than rewards that happen far in the future,
because you might not get those.
You're probably not going to live that long.

And so you could say, well, I'm for sure going to get r1,
because I'm here right now.
And then if I live, I'll get the future stuff.

And so, and then, if you just take these gammas
and multiply them out, what you end up
with is this exponential discounting of gammas.

So it's kind of cool.
Again, I'll let you ponder that.
Work through it more on your own.

And so now, what I want to do is define the Q
function, the same idea of a Q function,
but in the case of an infinite horizon discounted problem.

So let's just do this.
So Q-- I will leave off the star--
Q of s, a.
So now, assume same thing.
We know the MDP.
We know t.
We know r.
And we know gamma.
There's no h anymore, but there is gamma.
So the question is, I start in state s.
I take action a.
And I behave optimally in the future.
But now, I'm going to evaluate my future in that discounted
style.
So we say, OK, well, here I am.
I'm alive today.
And I'm in state s.
And I took action a.
So I totally get R of s, a.

Now, if I die, that's it.
So probably of 1 minus gamma, I get 0.
But if I live--
that is to say, with probability gamma--
then I get the future value.
OK, future value--
I'm not sure what state I'm going to end up in.
We've seen this now several times.
Not sure what state I'm going to end up in.
So I have to take an expectation over my next state.
If I arrive in state s prime, like I live until tomorrow,
then tomorrow I'm going to choose my action as best I can
according to this Q function, right?
So it's going to be max over a prime Q s prime, a prime.

So that's the optimal value function
for the discounted case.

And again, the optimal policy is this, right?
So I can just act greedily with respect to that policy.

OK, super cool.
So now we know a definition for Q.
But it's not obvious how to turn this into an algorithm.
Over there, it was pretty clear how to turn it-- especially,
it was clear how to turn it into a really slow algorithm,
because we could just execute the recursive definition,
or start with h and subroutine call down to 0.
Or we could start with horizon 0 and work our way up to h.
That's smarter.
But here, eh, it's not so clear, because we
got Q defined in terms of Q. So there's not
an obvious sequence of intermediate results
that we can compute.

OK, so here is the most awesome thing.
So I'm going to teach you a new algorithm.
It's called value iteration.

And value iteration-- there.
OK, so what do I mean?
What do I mean?
What I mean is that if you initialize your Q values--

so you're going to try to compute this function, Q of s,
a.
They're discrete.
You can initialize them to anything.
And it doesn't matter.
It's conventional to initialize them to 0.

Initialize Q values.
And then, just keep executing this equation, over and over,
for all your states and actions.
Over and over, just update.
Just, you start.
You initialize them to 0.
You could pick an s and an a at random if you want to.
Pick an s and an a at random.
And say, I'm going to update my Q value
estimate for this state and this action
by computing this using my terrible-- right now--
my terrible, horrible, Q value estimate that I started with,
which is that everything is 0.
I'm going to compute this.
And this is going to give me a new number for Q of s, a.
And that number is going to be, in a kind
of an interesting mathematical sense, a somewhat better number
than the one I had before.
And if I just keep doing this, over and over and over again,
I am guaranteed to converge to the optimal Q function.
This is like the simplest and most robust algorithm.
You can beat it with a stick.
You can run it in parallel, backward,
forward, at the same time.
It doesn't matter.
Just keep doing these updates.
Don't starve anyone, right?
You have to try all state-action pairs often enough.
It will converge to the optimal value.
It's awesome.


### Lecture: Sequential models - state machines

LESLIE KAELBLING: I want to just show you
an example because it's fun.
All right, so let me just set up this problem.
I also want to show you this because you're
going to see pictures that look like this
in the labs and stuff.
And so it's good to just get an idea.
So this is-- imagine we have a little robot that can
move around in a grid, right?
So it lives in a 10 by 10 grid.
Each one of these is a location.
So its state is its location.
And there are some locations in this grid
that are really tasty.
And we know the MDP.
So we know the transition model.
We know the rewards.
And what I'm showing here is the Q.
What I'm showing is the Q function,
the max over the Q's--
the max over the a's--
so basically the Q function, the maximum Q value in each square.
And the arrow shows the arg max.
It shows the best action to take.
So it shows the policy.
So to start with, I initialized it to all 0s.
And the arrows just point upward.
OK, so now I do one iteration.
So what I did here is I went through,
and I executed that Q update once
for all the states and actions--
just a for loop over states, a for loop over actions--
executed it once.
And what I found was that there are
some-- turns out that there are some states here
that have really high value.
OK, that's good.
And now I do another iteration.

I execute that equation again.
I compute it for every state and action pair.
And now, I find, oh, actually those
states that are near the really good ones,
they're not so bad themselves, because I
know that pretty soon in the future
I can get a nice pile of reward.
And you can see, also, that the actions are pointing in,
that the policy is updating.
So I do another iteration.
Looks like that.
And if I just keep iterating, eventually, it's interesting.
What happens is very--
sooner-- before the value function
has converged to the optimal value function,
the policy is already right.
At this point, the policy is good in all the squares,
I think.
Yes.
But if I keep going, oh, yes.
So now it converged to within whatever tolerance
I told it to converge to.
So that was pretty cool.
I'll show you one more demo, maybe.
Let's see.

So similar setup, but this time, there
are some walls in the world, so that I can't actually
go through the walls.
So there's one tasty goal location
in this particular setup that I made.
Now if I do one iteration, a value iteration,
I see that, well, it's good to be in those places to the left
and above this place.

And you can kind of think of it as the value propagating
outward.
So the way this particular domain works
is that there are walls.
And the robot can't go through the walls.
So it's learning a strategy to solve the maze,
basically from any initial location.
And you can see that it's gradually getting
lighter purple over there.
SUBJECT: Point at the end.
LESLIE KAELBLING: Excuse me.
SUBJECT: Point at the end [INAUDIBLE]..
LESLIE KAELBLING: Point at the end.
SUBJECT: Yellow dot [INAUDIBLE].
LESLIE KAELBLING: Oh, why is there an action out
of the yellow dot?
SUBJECT: [INAUDIBLE] I'm assuming the action out
of the yellow dot [INAUDIBLE]
LESLIE KAELBLING: The action out of the yellow
dot-- no, it's actually interesting.
Well, first of all, the robot doesn't have a stay action.
SUBJECT: [INAUDIBLE]
LESLIE KAELBLING: So in this case,
it has to go out and back.
Actually, the way this particular domain works
is after it hits the yellow dot, it goes into nirvana
where it stays forever.
And nothing happens anymore.
In this particular domain, that's how it works.
But, OK, so cool.
So what happened here was I put in a definition of an MDP.
I executed that update over and over and over.
That computed the Q function over the whole domain.
That let me compute the optimal policy.
So now I know how to make an optimal robot if I have
an MDP model of the domain.


## Source PDFs

- https://openlearninglibrary.mit.edu/assets/courseware/v1/caee9a9ff60ed183e4485869c2e88ac4/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Sequential_models.pdf
