A code directory that may be useful for doing this homework can be found here. You should implement your code
for Problem 1 (State Machines) in sm.py and Problem 5 (MDP Implementations) in mdp.py.
There are unit tests available for you run in tests.py.
Download this to your computer, or alternatively, use the colab notebook.
The colab notebook will also contain unit tests defined after each code question.

1) State Machines

We will implement state machines as sub-classes of the SM class, which specifies the start_state, transition_fn, and output_fn.

class SM:
start_state = None # default start state

def transition_fn(self, s, x):
'''s: the current state
x: the given input
returns: the next state'''
raise NotImplementedError

def output_fn(self, s):
'''s: the current state
returns: the corresponding output'''
raise NotImplementedError

def transduce(self, input_seq):
'''input_seq: the given list of inputs
returns: list of outputs given the inputs'''
raise NotImplementedError

An example of a sub-class is the Accumulator state machine, which adds up (accumulates) its input and outputs the sum. Convince yourself that the implementation works as expected before moving on.

class Accumulator(SM):
start_state = 0

def transition_fn(self, s, x):
return s + x

def output_fn(self, s):
return s

An example case of using Accumulator (with transduce)

sm = Accumulator()

# Output should = [-1, 1, 4, 2, 7, 13]
output = sm.transduce([-1, 2, 3, -2, 5, 6])

You can find these definitions in the sm.py file.

1.1) Transduce

Implement the transduce method for the SM class. It is given an
input sequence (a list) and returns an output sequence (a list) of the
outputs of the state machine on the input sequence. Assume self.transition_fn and self.output_fn are defined.

12345678class SM: start_state = None def transduce(self, input_seq): '''input_seq: a list of inputs to feed into SM returns: a list of outputs of SM''' pass # Your code hereXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

1.2) Binary Addition

Implement a Binary_Addition state machine that takes in a sequence
of pairs of binary digits (0,1) representing two reversed binary
numbers and returns a sequence of digits representing the reversed
sum. For instance, to sum two binary numbers 100 and 011, the input sequence will be [(0, 1), (0, 1), (1, 0)]. You will need to define start_state, transition_fn, and output_fn. Note that when transduced, the input sequence may need to be extended with an extra (0,0) to output the final carry. The inputs in the test cases have already been extended with the extra (0, 0).

An example case of using Binary_Addition is

sm = Binary_Addition()

# Output should = [0, 0, 1]
output = sm.transduce([(1, 1), (1, 0), (0, 0)])

1234567891011class Binary_Addition(SM): start_state = None # Change def transition_fn(self, s, x): # Your code here pass def output_fn(self, s): # Your code here passXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

1.3) Reverser

Implement a state machine that reverses a sequence. The input is a list of
the form:
sequence1 + ['end'] + sequence2 + refers to concatenation. sequence1 is
a list of strings, the 'end' string indicates termination, and sequence2 is arbitrary. The machine reverses sequence1: for each entry in the sequence1, the machine outputs None. For the 'end' input and each
entry in the second sequence, an item from the reversed sequence1 is
output, or None if no characters remain.
To view test cases, try submitting your code. An example case of using Reverser

sm = Binary_Addition()

# Output should = [None, None, None, 'bar', ' ', 'foo', None, None, None]
output = sm.transduce(['foo', ' ', 'bar'] + ['end'] + list(range(5)))

1234567891011class Reverser(SM): start_state = None # Change def transition_fn(self, s, x): # Your code here pass def output_fn(self, s): # Your code here passXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

1.4) RNN

A neural network is a (learned) function that maps input vectors
to output vectors. A recurrent neural network (RNN) is a
(learned) state machine that maps input sequences to output sequences.

In particular, an RNN has a transition function and an output
function, each of which is defined in terms of weight matrices, offset
vectors and activation functions, analogously to standard neural
networks.

* The inputs $x$ are $l \times 1$ vectors

* The states $s$ are $m \times 1$ vectors

* The outputs $y$ are $n \times 1$ vectors

The behavior is defined as follows:

\begin{align*} s_{t} & = f_1(W^{ss} s_{{t-1}} + W^{sx} x_{t} + W^{ss}_0) \\ y_{t} & = f_2(W^o s_{t} + W^o_0) \end{align*}
where $f_1$ and $f_2$ are two activation functions, such as linear, softmax, or tanh. This arrangement is shown below.

Note that each input i below has dimension l x 1. Implement the corresponding state machine, where the weights are given in __init__. Make sure to set an appropriate start_state.

1234567891011class RNN(SM): def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2): # Your code here pass def transition_fn(self, s, x): # Your code here pass def output_fn(self, s): # Your code here passXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

1.5) Accumulator Sign RNN

Enter the parameter matrices and vectors for an instance of the
RNN class such that the output is 1 if the cumulative sum
of the inputs is positive, -1 if the cumulative sum is negative and 0
if otherwise. Make sure that you scale the outputs so that the output
activation values are very close to 1, 0 and -1. Note that both the
inputs and outputs are 1x1.

Hint: np.tanh may be useful. Remember to convert your python lists to np.array.

123456789Wsx = # Your code hereWss = # Your code hereWo = # Your code hereWss_0 = # Your code hereWo_0 = # Your code heref1 = # Your code here, e.g. lambda x : xf2 = # Your code hereacc_sign = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

1.6) Autoregressive RNN

Enter the parameter matrices and vectors for an instance of the
RNN class such that it implements the following
autoregressive model:

$$
y_t = 1 y_{t-1} - 2 y_{t-2} + 3 y_{t-3}
$$
when $x_t = y_{t-1}$.

Note: unlike in the lab, as a result of your call to RNN, your initial (start) state
vector in the RNN will be initialized to all zeros (i.e., our RNN
implementation enforces this requirement). Instead, you should assume
that the initial value for $x$ will be provided , e.g., $x_1 = 5$ in the lab example,
and that each successive $x_t = y_{t-1}$ is also provided to
the transduce method.

123456789Wsx = # Your code hereWss = # Your code hereWo = # Your code hereWss_0 = # Your code hereWo_0 = # Your code heref1 = # Your code heref2 = # Your code hereauto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

2) MDP

We consider the same tiny MDP as in the exercises, with states (0, 1, 2, 3) and actions ('b', 'c'). The reward function is:

$$
R(s,a) = \begin{cases} 1& \text{if $s=1$} \\ 2& \text{if $s=3$} \\ 0& \text{otherwise} \end{cases}
$$

You get the reward associated with a state on the step when you exit that state. The transition function for each action is below, where $T[i,x,j]$ is the $P(s_{t+1}=j | a=x, s_t=i)$ (note rows correspond to the input states, and columns correspond to the output states).

$$
T(s_t,\text{'b'}, s_{t+1})= \begin{bmatrix} 0.0 & 0.9 & 0.1 & 0.0 \\ 0.9 & 0.1 & 0.0 & 0.0 \\ 0.0 & 0.0 & 0.1 & 0.9 \\ 0.9 & 0.0 & 0.0 & 0.1 \end{bmatrix}
$$

$$
T(s_t, \text{'c'},s_{t+1})= \begin{bmatrix} 0.0 & 0.1 & 0.9 & 0.0 \\ 0.9 & 0.1 & 0.0 & 0.0 \\ 0.0 & 0.0 & 0.1 & 0.9 \\ 0.9 & 0.0 & 0.0 & 0.1 \end{bmatrix}
$$

Note that the only effect of the action is to change the transition probability from state 0.

2.1) Limited horizons

Consider two policies: one that always takes action 'b' in state 0 and one that always takes action 'c'.

2.1.A)
Which policy is best if you're starting from state 0 with horizon 2?

--
Taking action 'b'
Taking action 'c'

2.1.B)
Which policy is best if you're starting from state 0 with horizon 3?

--
Taking action 'b'
Taking action 'c'

2.1.C)
What if we start in state 0 with horizon 5, take action 'c', land in
state 2 with horizon 4, land in state 3 with horizon 3 (and get
reward = 2!), and then land in state 0 with horizon 2? What action
should we take now?

--
Action 'b'
Action 'c'

2.2) At a discount

Now, in the same tiny MDP, we are interested in the infinite-horizon
discounted value, with discount factor 0.9. For a given policy, we
can write down a set of linear equations characterizing these values,
in the form:

\begin{align} v_0 & = r_0 + c_{00}v_0 + c_{01} v_1 + c_{02} v_2 + c_{03}v_3 \newline v_1 & = r_1 + c_{10}v_0 + c_{11} v_1 + c_{12} v_2 + c_{13}v_3 \newline v_2 & = r_2 + c_{20}v_0 + c_{21} v_1 + c_{22} v_2 + c_{23}v_3 \newline v_3 & = r_3 + c_{30}v_0 + c_{31} v_1 + c_{32} v_2 + c_{33}v_3 \end{align}

2.2.A)
Enter the vector of $r_{i}$ values, in the order shown above.

Enter the matrix as a list of 4 numbers:

2.2.B)
Enter the matrix of $c_{ij}$ values, in the order shown above, for the policy "always take action 'c'".

Enter the matrix as a list of 4 lists of numbers:

Solve for the value function under this policy by using numpy.linalg.solve to solve these equations.
For example, if you have a set of linear equations of the form:

1 v0 + 2 v1 + 3 v2 = 4
5 v0 + 6 v1 + 7 v2 = 8
9 v0 + 10 v1 + 11 v2 = 12

Then you can solve as follows:
A = np.matrix([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
b = np.matrix([[4], [8], [12]])
v = np.linalg.solve(A,b) # A v = b

2.2.C)

Enter a Python list of 4 numbers, accurate to at least 3 decimal places:

3) Tiny Q-value iteration

In the same example as above, with an infinite horizon and a discount factor of 0.9, compute three iterations of value iteration. Don't assume a particular policy. Assume that:

* All the value estimates start at 0 (meaning, at iteration 0, $Q(s, a) = 0$ for all $s, a$ pairs), and

* You operate synchronously (that is, on iteration $t$ of value iteration, you only use values that were computed on iteration $t-1$).

We recommend you compute the Q-value iteration by hand to get a better understanding of the algorithm.
For each iteration enter 8 numbers corresponding to:

$$
[Q(0,b), Q(0,c), Q(1,b), Q(1,c), Q(2,b), Q(2,c), Q(3,b), Q(3,c)]
$$
at that iteration, accurate to at least 3 decimal places.

3.A)
Iteration 1

Enter a Python list of 8 numbers:

3.B)
Iteration 2

Enter a Python list of 8 numbers:

3.C)
Iteration 3

Enter a Python list of 8 numbers:

3.D)
After the third iteration, what action would you select in state 0?

--
Action 'b'
Action 'c'

4) One D Grid

Consider a domain where states correspond to squares on a 1 by 10 grid. The state is the position of a robot on the board, $i$, which ranges from 1 to 10. The robot can move one square East or West. In this question, assume there is no randomness in the robot's motion. It cannot move off the board.

4.1) Negative Reward

There is a "goal" state $i_g$. Assume that the reward for taking any action from the goal state is 0, and that, once in the goal state, the robot "sticks" there forever. Further, assume the reward for all other actions is -1. $H$ is the horizon.

4.1.A)
If the goal is $10$, $H = 5$ and $\gamma = 1$, what is $V^*(1)$?

Enter a number:

4.1.B)
If the goal is $10$, $H = 15$ and $\gamma = 1$, what is $V^*(1)$?

Enter a number:

4.1.C)
If the goal is $10$, $H = \infty$ and $\gamma = 1$, what is $V^*(1)$?

Enter a number: