For this lab:

*
You will need to understand the material in the notes on
non-parametric methods.

*
A code directory that will be useful for doing this lab can be found
here. Download this to your computer.
Alternatively, the code is available in a colab notebook here.

1) Basic operation

Look at the code directory and open the file non_parametric.py. You
will see an implementation of the recursive (greedy) decision-tree
building algorithm described in lecture, and an implemention of a
nearest neighbor classifier.

We will be running these algorithms on the
Auto-MPG dataset that we used
in Lab 3 and HW 3; you should look back at the description on the
Lab 3 and your results from HW 3 to re-familiarize yourself with the data.
In the non_parametric.py file, towards the end is the code from HW 3
that we used to read the data into Numpy matrices. The one difference
is that we are reading the individual points into the rows of the
matrices rather than the columns as we did before.

Find the place where features are defined. As before, you
may choose there whether a feature is kept in its raw form,
converted to standard or to one_hot. Note that the call

auto_data, auto_labels = auto_data_and_labels(auto_data_all, features)

will create the data and labels matrices. Crucially, it will also
print which feature index corresponds to which original feature. For example:

0 cylinders 3.0 one_hot
1 cylinders 4.0 one_hot
2 cylinders 5.0 one_hot
3 cylinders 6.0 one_hot
4 cylinders 8.0 one_hot
5 displacement std
6 horsepower std
7 weight std
8 acceleration std
9 origin 1.0 one_hot
10 origin 2.0 one_hot
11 origin 3.0 one_hot

The first number is the feature index, which plays a prominent role in
the decision-tree construction. Remember that the one_hot
transformation produces a number of features with values either 0 or 1
corresponding to a discrete value of the original feature, so:
2 cylinders 5.0 one_hot

means that feature index 2 corresponds to 1 when cylinders is equal to
5.0 and 0 otherwise.
Note that by default we're ignoring model name and model year, as we
discussed in Lab 3.
Two useful functions defined in the file are:

*
auto_test(data, labels, pct=0.25) splits the input data (randomly) into training
and test sets of fractions 1-pct and pct, respectively, and
runs the decision tree construction algorithm on the training data
and evaluates the resulting tree on the test set.
By default, it prints the tree to depth 5 (see the
definition of dt_eval) and prints (and returns) the accuracy

*
auto_xval(data, labels, k=10) does k-fold cross validation. It also calls
dt_eval and print the trees and accuracy. At the end the function
returns the average accuracy across the runs.

A tree (that corresponds to the feature definitions above) might look like this
(note that n is the number of remaining points in the subtree):

Feat 7 < 0.0476422698221 [ n= 294.0 ]
Feat 7 < -0.251184192133 [ n= 171.0 ]
Feat 2 < 0.5 [ n= 142.0 ]
Feat 8 < -0.940473332238 [ n= 8.0 ]
=> 1.00 [ n= 4.0 ]
=> 0.00 [ n= 4.0 ]
Feat 7 < -0.887737602216 [ n= 134.0 ]
=> 1.00 [ n= 74.0 ]
Feat 7 < -0.442150215158 [ n= 60.0 ]
Feat 8 < 0.837889708284 [ n= 42.0 ]
Depth > 5
Depth > 5
=> 1.00 [ n= 18.0 ]
Feat 8 < 0.239053174231 [ n= 29.0 ]
Feat 5 < -0.628541123663 [ n= 17.0 ]
Feat 5 < -0.853672534932 [ n= 11.0 ]
=> 0.00 [ n= 4.0 ]
Feat 6 < 1.35064454164 [ n= 7.0 ]
Depth > 5
Depth > 5
=> 0.00 [ n= 6.0 ]
Feat 8 < 0.583837845352 [ n= 12.0 ]
=> 1.00 [ n= 4.0 ]
Feat 6 < 0.961017160039 [ n= 8.0 ]
=> 1.00 [ n= 2.0 ]
Feat 6 < 1.14084518232 [ n= 6.0 ]
Depth > 5
Depth > 5
Feat 8 < 1.54560561217 [ n= 123.0 ]
Feat 8 < 0.638277530266 [ n= 117.0 ]
=> 0.00 [ n= 102.0 ]
Feat 5 < -0.203108971485 [ n= 15.0 ]
=> 0.00 [ n= 14.0 ]
=> 1.00 [ n= 1.0 ]
Feat 8 < 2.36220088587 [ n= 6.0 ]
Feat 3 < 0.5 [ n= 4.0 ]
=> 0.00 [ n= 3.0 ]
=> 1.00 [ n= 1.0 ]
=> 1.00 [ n= 2.0 ]

The printing is done by recursively exploring the tree, printing the
test at a node (Feat i < th [ n= n ] , then going to the left child
(the Yes node) first and then the right child (the No node). When it
gets to a leaf in the tree, it prints '=> p [ n= n ]', where $p$ is the
prediction (probability of positive label).

Make sure that you can read the tree. Note: Why are some of the
thresholds negative values?

1A)
What do the classes that we are trying to predict in this problem mean?
(Review the description in
Lab 3)

1B)
Run a 10-fold cross validation with the one-hot and standardized features. This will
print 10 trees, each with an accuracy, then an overall accuracy.

* Look back to Section 4 in HW 3
and find the best accuracy we got with Average Perceptron for those features and compare.

* Look at the tests being used at the top layers of the trees; what features are
the tests splitting on?

* Look at the leaves of the trees; what does the float between 0 and 1
represent? How do you decide what class to output?

1C)
Uncomment out the small two-feature set in the file and repeat. Draw the top two layers
of the tree, what do they represent?

1D)
Does it change anything if we use raw for origin and/or
standardize weight? Explain.

1E)
When might we want to use a decision tree classifier as opposed
to an averaged perceptron or neural net classifier?

Check this box and submit when you have finished all parts of this question.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

2) Parameter Tuning

Note the three variables defined as class variables at the top of the
definition of the class DTN in the code file.

* N_THRESHOLD = 4 # don't split if node has fewer examples than this

* H_THRESHOLD = .01 # don't split if node has entropy less than this

* H_REDUCTION_THRESHOLD = .001 # don't split if it doesn't reduce entropy by this

2A)
By changing these parameters, theoretically what is the minimum training error you could get?
Is this a good thing?

2B)
From a learning algorithm perspective, what is the purpose of these values?

2C)
Try changing these values, for example N_THRESHOLD over [1, 2, 4], H_THRESHOLD
over [.01, .1, .25] and H_REDUCTION_THRESHOLD over [.001, .01, .1]. Note
the effect on the accuracy of 10-fold cross-validation and on the result trees. To
expedite the process, you can use the config input variable to auto_xval to change
the values, so for example you can set config = [4, .01, .001] to run auto_xval using
the default values. Be prepared to explain your findings.
How big are the trees on average with the lowest values of the parameters vs the
highest values of the parameters?

You can set verbose='first_tree' to output only the first tree.

2D)
Could you change these parameters such that no splits occur? If so, how?

Check this box and submit when you have finished all parts of this question.

Submit
View Answer Ask for Help

You have infinitely many submissions remaining.

3) Nearest Neighbors

We can set the decision_tree parameter in auto_xval to False in order to
run a 1-nearest neighbor model instead.

3A)
Run 10-fold cross validation using 1-nearest neighbors and record the
accuracy.

3B)
Repeat using the small two-feature sets from 1C and 1D. How does using the
unprocessed raw features from 1D compare to the processed features from 1C?

3C)
We can run k-nearest neighbors by setting the config parameter to desired
value of k. Note that this is different from the k parameter, which describes
the number of folds in cross-validation. Choosing odd values of k is preferred,
to avoid ties in choosing the majority classification.

Can you get a higher cross validation accuracy on the first feature set using a
different value of k? If so, what k did you use and what accuracy does it achieve?