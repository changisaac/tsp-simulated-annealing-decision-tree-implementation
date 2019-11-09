# CS486 Assignment 1
# Author: Isaac Chang

# Question 1 Instructions:
- The program is setup to currently run the 36 cities problem once using the temperature scheduling of exponential decrease with a base of 0.99999
- It takes around 25s to run through the problem.
- The results (final path, final distance) are printed to stdout.
- In addition, a plot of cost(distance) throughout the simulated annealing is shown.

## Custom Runs
- uncommenting lines 84, 85, or 88 excuslively will change the temperature schedule.
- modifying line 22 will increase the number of times the 36 cities problem is ran, if ran more than once, the lowest cost is shown at end of all runs.

# To Run:
- `cd src`
- `python q1.py`

# Queston 3 Instructions:
- The program is setup currently to first generate the decision tree using the training examples, then to run it on all examples in the test set. i
- The results (accuracy on training set, decision tree) is printed to stdout
- To interpret the output, it is printed sideways where the most left node i the root node.

# To Run:
- `cd src`
- `python q3.py`
