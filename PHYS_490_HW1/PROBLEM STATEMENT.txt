===PROBLEM STATEMENT===

In this assignment you implement a simple least square fitting using both the
analytic solution and stochastic gradient descent.

Deadline: End of day (Midnight) Eastern Time Zone Tuesday Jan 21, 2020

Create a git (Github.com) repository named Phys490HW1. The last commit to the
master branch before the deadline is used to test correctness of your code.
(Github's timestamp is a strict deadline for acceptance of your work).

Input file name is of the format *.in
Output file name is of the format *.out
Json file name is of the format *.json

The command to run your code should be of the format

python3 main.py data/file_name.in data/file_name.json

Therefore main.py from your implementation receives two flags (two file
names) and writes in a third file (not provided as a flag).

file_name is an identical name for both the input file (*.in) and the json
file (*.json). The json file is a dictionary of two hyperparameters (learning
rate and number of iterations of stochastic gradient descent). The output file
also should have the same name but appended with a .out extension.


==Example input file==

In example data/1.in the file contains the lines

----------1.in---------
14 20 69
16 3 -1
24 30 99
11 62 240
30 -4 -43
----------EOF----------

which should be interpreted as a training dataset of 5 samples (x, y) where x is
a two-dimensional real vector x = (x1, x2). Therefore each line of 1.in is of
the format:

x1 x2 y


==Example output file==

In the same example you are expected to create an output file with same path
and same file name but a .out extension (data/1.out) with an example content:

---------1.out---------
3.0000
-1.0000
4.0000

0.7180
-0.9106
4.0208
----------EOF----------

reporting two sets of numbers separated with an empty line in between and each
vector delimited by newline. Each number is printed in the file with 4 decimal
places. Therefore this file is of the format:

w_analytic1
w_analytic2
w_analytic3

w_gd1
w_gd2
w_gd3

The first vector (w_analytic) is the analytic solution to the least square
regression and the second (w_gd) is the returned solution from stochastic
gradient descent with learning rate and number of iterations determined in the
corresponding json file.


==Example json file==

The json file is a python dictionary with choices of the above-mentioned
hyperparameters. Depending on your implementation good choice of hyperparameters
might be different from these example values. By trial and error modify these
values so that your stochastic gradient descent produces results as close as
possible to the analytic solution.

---------1.json---------
{
	"learning rate": 0.0001,
	"num iter": 1000
}
-----------EOF----------



