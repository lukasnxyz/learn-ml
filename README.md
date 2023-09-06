### Machine-Learning-Practice
This is a repo for my practice in machine learning. I plan on making a machine learning library in C
called `libml` and make it an evergrowing project and write papers about it, but that is still far
from even barely working. For now, this is just basic ml implemented with linear algebra.

#### Quick Start

#### Notes
- https://machinelearningmastery.com/start-here/#getstarted
- Read `The Hundred Page Machine Learning Book`, it explains alot: https://themlbook.com/wiki/doku.php?id=contents
- gnuplot -> plot "cost.txt" with lines = graph out cost curve
- y = w(x) OR y = w(x) - b
- goal -> w(x) - b = 0
- square result from cost function to get more amplified result
- input is usually a feature vector
	1. Any value for w
	2. Give to w to cost function to get prediction precision (close to 0, the more precise and accurate)
	3. w - derivative of cost function (limit as h->0)
	4. Apply learning rate
	5. Iterate many times
