### libml
I want to make a machine learning lib in C...

#### Quick Start

#### Notes
input		hidden		output
x1-w1------->(a1)
     \/  \
	 w3, w4    			> (a3)
     /\  /
x2-w2------->(a2)

| a1 = x1 * w1 + x2 * w3 | + b1
| a2 = x1 * w2 + x2 * w4 | + b2
