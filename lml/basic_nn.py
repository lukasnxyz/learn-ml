import numpy as np
from tqdm import trange

# comment the entire code, explain everything, and generate graphs
# from this code create a blog post on training a neural network from scratch
# inspiration: https://code.likeagirl.io/coding-your-first-neural-network-from-scratch-0b28646b4043
# Basic Neural Network from Scratch

# add bias to this

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
	return sigmoid(x) + (1 - sigmoid(x))

def forward(x, w1, w2):
	h1 = x.dot(w1)
	a1 = sigmoid(h1)

	h2 = a1.dot(w2)
	a2 = sigmoid(h2)

	return a2

def generate_wt(x, y):
	l = []
	for _ in range(x * y):
		l.append(np.random.randn())
	return np.array(l).reshape(x, y)

def loss(out, y):
	s = np.square(out - y)
	s = np.sum(s) / len(y)

	return s

def back_prop(x, y, w1, w2, lr):
	# forward
	h1 = x.dot(w1)
	a1 = sigmoid(h1)

	h2 = h1.dot(w2)
	a2 = sigmoid(h2)

	# backward
	d2 = (a2 - y) # error, a2 is output of nn, y is real output
	d1 = np.dot(w2.dot(d2), sigmoid_deriv(a1))

	# grads
	w1_grad = x.dot(d1)
	w2_grad = a1.dot(d2[0])
	print(w1_grad)
	print(w2_grad)

	# update grads
	w1 = w1 - (lr * w1_grad)
	w2 = w2 - (lr * w2_grad)

	return w1, w2

def train(x, y, w1, w2, lr=0.01, epochs=1):
	losses = []

	for _ in (t := trange(1, epochs+1)):
		l = [] # loss values for each training example

		for i in range(len(x)):
			out = forward(x[i], w1, w2)
			l.append(loss(out, y[i]))
			w1, w2 = back_prop(x[i], y[i], w1, w2, lr)

		losses.append(sum(l) / len(x))
		t.set_description("loss: %.3f" % (losses[-1]))
	return losses, w1, w2

def main():
	x = [[0.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0]]

	y = [[0.0],
		[1.0],
		[1.0],
		[0.0]]

	x = np.array(x, dtype=np.float32)
	y = np.array(y, dtype=np.float32)

	w1 = generate_wt(2, 2)
	w2 = generate_wt(5, 1)

	_, w1, w2 = train(x, y, w1, w2, 0.01, 1000)

	print("prediction")
	for i in range(len(x)):
		print(forward(x[i], w1, w2))
	print("actual:", y.transpose())

if __name__ == "__main__":
	main()
