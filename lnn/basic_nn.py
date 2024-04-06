import numpy as np
import matplotlib.pyplot as plt
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
	h1 = x.dot(w1)
		a1 = sigmoid(h1)

		h2 = h1.dot(w2)
		a2 = sigmoid(h2) # output (4,1)

		#L = (a2 - y) # loss

		# gradients
		#dL = 1

		#da2_da1 = w2
		#dL_dw2 = a1

		#da1_dx = w1
		#da1_dw1 = x

		d2 = (a2 - y) # loss
		d1 = np.dot(w2.dot(d2.transpose()), sigmoid_deriv(a1))

		# gradients
		w1_grad = x.transpose().dot(d1.transpose())
		w2_grad = a1.transpose().dot(d2)

		w1 = w1 - (lr * w1_grad)
		w2 = w2 - (lr * w2_grad)

		return w1, w2

def train(x, y, w1, w2, lr=0.01, epochs=1):
	losses = []

		for _ in (t := trange(1, epochs+1)):
			l = []
				out = forward(x, w1, w2)
				l.append(loss(out, y))

				w1, w2 = back_prop(x, y, w1, w2, lr)

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

		w1 = generate_wt(2, 4)
		w2 = generate_wt(4, 1)

		_, w1, w2 = train(x, y, w1, w2, 0.01, 1000)

		print("prediction:", forward(x, w1, w2).transpose())
		print("prediction:", y.transpose())

if __name__ == "__main__":
	main()
