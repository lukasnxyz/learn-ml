import numpy as np
import random
from datetime import datetime

class Xor():
    def __init__(self, a0, w1, b1, a1, w2, b2, a2):
        self.a0 = a0
        self.w1 = w1
        self.b1 = b1
        self.a1 = a1
        self.w2 = w2
        self.b2 = b2
        self.a2 = a2
        self.a2 = self.a2.reshape(-1, 1)

    def __str__(self):
        return f"a0 = {self.a0}\nw1 = {self.w1}\nb1 = {self.b1}\na1 = {self.a1}\nw2 = {self.w2}\nb2 = {self.b2}\na2 = {self.a2}"

    def forward(self): # go through each neuron in neural network (1 at a time) when called
        self.a1 = self.a0 * self.w1
        self.a1 += self.b1
        self.a2 = sigmoid(self.a2)

    def cost(self, ti, to): # square difference between predicted and actual
        c = 0

        n = ti.shape[0]
        for i in range(n):
            x = ti[i].reshape(-1, 1)
            y = to[i].reshape(-1, 1)

            q = to.shape[1]
            for j in range(q):
                d = self.a2[0, j] - y[0, j]
                c += d*d

        return c/n

def finite_dif(model, g, eps, ti, to):
    c = model.cost(ti, to)

    for i in range(model.w1.shape[0]):
        for j in range(model.w1.shape[1]):
            saved = model.w1[i, j]
            model.w1[i, j] += eps
            g.w1[i, j] = (model.cost(ti, to) - c)/eps
            model.w1[i, j] = saved;

    for i in range(model.b1.shape[0]):
        for j in range(model.b1.shape[1]):
            saved = model.w1[i, j]
            model.w1[i, j] += eps
            g.w1[i, j] = (model.cost(ti, to) - c)/eps
            model.w1[i, j] = saved;

    for i in range(model.w2.shape[0]):
        for j in range(model.w2.shape[1]):
            saved = model.w1[i, j]
            model.w1[i, j] += eps
            g.w1[i, j] = (model.cost(ti, to) - c)/eps
            model.w1[i, j] = saved;

    for i in range(model.b2.shape[0]):
        for j in range(model.b2.shape[1]):
            saved = model.w1[i, j]
            model.w1[i, j] += eps
            g.w1[i, j] = (model.cost(ti, to) - c)/eps
            model.w1[i, j] = saved;

def learn(model, g, rate):
    for i in range(model.w1.shape[0]):
        for j in range(model.w1.shape[1]):
            model.w1[i, j] -= rate*g.w1[i, j]

    for i in range(model.b1.shape[0]):
        for j in range(model.b1.shape[1]):
            model.b1[i, j] -= rate*g.b1[i, j]

    for i in range(model.w2.shape[0]):
        for j in range(model.w2.shape[1]):
            model.w2[i, j] -= rate*g.w2[i, j]

    for i in range(model.b2.shape[0]):
        for j in range(model.b2.shape[1]):
            model.b2[i, j] -= rate*g.b2[i, j]

def sigmoid(z): # make number between -1.0 and 1.0
    return 1 / (1 + np.exp(-z))

def main():
    random.seed(datetime.now().timestamp())

    td = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 0]
        ])

    ti = td[:, :2]
    to = td[:, 2]
    to = to.reshape(-1, 1)

    model = Xor(
        np.array([0, 1]), # a0
        np.random.rand(2, 2), # w1
        np.random.rand(1, 2), # b1
        np.array([1, 1]), # a1
        np.random.rand(2, 1), # w2
        np.random.rand(1, 1), # b2
        np.array([1, 1]), # a2
        )

    g = Xor(
        np.array([0, 1]), # a0
        np.random.rand(2, 2), # w1
        np.random.rand(1, 2), # b1
        np.array([1, 1]), # a1
        np.random.rand(2, 1), # w2
        np.random.rand(1, 1), # b2
        np.array([1, 1]), # a2
        )

    eps = 1e-5
    rate = 1e-10

    print("cost: ", model.cost(ti, to))
    for i in range(100):
        model.forward()
        finite_dif(model, g, eps, ti, to)
        learn(model, g, rate)
        print(i, "cost: ", model.cost(ti, to))

    a = np.array([
            [model.a1[0, 0]],
            [model.a2[0, 0]],
             ])

    for i in range(2):
        for j in range(2):
            model.a0[0] = i
            model.a0[1] = j
            model.forward()
            y = model.a2[0]

            print(i, "^", j, y)

if __name__ == "__main__":
    main()
