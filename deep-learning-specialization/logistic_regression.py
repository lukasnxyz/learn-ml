import math
import numpy as np

# XOR
X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
    ])
Y = np.array([0, 1, 1, 0])

def sigmoid(x: float):
    return 1/(1+math.exp(-x))

def loss(yh: float, y: float) -> float:
    return -(y*math.log(yh)+(1-y)*math.log(1-yh))

def cost(yh: np.ndarray, y: np.ndarray):
    m, c = len(y), 0
    for i in range(m): 
        c += loss(yh[i], y[i])
    return c/m

def forward(x: np.ndarray, w: np.ndarray, b: float):
    out = []
    for i in range(len(x)):
        out.append(sigmoid(np.dot(x[i], w) + b)) # yh
    return out

def backward(x: np.ndarray, yh: np.ndarray, y: np.ndarray, w_dim: int):
    m = len(x)
    dz = yh - y
    dw = np.dot(x.T, dz.T) / m
    db = np.sum(dz) / m
    return dw, db

def train(iters: int, lr: float, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    for i in range(iters):
        # forward
        yh = forward(x, w, b)
        c = cost(yh, y)
        print(f'c: {c:.4f}, yh: {yh}')

        # backward
        dw, db = backward(x, yh, y, len(w))
        
        ## optimize
        w += -lr*dw
        b += -lr*db
    return c

if __name__ == '__main__':
    w = np.array([0.1, 0.1])
    b = 0.3

    loss = train(10, 0.1, X, Y, w, b)
