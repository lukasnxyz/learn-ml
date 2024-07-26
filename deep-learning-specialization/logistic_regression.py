import math
import numpy as np

# XOR
data = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0]
])
X = data[:-1, :]
Y = data[-1, :]

def sigmoid(x: np.ndarray):
    return 1/(1+np.exp(-x))

def loss(yh: float, y: float) -> float:
    return -(y*math.log(yh)+(1-y)*math.log(1-yh))

def cost(yh: np.ndarray, y: np.ndarray):
    m, c = len(y), 0
    for i in range(m): 
        c += loss(yh[i], y[i])
    return c/m

def forward(x: np.ndarray, w: np.ndarray, b: float):
    return sigmoid(np.dot(w, x) + b)

def backward(x: np.ndarray, yh: np.ndarray, y: np.ndarray, w_dim: int):
    m = len(x)
    dz = yh - y
    dw = 1/m*np.dot(x, dz.T)
    db = 1/m*np.sum(dz)
    return dw, db

def train(iters: int, lr: float, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    for i in range(iters):
        # forward
        yh = forward(x, w, b)
        c = cost(yh, y)
        print(f'c: {c:.4f}, yh: {yh}')

        # backward
        dw, db = backward(x, yh, y, len(w))
        
        # optimize
        w += -lr*dw
        b += -lr*db
    return c

if __name__ == '__main__':
    w = np.zeros([X.shape[0]])
    b = 0.0

    loss = train(2, 0.01, X, Y, w, b)
