import numpy as np

# AND
data = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 0, 0, 1]
])
X = data[:-1, :]
Y = data[-1, :]

def sigmoid(x: np.ndarray): return 1/(1+np.exp(-x))

def d_sigmoid(x: np.ndarray): return sigmoid(x)*(1-sigmoid(x))

def loss(yh: float, y: float): return -(y*np.log(yh)+(1-y)*np.log(1-yh))

def cost(yh: np.ndarray, y: np.ndarray):
    m, c = len(y), 0
    yh = yh.T
    for i in range(m): 
        c += loss(yh[i], y[i])
    return c/m

def forward(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, b: np.ndarray):
    x1 = sigmoid(np.dot(w1.T, x) + b[0])
    x2 = sigmoid(np.dot(w2.T, x1) + b[1])
    return x2

def backward(x: np.ndarray, yh: np.ndarray, y: np.ndarray, w1: np.ndarray, w2: np.ndarray, b: np.ndarray):
    m = x.shape[1]
    z1 = np.dot(w1.T, x) + b[0]
    a1 = sigmoid(z1)

    dz2 = yh - y
    dw2 = 1/m*np.dot(dz2, a1.T)
    db2 = 1/m*np.sum(dz2)

    dz1 = np.dot(w2, dz2) * d_sigmoid(z1)
    dw1 = 1/m*np.dot(dz1, x.T)
    db1 = 1/m*np.sum(dz1)

    return dw1, db1, dw2, db2

def train(iters: int, ieval: int, lr: float, x: np.ndarray, y: np.ndarray, w1: np.ndarray, w2: np.ndarray, b: np.ndarray):
    for i in range(iters):
        # forward
        yh = forward(x, w1, w2, b)
        c = cost(yh, y)
        if i % ieval == 0: print(f'c: {c.item():.4f}, yh: {yh}')

        # backward
        dw1, db1, dw2, db2 = backward(x, yh, y, w1, w2, b)

        # step
        w1 += -lr*dw1.T
        b[0] += -lr*db1
        w2 += -lr*dw2.T
        b[1] += -lr*db2
    return c, yh

if __name__ == '__main__':
    # layer 1
    w1 = np.zeros((2, 4)) # n_weights x n_neurons 
    # layer 2
    w2 = np.zeros((4, 1)) # n_weights x n_neurons
    b = np.zeros((2, 1)) # n_layers x n_biases (always 1)

    cost, yh = train(5000, 500, 0.3, X, Y, w1, w2, b)
    print(f'{yh[0][3]:.4f}')
    print(np.round(yh))
