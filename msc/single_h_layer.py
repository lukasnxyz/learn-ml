import numpy as np

np.random.seed(42)

# OR
data = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 1]])
Xtr = data[:-1, :] # (2, 4)
Ytr = data[-1, :] # (4,)

def relu(x: np.ndarray): return np.maximum(0, x)
def d_relu(x: np.ndarray): return (x>0)*1
def sigmoid(x: np.ndarray): return 1/(1+np.exp(-x))
def loss(yh: float, y: float): return -(y*np.log(yh)+(1-y)*np.log(1-yh))

def cost(yh: np.ndarray, y: np.ndarray):
    m, c, yh = len(y), 0, yh.T
    for i in range(m): c += loss(yh[i], y[i]) 
    return (c/m).squeeze()

def forward(x: np.ndarray, w1: np.ndarray, w2: np.ndarray, b1: np.ndarray, b2: float):
    l1 = relu(np.dot(w1, x) + b1)
    l2 = sigmoid(np.dot(w2, l1) + b2)
    return l2

def backward(x: np.ndarray, yh: np.ndarray, y: np.ndarray, W1: np.ndarray, W2: np.ndarray, b1: np.ndarray):
    m = x.shape[1]
    z1 = np.dot(W1, x) + b1
    a1 = relu(z1)

    dz2 = yh - y
    dw2 = 1/m*np.dot(dz2, a1.T)
    db2 = 1/m*np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(W2, dz2.T) * d_relu(z1)
    dw1 = 1/m*np.dot(dz1, x.T)
    db1 = 1/m*np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2

def train(iters: int, ieval: int, lr: float, x: np.ndarray, y: np.ndarray, W1: np.ndarray, W2: np.ndarray, b1: np.ndarray, b2: float):
    c = 0
    for i in range(iters):
        # forward
        yh = forward(x, W1, W2, b1, b2)
        c = cost(yh, y)
        if i % ieval == 0: print(f'c: {c:.4f}, yh: {yh}')

        # backward
        dw1, db1, dw2, db2 = backward(x, yh, y, W1, W2, b1)

        # step
        W1 += -lr*dw1
        b1 += -lr*db1
        W2 += -lr*dw2
        b2 += -lr*db2
    return c, yh

if __name__ == '__main__':
    # layer 1
    W1 = np.random.randn(4, 2) #* 0.01 # n_neurons x n_inputs
    b1 = np.zeros((4, 1)) # n_neurons x n_inputs
    # layer 2
    W2 = np.random.randn(1, 4) #* 0.01 # n_neurons x n_inputs
    b2 = 0 

    cost, yh = train(2000, 100, 1e-3, Xtr, Ytr, W1, W2, b1, b2)
    print(np.round(yh))
