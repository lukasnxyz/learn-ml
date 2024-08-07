import numpy as np
from numpy import load
from tqdm import tqdm

np.random.seed(42)

mnist_data = dict(load('data/mnist.npz'))
trlen = mnist_data['y_train'].shape[0]
testlen = mnist_data['y_test'].shape[0]
# split
tds = int(testlen*0.5)
# train
Xtr = mnist_data['x_train'].reshape(60000, -1).T
Ytr = mnist_data['y_train'].reshape(-1, 1).T
# dev
Xdev = mnist_data['x_test'].reshape(10000, -1).T[:, :tds]
Ydev = mnist_data['y_test'].reshape(-1, 1).T[:, :tds]
# test
Xtest = mnist_data['x_test'].reshape(10000, -1).T[:, tds:]
Ytest = mnist_data['y_test'].reshape(-1, 1).T[:, tds:]

print('shapes', Xtr.shape, Ytr.shape, Xdev.shape, Ydev.shape, Xtest.shape, Ytest.shape)

def onehot(x: int):
    z = np.zeros((10, 1))
    z[x] = 1
    return z
#def softmax(z: np.ndarray): return np.exp(z)/np.sum(np.exp(z))
def softmax(x: np.ndarray):
    #print('x', x)
    ex = np.exp(x)
    #print('ex', ex)
    s = np.sum(ex, axis=0).reshape(-1, 1).T
    ret = ex/s # (dist, samples)
    return ret
def relu(x: np.ndarray): return np.maximum(0, x)
def d_relu(x: np.ndarray): return (x>0)*1

def sigmoid(x: np.ndarray): return 1/(1+np.exp(-x))
def d_sigmoid(x: np.ndarray): return sigmoid(x)*(1-sigmoid(x))

def closs(yh: np.ndarray, y: np.ndarray, beta=1e-15): # cross-entropy
    return -np.sum(y*np.log(yh+beta))
def cost(yh: np.ndarray, y: np.ndarray):
    m, tot_loss = yh.shape[1], 0
    for i in range(m): tot_loss += closs(yh[:, i].reshape(-1, 1), onehot(y[:, i]))
    return tot_loss/m

def accuracy(yh: np.ndarray, y: np.ndarray):
    m = len(y)
    t = 0
    for i in range(m):
        if yh[i] == y[i]: t += 1
    return t/m

class Model:
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        self.W1 = np.random.randn(n_hidden, n_in) * (2/n_in)**0.5
        self.b1 = np.zeros((n_hidden, 1))
        # add a dropout maybe
        self.a1 = sigmoid
        self.W2 = np.random.randn(n_out, n_hidden) * (2/n_hidden)**0.5
        self.b2 = np.zeros((n_out, 1))
        self.a2 = softmax

    def forward(self, x: np.ndarray):
        l1 = self.a1(np.dot(self.W1, x) + self.b1)
        l2 = self.a2(np.dot(self.W2, l1) + self.b2)
        return l2

    def backward(self, x: np.ndarray, yh: np.ndarray, y: np.ndarray):
        m = x.shape[1] # (784, num of examples)
        print('W1', self.W1)
        print('x', x)
        z1 = np.dot(self.W1, x) + self.b1
        ai = self.a1(z1)

        dz2 = yh - onehot(y) 
        print('ai', ai)
        self.dW2 = 1/m*np.dot(dz2, ai.T)
        self.db2 = 1/m*np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(self.W2.T, dz2) * d_sigmoid(z1)
        self.dW1 = 1/m*np.dot(dz1, x.T)
        self.db1 = 1/m*np.sum(dz1, axis=1, keepdims=True)

    def step(self, lr: float):
        self.W1 += -lr*self.dW1
        self.W2 += -lr*self.dW2
        self.b1 += -lr*self.db1
        self.b2 += -lr*self.db2

m = Model(784, 128, 10)
batch_size = 32
epochs = 2
iepoch = 1
lossi = []

for i in (t := tqdm(range(epochs))):
    ix = np.random.randint(0, Xtr.shape[1], (batch_size,))
    Xb, Yb = Xtr[:, ix], Ytr[:, ix]

    yh = m.forward(Xb)
    loss = cost(yh, Yb)

    m.backward(Xb, yh, Yb)
    m.step(0.001)

    print(m.W2)

    t.set_description(f'loss: {loss:.4f}')
    lossi.append(loss)
    #if i % iepoch == 0:
        #print(np.argmax(yh, axis=0))

#yhs = []
#mi = Xdev.shape[1]
#for i in range(mi):
#    Xb = Xdev[:, i].reshape(-1, 1)
#    iyh = m.forward(Xb)
#    yhs.append(np.argmax(iyh))
#
#devprob = accuracy(yhs, Ydev.T)*100
#print(f'acc: {devprob:.2f}%')

#import matplotlib.pyplot as plt
#plt.plot(lossi)
#plt.show()
