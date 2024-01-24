import numpy as np
from tqdm import tqdm

from scipy.spatial import distance # for Guassian kernel

# note: can use something like
# @SVMClass
# def fit(self, X, y):
# to implement a method outside of it's parent class

def accuracy(true, pred):
    return np.sum(true == pred) / len(true)

class SVM:
    k_linear = lambda x1, x2, c=0: np.dot(x1, x2.T)
    k_polynomial = lambda x1, x2, q=5: (1 + np.dot(x2.T, x1)) ** q
    # radial basis function (Guassian kernel)
    k_rbf = lambda x1, x2, y=10: np.exp(-y * distance.cdist(x1, x2, "sqeuclidean"))
    # kernel --> K(x1, x2)
    kernel_funcs = {"linear": k_linear, "polynomial": k_polynomial, "rbf": k_rbf}

    def __init__(self, kernel="linear", k=2, lr=0.001, lam=0.01, epochs=500):
        self.lr = lr
        self.lam = lam # regularization param (hardness of margin)
        self.epochs = epochs

        self.weight = None
        self.bias = None

        self.kernel_name = kernel # store kernel name as string
        self.kernel = SVM.kernel_funcs[kernel] # kernel type
        self.k = k # kernel param for kernel function (polynomial, rbf)

        # multi-class
        #self.multiclass = False
        #self.clfs = []

    def fit(self, X, y): # multi_class=False
        # check if multiclass
        #if len(np.unique(y)) > 2:
            #return multi_fit(X, y)

        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)
        self.K = self.kernel(X, X, self.k)

        self.weights = np.zeros(n_features)
        self.bias = 0

        print(X.shape)
        print(self.K.shape)

        for _ in tqdm(range(self.epochs), desc="Training SVM"):
            for idx, x_i in enumerate(X):
                decision_function = np.dot(self.K[idx], self.weights) - self.bias
                condition = y[idx] * decision_function >= 1

                if condition[0]: # line is overfitted
                    self.weights -= self.lr * (2 * self.lam * self.weights)
                else: # line is underfitted
                    gradient = (2 * self.lam * self.weights) - (y[idx] * self.K)
                    self.weights -= self.lr * gradient
                    self.bias -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)
