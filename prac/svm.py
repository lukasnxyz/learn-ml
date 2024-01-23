import numpy as np
from tqdm import tqdm

def accuracy(true, pred):
    return np.sum(test == pred) / len(test)

class SVM:
    def __init__(self, lr=0.001, lam=0.01, epochs=500):
        self.lr = lr
        self.lam = lam
        self.epochs = epochs
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)
