import numpy as np
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# a perceptron is the single unit if a neural network
# also known as the prototype for neural networks
# a single layer perceptron: can learn only linearly separable patterns
# multilayer perceptron (neural network): can learn more complex patterns
# a simplified model of a biological neuron, one cell

# inputs * weights -> net input function -> activation function (1 or 0)
#                                           neuron fires or not fires
# in single layer perceptron, class label 1 and 0

# actiation funtion: unit step function -. g(x)
# wx + b = f(x)
# g(f(x))

# for each training sample
# w = w + change in w
# b = b + change in bias
# change in w = lr (yi - yhati) xi
# change in b = lr (yi - yhati)
# lr in [0,1]
# yi actual class
# yhati approximation of yi with g(f(x))

# update rule explained
# y | yhat = y - yhat (change in w/b)
# 1 | 1    = 0
# 1 | 0    = 1
# 0 | 0    = 0
# 0 | 1    = -1
# The weights are pushed towards the target class in case of missclassification

# implemented this completely alone, I'm coming along!

def accuracy(true, pred):
    a = np.sum(true == pred) / len(true)
    return a

def unit_step(x):
    # 0 where x < 0, 1 where x > 0
    return np.where(x > 0, 1, 0)

def relu(x):
    # rectified linear unit
    # 0 where x < 0, x where x > 0
    return np.where(x > 0, x, 0)

class Perceptron:
    def __init__(self, lr=0.001, epochs=500, activ_func=unit_step):
        self.lr = lr
        self.epochs = epochs
        self.activ_func = activ_func # activation function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = [relu(y_i) for y_i in y]

        for _ in tqdm(range(self.epochs)):
            for id, x_i in enumerate(X):
                f = np.dot(x_i, self.weights) + self.bias
                fhat = self.activ_func(f)
                upd = self.lr * (y_[id] - fhat)

                self.weights += upd * x_i
                self.bias += upd

    def predict(self, X):
        preds = np.dot(X, self.weights) + self.bias
        y_preds = self.activ_func(preds)
        return y_preds

if __name__ == "__main__":
    '''
    with open("../data/height-weight.csv", "r") as file:
        reader = csv.reader(file)
        data_csv = list(reader)

    data = np.array(data_csv)
    data = np.delete(data, (0), axis=0)
    data = data.astype(float)

    X = data[:, :-1]
    y = data[:, -1]
    '''

    from sklearn import datasets
    X, y = datasets.make_blobs(
            n_samples=3000, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    perc = Perceptron(lr=0.0001, epochs=500, activ_func=relu)
    perc.fit(X, y)

    preds = perc.predict(X_test)
    a = accuracy(y_test, preds)

    print("Accuracy: %.3f" % (a))
