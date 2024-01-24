import numpy as np
from tqdm import tqdm

from scipy.spatial import distance # for Guassian kernel
from scipy.interpolate import Rbf

# note: can use something like
#   @SVMClass
#   def fit(self, X, y):
#       implement_func()
# to implement a method outside of it's parent class

# Practice:
# Implement working kernels
# Implement multiclass (>= 3)

def accuracy(true, pred):
    return np.sum(true == pred) / len(true)

class SVM:
    k_linear = lambda x, xp, q=0: np.dot(x.T, xp)
    k_polynomial = lambda x, xp, q=2: (1 + np.dot(x.T, xp)) ** q
    #k_rbf = lambda x, xp, q=10: np.exp(-q * distance.cdist(x, xp, "sqeuclidean"))
    k_rbf = lambda x, xp, q=10: np.exp(-q*np.sum((xp-x[:,np.newaxis])**2,axis=-1))
    # kernel --> K(x, xp)
    kernel_funcs = {"linear": k_linear, "polynomial": k_polynomial, "rbf": k_rbf}

    def __init__(self, kernel="linear", k=2, lr=0.001, lam=0.1, epochs=500):
        self.lr = lr
        self.lam = lam # regularization param (hardness of margin)
        self.epochs = epochs

        self.weight = None
        self.bias = None

        self.kernel_name = kernel # store kernel name as string
        self.kernel = SVM.kernel_funcs[kernel] # kernel type
        self.k = k # kernel param for kernel function (polynomial, rbf)
        self.K = None

        # multi-class
        #self.multiclass = False
        #self.clfs = []

    def fit(self, X, y): # multi_class=False
        # check if multiclass
        #if len(np.unique(y)) > 2:
            #return multi_fit(X, y)

        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in tqdm(range(self.epochs), desc="Training SVM"):
            for idx, x_i in enumerate(X):
                self.K = self.kernel(X, X,  q=10)

                decision_function = np.dot(self.K, self.weights) - self.bias
                condition = y[idx] * decision_function >= 1

                if condition[0]: # line is overfitted
                    self.weights -= self.lr * (2 * self.lam * self.weights)
                else: # line is underfitted
                    gradient = (2 * self.lam * self.weights) - (y[idx] * self.K)
                    self.weights -= float(self.lr) * gradient
                    self.bias -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)

if __name__ == "__main__":
    from csv import reader
    from sklearn.model_selection import train_test_split

    with open("../data/height-weight.csv", "r") as file:
        next(file)
        reader = reader(file)
        data_csv = list(reader)

    data = np.array(data_csv)
    data = data.astype(float)

    X = data[:, :-1]
    y = data[:, -1]
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = SVM(lr=0.001, kernel="linear", lam=5, epochs=750)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    a = accuracy(y_test, predictions) * 100
    print("Accuracy: " + "{:.2f}%".format(a))
