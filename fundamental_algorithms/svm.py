import numpy as np
from sklearn.model_selection import train_test_split, train_test_split
#from sklearn import datasets
from tqdm import tqdm
import csv
#import matplotlib.pyplot as plt

# You want to categorize objects into two or more classes
# Dog or cat? Stock up or down?
# Supervised learning
class SVM:
    def __init__(self, rate=1e-4, lam=1e-2, epochs=500):
        self.rate = rate
        self.lam = lam
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        _, n_features = X.shape # _ = n_samples

        y_ = np.where(y <= 0, -1, 1)

        # Init weights
        # It would be better to randomly initialize the values
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in tqdm(range(self.epochs), desc="Training SVM"):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.rate * (2 * self.lam * self.weights)
                else:
                    self.weights -= self.rate * (2 * self.lam * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)

def accuracy(true, pred):
    accuracy = np.sum(true == pred) / len(true)
    return accuracy

def main():
    '''
    # dummy data
    X, y = datasets.make_blobs(
            n_samples=1000, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
    )
    '''

    with open("../data/height-weight.csv", "r") as file:
        reader = csv.reader(file)
        data_csv = list(reader)

    data = np.array(data_csv)
    data = np.delete(data, (0), axis=0)
    data = data.astype(float)

    X = data[:, 1:]
    y = data[:, 0]
    y = np.where(y == 0, -1, 1)

    '''
    rows, _ = X.shape
    for i in tqdm(range(rows), desc="Plotting"):
        plt.plot(X[i][0], X[i][1], 'ro')

    plt.xlabel("Height")
    plt.ylabel("Weight")
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = SVM(rate=1e-4, epochs=1000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    a = accuracy(y_test, predictions) * 100
    print("Accuracy: " + "{:.2f}%".format(a))

    #plt.show()

if __name__ == "__main__":
    main()
