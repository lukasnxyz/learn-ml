import numpy as np
from sklearn.model_selection import train_test_split, train_test_split
from tqdm import tqdm
import csv

# You want to categorize objects into two or more classes
# Dog or cat? Stock up or down?
# Supervised learning
class SVM():
    def __init__(self, lr=1e-4, lam=1e-2, epochs=500):
        self.lr = lr
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
                    self.weights -= self.lr * (2 * self.lam * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lam * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)

def accuracy(true, pred):
    accuracy = np.sum(true == pred) / len(true)
    return accuracy

def main():
    with open("../data/height-weight.csv", "r") as file:
        next(file)
        reader = csv.reader(file)
        data_csv = list(reader)

        for row in data_csv:
            for i, x in enumerate(row):
                if len(x)< 1:
                    x = row[i] = "0.0"

    data = np.array(data_csv)
    data = data.astype(float)

    X = data[:, :-1]
    y = data[:, -1]
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = SVM(lr=0.001, epochs=1000)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    a = accuracy(y_test, predictions) * 100
    print("Accuracy: " + "{:.2f}%".format(a))

if __name__ == "__main__":
    main()
