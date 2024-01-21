import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def accuracy(ys_pred, ys_test):
    return np.sum(ys_pred == ys_test) / len(ys_test)

class LogisiticRegression():
    def __init__(self, rate=1e-4, epochs=250):
        self.rate = rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, features, y): # features is a height and weight vector, y is a scalar 1 or 0 for male or female
        n_samples, n_features = features.shape # .shape gives number of rows, number of columns
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epochs):
            print("Epoch:", i)
            linear_pred = np.dot(features, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(features.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            print(dw.shape)
            print(dw)
            print(self.weights.shape)
            print(self.weights)
            self.weights = self.weights - dw * self.rate
            self.bias = self.bias - db* self.rate
            return

    def predict(self, features):
        linear_pred = np.dot(features, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y  in y_pred]
        return class_pred

def main():
    '''
    bc = datasets.load_breast_cancer()
    features, ys = bc.data, bc.target
    features_train, features_test, ys_train, ys_test = train_test_split(features, ys, test_size=0.2, random_state=1234)

    clf = LogisiticRegression(epochs=1000)
    print(features_train.shape)
    clf.fit(features_train, ys_train)
    ys_pred = clf.predict(features_test)
    print(accuracy(ys_pred, ys_test))

'''
    data = pd.read_csv("../data/height-weight.csv")

    features = np.full((len(data.height), 2), 0.0)
    ys = np.full((len(data.gender), 1), 0.0)
    for i in range(2):
        for x in range(len(data.height)):
            if i == 0:
                features[x][i] = data.height[x]
            else:
                features[x][i] = data.weight[x]

    for i in range(len(data.gender)):
        ys[i] = data.gender[i]

    clf = LogisiticRegression()
    clf.fit(features, ys)

    features_test = np.array([66.786927239528, 165.431242225646])
    ys_test = np.array([1])

    ys_pred = clf.predict(features_test)
    print(accuracy(ys_test, ys_pred))

    # need to use logistic regression for true or false problems
    # this is more of a true or false algo because of the sigmoid function
    # more for categorical data and classification

if __name__ == "__main__":
    main()
