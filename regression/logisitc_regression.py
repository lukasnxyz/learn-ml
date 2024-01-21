import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

'''
def gradient_descent(m_now, b_now, points, rate):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = float(points.iloc[i].x)
        y = float(points.iloc[i].y)

        linear_pred = m_now * x + b_now
        pred = sigmoid(linear_pred)

        m_gradient = (1/n) * (x * (pred - y))
        b_gradient = (1/n) * (pred - y)

    m = m_now - m_gradient * rate
    b = b_now - b_gradient * rate

    return m, b

def train(m, b, points, rate, epochs):
    for i in range(epochs):
        m, b = gradient_descent(m, b, points, rate)
        print(f"{i}: m = {m}, b = {b}")

    return m, b

def predict(m, b, x):
    linear_pred = m * x + b
    pred = sigmoid(linear_pred)

    if pred <= 0.5:
        return 0
    else:
        return 1
'''

class LogisiticRegression():
    def __init__(self, rate=1e-3, epochs=250):
        self.rate = rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, features, y): # features is a height and weight vector, y is a scalar 1 or 0 for male or female
        n_samples, n_features = features.shape # .shape gives number of rows, number of columns
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_pred = np.dot(features, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(features.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights = self.weights - dw * self.rate
            self.bias = self.bias - db * self.rate

'''
    def predict(self, features):
        lienar_pred = np.dot(features, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y  in y_pred]
        return class_pred
    '''

def main():
    data = read_csv("../data/height-weight.csv")

    # need to use logistic regression for true or false problems
    # this is more of a true or false algo because of the sigmoid function
    # more for categorical data and classification

    features = np.array([data.height, data.weight])
    print(features)

if __name__ == "__main__":
    main()
