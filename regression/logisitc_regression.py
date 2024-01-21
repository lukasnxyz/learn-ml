import numpy as np
import csv

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

            self.weights = self.weights - dw * self.rate
            self.bias = self.bias - db* self.rate

    def predict(self, features):
        linear_pred = np.dot(features, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred

def main():
    #data = pd.read_csv("../data/height-weight.csv")
    with open("../data/height-weight.csv", "r") as file:
        reader = csv.reader(file)
        data_csv = list(reader)

    data = np.array(data_csv)
    data = np.delete(data, (0), axis=0)
    data = data.astype(float)

    features = data[:, 1:]
    ys = data[:, 0]

    clf = LogisiticRegression()
    clf.fit(features, ys)

    features_test = np.array([[66.786927239528, 165.431242225646],
        [72.6331067350844, 208.066256946822],
        [63.360509749283, 134.955817345201],
        [61.2018091899706, 102.179562817238]])
    ys_test = np.array([1, 1, 0, 0])

    ys_pred = clf.predict(features_test)
    print(accuracy(ys_test, ys_pred))

    # need to use logistic regression for true or false problems
    # this is more of a true or false algo because of the sigmoid function
    # more for categorical data and classification

if __name__ == "__main__":
    main()
