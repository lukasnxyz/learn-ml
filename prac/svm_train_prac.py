from multi_class_svm import SVM, accuracy
import numpy as np

from csv import reader
from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
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

    clf = SVM(lr=0.001, kernel="rbf")
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    a = accuracy(y_test, predictions) * 100
    print("Accuracy: " + "{:.2f}%".format(a))
