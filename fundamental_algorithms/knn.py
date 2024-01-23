import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)

# Not really an algo, more of a classification given an input
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Returns class based on distances
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    # Computes distances
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indicies = np.argsort(distances)[:self.k] # Indices of the closest k neighbors, bc it's sorted
        k_nearest_labels = [self.y_train[i] for i in k_indicies]

        most_common = Counter(k_nearest_labels).most_common()

        return most_common[0][0]

if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = KNN(k=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = np.sum(predictions == y_test) / len(predictions)
    print(acc)
