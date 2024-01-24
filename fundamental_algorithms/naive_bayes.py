import numpy as np

# probability
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calc mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c] # get all sampels of current class c
            self._mean[idx, :] = X_c.mean(axis=0) # mean of all features of class c
            self._var[idx, :] = X_c.var(axis=0) # variance of all features of class c
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        # calc posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x))) # posterior function
            posterior += prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)] # highest posterior probability

    def _pdf(self, idx, x):
        mean = self._mean[idx]
        var = self._var[idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    X, y = datasets.make_classification(
            n_samples=1000, n_features=10, n_classes=2, random_state=123
    )

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
    )

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(accuracy(y_test, predictions))
