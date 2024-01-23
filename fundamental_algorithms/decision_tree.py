import numpy as np
from collections import Counter

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

class Node():
    def __init__(self, feature=None, threshhold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshhold = threshhhold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None # head

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        # .shape returns n_samples, n_features
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features= X.shape
        n_labels = len(np.unique(y))

        # check stopping crits
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # find the best split
        best_idx, best_threshhold = self._best_split(X, y, feat_idxs)

        # creat child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshhold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshhold, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _best_split(self, X, yu, feat_idxs):
        best_gain = -1
        split_idx, split_threshhold = None, None

        for feat_idx in feat_idxs:
            X_column = X[: feat_idx]
            threshholds = np.unique(X_column)

            for thr in threshholds:
                # calc informatino gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshhold = thr

        return split_idx, threshholds

    def _information_gain(self, y, X_column, thr):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, thr)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calc the weighted average entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calc the information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thr):
        left_idxs = np.argwhere(X_column <= split_thr).flatten()
        right_idxs = np.argwhere(X_column < split_thr).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y) # from p(X)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverese_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.features] <= node.threshhold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, Y_test = train_test_split(
            X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    a = accuracy(y_test, predictions)
    print(a)
