import numpy as np
from collections import Counter

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# not working

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = left
        self.value = value # if leaf node

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features # adds randomness to only use a subset of features in tree
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth):
        # check stopping crit
        # find best split
        # create child nodes
        n_samples, n_features = X.shape
        n_classes = np.unique(y)

        if(depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # get the features used to consider making the new split
        # randomness in decision trees
        feat_ids = np.random.choice(n_features, self.n_features, replace=False)

        # find best split
        best_thresh, best_feature = self._best_split(X, y, feat_ids)

    def _best_split(self, X, y, feat_ids):
        # find the best information gain
        split_threshold, split_id = None, None

        for feat_id in feat_ids:
            X_column = X[:, feat_id]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # calc information gain
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_id = feat_id
                    split_threshold = threshold

        return split_threshold, split_id

    def _information_gain(self, y):
        # entropy of parent - weight_average * entropy of children
        # parent entropy
        # create children
        # calc weighted entropy of children
        # calc information gain
        parent_entropy = self._entropy(y)

        left_ids, right_ids = self._split(X_column, threshold)
        if len(left_ids) == 0 or len(right_ids) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_ids), len(right_ids)
        e_l, e_r = self._entropy(y[left_ids]), self._entropy(y[right_ids])

    def _entropy(self, y):
        xs = np.bincount(y)
        ps = xs / len(y)
        return -np.sum[p*np.log(p) for p in ps if p>0]

    def _split(self, X_column, split_threshold):
        left_ids = np.argwhere(X_column <= split_threshold).flatten()
        right_ids = np.argwhere(X_column > split_threshold).flatten()
        return left_ids, right_ids

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self):

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
    print("Accuracy: %.2f" % (a))
