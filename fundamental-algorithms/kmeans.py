import numpy as np
import matplotlib.pyplot as plt

# unsupervised (unlabeled data)
# clusteres data into k clusters with nearest means
# random cluster centers at start
# euclidean distance

def euclidean_distance(sample, point):
    return np.sqrt(np.sum((point - sample) ** 2))

class KM:
    def __init__(self, K=5, epochs=100, plot_steps=False):
        self.K = K
        self.epochs = epochs
        self.plot_steps = plot_steps

        # list of sample indicies for each cluster (class)
        self.clusters = [[] for _ in range(self.K)]

        # store centers (mean vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # init
        random_sample_ids = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[id] for id in random_sample_ids]

        # optimize clusters
        for _ in range(self.epochs):
            # assign samples to closest samples (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calc new centroids from clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classifiy samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        # assign the sampels to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for id, sample in enumerate(self.X):
            centroid_id = self._closest_centroid(sample, centroids)
            clusters[centroid_id].append(id)

        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_id = np.argmin(distances)
        return closest_id # closest centroid

    def _get_centroids(self, clusters):
        # assign the mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_id, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_id] = cluster_mean

        return centroids

    def _is_converged(self, centroids_old, centroids):
        # check the distances between old and new centroids for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_id, cluster in enumerate(clusters):
            for sample_id in cluster:
                labels[sample_id] = cluster_id

        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, id in enumerate(self.clusters):
            point = self.X[id].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

# testing
if __name__ == "__main__":
    np.random.seed(42)
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )

    clusters = len(np.unique(y))
    print(clusters)

    k = KM(K=clusters, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()
