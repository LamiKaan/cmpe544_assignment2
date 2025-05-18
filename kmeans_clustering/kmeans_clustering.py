import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from utils import compile_data_for_SVM, upscale_and_enhance_image, calculate_clustering_accuracy
import time
import cv2

class KMeansClustering:
    def __init__(self, n_clusters=5, initialization='random', distance_metric='euclidean', max_iter=1000, tol=1e-5, random_state=544):
        # Initialize the variables for KMeans clustering algorithm
        self.k = n_clusters
        self.initialization = initialization
        self.distance_metric = distance_metric
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        # Variables to hold the centroids and labels
        self.centroids = None
        self.labels = None
        # Create an instance variable to hold numpy random number generator (with a seed for reproducibility)
        self.rng = np.random.default_rng(self.random_state)
        # Variable to store the results
        self.results = {'internal': {}, 'external': {}}

    def initialize_centroids(self, X):
        # Randomly select k samples as initial centroids
        indices = self.rng.choice(len(X), self.k, replace=False)
        return X[indices]

    def compute_distances(self, X, centroids):
        if self.distance_metric == 'euclidean':
            # Euclidean distance from each point to each centroid
            return np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        elif self.distance_metric == 'manhattan':
            pass
        elif self.distance_metric == 'cosine':
            pass
        else:
            pass

    def assign_clusters(self, distances):
        # Assign each sample to the closest centroid
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        # Create empty list to hold new centroids
        new_centroids = []

        # For each cluster, compute the mean of the points assigned to it
        for i in range(self.k):
            # Get points assigned to the current cluster
            cluster_points = X[labels == i]

            # If there are points in the cluster, compute the mean
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
            else:
                # Reinitialize empty cluster centroid randomly
                new_centroid = X[self.rng.choice(len(X))]
            
            # Append the new centroid to the list
            new_centroids.append(new_centroid)
        
        # Convert list to numpy array
        return np.array(new_centroids)

    def fit(self, X):
        # Initialize centroids from the dataset
        self.centroids = self.initialize_centroids(X)

        # For a maximum number of iterations or until convergence
        for i in range(self.max_iter):
            # Store old/current centroids for convergence check
            old_centroids = self.centroids.copy()

            # Compute distances from each point to each centroid
            distances = self.compute_distances(X, self.centroids)
            # Assign each sample to the cluster with the nearest centroid
            self.labels = self.assign_clusters(distances)
            # Update centroids based on the assigned clusters
            self.centroids = self.update_centroids(X, self.labels)

            # Check convergence
            shift = np.linalg.norm(self.centroids - old_centroids)
            if shift < self.tol:
                break

        return self

    def predict(self, X):
        distances = self.compute_distances(X, self.centroids)
        return self.assign_clusters(distances)

    def calculate_results(self, X, y):
        # FIRST, FOR INTERNAL METRICS

        # SSE (Sum of Squared Errors)
        # Calculate the distance of each point to cluster centers
        distances = self.compute_distances(X, self.centroids)
        # Get the minimum distance for each point to its assigned cluster center, and sum their squares
        sse = np.sum((np.min(distances, axis=1))**2)
        self.results['internal']['sse'] = sse

        # Silhouette Score
        ss = silhouette_score(X=X, labels=self.labels, metric=self.distance_metric, random_state=self.random_state)
        self.results['internal']['silhouette_score'] = ss

        # Davies-Bouldin Score
        dbs = davies_bouldin_score(X=X, labels=self.labels)
        self.results['internal']['davies_bouldin_score'] = dbs

        # SECOND, FOR EXTERNAL METRICS
       
        # Clustering accuracy
        ca = calculate_clustering_accuracy(y_true=y, y_pred=self.labels)
        self.results['external']['clustering_accuracy'] = ca




        # Within-cluster sum of squares (WCSS)
        distances = self.compute_distances(X, self.centroids)
        return np.sum((np.min(distances, axis=1))**2)
    

if __name__ == "__main__":

    # Paths to the training data
    train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_features.npy"))
    train_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_labels.npy"))
    train_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_images.npy"))

    # Paths to the test data
    test_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_features.npy"))
    test_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_labels.npy"))
    test_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_images.npy"))

    # Load the training and test features, labels and images
    train_features = np.load(train_features_path)
    test_features = np.load(test_features_path)

    train_labels = np.load(train_labels_path)
    test_labels = np.load(test_labels_path)

    train_images = np.load(train_images_path)
    test_images = np.load(test_images_path)

    X = train_features
    y = train_labels

    # Create a KMeansClustering object
    kmeans = KMeansClustering(n_clusters=5)
    # Fit with training data
    kmeans.fit(X)

    print("WCSS:", kmeans.inertia(X))
    print("Cluster labels:", kmeans.labels)