from PIL import Image
import numpy as np
from sklearn.cluster import KMeans as kmauto
import matplotlib.pyplot as plt
import os

os.environ['OMP_NUM_THREADS'] = '4'


def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))



class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None

    def fit(self, X):
        self.cluster_centers_ = self._initialize_centers(X)
        for _ in range(self.max_iter):
            clusters = self._assign_clusters(X)
            new_centers = self._update_centers(X, clusters)
            if np.allclose(self.cluster_centers_, new_centers):
                break
            self.cluster_centers_ = new_centers

    def _initialize_centers(self, X):
        n_samples, _ = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centers = X[random_indices]
        return centers

    def _assign_clusters(self, X):
        clusters = []
        for sample in X:
            distances = [euclidean_distance(sample, center) for center in self.cluster_centers_]
            cluster_id = np.argmin(distances)
            clusters.append(cluster_id)
        return np.array(clusters)

    def _update_centers(self, X, clusters):
        new_centers = []
        for cluster_id in range(self.n_clusters):
            cluster_samples = X[clusters == cluster_id]
            if len(cluster_samples) > 0:
                center = np.mean(cluster_samples, axis=0)
            else:
                center = self.cluster_centers_[cluster_id]
            new_centers.append(center)
        return np.array(new_centers)


def perform_kmeans_clustering(k, data_points):
    kmeans = kmauto(n_clusters=k)
    kmeans.fit(data_points)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids


""""
def manual_kmeans(k,data_points):
    kmeans=Kmeans(k=k)
    labels,centroids=kmeans.predict(data_points)
    return labels,centroids
"""
def compute_likelihood(pixel, centroids):
    distances = np.linalg.norm(pixel - centroids, axis=1)
    likelihoods = np.exp(-distances)
    return likelihoods





def assign_pixel_class(pixel, foreground_centroids, background_centroids, foreground_weights, background_weights):
    foreground_likelihoods = compute_likelihood(pixel, foreground_centroids)
    background_likelihoods = compute_likelihood(pixel, background_centroids)

    foreground_probability = np.dot(foreground_weights, foreground_likelihoods)
    background_probability = np.dot(background_weights, background_likelihoods)

    if foreground_probability > background_probability:
        return 1  # Foreground class
    else:
        return 0  # Background class


def lazy_snapping(image_path, seed_image_path, num_clusters,user_kmeans=True):
    # Load the original image
    image = Image.open(image_path)
    data = np.array(image)

    # Load the seed image
    seed_image = Image.open(seed_image_path)
    seed_data = np.array(seed_image)
    if seed_data.shape[-1] == 4:
        seed_data = seed_data[:, :, :3]


    if user_kmeans:
        # Extract seed pixels and cluster centroids using manual kmeans
        foreground_indices, background_indices, foreground_centroids, background_centroids = extract_seed_pixels_manual(
            data, seed_data, num_clusters
        )
    else:
        # Extract seed pixels and cluster centroids using k-means clustering in scikit-learn
        foreground_indices, background_indices, foreground_centroids, background_centroids = extract_seed_pixels(
            data, seed_data, num_clusters
        )



    # Compute weights for each cluster
    total_foreground_pixels = foreground_indices.shape[0]
    total_background_pixels = background_indices.shape[0]

    foreground_weights = np.bincount(foreground_indices, minlength=num_clusters) / total_foreground_pixels
    background_weights = np.bincount(background_indices, minlength=num_clusters) / total_background_pixels

    # Apply lazy snapping to classify each pixel
    segmented_image = np.zeros(data.shape, dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pixel = data[i, j]
            class_label = assign_pixel_class(pixel, foreground_centroids, background_centroids,
                                             foreground_weights, background_weights)
            segmented_image[i, j] = pixel if class_label else [0, 0, 0]

    return segmented_image


def extract_seed_pixels(data, seed_data, num_clusters):
    # Extract foreground and background seed pixels
    foreground_indices = np.where((seed_data[:, :, 0] != 0) & (seed_data[:, :, -1] == 0))
    background_indices =  np.where((seed_data[:, :, 0] != 0) & (seed_data[:, :, -1] != 0))



# Extract RGB color values of seed pixels
    foreground_pixels = data[foreground_indices]
    background_pixels = data[background_indices]

    # Perform k-means clustering
    foreground_labels, foreground_centroids = perform_kmeans_clustering(num_clusters, foreground_pixels)
    background_labels, background_centroids = perform_kmeans_clustering(num_clusters, background_pixels)


    return foreground_labels, background_labels, foreground_centroids, background_centroids



def extract_seed_pixels_manual(data, seed_data, num_clusters):
    foreground_indices = np.where(np.all(seed_data == [255, 0, 0], axis=-1))
    background_indices = np.where(np.all(seed_data == [6, 0, 255], axis=-1))

    foreground_pixels = data[foreground_indices]
    background_pixels = data[background_indices]

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(foreground_pixels)
    foreground_labels = kmeans._assign_clusters(foreground_pixels)
    foreground_centroids = kmeans.cluster_centers_

    kmeans.fit(background_pixels)
    background_labels = kmeans._assign_clusters(background_pixels)
    background_centroids = kmeans.cluster_centers_

    return foreground_labels, background_labels, foreground_centroids, background_centroids





image_path = "lady.PNG"
seed_image_path = "lady stroke 2.png"
num_clusters = 1


segmented_image_auto = lazy_snapping(image_path, seed_image_path, num_clusters, user_kmeans=False)
segmented_image_manual = lazy_snapping(image_path, seed_image_path, num_clusters, user_kmeans=True)

# Save the auto segmented image
result_image = Image.fromarray(segmented_image_auto)
result_image.save("lady_2_auto.png")

# Save the manual segmented image
result_image = Image.fromarray(segmented_image_manual)
result_image.save("lady_2_manaual.png")
