import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from collections import Counter

style.use('ggplot')

def mean_shift(points, bandwidth=1, max_iterations=300):
    points = np.array(points)
    centroids = points.copy()
    for _ in range(max_iterations):
        new_centroids = []
        for centroid in centroids:
            in_bandwidth = points[
                np.linalg.norm(points - centroid, axis=1) < bandwidth
            ]
            if len(in_bandwidth) > 0:
                new_centroids.append(in_bandwidth.mean(axis=0))
            else:
                new_centroids.append(centroid)
        new_centroids = np.array(new_centroids)
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < 1e-5):
            break
        centroids = new_centroids
    return centroids

def merge_centroids(centroids, threshold=1.5):
    merged = []
    for centroid in centroids:
        if not any(np.linalg.norm(centroid - m) < threshold for m in merged):
            merged.append(centroid)
    return np.array(merged)

def remove_isolated_points(points, threshold=10.0):
    points = np.array(points)
    non_isolated_points = [point for point in points if np.any((np.linalg.norm(points - point, axis=1) < threshold) & (np.linalg.norm(points - point, axis=1) > 0))]
    return np.array(non_isolated_points)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_flat = x_train.reshape(-1, 28*28)

tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
x_train_tsne = tsne.fit_transform(x_train_flat[:5000])

x_train_tsne = remove_isolated_points(x_train_tsne, threshold=5.0)

centroids = mean_shift(x_train_tsne, bandwidth=8)
centroids = merge_centroids(centroids, threshold=8.0)

plt.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], color='blue', label='Points')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')

for centroid in centroids:
    circle = plt.Circle((centroid[0], centroid[1]), 8, color='red', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Mean Shift Clustering on MNIST data with t-SNE')
plt.legend()
plt.show()

cluster_labels = {}
for i, centroid in enumerate(centroids):
    cluster_points = x_train_tsne[np.linalg.norm(x_train_tsne - centroid, axis=1) < 8]
    original_indices = np.where(np.isin(x_train_tsne, cluster_points).all(axis=1))[0]
    true_labels = y_train[:5000][original_indices]
    most_common_label = Counter(true_labels).most_common(1)[0][0]
    cluster_labels[i] = most_common_label

predicted_labels = []
for point in x_train_tsne:
    distances = np.linalg.norm(point - centroids, axis=1)
    nearest_centroid_index = np.argmin(distances)
    predicted_labels.append(cluster_labels[nearest_centroid_index])

accuracy = accuracy_score(y_train[:5000], predicted_labels)
print(f"Dokladnosc: {accuracy}")