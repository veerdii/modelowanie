import numpy as np
from sklearn.cluster import MeanShift
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.manifold import TSNE
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_flat = x_train.reshape(-1, 28*28)
tsne = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42)
x_train_tsne = tsne.fit_transform(x_train_flat[:5000])

bandwidth = 8  # Adjust as needed
clustering = MeanShift(bandwidth=bandwidth).fit(x_train_tsne)
cluster_centers = clustering.cluster_centers_
labels = clustering.labels_

cluster_labels = {}
for i in range(len(cluster_centers)):
    cluster_points_indices = np.where(labels == i)[0]
    true_labels = y_train[:5000][cluster_points_indices]
    most_common_label = Counter(true_labels).most_common(1)[0][0]
    cluster_labels[i] = most_common_label

predicted_labels = [cluster_labels[label] for label in labels]

plt.figure(figsize=(10, 8))
for i in range(len(cluster_centers)):
    cluster_points_indices = np.where(labels == i)[0]
    plt.scatter(x_train_tsne[cluster_points_indices, 0],
                x_train_tsne[cluster_points_indices, 1],
                label=f"Cluster {i} (Digit: {cluster_labels[i]})")

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='x', s=200, color='black', label='Centroids')

plt.title('Mean Shift Clustering on MNIST (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

accuracy = accuracy_score(y_train[:5000], predicted_labels)
print(f"Dokladnosc: {accuracy}")