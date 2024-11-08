import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import random

style.use('ggplot')

def draw(n, range_x, range_y):
    points = []
    for _ in range(n):
        x = round(random.uniform(range_x[0], range_x[1]), 1)
        y = round(random.uniform(range_y[0], range_y[1]), 1)
        points.append((x, y))
    return np.array(points)

def mean_shift(points, bandwidth=1, max_iterations=100):
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

def merge_centroids(centroids, points, threshold=1.0):
    merged = []
    for centroid in centroids:
        if not any(np.linalg.norm(centroid - m) < threshold for m in merged):
            merged.append(centroid)
    merged_with_points = [centroid for centroid in merged if any(np.linalg.norm(points - centroid, axis=1) < threshold)]
    return np.array(merged_with_points)

def remove_isolated_points(points, threshold=1.0):
    points = np.array(points)
    non_isolated_points = [point for point in points if np.any((np.linalg.norm(points - point, axis=1) < threshold) & (np.linalg.norm(points - point, axis=1) > 0))]
    return np.array(non_isolated_points)
range_x = (0, 10)
range_y = (0, 10)
points = draw(20, range_x, range_y)
points = remove_isolated_points(points, threshold=1.5)
centroids = mean_shift(points, bandwidth=1.5)
centroids = merge_centroids(centroids, points, threshold=1.0)

x_values = points[:, 0]
y_values = points[:, 1]

plt.scatter(x_values, y_values, color='blue', label='Punkty')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroidy')

for centroid in centroids:
    circle = plt.Circle((centroid[0], centroid[1]), 1.5, color='red', fill=False, linestyle='--')
    plt.gca().add_artist(circle)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Klasteryzacja Mean Shift')
plt.legend()
plt.show()