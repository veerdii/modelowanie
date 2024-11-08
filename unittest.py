import unittest
import numpy as np
from mean_shift_clustering import draw, mean_shift, merge_centroids, remove_isolated_points

class TestMeanShiftClustering(unittest.TestCase):

    def test_draw(self):
        # Sprawdzenie, czy funkcja zwraca poprawną liczbę punktów
        n = 10
        range_x = (0, 10)
        range_y = (0, 10)
        points = draw(n, range_x, range_y)
        self.assertEqual(points.shape, (n, 2))  # Powinno zwrócić n punktów (x, y)
        # Sprawdzenie, czy punkty mieszczą się w określonych zakresach
        self.assertTrue(np.all(points[:, 0] >= range_x[0]) and np.all(points[:, 0] <= range_x[1]))
        self.assertTrue(np.all(points[:, 1] >= range_y[0]) and np.all(points[:, 1] <= range_y[1]))

    def test_mean_shift(self):
        # Prosty przypadek testowy z dwoma skupiskami punktów
        points = np.array([[1, 1], [1.1, 1], [5, 5], [5.1, 5.1]])
        centroids = mean_shift(points, bandwidth=1)
        # Predykcja dwóch centroidów: jednego w pobliżu (1, 1), a drugiego w pobliżu (5, 5)
        self.assertEqual(len(centroids), len(points))
        # Centroid powinien być blisko początkowych punktów w każdym skupisku
        self.assertTrue(np.linalg.norm(centroids[0] - np.array([1, 1])) < 0.2)
        self.assertTrue(np.linalg.norm(centroids[2] - np.array([5, 5])) < 0.2)

    def test_merge_centroids(self):
        # Testowanie merging centroidów
        centroids = np.array([[1, 1], [1.05, 1], [5, 5], [5.1, 5.1]])
        points = np.array([[1, 1], [5, 5]])
        merged = merge_centroids(centroids, points, threshold=0.2)
        self.assertEqual(len(merged), 2)  # Powinny zostać zmergowane dwie pary punktów
        self.assertTrue(np.allclose(merged[0], np.array([1, 1]), atol=0.1))
        self.assertTrue(np.allclose(merged[1], np.array([5, 5]), atol=0.1))

    def test_remove_isolated_points(self):
        # Sprawdzenie czy izolowane punkty są poprawnie usuwane
        points = np.array([[1, 1], [2, 2], [10, 10]])
        non_isolated = remove_isolated_points(points, threshold=1.5)
        self.assertEqual(len(non_isolated), 2)  # Punkt [10, 10] jest zbyt odosobniony, więc powinien zostać usunięty
        self.assertTrue(np.array_equal(non_isolated, np.array([[1, 1], [2, 2]])))