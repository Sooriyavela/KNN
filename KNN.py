import numpy as np


class KNN:
    def __init__(self, k=5, distance_metric='euclidean', p=2):
        self.k = k
        self.distance_metric = distance_metric
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        self.X_test = X_test
        y_pred = [self._predict(x) for x in self.X_test]
        return np.array(y_pred)

    def _predict(self, x):
        if self.distance_metric == 'euclidean':
            distances = [np.sqrt(np.sum((x_train - x)**2))
                         for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [np.sum(np.abs(x_train - x))
                         for x_train in self.X_train]
        elif self.distance_metric == 'minkowski':
            distances = [np.sum(np.abs(x_train - x)**self.p)**(1/self.p)
                         for x_train in self.X_train]
        else:
            ValueError("Invalid distance metric")
        Indices = np.argsort(distances)[:self.k]
        K_dist = [self.y_train[i] for i in Indices]
        prediction = np.mean(K_dist)
        return prediction
