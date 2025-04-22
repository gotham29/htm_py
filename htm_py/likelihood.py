import math
import numpy as np
from collections import deque

class AnomalyLikelihood:
    def __init__(self, learning_period=300, estimation_samples=100, reestimation_period=100):
        self.learning_period = learning_period
        self.estimation_samples = estimation_samples
        self.reestimation_period = reestimation_period
        self.window = deque(maxlen=estimation_samples)
        self.mean = 0.0
        self.variance = 1.0
        self.count = 0

    def update(self, anomaly_score):
        self.window.append(anomaly_score)
        self.count += 1

        if self.count <= self.learning_period:
            return 0.5  # Default likelihood until learning is complete

        if self.count % self.reestimation_period == 0 or self.count == self.learning_period + 1:
            self._recalculate_distribution()

        z_score = (anomaly_score - self.mean) / math.sqrt(self.variance + 1e-9)
        likelihood = 1.0 - self._gaussian_cdf(z_score)
        return likelihood

    def _recalculate_distribution(self):
        data = np.array(self.window)
        self.mean = np.mean(data)
        self.variance = np.var(data)

    def _gaussian_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
