#TODO:
# Affinity Propagation
# Agglomerative Clustering
# BIRCH
# DBSCAN
# K-Means
# Mini-Batch K-Means
# Mean Shift
# OPTICS
# Spectral Clustering
# Gaussian Mixture Model

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import euclidean
from encoders import *
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, Birch,DBSCAN, MiniBatchKMeans,KMeans

class AdversarialClassifier:
    def __init__(self, K, threshold, max_memory_size):
        self.K = K
        self.threshold = threshold
        self.buffer = []
        self.memory = []
        self.detected = []
        self.nr_samples = 0
        self.max_sampling_size = max_memory_size
        self.history = []

    @abstractmethod
    def is_attack(self, x, neighbors):
        pass

    def clear_memory(self):
        self.memory = []
        self.buffer = []

    def process_query(self, query):

        if len(self.memory) and len(self.buffer) < self.K:
            self.buffer.append(query)
            self.nr_samples += 1
            return False

        if len(self.buffer) > 0:
            all_queries = []
            self.nr_samples += 1
            all_queries.extend(self.buffer)
            all_queries.extend(self.memory)
            is_attack, avg = self.is_attack(query, np.stack(all_queries, axis=1))
            if is_attack:
                self.detected.append(avg)
                self.history.append(self.nr_samples)
                self.nr_samples = 0
                self.clear_memory()
                return True
                pass
            else:
                self.buffer.append(query)
        if len(self.buffer) >= self.max_sampling_size:
            self.memory.extend(self.buffer)
            self.buffer = []


class Detector:
    def __init__(self, encoder: Encoder, classifier: AdversarialClassifier):
        self.encoder = encoder
        self.classifier = classifier
        self.memory = []
        self.adversarial_samples = []
        pass

    @abstractmethod
    def process_query(self):
        pass


class K_Means_Classifier(AdversarialClassifier):
    def __init__(self, K, threshold, max_memory_size):
        super().__init__(K, threshold, max_memory_size)

    def get_mean_distances(self, part):
        means = []
        for i in range(part.shape[1]):
            mean = []
            for v in np.concatenate((part[:, :i], part[:, i + 1:]), axis=1).T:
                mean.append(euclidean(part[:, i], v))
            means.append(np.mean(mean))
        return np.percentile(means,self.threshold*100)

    def is_attack(self, x, neighbors):

        x_distances = [euclidean(x, neighbors[:, i]) for i in range(neighbors.shape[1])]
        euc_dist_index = np.argsort(x_distances)
        neighbors_sorted_by_distance = neighbors[:, euc_dist_index][:self.K]
        neighbors_mean = self.get_mean_distances(neighbors_sorted_by_distance)
        x_mean_distance = np.mean(np.sort(x_distances)[:self.K])
        return x_mean_distance <= neighbors_mean, x_mean_distance

class AgglomerateClassifier(AdversarialClassifier):

    def __init__(self, k, threshold=None, max_memory_size=1000):
        super().__init__(k, threshold, max_memory_size)
        self.model = AgglomerativeClustering(n_clusters=2, compute_distances=True)

    def is_attack(self, x, neighbors):
        x_distances = [euclidean(x, neighbors[:, i]) for i in range(neighbors.shape[1])]
        euc_dist_index = np.argsort(x_distances)
        neighbors_sorted_by_distance = neighbors[:, euc_dist_index]
        predictions = self.model.fit_predict(np.concatenate((x, neighbors_sorted_by_distance), axis=1))[:self.K+1]
        if len(predictions[1:][predictions[1:] == predictions[0]]) >= self.threshold*len(predictions[1:]):
            return True, np.mean(x_distances)

        return False




