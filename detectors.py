#TODO:
# DBSCAN
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
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, Birch,DBSCAN, MiniBatchKMeans, KMeans,MeanShift, SpectralClustering
import matplotlib.pyplot as plt
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
            is_attack, avg = self.is_attack(query, np.stack(all_queries, axis=0))
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

    @staticmethod
    def order_neighbors(x, neighbors):
        x_distances = [euclidean(x, neighbors[i, :]) for i in range(neighbors.shape[0])]
        euc_dist_index = np.argsort(x_distances)
        return neighbors[euc_dist_index, :], x_distances

    def cluster_polling(self, predictions, x_distances):
        if len(predictions[1:][predictions[1:] == predictions[0]]) >= self.threshold*len(predictions[1:]):
            return True, np.mean(np.sort(x_distances)[:self.K])
        else:
            return False, None


class Detector:
    def __init__(self, encoder: Encoder, classifier: AdversarialClassifier):
        self.encoder = encoder
        self.classifier = classifier
        self.memory = []
        self.adversarial_samples = []
        pass
        self.encoder = encoder
        self.classifier = classifier

    def encode(self, x) -> np.array:
        return self.encoder.encode(x)

    def check_query(self, query: np.array) -> bool:
        x = self.encode(query)
        return self.classifier.process_query(x)
        pass


class KMeansClassifier(AdversarialClassifier):
    def __init__(self, K, threshold, max_memory_size):
        super().__init__(K, threshold, max_memory_size)

    def get_mean_distances(self, part):
        means = []
        for i in range(part.shape[0]):
            mean = []
            for v in np.concatenate((part[:i, :], part[i + 1:,:]), axis=0):
                mean.append(euclidean(part[i, :], v))
            means.append(np.mean(mean))
        return np.percentile(means,self.threshold*100)

    def is_attack(self, x, neighbors):

        neighbors_sorted_by_distance, x_distances = self.order_neighbors(x, neighbors)
        x_mean_distance = np.mean(np.sort(x_distances)[:self.K])
        neighbors_mean = self.get_mean_distances(neighbors_sorted_by_distance)
        return x_mean_distance <= neighbors_mean, x_mean_distance


class AgglomerateClassifier(AdversarialClassifier, AgglomerativeClustering):

    def __init__(self, k, threshold=None, max_memory_size=1000, *args, **kwargs):
        AdversarialClassifier.__init__(self, k, threshold, max_memory_size)
        AgglomerativeClustering.__init__(self, *args, **kwargs)

    def is_attack(self, x, neighbors):
        neighbors_sorted_by_distance, x_distances = self.order_neighbors(x, neighbors)
        predictions = self.fit_predict(np.concatenate((x, neighbors_sorted_by_distance), axis=0))
        is_similar, mean = self.cluster_polling(predictions, x_distances)
        if is_similar:
            return is_similar, mean

        return False, None


class BIRCHClassifier(AdversarialClassifier, Birch):

    def __init__(self, k, threshold=None, max_memory_size=1000, *args, **kwargs):
        AdversarialClassifier.__init__(self,k, threshold, max_memory_size)
        Birch.__init__(self, *args, **kwargs)

    def is_attack(self, x, neighbors):
        neighbors_sorted_by_distance, x_distances = self.order_neighbors(x, neighbors)
        if self.nr_samples > self.K+1:
            self.partial_fit_(np.concatenate((x, neighbors_sorted_by_distance), axis=0))
            predictions = self.predict(np.concatenate((x, neighbors_sorted_by_distance), axis=0))
        else:
            predictions = self.fit_predict(np.concatenate((x, neighbors_sorted_by_distance), axis=0))

        is_similar, mean = self.cluster_polling(predictions, x_distances)
        if is_similar:
            return is_similar, mean
        return False, None


class MeanShiftClassifier(AdversarialClassifier, MeanShift):

    def __init__(self, k, threshold=None, max_memory_size=1000, *args, **kwargs):
        AdversarialClassifier.__init__(k, threshold, max_memory_size)
        MeanShift.__init__(self, *args, **kwargs)

    def is_attack(self, x, neighbors):
        neighbors_sorted_by_distance, x_distances = self.order_neighbors(x, neighbors)
        predictions = self.fit_predict(np.concatenate((x, neighbors_sorted_by_distance), axis=0))
        is_similar, mean = self.cluster_polling(predictions, x_distances)
        if is_similar:
            return is_similar, mean

        return False, None


class DBSCANClassifier(AdversarialClassifier, DBSCAN):

    def __init__(self, k, threshold=None, max_memory_size=1000, *args,**kwargs):
        AdversarialClassifier.__init__(self,k, threshold, max_memory_size)
        DBSCAN.__init__(self, *args, **kwargs)

    def is_attack(self, x, neighbors):
        neighbors_sorted_by_distance, x_distances = self.order_neighbors(x, neighbors)
        predictions = self.fit_predict(np.concatenate((x, neighbors_sorted_by_distance), axis=0))
        is_similar, mean = self.cluster_polling(predictions, x_distances)
        if is_similar:
            return is_similar,mean
        return False, None


class SpectralClusteringClassifier(AdversarialClassifier, SpectralClustering):

    def __init__(self, k, threshold, max_memory_size=1000, *args, **kwargs):
        AdversarialClassifier.__init__(self, k, threshold, max_memory_size)
        SpectralClustering.__init__(self, *args, **kwargs)

    def is_attack(self, x, neighbors):
        x_distances = [euclidean(x, neighbors[i, :]) for i in range(neighbors.shape[0])]
        euc_dist_index = np.argsort(x_distances)
        neighbors_sorted_by_distance = neighbors[euc_dist_index, :]

        predictions = self.fit_predict(np.concatenate((x, neighbors_sorted_by_distance), axis=0))
        labels = np.unique(predictions)
        for l in labels:
            args = np.where(predictions[1:] == l)
            plt.scatter(neighbors_sorted_by_distance[args, 0], neighbors_sorted_by_distance[args, 1])
        plt.scatter(x[:, 0], x[:, 1])
        plt.show()
        is_similar, mean = self.cluster_polling(predictions, x_distances)
        if is_similar:
            return is_similar, mean
        return False, None