import unittest
from detectors import *

class MyTestCase(unittest.TestCase):
    def test_something(self):
        samples1 = np.random.normal(0, 10, (2, 200))
        samples2 = np.random.normal(2, 10, (2, 200))
        samples = np.concatenate((samples1, samples2), axis=1)
        new_sample = np.random.normal(25, 1, (2, 1))
        SClustering = AgglomerateClassifier(k=50, threshold=1, n_clusters=2, linkage='average')
        print(SClustering.is_attack(new_sample.T, samples.T))


if __name__ == '__main__':
    unittest.main()
