import numpy as np
from scipy.linalg import norm


def normal_noise(x):
    noise = np.random.normal(0,10,x.shape)/255.
    noisy = norm(x + noise)
    return np.clip(noisy,0,1)


def poisson_noise(x):
    noise = np.random.poisson(1/2, x.shape) / .255
    return np.clip(x + noise, 0,1)



