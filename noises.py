import numpy as np
from skimage.util import random_noise

def normal_noise(x):
    return random_noise(x, mode='gaussian', clip=True)


def s_and_p_noise(x):
    return random_noise(x, mode='s&p', clip=True)


def poisson_noise(image):
    noisy = random_noise(image, mode="poisson", clip=True)
    return noisy


def apply_noise(image: np.array, noise: str):
    if noise == 'poisson':
        return poisson_noise(image)
    elif noise == 's_p':
        return s_and_p_noise(image)
    elif noise == 'normal':
        return normal_noise(image)


def apply_random_noise(image):
    return apply_noise(image, np.random.choice(['poisson', 's_p' 'normal']))