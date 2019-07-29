import numpy as np
import random
import torch
from torchvision.transforms import Compose

class RandomFlip(object):
    def __init__(self, random_state, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)

    def __call__(self, m):
        for axis in self.axes:
            if self.random_state.uniform() > 0.5:
                m = np.flip(m, axis)
        return m

class RandomRotate90(object):
    def __init__(self, degree, random_state, **kwargs):
        self.random_state = random_state
        self.degree = degree

    def __call__(self, m):
        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if self.degree == 90:
            m = np.rot90(m, k, (1, 2))

        return m

class GaussianNoise(object):
    def __init__(self, random_state, max_sigma, max_value=255, **kwargs):
        self.random_state = random_state
        self.max_sigma = max_sigma
        self.max_value = max_value

    def __call__(self, m):
        # pick std dev from [0; max_sigma]
        std = self.random_state.randint(self.max_sigma)
        gaussian_noise = self.random_state.normal(0, std, m.shape)
        noisy_m = m + gaussian_noise
        return np.clip(noisy_m, 0, self.max_value).astype(m.dtype)

class Randomcrop(object):
    def __call__(self, img):
        dim = img.size()
        x = random.randint(1, dim[0]/2)
        y = random.randint(1, dim[1]/2)
        z = random.randint(1, dim[2]/2)
        cropImg = img[(x):(x + dim[0]/2), (y):(y + dim[1]/2), (z):(z + dim[2]/2)]

        return cropImg

def create_transform():
    seed = 47
    random_state = np.random.RandomState(seed)
    return Compose([
        RandomFlip(random_state),
        GaussianNoise(random_state, 0.7),
        Randomcrop(),
    ])
