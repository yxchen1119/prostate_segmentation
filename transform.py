import numpy as np
from torchvision.transforms import Compose
from scipy.ndimage import rotate
import torch

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

class RandomContrast:
    """
        Adjust the brightness of an image by a random factor.
    """

    def __init__(self, random_state, factor=0.5, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        self.factor = factor
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            brightness_factor = self.factor + self.random_state.uniform()
            return np.clip(m * brightness_factor, 0, 1)

        return m

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='constant', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)

        return m

class GaussianNoise(object):
    def __init__(self, random_state, min_sigma, max_sigma, max_value=255, **kwargs):
        self.random_state = random_state
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.max_value = max_value

    def __call__(self, m):
        # pick std dev from [0; max_sigma]
        std = self.random_state.uniform(self.min_sigma, self.max_sigma)
        gaussian_noise = self.random_state.normal(0, std, m.shape)
        noisy_m = m + gaussian_noise

        return np.clip(noisy_m, 0, self.max_value).astype(m.dtype)

class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor.
    """

    def __call__(self, m):
        return torch.from_numpy(m)


class Transformer:
    def __init__(self):
        self.seed = 47

    def create_transform(self):

        random_state = np.random.RandomState(self.seed)

        return Compose([
            RandomFlip(random_state),
            RandomContrast(random_state),
            RandomRotate(random_state),
            GaussianNoise(random_state, 0.3, 0.7),
            ToTensor()
        ])

def get_transformer():
    return Transformer()