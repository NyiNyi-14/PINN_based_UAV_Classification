# %%
import numpy as np

# %%
class Noise:
    def __init__(self, noise_type):
        self.noise_type = noise_type

    def generate_noise(self, data, mean, std):
        if self.noise_type == 'gaussian':
            noise = np.random.normal(loc = mean, scale = std, size = data.shape)
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-std, std, size = data.shape)
        else:
            raise ValueError("Noise type must be 'gaussian' or 'uniform'")
        return data + noise

# %%
