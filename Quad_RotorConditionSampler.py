# %%
import numpy as np

# %%
class Quad_RotorConditionSampler:
    def __init__(self, hover_omega, delta, max_omega):
        """
        Initializes the sampler.
        :param hover_omega: Hover rotor speed (rad/s)
        :param delta: Magnitude of change from hover for maneuvers
        :param max_omega: Maximum allowable rotor speed (rad/s)
        """
        self.hover_omega = hover_omega
        self.delta = delta
        self.max_omega = max_omega
        self.conditions = [
            'hover', 'climb', 'descend',
            'yaw_left', 'yaw_right',
            'roll_left', 'roll_right',
            'pitch_forward', 'pitch_back',
            'disturbed'
        ]

    def sample(self, n_samples):
        omega_data = []
        labels = []

        for _ in range(n_samples):
            condition = np.random.choice(self.conditions)
            omega = self._generate_omega(condition)
            omega_data.append(omega)
            labels.append(condition)

        return np.array(omega_data), labels

    def _generate_omega(self, condition):
        base = self.hover_omega
        delta = self.delta
        ω = np.zeros(4)

        if condition == 'hover':
            ω[:] = base

        elif condition == 'climb':
            ω[:] = base + delta * np.random.uniform(0.1, 1)

        elif condition == 'descend':
            ω[:] = base - delta * np.random.uniform(0.1, 1)

        elif condition == 'yaw_left':
            ω[0], ω[2] = base + delta, base + delta
            ω[1], ω[3] = base - delta, base - delta

        elif condition == 'yaw_right':
            ω[1], ω[3] = base + delta, base + delta
            ω[0], ω[2] = base - delta, base - delta

        elif condition == 'roll_right':
            ω[1], ω[2] = base + delta, base + delta
            ω[0], ω[3] = base - delta, base - delta

        elif condition == 'roll_left':
            ω[0], ω[3] = base + delta, base + delta
            ω[1], ω[2] = base - delta, base - delta

        elif condition == 'pitch_forward':
            ω[2], ω[3] = base + delta, base + delta
            ω[0], ω[1] = base - delta, base - delta

        elif condition == 'pitch_back':
            ω[0], ω[1] = base + delta, base + delta
            ω[2], ω[3] = base - delta, base - delta

        elif condition == 'disturbed':
            ω = np.random.uniform(base - 1.5 * delta, base + 1.5 * delta, size=4)

        return np.clip(ω, 0, self.max_omega)

# %%
