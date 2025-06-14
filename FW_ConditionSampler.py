# %%
import numpy as np

# %%
class FW_ConditionSampler:
    def __init__(self, base_throttle, delta):
        """
        Initializes the sampler.
        :param base_throttle: Trim throttle for level flight
        :param delta: Max deviation for surface deflections
        """
        self.base_throttle = base_throttle
        self.delta = delta
        self.conditions = [
            'cruise', 'climb', 'descend',
            'yaw_left', 'yaw_right',
            'roll_left', 'roll_right',
            'pitch_up', 'pitch_down',
            'disturbed'
        ]

    def sample(self, n_samples):
        control_data = []
        labels = []

        for _ in range(n_samples):
            condition = np.random.choice(self.conditions)
            controls = self._generate_controls(condition)
            control_data.append(controls)
            labels.append(condition)

        return np.array(control_data), labels

    def _generate_controls(self, condition):
        throttle = self.base_throttle
        delta = self.delta  # now defines max range
        max_a, max_e, max_r = 0.35, 0.44, 0.52  # aileron, elevator, rudder limits in rad

        # Default neutral
        delta_a = 0.0
        delta_e = 0.0
        delta_r = 0.0

        if condition == 'cruise':
            throttle = self.base_throttle

        elif condition == 'climb':
            throttle += np.random.uniform(0, delta)
            delta_e = -np.random.uniform(0, max_e)

        elif condition == 'descend':
            throttle -= np.random.uniform(0, delta)
            delta_e = np.random.uniform(0, max_e)

        elif condition == 'yaw_left':
            delta_r = np.random.uniform(0, max_r)

        elif condition == 'yaw_right':
            delta_r = -np.random.uniform(0, max_r)

        elif condition == 'roll_left':
            delta_a = np.random.uniform(0, max_a)

        elif condition == 'roll_right':
            delta_a = -np.random.uniform(0, max_a)

        elif condition == 'pitch_up':
            delta_e = -np.random.uniform(0, max_e)

        elif condition == 'pitch_down':
            delta_e = np.random.uniform(0, max_e)

        elif condition == 'disturbed':
            throttle = np.clip(self.base_throttle + np.random.uniform(-delta, delta), 0.0, 1.0)
            delta_a = np.random.uniform(-max_a, max_a)
            delta_e = np.random.uniform(-max_e, max_e)
            delta_r = np.random.uniform(-max_r, max_r)

        return [throttle, delta_a, delta_e, delta_r]

# %%
