# %% Import libraries
import numpy as np
# %% Generate scenarios for fixed-wing UAVs
class FW_ScenarioSampler:
    def __init__(self, base_throttle, delta, seed=None):
        self.base_throttle = base_throttle
        self.delta = delta
        if seed is not None:
            np.random.seed(seed)

        self.conditions = [
            'cruise', 'climb', 'descend',
            'yaw_left', 'yaw_right',
            'roll_left', 'roll_right',
            'pitch_up', 'pitch_down',
            'disturbed'
        ]

    def sample(self, n_samples):
        data = []
        for _ in range(n_samples):
            cond = np.random.choice(self.conditions)
            control, state, condition = self._generate_conditions(cond)
            data.append((control, state, condition))
        return data

    def _generate_conditions(self, condition):
        throttle = self.base_throttle
        delta = self.delta
        max_a, max_e, max_r = 0.01, 0.01, 0.02

        delta_a = 0.0
        delta_e = 0.0
        delta_r = 0.0

        state = np.zeros(12)
        state[0] = 20  # u
        state[1:3] = 0  # v, w
        state[3:6] = 0  # p, q, r
        state[6:9] = [0, np.deg2rad(3), 0]  # phi, theta, psi
        state[6:9] = 0
        state[9:11] = np.random.uniform(-10, 10, size=2)  # x, y
        state[11] = np.random.uniform(-120, -100)  # z (NED altitude)

        if condition == 'cruise':
            delta_e = -0.005

        elif condition == 'climb':
            throttle += np.random.uniform(0, delta)
            delta_e = -np.random.uniform(0, max_e) * 0.5

            state[2] = -np.random.uniform(0.5, 1.0) # w
            state[7] = np.deg2rad(20) # theta
            state[4] = np.random.uniform(0.02, 0.08) # q

        elif condition == 'descend':
            throttle -= np.random.uniform(0, delta)
            delta_e = np.random.uniform(0, max_e) * 0.5

            state[2] = np.random.uniform(0.5, 1.0)
            state[7] = np.deg2rad(-20)
            state[4] = -np.random.uniform(0.02, 0.08)

        elif condition == 'yaw_left':
            delta_r = +np.random.uniform(0.01, max_r) * 0.5
            delta_a, delta_e = 0.008, -0.008

            state[3] = np.deg2rad(2) # p
            state[5] = np.deg2rad(3) # r
            state[6] = np.deg2rad(15) # phi

        elif condition == 'yaw_right':
            delta_r = -np.random.uniform(0.01, max_r) * 0.5
            delta_a, delta_e = -0.008, -0.008

            state[3] = -np.deg2rad(2) # p
            state[5] = -np.deg2rad(3) # r
            state[6] = -np.deg2rad(15) # phi

        elif condition == 'roll_left':
            delta_a = np.random.uniform(0.001, max_a) * 0.5

            state[6] = np.deg2rad(15)
            state[3] = np.random.uniform(0.05, 0.1)

        elif condition == 'roll_right':
            delta_a = -np.random.uniform(0.001, max_a) * 0.5

            state[6] = np.deg2rad(-15)
            state[3] = -np.random.uniform(0.05, 0.1)

        elif condition == 'pitch_up':
            delta_e = -np.random.uniform(0, max_e) * 0.5

            state[7] = np.deg2rad(6)
            state[4] = np.random.uniform(0.03, 0.08)

        elif condition == 'pitch_down':
            delta_e = np.random.uniform(0, max_e) * 0.5

            state[7] = np.deg2rad(-6)
            state[4] = -np.random.uniform(0.03, 0.08)

        elif condition == 'disturbed':
            throttle = np.clip(self.base_throttle + np.random.uniform(-delta, delta), 0.0, 1.0)
            delta_a = np.random.uniform(-max_a, max_a) * 0.5
            delta_e = np.random.uniform(-max_e, max_e) * 0.5
            delta_r = np.random.uniform(-max_r, max_r) * 0.5

            state = np.zeros(12)
            state[0] = np.random.uniform(15, 25)
            state[1:3] = np.random.uniform(-0.5, 0.5, size=2)
            state[3:6] = np.random.uniform(-0.3, 0.3, size=3)
            state[6:9] = np.random.uniform(-0.2, 0.2, size=3)
            state[11] = np.random.uniform(-150, -50)

        throttle = np.clip(throttle, 0.0, 1.0)
        return np.array([throttle, delta_a, delta_e, delta_r]), state, condition

# %%
