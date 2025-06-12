# %%
import numpy as np

# %%
class Quad_InitialStateSampler:
    def __init__(self):
        self.conditions = [
            'hover', 'climb', 'descend',
            'yaw_left', 'yaw_right',
            'roll_left', 'roll_right',
            'pitch_forward', 'pitch_back',
            'disturbed'
        ]

    def sample(self, n_samples=1000):
        state_data = []
        labels = []

        for _ in range(n_samples):
            cond = np.random.choice(self.conditions)
            state = self._generate_initial_state(cond)
            state_data.append(state)
            labels.append(cond)

        return np.array(state_data), labels

    def _generate_initial_state(self, condition):
        # state = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        state = np.zeros(12)

        # Position: wider 3D space
        state[0] = np.random.uniform(-50, 50)  # x
        state[1] = np.random.uniform(-50, 50)  # y
        state[2] = np.random.uniform(0, 100)   # z

        if condition == 'climb':
            state[5] = -np.random.uniform(0.5, 1.0)  # vz upward

        elif condition == 'descend':
            state[5] = np.random.uniform(0.5, 1.0)  # vz downward

        elif condition == 'yaw_left':
            state[8] = np.deg2rad(np.random.uniform(-10, 10))  # psi
            state[11] = 0.5  # r

        elif condition == 'yaw_right':
            state[8] = np.deg2rad(np.random.uniform(-10, 10))
            state[11] = -0.5  # r

        elif condition == 'roll_left':
            state[6] = np.deg2rad(5)  # phi
            state[9] = 0.5  # p

        elif condition == 'roll_right':
            state[6] = np.deg2rad(-5)
            state[9] = -0.5

        elif condition == 'pitch_forward':
            state[7] = np.deg2rad(5)  # theta
            state[10] = 0.5  # q

        elif condition == 'pitch_back':
            state[7] = np.deg2rad(-5)
            state[10] = -0.5

        elif condition == 'disturbed':
            state = np.random.uniform(-2, 2, size=12)

        return state

# %%