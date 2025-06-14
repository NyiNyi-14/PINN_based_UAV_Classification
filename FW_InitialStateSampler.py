# %%
import numpy as np

# %%
class FW_InitialStateSampler:
    def __init__(self):
        self.conditions = [
            'cruise', 'climb', 'descend',
            'yaw_left', 'yaw_right',
            'roll_left', 'roll_right',
            'pitch_up', 'pitch_down',
            'disturbed'
        ]

    def sample(self, n_samples):
        state_data = []
        labels = []

        for _ in range(n_samples):
            cond = np.random.choice(self.conditions)
            state = self._generate_initial_state(cond)
            state_data.append(state)
            labels.append(cond)

        return np.array(state_data), labels

    def _generate_initial_state(self, condition):
        # state = [u, v, w, p, q, r, phi, theta, psi, x, y, z]
        state = np.zeros(12)

        # Position (fixed-wing starts airborne in NED)
        state[9] = 0.0   # x
        state[10] = 0.0  # y
        state[11] = np.random.uniform(-150, -50)  # z (altitude, negative in NED)

        # Forward velocity (always nonzero)
        state[0] = np.random.uniform(12, 25)  # u: forward speed

        # Other baseline motions
        state[1] = np.random.uniform(-1, 1)   # v: sideslip
        state[2] = np.random.uniform(-1, 1)   # w: vertical

        if condition == 'cruise':
            pass  # default is already near trimmed

        elif condition == 'climb':
            state[2] = -np.random.uniform(1.0, 2.0)     # w upward
            state[7] = np.deg2rad(8)                    # theta (pitch up)
            state[4] = np.random.uniform(0.1, 0.3)       # q (pitch rate)

        elif condition == 'descend':
            state[2] = np.random.uniform(1.0, 2.0)       # w downward
            state[7] = np.deg2rad(-8)                    # theta (pitch down)
            state[4] = -np.random.uniform(0.1, 0.3)      # q

        elif condition == 'yaw_left':
            state[8] = np.random.uniform(0, 2*np.pi)     # initial heading
            state[5] = np.random.uniform(0.1, 0.3)       # positive yaw rate (r)

        elif condition == 'yaw_right':
            state[8] = np.random.uniform(0, 2*np.pi)
            state[5] = -np.random.uniform(0.1, 0.3)      # negative yaw rate

        elif condition == 'roll_left':
            state[6] = np.deg2rad(15)                    # phi (roll left)
            state[3] = np.random.uniform(0.1, 0.3)       # p (roll rate)

        elif condition == 'roll_right':
            state[6] = np.deg2rad(-15)                   # phi (roll right)
            state[3] = -np.random.uniform(0.1, 0.3)      # p

        elif condition == 'pitch_up':
            state[7] = np.deg2rad(10)                    # theta
            state[4] = np.random.uniform(0.1, 0.3)       # q

        elif condition == 'pitch_down':
            state[7] = np.deg2rad(-10)
            state[4] = -np.random.uniform(0.1, 0.3)

        elif condition == 'disturbed':
            state = np.random.uniform(-2, 2, size=12)
            state[0] = np.random.uniform(12, 25)  # keep u positive
            state[11] = np.random.uniform(-150, -50)  # ensure altitude

        return state

# %%
