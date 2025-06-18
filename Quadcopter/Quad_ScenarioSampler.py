# %%
import numpy as np

# %%
class Quad_ScenarioSampler:
    def __init__(self, hover_omega, delta, max_omega, seed = None):
        self.hover_omega = hover_omega
        self.delta = delta
        self.max_omega = max_omega
        if seed is not None:
            np.random.seed(seed)
        self.conditions = [
            'hover', 'climb', 'descend',
            'yaw_left', 'yaw_right',
            'roll_left', 'y_negative', 'roll_right', 'y_positive',
            'pitch_forward', 'x_positive', 'pitch_back', 'x_negative',
            'disturbed'
        ]

    def sample(self, n_samples):
        data = []
        for _ in range(n_samples):
            cond = np.random.choice(self.conditions)
            control, state, label = self._generate_conditions(cond)
            data.append((control, state, label))
        return data

    def _generate_conditions(self, condition):
        base = self.hover_omega
        delta = self.delta
        w = np.zeros(4)
        state = np.zeros(12)

        # position, ENU
        state[0] = np.random.uniform(-20, 20)   # x
        state[1] = np.random.uniform(-20, 20)   # y
        state[2] = np.random.uniform(10, 100)    # z

        # baseline
        state[3] = np.random.uniform(-0.2, 0.2)  # vx
        state[4] = np.random.uniform(-0.2, 0.2)  # vy
        state[5] = np.random.uniform(-0.2, 0.2)  # vz

        if condition == 'hover':
            w[:] = base

            state[3] = 0  # vx
            state[4] = 0  # vy
            state[5] = 0  # vz

        elif condition == 'climb':
            w[:] = base + delta

            state[3] = 0  # vx
            state[4] = 0  # vy
            state[5] = np.random.uniform(0.2, 0.5)  # vz (positive = climbing up in ENU)

        elif condition == 'descend':
            w[:] = base - delta

            state[3] = 0  # vx
            state[4] = 0  # vy
            state[5] = -np.random.uniform(0.2, 0.5)  # vz (negative = going down in ENU)

        elif condition == 'yaw_left':
            w[0], w[2] = base + 0.5 * delta, base + 0.5 * delta
            w[1], w[3] = base - 0.5 * delta, base - 0.5 * delta
            state[11] = np.random.uniform(0.1, 0.3)  # r (reduced rate)
            state[8] = np.random.uniform(0, 2*np.pi)
    
        elif condition == 'yaw_right':
            w[1], w[3] = base + 0.5 * delta, base + 0.5 * delta
            w[0], w[2] = base - 0.5 * delta, base - 0.5 * delta
            state[11] = -np.random.uniform(0.1, 0.3) # -r
            state[8] = np.random.uniform(0, 2*np.pi)

        elif condition == 'roll_left':
            w[0], w[3] = base + 0.5 * delta, base + 0.5 * delta
            w[1], w[2] = base - 0.5 * delta, base - 0.5 * delta
            state[6] = np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # phi
            state[9] = np.random.uniform(0.1, 0.3)  # p

        elif condition == 'y_negative':
            w[:] = base 
            state[6] = np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # phi
            state[9] = np.random.uniform(0.1, 0.3)  # p

        elif condition == 'roll_right':
            w[1], w[2] = base + 0.5 * delta, base + 0.5 * delta
            w[0], w[3] = base - 0.5 * delta, base - 0.5 * delta
            state[6] = -np.random.uniform(np.deg2rad(5), np.deg2rad(10))
            state[9] = -np.random.uniform(0.1, 0.3)
        
        elif condition == 'y_positive':
            w[:] = base 
            state[6] = -np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # phi
            state[9] = -np.random.uniform(0.1, 0.3)  # p

        elif condition == 'pitch_forward':
            w[2], w[3] = base + 0.5 * delta, base + 0.5 * delta
            w[0], w[1] = base - 0.5 * delta, base - 0.5 * delta
            state[7] = np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # theta
            state[10] = np.random.uniform(0.1, 0.3)  # q
        
        elif condition == 'x_positive':
            w[:] = base 
            state[7] = np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # theta
            state[10] = np.random.uniform(0.1, 0.3)  # q

        elif condition == 'pitch_back':
            w[0], w[1] = base + 0.5 * delta, base + 0.5 * delta
            w[2], w[3] = base - 0.5 * delta, base - 0.5 * delta
            state[7] = -np.random.uniform(np.deg2rad(5), np.deg2rad(10))
            state[10] = -np.random.uniform(0.1, 0.3)

        elif condition == 'x_negative':
            w[:] = base 
            state[7] = -np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # theta
            state[10] = -np.random.uniform(0.1, 0.3)  # q

        elif condition == 'disturbed':
            w = np.random.uniform(base - delta, base + delta, size=4)
            state = np.random.uniform(-0.5, 0.5, size=12)
            state[0:2] = np.random.uniform(-5, 5, 2) # x, y
            state[2] = np.random.uniform(5, 30) # z

        w = np.clip(w, 0.1 * self.max_omega, self.max_omega)  # Motor safety

        return w, state, condition

# %%