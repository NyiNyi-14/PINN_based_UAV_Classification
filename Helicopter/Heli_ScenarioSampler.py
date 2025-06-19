# %%
import numpy as np

# %%
class Heli_ScenarioSampler:
    def __init__(self, m, g, seed = None):
        self.m = m
        self.g = g
        if seed is not None:
            np.random.seed(seed)
        self.conditions = [
            'hover', 'ascend', 'descend',
            'cyclicPitch_forward', 'x_positive',
            'cyclicPitch_backward', 'x_negative',
            'cyclicRoll_left', 'y_negative',
            'cyclicRoll_right', 'y_positive',
            'yaw_left', 'yaw_right', 'disturbed'
        ]

    def sample(self, n_samples):
        data = []
        for _ in range(n_samples):
            cond = np.random.choice(self.conditions)
            control, state, label = self._generate_conditions(cond)
            data.append((control, state, label))
        return data

    def _generate_conditions(self, condition):
        m = self.m
        g = self.g

        state = np.zeros(12)  # x, y, z, vx, vy, vz, phi, theta, psi, p, q, r
        thrust_data = np.zeros(2)  # [main_thrust, tail_thrust]
        cyclic_data = np.zeros(2)  # [cyclic_roll, cyclic_pitch]
        control = np.zeros(4)      # [main_thrust, tail_thrust, cyclic_roll, cyclic_pitch]

        # Randomize position
        state[0] = np.random.uniform(-20, 20)
        state[1] = np.random.uniform(-20, 20)
        state[2] = np.random.uniform(10, 100)

        # Randomize base velocity
        state[3] = np.random.uniform(-0.2, 0.2)  # vx
        state[4] = np.random.uniform(-0.2, 0.2)  # vy
        state[5] = np.random.uniform(-0.2, 0.2)  # vz

        rand_angle = np.radians(np.random.uniform(10, 20))
        rand_force = np.random.uniform(0.15, 0.25) * m * g
        hover_thrust = m * g

        if condition == 'hover':
            thrust_data[0] = hover_thrust
            state[3:6] = 0

        elif condition == 'ascend':
            thrust_data[0] = hover_thrust + rand_force
            state[5] = np.random.uniform(0.2, 0.5)

        elif condition == 'descend':
            thrust_data[0] = hover_thrust - rand_force
            state[5] = -np.random.uniform(0.2, 0.5)

        elif condition in ['cyclicPitch_forward']:
            cyclic_data[1] = rand_angle
            state[7] = rand_angle
            state[3] = np.random.uniform(0.2, 0.5)
            thrust_data[0] = hover_thrust / np.cos(rand_angle)

        elif condition in ['x_positive']:
            state[7] = rand_angle
            state[3] = np.random.uniform(0.2, 0.5)
            thrust_data[0] = hover_thrust / np.cos(rand_angle)
        
        elif condition in ['cyclicPitch_backward']:
            cyclic_data[1] = -rand_angle
            state[7] = -rand_angle
            state[3] = -np.random.uniform(0.2, 0.5)
            thrust_data[0] = hover_thrust / np.cos(rand_angle)

        elif condition in ['x_negative']:
            state[7] = -rand_angle
            state[3] = -np.random.uniform(0.2, 0.5)
            thrust_data[0] = hover_thrust / np.cos(rand_angle)

        elif condition in ['cyclicRoll_left']:
            cyclic_data[0] = rand_angle
            state[6] = rand_angle
            state[4] = np.random.uniform(0.2, 0.5)
            thrust_data[0] = hover_thrust / np.cos(rand_angle)

        elif condition in ['y_negative']:
            state[6] = rand_angle
            state[4] = np.random.uniform(0.2, 0.5)
            thrust_data[0] = hover_thrust / np.cos(rand_angle)

        elif condition in ['cyclicRoll_right']:
            cyclic_data[0] = -rand_angle
            state[6] = -rand_angle
            state[4] = np.random.uniform(0.2, 0.5)
            thrust_data[0] = hover_thrust / np.cos(rand_angle)

        elif condition in ['y_positive']:
            state[6] = -rand_angle
            state[4] = -np.random.uniform(0.2, 0.5)
            thrust_data[0] = hover_thrust / np.cos(rand_angle)

        elif condition == 'yaw_left':
            thrust_data[0] = hover_thrust
            thrust_data[1] = rand_force * 5
            cyclic_data[1] = rand_angle
            state[7] = rand_angle

        elif condition == 'yaw_right':
            thrust_data[0] = hover_thrust
            thrust_data[1] = -rand_force * 5
            cyclic_data[1] = rand_angle
            state[7] = rand_angle

        elif condition == 'disturbed':
            thrust_data[0] = hover_thrust + np.random.uniform(-0.1, 0.1) * m * g
            thrust_data[1] = rand_force * np.random.uniform(-3, 3)
            cyclic_data[0] = np.random.uniform(-1, 1) * rand_angle
            cyclic_data[1] = np.random.uniform(-1, 1) * rand_angle
            state[0:2] = np.random.uniform(-5, 5, 2)
            state[2] = np.random.uniform(5, 30)

        # Final control format: [main_thrust, tail_thrust, cyclic_roll, cyclic_pitch]
        control = np.concatenate((thrust_data, cyclic_data))
        return control, state, condition

# %%