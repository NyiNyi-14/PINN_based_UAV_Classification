# %%
import numpy as np

# %%
class FW_ScenarioSampler:
    def __init__(self, base_throttle, delta, seed = None):
        """
        Unified sampler for initial state and control input.
        :param base_throttle: Trim throttle for level flight
        :param delta: Max deviation for control surface deflections and throttle
        """
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
        """
        Samples a list of scenarios.
        Returns:
            data: List of tuples (control_input, initial_state, label)
        """
        data = []

        for _ in range(n_samples):
            cond = np.random.choice(self.conditions)
            control, state, condition = self._generate_conditions(cond)
            data.append((control, state, condition))

        return data

    def _generate_conditions(self, condition):
        throttle = self.base_throttle
        delta = self.delta
        max_a, max_e, max_r = 0.35, 0.44, 0.52

        delta_a = 0.0
        delta_e = 0.0
        delta_r = 0.0

        # state = [u, v, w, p, q, r, phi, theta, psi, x, y, z]
        state = np.zeros(12)

        # Position (fixed-wing starts airborne in NED)
        state[9] = np.random.uniform(-10, 10)   # x
        state[10] = np.random.uniform(-10, 10)  # y
        state[11] = np.random.uniform(-150, -50)  # z (altitude, negative in NED)

        # Forward velocity (always nonzero)
        state[0] = np.random.uniform(18, 25)  # u: forward speed

        # Other baseline motions
        state[1] = np.random.uniform(-0.05, 0.05)    # v: minimal sideslip
        state[2] = np.random.uniform(-0.05, 0.05)    # w: minimal vertical motion

        if condition == 'cruise':
            throttle = self.base_throttle

        elif condition == 'climb':
            throttle += np.random.uniform(0, delta)
            delta_e = -np.random.uniform(0, max_e) * 0.5

            state[2] = -np.random.uniform(1.0, 2.0)     # w upward
            state[7] = np.deg2rad(8)                    # theta (pitch up)
            state[4] = np.random.uniform(0.05, 0.15)       # q (pitch rate)

        elif condition == 'descend':
            throttle -= np.random.uniform(0, delta)
            delta_e = np.random.uniform(0, max_e) * 0.5

            state[2] = np.random.uniform(1.0, 2.0)       # w downward
            state[7] = np.deg2rad(-8)                    # theta (pitch down)
            state[4] = -np.random.uniform(0.05, 0.15)       # q

        elif condition == 'yaw_left':
            delta_r = np.random.uniform(0, max_r) * 0.5

            state[8] = np.random.uniform(0, np.pi)     # initial heading
            state[5] = np.random.uniform(0.05, 0.15)       # positive yaw rate (r)

        elif condition == 'yaw_right':
            delta_r = -np.random.uniform(0, max_r) * 0.5

            state[8] = np.random.uniform(0, np.pi)
            state[5] = -np.random.uniform(0.05, 0.15)      # negative yaw rate

        elif condition == 'roll_left':
            delta_a = np.random.uniform(0, max_a) * 0.5

            state[6] = np.deg2rad(15)                    # phi (roll left)
            state[3] = np.random.uniform(0.1, 0.3)       # p (roll rate)

        elif condition == 'roll_right':
            delta_a = -np.random.uniform(0, max_a) * 0.5

            state[6] = np.deg2rad(-15)                   # phi (roll right)
            state[3] = -np.random.uniform(0.1, 0.3)      # p

        elif condition == 'pitch_up':
            delta_e = -np.random.uniform(0, max_e) * 0.5

            state[7] = np.deg2rad(10)                    # theta
            state[4] = np.random.uniform(0.05, 0.15)       # q

        elif condition == 'pitch_down':
            delta_e = np.random.uniform(0, max_e) * 0.5

            state[7] = np.deg2rad(-10)
            state[4] = -np.random.uniform(0.05, 0.15)

        elif condition == 'disturbed':
            throttle = np.clip(self.base_throttle + np.random.uniform(-delta, delta), 0.0, 1.0)
            delta_a = np.random.uniform(-max_a, max_a) * 0.5  # Reduced aggression
            delta_e = np.random.uniform(-max_e, max_e) * 0.5
            delta_r = np.random.uniform(-max_r, max_r) * 0.5
            
            state = np.zeros(12)
            state[0] = np.random.uniform(15, 25)  # u
            state[1:3] = np.random.uniform(-0.5, 0.5, size=2)  # v, w
            state[3:6] = np.random.uniform(-0.3, 0.3, size=3)  # p, q, r
            state[6:9] = np.random.uniform(-0.2, 0.2, size=3)  # phi, theta, psi
            state[11] = np.random.uniform(-150, -50)  # z
        
        throttle = np.clip(throttle, 0.0, 1.0)
        return [throttle, delta_a, delta_e, delta_r], state, condition

# # %% Trim
# class FW_ScenarioSampler:
#     def __init__(self, base_throttle, delta, seed=None):
#         """
#         Unified sampler with comprehensive trim logic for all flight conditions.
#         :param base_throttle: Trim throttle for level flight
#         :param delta: Max deviation for control surface deflections and throttle
#         """
#         self.base_throttle = base_throttle
#         self.delta = delta
#         if seed is not None:
#             np.random.seed(seed)
            
#         self.conditions = [
#             'cruise', 'climb', 'descend',
#             'yaw_left', 'yaw_right',
#             'roll_left', 'roll_right',
#             'pitch_up', 'pitch_down',
#             'disturbed'
#         ]

#         # Comprehensive trim parameters for all conditions
#         self.trim_params = {
#             # Steady flight conditions
#             'cruise': {'delta_e': -0.05, 'theta': 0.05, 'delta_a': 0.0, 'delta_r': 0.0},
#             'climb': {'delta_e': -0.15, 'theta': 0.15, 'delta_a': 0.0, 'delta_r': 0.0},
#             'descend': {'delta_e': 0.10, 'theta': -0.10, 'delta_a': 0.0, 'delta_r': 0.0},
            
#             # Turning conditions (coordinated turns)
#             'yaw_left': {'delta_r': 0.25, 'delta_a': 0.1, 'phi': 0.17, 'theta': 0.05},
#             'yaw_right': {'delta_r': -0.25, 'delta_a': -0.1, 'phi': -0.17, 'theta': 0.05},
            
#             # Banking conditions
#             'roll_left': {'delta_a': 0.2, 'phi': 0.26, 'theta': 0.05},
#             'roll_right': {'delta_a': -0.2, 'phi': -0.26, 'theta': 0.05},
            
#             # Pitch conditions
#             'pitch_up': {'delta_e': -0.2, 'theta': 0.17},
#             'pitch_down': {'delta_e': 0.15, 'theta': -0.13}
#         }

#     def sample(self, n_samples):
#         """
#         Samples a list of scenarios.
#         Returns:
#             data: List of tuples (control_input, initial_state, label)
#         """
#         data = []

#         for _ in range(n_samples):
#             cond = np.random.choice(self.conditions)
#             control, state, condition = self._generate_conditions(cond)
#             data.append((control, state, condition))

#         return data

#     def _generate_conditions(self, condition):
#         # Initialize with trimmed values from params
#         trim = self.trim_params.get(condition, {})
        
#         # Base control inputs
#         throttle = self.base_throttle
#         delta_a = trim.get('delta_a', 0.0)
#         delta_e = trim.get('delta_e', 0.0)
#         delta_r = trim.get('delta_r', 0.0)
        
#         # Initialize state (NED frame)
#         state = np.zeros(12)
#         state[9:11] = np.random.uniform(-10, 10, 2)  # x, y
#         state[11] = np.random.uniform(-150, -50)     # z
#         state[0] = np.random.uniform(18, 25)         # u
#         state[1:3] = np.random.uniform(-0.05, 0.05, 2)  # v, w
        
#         # Set trimmed attitudes
#         state[6] = trim.get('phi', 0.0)    # phi
#         state[7] = trim.get('theta', 0.0)  # theta
#         state[8] = np.random.uniform(0, 2*np.pi)  # psi (random heading)

#         # Condition-specific adjustments
#         if condition == 'climb':
#             throttle += np.random.uniform(0, self.delta)
#             state[2] = -np.random.uniform(1.0, 2.0)  # Upward velocity
#             state[4] = np.random.uniform(0.05, 0.15)  # q
            
#         elif condition == 'descend':
#             throttle = np.clip(throttle - np.random.uniform(0, self.delta), 0.1, 1.0)
#             state[2] = np.random.uniform(1.0, 2.0)  # Downward velocity
#             state[4] = -np.random.uniform(0.05, 0.15)
            
#         elif condition in ['yaw_left', 'yaw_right']:
#             state[5] = np.random.uniform(0.05, 0.15) * (-1 if 'right' in condition else 1)
            
#         elif condition in ['roll_left', 'roll_right']:
#             state[3] = np.random.uniform(0.1, 0.3) * (-1 if 'right' in condition else 1)
            
#         elif condition == 'disturbed':
#             throttle = np.clip(throttle + np.random.uniform(-self.delta, self.delta), 0.1, 1.0)
#             delta_a = np.random.uniform(-0.2, 0.2)
#             delta_e = np.random.uniform(-0.15, 0.15)
#             delta_r = np.random.uniform(-0.15, 0.15)
#             state[3:6] = np.random.uniform(-0.3, 0.3, 3)  # p, q, r
#             state[6:9] = np.random.uniform(-0.2, 0.2, 3)  # phi, theta, psi

#         return [throttle, delta_a, delta_e, delta_r], state, condition

# %%