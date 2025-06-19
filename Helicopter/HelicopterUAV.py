# %% Import Libraries
import numpy as np

# %% Helicopter Dynamics
class HelicopterUAV:
    def __init__(self, m, g, l, d, I, k_drag_linear, k_drag_angular): 
        self.m = m
        self.g = g
        self.l = l # moment arm
        self.d = d # distance from main rotor to rear
        self.I = I # diagonal inertia matrix
        self.k_drag_linear = k_drag_linear
        self.k_drag_angular = k_drag_angular

    def R_matrix(self, phi, theta, psi):
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi),  np.cos(psi), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi),  np.cos(phi)]])
        return Rz @ Ry @ Rx
    
    def W_matrix(self, phi, theta):
        return np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])
    
    def thrust_and_torques(self, thrust_and_cyclic):
        main_thrust = thrust_and_cyclic[0]
        rear_thrust = thrust_and_cyclic[1]
        cyclicAngle_roll = thrust_and_cyclic[2]
        cyclicAngle_pitch = thrust_and_cyclic[3]
        u1 = main_thrust
        u2 = main_thrust * self.l * np.sin(cyclicAngle_roll)
        u3 = main_thrust * self.l * np.sin(cyclicAngle_pitch)
        u4 = rear_thrust * self.d
       # print(f"[DEBUG] u2 (roll torque): {u2:.4f}, u3 (pitch torque): {u3:.4f}")
        thrust_vec = np.array([0,0, u1])
        torque_vec = np.array([u2, u3, u4])

        return thrust_vec, torque_vec

    def dynamics(self, t, state, thrust_and_cyclic):
        thrust, torque = self.thrust_and_torques(thrust_and_cyclic)
        m, g, I = self.m, self.g, self.I
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state

        omega_body = np.array([p,q,r])
        vel = np.array([vx, vy, vz])
        ang = np.array([phi,theta,psi])

        drag_world = -self.k_drag_linear * vel
        damping_world = -self.k_drag_angular * omega_body

        R = self.R_matrix(phi,theta,psi)
        gravity = np.array([0,0,-g])

        accel = ((1/m) * (R @ thrust  +  drag_world)) + gravity

        omega_dot = np.linalg.inv(I) @ (torque + damping_world - np.cross(omega_body, I @ omega_body))
        euler_dot = self.W_matrix(phi, theta) @ omega_body

        dstate = np.zeros(12)
        dstate[0:3] = vel
        dstate[3:6] = accel
        dstate[6:9] = euler_dot
        dstate[9:12] = omega_dot
        return dstate

# %%



       

# %%
