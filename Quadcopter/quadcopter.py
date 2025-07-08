# %% Import Libraries
import numpy as np

# %% Quadcopter Dynamics
class quadcopter:
    def __init__(self, m, g, l, I, kD, kT, k_drag_linear, k_drag_angular):
        self.m = m
        self.g = g
        self.l = l
        self.I = I
        self.kD = kD  # drag torque constant
        self.kT = kT  # thrust coefficient
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

    def rotor_forces(self, omega):
        T = self.kT * omega**2
        tau = self.kD * omega**2
        return T, tau

    def Rotor_torque(self, T, tau):
        # T and tau are arrays of length 4: [T1, T2, T3, T4], [τ1, τ2, τ3, τ4]
        u1 = np.sum(T)
        u2 = self.l * (-T[1] + T[3])
        u3 = self.l * (-T[2] + T[0])
        u4 = tau[0] - tau[1] + tau[2] - tau[3]
        thrust = np.array([0, 0, u1])
        torque = np.array([u2, u3, u4])
        return thrust, torque

    def dynamics(self, t, state, omega):
        omega = np.array(omega)
        T, tau = self.rotor_forces(omega)
        thrust, torque = self.Rotor_torque(T, tau)

        m, g, I = self.m, self.g, self.I
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        vel = np.array([vx, vy, vz])
        ang = np.array([phi, theta, psi])
        omega_body = np.array([p, q, r])

        R = self.R_matrix(phi, theta, psi)
        gravity = np.array([0, 0, -g])
        drag_world = -self.k_drag_linear * vel  
        acc = (1/m) * (R @ thrust + drag_world) + gravity

        damping = -self.k_drag_angular * omega_body
        omega_dot = np.linalg.inv(I) @ (torque + damping - np.cross(omega_body, I @ omega_body))
        euler_dot = self.W_matrix(phi, theta) @ omega_body

        dstate = np.zeros(12)
        dstate[0:3] = vel
        dstate[3:6] = acc
        dstate[6:9] = euler_dot
        dstate[9:12] = omega_dot
        return dstate
    
# %%