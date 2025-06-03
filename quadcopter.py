# %% Import Libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sampling import sampling

# %% Quadcopter Dynamics
class Quadcopter:
    def __init__(self, m, g, l, I, kD):
        self.m = m
        self.g = g
        self.l = l
        self.I = I
        self.kD = kD # drag torque constant

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

    def Rotor_torque(self, T1, T2, T3, T4):
        u1 = T1 + T2 + T3 + T4
        u2 = self.l * (T2 - T4)
        u3 = self.l * (T3 - T1)
        u4 = self.kD * (T1 - T2 + T3 - T4)
        thrust = np.array([0, 0, u1])
        torque = np.array([u2, u3, u4])
        return thrust, torque

    def dynamics(self, t, state, T1234):
        T1, T2, T3, T4 = T1234
        thrust, torque = self.Rotor_torque(T1, T2, T3, T4)
        m, g, I = self.m, self.g, self.I
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state

        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        ang = np.array([phi, theta, psi])
        omega = np.array([p, q, r])

        R = self.R_matrix(phi, theta, psi)
        acc = (1/m) * R @ thrust - np.array([0, 0, g])

        omega_dot = np.linalg.inv(I) @ (torque - np.cross(omega, I @ omega))
        euler_dot = self.W_matrix(phi, theta) @ omega

        dstate = np.zeros(12)
        dstate[0:3] = vel
        dstate[3:6] = acc
        dstate[6:9] = euler_dot
        dstate[9:12] = omega_dot
        return dstate

# %% Parameters
mass = 0.5
g = 9.81
l = 0.2
kD = 1e-7
Ixx = 5e-3
Iyy = 5e-3
Izz = 9e-3
I = np.diag([Ixx, Iyy, Izz])

# time frame
duration = 10
dt = 0.01
time = np.arange(0, duration, dt)

y0 = np.zeros(12)
T = {
    "T1": (2.0, 0.5),
    "T2": (2.0, 0.5),
    "T3": (2.0, 0.5),
    "T4": (2.0, 0.5),
}
samples = 100 
sample_init = sampling(T, samples)
T_dict = sample_init.LHS_sampling()
# sampling.plot_distributions(T_dict, "Latin Hypercube")
T_array = np.vstack([T_dict["T1"], T_dict["T2"], T_dict["T3"], T_dict["T4"]]).T

quad = Quadcopter(m = mass, g = g, l = l, I = I, kD = kD)
trajectories = []
for i in range(samples):
    T1234 = T_array[i]
    sol = solve_ivp(quad.dynamics, [0, duration], y0, args = (T1234,), t_eval = time)
    trajectories.append(sol.y)

# %% Visualization, Tests
plt.figure(figsize=(12, 6))
for i, states in enumerate(trajectories[:5]):  # plot first 5 only
    plt.plot(time, states[0], label=f"Case {i+1} (x pos)")
plt.xlabel("Time [s]")
plt.ylabel("Z Position [m]")
plt.title("Quadcopter Altitude with Different Constant Thrusts")
plt.legend()
plt.grid()
plt.show()

# %%