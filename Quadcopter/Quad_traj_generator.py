# %% Import Libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn

import os
os.chdir("/Users/nyinyia/Documents/09_LSU_GIT/PINN_based_UAV_Classification/Quadcopter")
print("Current working directory:", os.getcwd())
print("Files in the directory:", os.listdir(os.getcwd()))

from quadcopter import quadcopter
from Quad_ScenarioSampler import Quad_ScenarioSampler

# %% ============================ Simulation parameter ============================
duration = 10
dt = 0.01
time = np.arange(0, duration, dt)
sample = 1000
seeding = 14

q_mass = 0.5 # kg
g = 9.81 # m/s²
q_l = 0.2 # m
kD = 1e-6 # drag torque coefficient
kT = 3e-5 # thrust coefficient
k_drag_linear=0.5
k_drag_angular=0.02
Ixx, Iyy = 5e-3, 5e-3 # kg·m²
Izz = 9e-3 # kg·m²
I = np.diag([Ixx, Iyy, Izz])

# x, y, z, vx, vy, vz, phi, theta, psi, p, q, r
hover = np.sqrt(q_mass * g / (4 * kT))
delta_speed = 25 # rad/s
max_speed = 400

Quad_scene = Quad_ScenarioSampler(hover_omega = hover, delta = delta_speed, max_omega = max_speed, seed = seeding)
Quad_conditions = Quad_scene.sample(n_samples = sample)
Quad = quadcopter(m = q_mass, 
                  g = g, 
                  l = q_l, 
                  I = I, 
                  kD = kD, 
                  kT = kT, 
                  k_drag_linear = k_drag_linear, 
                  k_drag_angular = k_drag_angular)

print(f"Quad Condition: \n {Quad_conditions}")

# %% Manual testing
duration = 100
dt = 0.01
time = np.arange(0, duration, dt)

# x, y, z, vx, vy, vz, phi, theta, psi, p, q, r
init_test = np.zeros(12)
init_test[2] = 100
ds = 25

# init_test[11] = -np.random.uniform(0.1, 0.3)  # r (reduced rate)
# init_test[8] = np.random.uniform(0, 2*np.pi) #psi
# init_test[6] = np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # phi
# init_test[6] = -np.deg2rad(20)
# init_test[9] = -np.random.uniform(0.1, 0.3)  # p

init_test[3] = np.random.uniform(-0.2, 0.2)  # vx
init_test[4] = np.random.uniform(-0.2, 0.2)  # vy
init_test[5] = np.random.uniform(-0.2, 0.2)  # vz

# init_test = np.random.uniform(-0.5, 0.5, size=12)
# init_test[0:2] = np.random.uniform(-5, 5, 2) # x, y
# init_test[2] = np.random.uniform(5, 30) # z

# init_test[11], init_test[9], init_test[10] = np.random.uniform(-0.3, 0.3, size = 3)  # r, p
# init_test[8] = np.random.uniform(0, 2*np.pi)
# init_test[6], init_test[7] = np.random.uniform(np.deg2rad(-10), np.deg2rad(10), size = 2)  # phi

# init_test[7] = np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # theta
# init_test[7] = np.deg2rad(25)
# init_test[10] = np.random.uniform(0.1, 0.3)  # q

# init_test[6] = -np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # phi
# init_test[9] = -np.random.uniform(0.1, 0.3)  # p

init_test[7] = np.random.uniform(np.deg2rad(5), np.deg2rad(10))  # theta
init_test[10] = np.random.uniform(0.1, 0.3)  # q

test_speed = [hover + 0/5, hover + 0/5, hover + 0/5, hover + 0/5]

sol = solve_ivp(lambda t, y: Quad.dynamics(t, y, test_speed), [0, duration], init_test, t_eval=time)

plt.plot(time, sol.y[0], label = "x")
plt.plot(time, sol.y[1], label = "y")
plt.plot(time, sol.y[2], label = "z")
# plt.plot(time, sol.y[7], label = "pitch angel")
plt.legend()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2])

# %% Simulate
q_state = []
for i in range(sample):
    omega = Quad_conditions[i][0]
    init = Quad_conditions[i][1]
    sol = solve_ivp(lambda t, y: Quad.dynamics(t, y, omega), [0, duration], init, t_eval=time)
    q_state.append(sol.y.T)

#  Visualization for Quad
few_samples = 10
n_plot = min(few_samples, len(q_state))

# 2D Plots
fig_pos, ax_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_pos.suptitle("Position over Time")

fig_vel, ax_vel = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_vel.suptitle("Velocity over Time")

for i in range(n_plot):
    state = q_state[i]
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]

    ax_pos[0].plot(time, x, label=f"{Quad_conditions[i][2]}")
    ax_pos[1].plot(time, y,)
    ax_pos[2].plot(time, z,)

    ax_vel[0].plot(time, vx, label=f"{Quad_conditions[i][2]}")
    ax_vel[1].plot(time, vy, )
    ax_vel[2].plot(time, vz, )

ax_pos[0].set_ylabel('X (m)')
ax_pos[1].set_ylabel('Y (m)')
ax_pos[2].set_ylabel('Z (m)')
ax_pos[2].set_xlabel('Time (s)')
ax_pos[0].legend()

ax_vel[0].set_ylabel('Vx (m/s)')
ax_vel[1].set_ylabel('Vy (m/s)')
ax_vel[2].set_ylabel('Vz (m/s)')
ax_vel[2].set_xlabel('Time (s)')
ax_vel[0].legend()

plt.tight_layout()
plt.show()

# 3D Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(n_plot):
    state = q_state[i]
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
    ax.plot(x, y, z, label=f"{Quad_conditions[i][2]}")

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Direction Plot')
ax.legend()
# ax.auto_scale_xyz(x, y, z)
plt.show()

# %% Data Format
x_quad = q_state
label_quad = [0] * len(x_quad)

u_quad = []
for i in range(sample):
    omega = Quad_conditions[i][0]          
    T = len(time)
    u_traj = np.tile(omega.reshape(-1, 1), (1, T)) 
    u_quad.append(u_traj.T)

dx_quad = []
for x_traj, omega in zip(x_quad, u_quad): 
    dx_traj = np.zeros_like(x_traj)   
    for i in range(x_traj.shape[0]):
        dx_traj[i, :] = Quad.dynamics(0, x_traj[i, :], omega[i, :])
    dx_quad.append(dx_traj)

# %% Saving dataset
x_quad = np.array(x_quad)      
u_quad = np.array(u_quad)      
dx_quad = np.array(dx_quad)   
label_quad = np.array(label_quad)

np.savez_compressed("quad_dataset.npz", x=x_quad, u=u_quad, dx=dx_quad, label=label_quad)

# %%
