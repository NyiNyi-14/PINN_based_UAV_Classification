# %% Import Libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sampling import sampling
from mpl_toolkits.mplot3d import Axes3D

import os
os.chdir("/Users/nyinyia/Documents/09_LSU_GIT/PINN_based_UAV_Classification")
print("Current working directory:", os.getcwd())
print("Files in the directory:", os.listdir(os.getcwd()))

from quadcopter import quadcopter 
from Quad_InitialStateSampler import Quad_InitialStateSampler
from Quad_RotorConditionSampler import Quad_RotorConditionSampler

# %% Quadcopter Parameters
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

hover = np.sqrt(q_mass * g / (4 * kT))
delta_speed = 25 # rad/s
max_speed = 400
sample = 100
omega_generator = Quad_RotorConditionSampler(hover_omega = hover, delta = delta_speed, max_omega = max_speed)
omega_set, omega_labels = omega_generator.sample(n_samples = sample)

init_generator = Quad_InitialStateSampler()
init_set, init_labels = init_generator.sample(n_samples = sample)

# %% Simulate
# time frame
duration = 10
dt = 0.01
time = np.arange(0, duration, dt)
quad = quadcopter(m = q_mass, g = g, l = q_l, I = I, kD = kD, kT = kT, k_drag_linear = k_drag_linear, k_drag_angular = k_drag_angular)
q_state = []

for i in range(sample):
    init = init_set[i]
    omega = omega_set[i]
    sol = solve_ivp(lambda t, y: quad.dynamics(t, y, omega), [0, duration], init, t_eval=time)
    q_state.append(sol.y)

# %% Visualization
n_plot = min(50, len(q_state))  # plot up to 5 samples

fig_pos, ax_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_pos.suptitle("Position over Time")

fig_vel, ax_vel = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_vel.suptitle("Velocity over Time")

# Loop through each sample and plot
for i in range(n_plot):
    state = q_state[i]
    x, y, z = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]

    ax_pos[0].plot(time, x, label=f"Sample {i+1}")
    ax_pos[1].plot(time, y, label=f"Sample {i+1}")
    ax_pos[2].plot(time, z, label=f"Sample {i+1}")

    ax_vel[0].plot(time, vx, label=f"Sample {i+1}")
    ax_vel[1].plot(time, vy, label=f"Sample {i+1}")
    ax_vel[2].plot(time, vz, label=f"Sample {i+1}")

# Label position axes
ax_pos[0].set_ylabel('X (m)')
ax_pos[1].set_ylabel('Y (m)')
ax_pos[2].set_ylabel('Z (m)')
ax_pos[2].set_xlabel('Time (s)')
ax_pos[0].legend()

# Label velocity axes
ax_vel[0].set_ylabel('Vx (m/s)')
ax_vel[1].set_ylabel('Vy (m/s)')
ax_vel[2].set_ylabel('Vz (m/s)')
ax_vel[2].set_xlabel('Time (s)')
ax_vel[0].legend()

plt.tight_layout()
plt.show()

# %% Test one by one
sol = solve_ivp(lambda t, y: quad.dynamics(t, y, omega_set[2]), [0, duration], init_set[2], t_eval=time)
plt.plot(sol.t, sol.y[0], label='X Position')
plt.plot(sol.t, sol.y[1], label='Y Position')
plt.plot(sol.t, sol.y[2], label='Z Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Quadcopter Position Over Time')
plt.legend()
plt.grid()
plt.show()

# %% #D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='3D trajectory')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Final Direction Vector')
ax.legend()
ax.auto_scale_xyz(x, y, z)
plt.show()

# %%
