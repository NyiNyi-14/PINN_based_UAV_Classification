# %% Import Libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Assuming your heli_dynamics class is in helicopter/HelicopterUAV.spy
from HelicopterUAV import HelicopterUAV
from Heli_ScenarioSampler import Heli_ScenarioSampler

import os
os.chdir("/Users/nyinyia/Documents/09_LSU_GIT/PINN_based_UAV_Classification")
print("Current working directory:", os.getcwd())
print("Files in the directory:", os.listdir(os.getcwd()))

# %% ============================ Simulation parameter ============================
duration = 10
dt = 0.01
time = np.arange(0, duration, dt)
sample = 5

m = 1.5
g = 9.81
l = 0.3
d = 0.9
Ixx, Iyy, Izz = 0.03, 0.04, 0.05
I = np.diag([Ixx, Iyy, Izz])
k_drag_linear = 0.5
k_drag_angular = 0.05

Heli_scene = Heli_ScenarioSampler(m, g)
Heli_conditions = Heli_scene.sample(n_samples = sample)

Heli = HelicopterUAV(m = m, 
                     g = g, 
                     l = l, 
                     d = d, 
                     I = I, 
                     k_drag_linear = k_drag_linear, 
                     k_drag_angular = k_drag_angular)

# x, y, z, vx, vy, vz, phi, theta, psi, p, q, r
print(f"Heli Condition: \n {Heli_conditions}")

# %%
heli_state = []
for i in range(sample):
    thrust_cyclic = Heli_conditions[i][0]
    init = Heli_conditions[i][1]
    sol = solve_ivp(lambda t, y: Heli.dynamics(t, y, thrust_cyclic), [0, duration], init, t_eval=time)
    heli_state.append(sol.y.T)

# %%
#  Visualization for Quad
few_samples = 10
n_plot = min(few_samples, len(heli_state))

# 2D Plots
fig_pos, ax_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_pos.suptitle("Position over Time")

fig_vel, ax_vel = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_vel.suptitle("Velocity over Time")

for i in range(n_plot):
    state = heli_state[i]
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]

    ax_pos[0].plot(time, x, label=f"{Heli_conditions[i][2]}")
    ax_pos[1].plot(time, y,)
    ax_pos[2].plot(time, z,)

    ax_vel[0].plot(time, vx, label=f"{Heli_conditions[i][2]}")
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
    state = heli_state[i]
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
    ax.plot(x, y, z, label=f"{Heli_conditions[i][2]}")

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Direction Plot')
ax.legend()
# ax.auto_scale_xyz(x, y, z)
plt.show()

# %% Data Format
x_heli = heli_state
label_heli = [2] * len(x_heli)

u_heli = []
for i in range(sample):
    thrust_cyclic = Heli_conditions[i][0]          
    T = len(time)
    u_traj = np.tile(thrust_cyclic.reshape(-1, 1), (1, T)) 
    u_heli.append(u_traj.T)

dx_heli = []
for x_traj, thrust_cyclic in zip(x_heli, u_heli): 
    dx_traj = np.zeros_like(x_traj)   
    for i in range(x_traj.shape[0]):
        dx_traj[i, :] = Heli.dynamics(0, x_traj[i, :], thrust_cyclic[i, :])
    dx_heli.append(dx_traj)

# %% Saving dataset
x_heli = np.array(x_heli)      
u_heli = np.array(u_heli)      
dx_heli = np.array(dx_heli)   
label_heli = np.array(label_heli)

np.savez_compressed("heli_dataset.npz", x=x_heli, u=u_heli, dx=dx_heli, label=label_heli)

# %%
