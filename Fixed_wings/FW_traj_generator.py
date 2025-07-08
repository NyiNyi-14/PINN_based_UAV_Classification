# %% Import Libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn

import os
os.chdir("...")
print("Current working directory:", os.getcwd())
print("Files in the directory:", os.listdir(os.getcwd()))

from FixedWingUAV import FixedWingUAV
from FW_ScenarioSampler import FW_ScenarioSampler

# %% Simulation parameter 
duration = 10
dt = 0.01
time = np.arange(0, duration, dt)
sample = 1000
seeding = 14

params = [
    1.0,        # [0] mass (kg) — small electric fixed-wing UAV
    9.81,       # [1] gravity (m/s²)
    0.03,       # [2] Ixx (kg·m²) — roll inertia (reduced for 1 kg aircraft)
    0.04,       # [3] Iyy (kg·m²) — pitch inertia
    0.05,       # [4] Izz (kg·m²) — yaw inertia (Ixx + Iyy approx)
    0.01,       # [5] Ixz (kg·m²) — cross-inertia, small

    1.225,      # [6] air density (kg/m³)

    0.3,        # [7] wing area S (m²) — suitable for 1 kg UAV
    1.5,        # [8] wingspan b (m) — gives AR ≈ 7.5
    0.2,        # [9] mean chord c (m) — S / b = 0.2

    0.02,       # [10] CD0 — parasitic drag
    0.1,        # [11] CL0 — lift offset at α = 0
    0.0,        # [12] Cm0 — pitching moment at trim
    0.0,        # [13] CY0 — side force at β = 0
    0.0,        # [14] Cl0 — roll moment at zero sideslip
    0.0,        # [15] Cn0 — yaw moment at zero sideslip

    0.05,       # [16] CD_alpha — induced drag slope (reduced from 0.1)
    5.7,        # [17] CL_alpha — lift slope per rad (~2π for thin airfoils)
    -0.8,       # [18] Cm_alpha — pitch stability derivative

    -0.1,       # [19] CY_beta — lateral force due to sideslip
    -0.1,       # [20] Cl_beta — roll moment from β (dihedral effect)
     0.1,       # [21] Cn_beta — yaw moment from β (fin effect)

    0.0,        # [22] CD_q — drag from pitch rate (usually negligible)
    0.0,        # [23] CL_q — lift from pitch rate (often small)
   -2.0,        # [24] Cm_q — pitch damping (good UAV value)

    0.0,        # [25] CY_p — lateral force from roll rate
   -0.5,        # [26] Cl_p — roll damping
   -0.02,       # [27] Cn_p — yaw damping from roll rate

    0.0,        # [28] CD_delta_e — elevator effect on drag
    0.5,        # [29] CL_delta_e — elevator lift effectiveness
   -1.0,        # [30] Cm_delta_e — elevator moment (negative = nose-down)

    0.1,        # [31] CY_delta_r — rudder side force
    0.3,        # [32] Cl_delta_a — aileron roll effectiveness
   -0.1,        # [33] Cn_delta_r — rudder yaw effectiveness

    8000,       # [34] omega (RPM max) — realistic electric prop speed
    0.25,       # [35] propeller diameter (m) — 10-inch
    0.1         # [36] Ct — thrust coefficient for propeller
]

FW = FixedWingUAV(params = params)
base_throttle = 0.5
delta_throttle = 0.15
FW_sampler = FW_ScenarioSampler(base_throttle = base_throttle, delta = delta_throttle, seed = seeding)
FW_conditions = FW_sampler.sample(n_samples = sample)
print(f"FW condition: \n {FW_conditions}")

# %% Manual testing
# init_test = np.zeros(12)
# init_test[-1] = -100 # z
# init_test[0] = 50 # u
duration = 10
dt = 0.01
time = np.arange(0, duration, dt)
trim_state = np.zeros(12)
trim_state[0] = 20 # u = forward velocity (m/s)
trim_state[1] = 0 # v
trim_state[2] = 0 # w
trim_state[3] = np.deg2rad(0) # p
trim_state[4] = 0.08 # q
trim_state[5] = np.deg2rad(0) # r
trim_state[6] = np.deg2rad(0) # phi
trim_state[7] = np.deg2rad(3) # theta
trim_state[8] = 0 # psi
trim_state[9] = 0 # x
trim_state[10] = 0 # y
trim_state[11] = -100 # z = altitude in NED

trim_input = [0.5, -0.00, -0.001, 0.00]

sol = solve_ivp(lambda t, y: FW.dynamics(t, y, trim_input), [0, duration], trim_state, t_eval=time)

plt.plot(time, sol.y[9], label = "x")
plt.plot(time, sol.y[10], label = "y")
plt.plot(time, -sol.y[11], label = "z")
plt.legend()
plt.grid()
plt.show()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[9], sol.y[10], -sol.y[11])

# %% Simulate for multiple scenarios
duration = 10
dt = 0.01
time = np.arange(0, duration, dt)
fw_state = []
for i in range(sample):  
    control = FW_conditions[i][0] # [throttle, delta_a, delta_e, delta_r]
    init = FW_conditions[i][1]         

    sol = solve_ivp(lambda t, y: FW.dynamics(t, y, control), [0, duration], init, t_eval=time)
    fw_state.append(sol.y.T) 

# 2D Plots
few_samples = 10
n_plot = min(few_samples, len(fw_state)) 

fw_pos, fwa_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fw_pos.suptitle("FW Position over Time")

fw_vel, fwa_vel = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fw_vel.suptitle("FW Velocity over Time")

for i in range(n_plot):
    state = fw_state[i]
    x, y, z = state[:, 9], state[:, 10], -state[:, 11]
    vx, vy, vz = state[:, 0], state[:, 1], -state[:, 2]

    fwa_pos[0].plot(time, x, label=f"{FW_conditions[i][2]}")
    fwa_pos[1].plot(time, y,)
    fwa_pos[2].plot(time, z,)

    fwa_vel[0].plot(time, vx, label=f"{FW_conditions[i][2]}")
    fwa_vel[1].plot(time, vy, )
    fwa_vel[2].plot(time, vz, )

fwa_pos[0].set_ylabel('X (m)')
fwa_pos[1].set_ylabel('Y (m)')
fwa_pos[2].set_ylabel('Z (m)')
fwa_pos[2].set_xlabel('Time (s)')
fwa_pos[0].legend()

fwa_vel[0].set_ylabel('Vx (m/s)')
fwa_vel[1].set_ylabel('Vy (m/s)')
fwa_vel[2].set_ylabel('Vz (m/s)')
fwa_vel[2].set_xlabel('Time (s)')
fwa_vel[0].legend()

plt.tight_layout()
plt.show()

# 3D Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(n_plot):
    state = fw_state[i]
    x, y, z = state[:, 9], state[:, 10], -state[:, 11]
    ax.plot(x, y, z, label=f"{FW_conditions[i][2]}")

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Direction Plot')
ax.legend()
plt.show()

# %% Data Format
x_fw = fw_state
label_fw = [1] * len(x_fw)

u_fw = []
for i in range(sample):
    throttle = FW_conditions[i][0]           
    T = len(time)
    u_traj = np.tile(throttle.reshape(-1, 1), (1, T)) 
    u_fw.append(u_traj.T)

dx_fw = []
for x_traj, throttle in zip(x_fw, u_fw):  
    dx_traj = np.zeros_like(x_traj)        
    for i in range(x_traj.shape[0]):      
        dx_traj[i, :] = FW.dynamics(0, x_traj[i, :], throttle[i, :])
    dx_fw.append(dx_traj)

# %% Saving dataset
x_fw = np.array(x_fw)      
u_fw = np.array(u_fw)     
dx_fw = np.array(dx_fw)   
label_fw = np.array(label_fw)  

np.savez_compressed("fw_dataset.npz", x=x_fw, u=u_fw, dx=dx_fw, label=label_fw)

# %%