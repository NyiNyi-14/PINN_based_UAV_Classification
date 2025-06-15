# %% Import Libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.chdir("/Users/nyinyia/Documents/09_LSU_GIT/PINN_based_UAV_Classification")
print("Current working directory:", os.getcwd())
print("Files in the directory:", os.listdir(os.getcwd()))

from Quadcopter.quadcopter import quadcopter
from Quadcopter.Quad_ScenarioSampler import Quad_ScenarioSampler

from Fixed_wings.FixedWingUAV import FixedWingUAV
from Fixed_wings.FW_ScenarioSampler import FW_ScenarioSampler

# %%
# Simulation parameter
duration = 10
dt = 0.01
time = np.arange(0, duration, dt)
sample = 5

# %% ============================ Quadcopter ============================
# Parameters
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

Quad_scene = Quad_ScenarioSampler(hover_omega = hover, delta = delta_speed, max_omega = max_speed)
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
init_test = np.zeros(12)
init_test[2] = -10

sol = solve_ivp(lambda t, y: Quad.dynamics(t, y, [hover+ 10]*4), [0, duration], init_test, t_eval=time)

plt.plot(time, sol.y[0])
plt.plot(time, sol.y[1])
plt.plot(time, sol.y[2])

# %% Simulate
q_state = []
for i in range(sample):
    omega = Quad_conditions[i][0]
    init = Quad_conditions[i][1]
    sol = solve_ivp(lambda t, y: Quad.dynamics(t, y, omega), [0, duration], init, t_eval=time)
    q_state.append(sol.y)

#  Visualization for Quad
few_samples = 10
n_plot = min(few_samples, len(q_state))  # plot up to 5 samples

# 2D Plots
fig_pos, ax_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_pos.suptitle("Position over Time")

fig_vel, ax_vel = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_vel.suptitle("Velocity over Time")

# Loop through each sample and plot
for i in range(n_plot):
    state = q_state[i]
    x, y, z = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]

    ax_pos[0].plot(time, x, label=f"{Quad_conditions[i][2]}")
    ax_pos[1].plot(time, y,)
    ax_pos[2].plot(time, z,)

    ax_vel[0].plot(time, vx, label=f"{Quad_conditions[i][2]}")
    ax_vel[1].plot(time, vy, )
    ax_vel[2].plot(time, vz, )

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

# 3D Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(n_plot):
    state = q_state[i]
    x, y, z = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]
    ax.plot(x, y, z, label=f"{Quad_conditions[i][2]}")

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Direction Plot')
ax.legend()
# ax.auto_scale_xyz(x, y, z)
plt.show()

# %% ============================ Fixed Wings ============================
# Parameters
params = [
    # === Mass and Inertia ===
    1.0,        # [0] mass (kg) — small electric fixed-wing UAV
    9.81,       # [1] gravity (m/s²)
    0.03,       # [2] Ixx (kg·m²) — roll inertia (reduced for 1 kg aircraft)
    0.04,       # [3] Iyy (kg·m²) — pitch inertia
    0.05,       # [4] Izz (kg·m²) — yaw inertia (Ixx + Iyy approx)
    0.01,       # [5] Ixz (kg·m²) — cross-inertia, small

    # === Environmental Constants ===
    1.225,      # [6] air density (kg/m³)

    # === Geometry ===
    0.3,        # [7] wing area S (m²) — suitable for 1 kg UAV
    1.5,        # [8] wingspan b (m) — gives AR ≈ 7.5
    0.2,        # [9] mean chord c (m) — S / b = 0.2

    # === Aerodynamic Coefficients ===
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

    # === Propulsion ===
    8000,       # [34] omega (RPM max) — realistic electric prop speed
    0.25,       # [35] propeller diameter (m) — 10-inch
    0.1         # [36] Ct — thrust coefficient for propeller
]

base_throttle = 0.7
delta_throttle = 0.15

FW_sampler = FW_ScenarioSampler(base_throttle = base_throttle, delta = delta_throttle)
FW_conditions = FW_sampler.sample(n_samples = sample)
FW = FixedWingUAV(params = params)
print(f"FW condition: \n {FW_conditions}")

# %% Manual testing
# init_test = np.zeros(12)
# init_test[-1] = -100 # z
# init_test[0] = 50 # u

# Trimmed state = [u, v, w, p, q, r, phi, theta, psi, x, y, z]
trim_state = np.zeros(12)
trim_state[0] = 15.0                  # u = forward velocity (m/s)
trim_state[7] = np.deg2rad(2)       # theta = small pitch angle (~4–5 deg)
trim_state[11] = -100                 # z = altitude in NED

# Inputs = [throttle, delta_a, delta_e, delta_r]
trim_input = [0.6, -0.02, -0., 0.]   # Elevator deflected slightly down

# sol = solve_ivp(lambda t, y: FW.dynamics(t, y, [base_throttle, 0, 0, 0]), [0, duration], init_test, t_eval=time)
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


# %% FW Simulate
fw_state = []
for i in range(sample):  # or 'sample'
    control = FW_conditions[i][0]        # [throttle, delta_a, delta_e, delta_r]
    init = FW_conditions[i][1]          # initial state

    sol = solve_ivp(lambda t, y: FW.dynamics(t, y, control), [0, duration], init, t_eval=time)
    fw_state.append(sol.y) 

# 2D Plots
few_samples = 10
n_plot = min(few_samples, len(fw_state)) 

fw_pos, fwa_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fw_pos.suptitle("FW Position over Time")

fw_vel, fwa_vel = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fw_vel.suptitle("FW Velocity over Time")

# Loop through each sample and plot
for i in range(n_plot):
    state = fw_state[i]
    x, y, z = state[9], state[10], -state[11]
    vx, vy, vz = state[0], state[1], -state[2]

    fwa_pos[0].plot(time, x, label=f"{FW_conditions[i][2]}")
    fwa_pos[1].plot(time, y,)
    fwa_pos[2].plot(time, z,)

    fwa_vel[0].plot(time, vx, label=f"{FW_conditions[i][2]}")
    fwa_vel[1].plot(time, vy, )
    fwa_vel[2].plot(time, vz, )

# Label position axes
fwa_pos[0].set_ylabel('X (m)')
fwa_pos[1].set_ylabel('Y (m)')
fwa_pos[2].set_ylabel('Z (m)')
fwa_pos[2].set_xlabel('Time (s)')
fwa_pos[0].legend()

# Label velocity axes
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
    x, y, z = state[9], state[10], -state[11]
    # vx, vy, vz = state[3], state[4], state[5]
    ax.plot(x, y, z, label=f"{FW_conditions[i][2]}")

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Direction Plot')
ax.legend()
# ax.auto_scale_xyz(x, y, z)
plt.show()

# %%
