# %%
import numpy as np
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation as R

# %%
def fixed_wing_6dof(X, t, U, params):
    """
    6-DOF nonlinear dynamics for a fixed-wing UAV.
    States: [u, v, w, p, q, r, phi, theta, psi, x, y, z]
    Inputs: [throttle (T), aileron (delta_a), elevator (delta_e), rudder (delta_r)]
    """
    u, v, w = X[0], X[1], X[2]    # Body-frame velocities (m/s)
    p, q, r = X[3], X[4], X[5]    # Angular rates (rad/s)
    phi, theta, psi = X[6], X[7], X[8]  # Euler angles (rad)
    x, y, z = X[9], X[10], X[11]   # Inertial position (m)
    
    throttle, delta_a, delta_e, delta_r = U
    
    m, g, Ixx, Iyy, Izz, Ixz, rho, S, b, c = params[0:10]
    CD0, CL0, Cm0, CY0, Cl0, Cn0 = params[10:16]  # Aerodynamic coefficients
    CD_alpha, CL_alpha, Cm_alpha, CY_beta, Cl_beta, Cn_beta = params[16:22]
    CD_q, CL_q, Cm_q, CY_p, Cl_p, Cn_p = params[22:28]
    CD_delta_e, CL_delta_e, Cm_delta_e, CY_delta_r, Cl_delta_a, Cn_delta_r = params[28:34]
    
    V = np.sqrt(u**2 + v**2 + w**2)  # Airspeed (m/s)
    alpha = np.arctan2(w, u)          # Angle of attack (rad)
    beta = np.arcsin(v / V)           # Sideslip angle (rad)
    q_bar = 0.5 * rho * V**2          # Dynamic pressure (Pa)
    
    # Lift and Drag Coefficients (Nonlinear)
    CL = CL0 + CL_alpha * alpha + CL_q * (q * c / (2 * V)) + CL_delta_e * delta_e
    CD = CD0 + CD_alpha * alpha + CD_q * (q * c / (2 * V)) + CD_delta_e * delta_e
    CY = CY0 + CY_beta * beta + CY_p * (p * b / (2 * V)) + CY_delta_r * delta_r
    
    # Dimensionless Moments Coefficients
    Cl = Cl0 + Cl_beta * beta + Cl_p * (p * b / (2 * V)) + Cl_delta_a * delta_a
    Cm = Cm0 + Cm_alpha * alpha + Cm_q * (q * c / (2 * V)) + Cm_delta_e * delta_e
    Cn = Cn0 + Cn_beta * beta + Cn_p * (p * b / (2 * V)) + Cn_delta_r * delta_r
    
    # Forces in Body Frame
    Fx_aero = q_bar * S * (-CD * np.cos(alpha) + CL * np.sin(alpha))
    Fy_aero = q_bar * S * CY
    Fz_aero = q_bar * S * (-CD * np.sin(alpha) - CL * np.cos(alpha))
    
    # Moments in Body Frame
    tau_x = q_bar * S * b * Cl
    tau_y = q_bar * S * c * Cm
    tau_z = q_bar * S * b * Cn
    
    # Thrust Force (Simplified)
    Fx_prop = throttle * 10.0  # Thrust magnitude (adjust as needed)
    
    # Gravity Force in Body Frame
    Fx_grav = -m * g * np.sin(theta)
    Fy_grav = m * g * np.cos(theta) * np.sin(phi)
    Fz_grav = m * g * np.cos(theta) * np.cos(phi)
    
    # Total Forces/Moments
    Fx = Fx_aero + Fx_prop + Fx_grav
    Fy = Fy_aero + Fy_grav
    Fz = Fz_aero + Fz_grav
    tau = np.array([tau_x, tau_y, tau_z])
    
    # --- Rigid-Body Dynamics ---
    # Translational Acceleration
    u_dot = (Fx / m) - (q * w) + (r * v)
    v_dot = (Fy / m) - (r * u) + (p * w)
    w_dot = (Fz / m) - (p * v) + (q * u)
    
    # Rotational Acceleration (Euler's Equations)
    omega = np.array([p, q, r])
    I = np.array([[Ixx, 0, -Ixz],
                  [0, Iyy, 0],
                  [-Ixz, 0, Izz]])
    omega_dot = np.linalg.inv(I) @ (tau - np.cross(omega, I @ omega))
    
    # Euler Angle Rates
    phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)
    
    # Position Rates (Inertial Frame)
    rot_B_to_I = R.from_euler('ZYX', [psi, theta, phi]).as_matrix()
    vel_I = rot_B_to_I @ np.array([u, v, w])
    x_dot, y_dot, z_dot = vel_I[0], vel_I[1], vel_I[2]
    
    # State Derivatives
    X_dot = np.array([u_dot, v_dot, w_dot,
                      omega_dot[0], omega_dot[1], omega_dot[2],
                      phi_dot, theta_dot, psi_dot,
                      x_dot, y_dot, z_dot])
    
    return X_dot

# %% Simulation Parameters
params = [
    1.0,        # Mass (kg)
    9.81,       # Gravity (m/s²)
    0.1,        # Ixx (kg·m²)
    0.1,        # Iyy (kg·m²)
    0.1,        # Izz (kg·m²)
    0.01,       # Ixz (kg·m²)
    1.225,      # Air density (kg/m³)
    0.5,        # Wing area (m²)
    2.0,        # Wingspan (m)
    0.2,        # Mean chord (m)
    0.02,       # CD0
    0.1,        # CL0
    0.0,        # Cm0
    0.0,        # CY0
    0.0,        # Cl0
    0.0,        # Cn0
    0.1,        # CD_alpha
    5.0,        # CL_alpha
    -0.5,       # Cm_alpha
    -0.1,       # CY_beta
    -0.1,       # Cl_beta
    0.1,        # Cn_beta
    0.0,        # CD_q
    0.0,        # CL_q
    -10.0,      # Cm_q
    0.0,        # CY_p
    -0.5,       # Cl_p
    -0.1,       # Cn_p
    0.0,        # CD_delta_e
    0.5,        # CL_delta_e
    -1.0,       # Cm_delta_e
    0.1,        # CY_delta_r
    0.1,        # Cl_delta_a
    -0.1        # Cn_delta_r
]

# Initial Conditions (Trimmed Flight)
X0 = [10.0, 0.0, 0.0,  # u, v, w
      0.0, 0.0, 0.0,    # p, q, r
      0.0, 0.1, 0.0,    # phi, theta, psi
      0.0, 0.0, -100.0] # x, y, z (z = -100 m, altitude)

# Inputs: [throttle, aileron, elevator, rudder]
U = [0.5, 0.0, 0.0, 0.0]  # 50% throttle, no control deflections

# Time Vector
t = np.linspace(0, 10, 1000)  # 10 seconds simulation

# Solve ODE
X = odeint(fixed_wing_6dof, X0, t, args=(U, params))

# %% Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, X[:, 0], label='u (m/s)')
plt.plot(t, X[:, 1], label='v (m/s)')
plt.plot(t, X[:, 2], label='w (m/s)')
plt.legend()
plt.title('Body-Frame Velocities')

plt.subplot(3, 1, 2)
plt.plot(t, np.degrees(X[:, 6]), label='Roll (deg)')
plt.plot(t, np.degrees(X[:, 7]), label='Pitch (deg)')
plt.plot(t, np.degrees(X[:, 8]), label='Yaw (deg)')
plt.legend()
plt.title('Euler Angles')

plt.subplot(3, 1, 3)
plt.plot(t, -X[:, 11], label='Altitude (m)')
plt.legend()
plt.title('Position')
plt.tight_layout()
plt.show()

# %%
