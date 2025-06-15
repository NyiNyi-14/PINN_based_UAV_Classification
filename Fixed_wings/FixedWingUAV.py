# %%
import numpy as np

# %%
class FixedWingUAV:
    def __init__(self, params):
        """
        Initialize fixed-wing parameters.
        params: list of constants and aerodynamic coefficients in specific order.
        """
        self.m, self.g, self.Ixx, self.Iyy, self.Izz, self.Ixz, self.rho, self.S, self.b, self.c = params[0:10]
        self.CD0, self.CL0, self.Cm0, self.CY0, self.Cl0, self.Cn0 = params[10:16]
        self.CD_alpha, self.CL_alpha, self.Cm_alpha, self.CY_beta, self.Cl_beta, self.Cn_beta = params[16:22]
        self.CD_q, self.CL_q, self.Cm_q, self.CY_p, self.Cl_p, self.Cn_p = params[22:28]
        self.CD_delta_e, self.CL_delta_e, self.Cm_delta_e, self.CY_delta_r, self.Cl_delta_a, self.Cn_delta_r = params[28:34]
        self.omega, self.D, self.Ct = params[34:37]

        self.I = np.array([
            [self.Ixx, 0, -self.Ixz],
            [0, self.Iyy, 0],
            [-self.Ixz, 0, self.Izz]
        ])

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

    def dynamics(self, t, X, U):
        u, v, w = X[0:3] # Body-frame velocities (m/s)
        p, q, r = X[3:6] # Angular rates (rad/s)
        phi, theta, psi = X[6:9] # Euler angles (rad)
        x, y, z = X[9:12] # Inertial position (m)
        throttle, delta_a, delta_e, delta_r = U

        epsilon = 1e-3  # Small number to prevent divide-by-zero
        V = np.sqrt(u**2 + v**2 + w**2) # Airspeed (m/s)
        if V < epsilon:
            V = epsilon

        alpha = np.arctan2(w, u) # Angle of attack (rad)
        beta = np.clip(v / V, -1.0, 1.0) # Sideslip angle (rad)
        beta = np.arcsin(beta)
        q_bar = 0.5 * self.rho * V**2 # Dynamic pressure (Pa)

        # Lift and Drag Coefficients (Nonlinear)
        CL = self.CL0 + self.CL_alpha * alpha + self.CL_q * (q * self.c / (2 * V)) + self.CL_delta_e * delta_e
        CD = self.CD0 + self.CD_alpha * alpha + self.CD_q * (q * self.c / (2 * V)) + self.CD_delta_e * delta_e
        CY = self.CY0 + self.CY_beta * beta + self.CY_p * (p * self.b / (2 * V)) + self.CY_delta_r * delta_r

        # Dimensionless Moments Coefficients
        Cl = self.Cl0 + self.Cl_beta * beta + self.Cl_p * (p * self.b / (2 * V)) + self.Cl_delta_a * delta_a
        Cm = self.Cm0 + self.Cm_alpha * alpha + self.Cm_q * (q * self.c / (2 * V)) + self.Cm_delta_e * delta_e
        Cn = self.Cn0 + self.Cn_beta * beta + self.Cn_p * (p * self.b / (2 * V)) + self.Cn_delta_r * delta_r

        # Forces in Body Frame
        Fx_aero = q_bar * self.S * (-CD * np.cos(alpha) + CL * np.sin(alpha))
        Fy_aero = q_bar * self.S * CY
        Fz_aero = q_bar * self.S * (-CD * np.sin(alpha) - CL * np.cos(alpha))

        # Thrust Force
        Omega_max = self.omega * 2 * np.pi / 60  # Max RPM → rad/s
        Omega_p = throttle * Omega_max     # Actual RPM based on throttle
        Fx_prop = (self.rho * self.D**4 / (4 * np.pi**2)) * (Omega_p**2) * self.Ct

        # Gravity Force in Body Frame
        Fx_grav = -self.m * self.g * np.sin(theta)
        Fy_grav =  self.m * self.g * np.cos(theta) * np.sin(phi)
        Fz_grav =  self.m * self.g * np.cos(theta) * np.cos(phi)

        # Total Forces/Moments
        Fx = Fx_aero + Fx_prop + Fx_grav
        Fy = Fy_aero + Fy_grav
        Fz = Fz_aero + Fz_grav

        # Moments in Body Frame
        tau_x = q_bar * self.S * self.b * Cl
        tau_y = q_bar * self.S * self.c * Cm
        tau_z = q_bar * self.S * self.b * Cn
        tau = np.array([tau_x, tau_y, tau_z])

        # Translational Acceleration
        u_dot = (Fx / self.m) - (q * w) + (r * v)
        v_dot = (Fy / self.m) - (r * u) + (p * w)
        w_dot = (Fz / self.m) - (p * v) + (q * u)

        # Rotational Acceleration
        omega = np.array([p, q, r])
        omega_dot = np.linalg.inv(self.I) @ (tau - np.cross(omega, self.I @ omega))

        # Euler Angle Rates
        phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

        # Position Rates (Inertial)
        R_bi = self.R_matrix(phi, theta, psi)
        vel_I = R_bi @ np.array([u, v, w])
        x_dot, y_dot, z_dot = vel_I

        return np.array([
            u_dot, v_dot, w_dot,
            omega_dot[0], omega_dot[1], omega_dot[2],
            phi_dot, theta_dot, psi_dot,
            x_dot, y_dot, z_dot
        ])

# %%
