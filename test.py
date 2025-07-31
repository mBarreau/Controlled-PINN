# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are


# %%
class Pendulum:
    def __init__(self, l, m, b, Q, R, dt, T, x0, theta_ref):
        self.g = 9.81
        self.l = l
        self.m = m
        self.b = b

        self.Q = Q
        self.R = R

        self.dt = dt
        self.time = np.arange(0, T, dt)
        self.xs = [x0]

        self.theta_ref = theta_ref

        self.linearize_pendulum()
        self.design_lqr_controller()
        self.u = [0]

    def design_lqr_controller(self):
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ P

    def linearize_pendulum(self):
        theta = self.xs[-1][0]
        A = np.array(
            [
                [0, 1],
                [-self.g / self.l * np.cos(theta), -self.b / (self.m * self.l**2)],
            ]
        )
        B = np.array([[0], [1 / (self.m * self.l**2)]])
        C = np.array([[1, 0]])  # output y = theta
        self.A = A
        self.B = B
        self.C = C

    def pendulum_dynamics_lqr(self):
        theta = self.xs[-1][0]
        omega = self.xs[-1][1]
        z = self.xs[-1][2]

        x = np.array([[theta - theta_ref], [omega]])

        # Add integral control: u = -Kx - Ki*z
        Ki = 0.5
        u = -self.K @ x - Ki * z
        self.u.append(u[0, 0])

        dtheta = omega
        domega = (
            -(self.g / self.l) * np.sin(theta)
            - self.b * omega
            + u[0, 0] / (self.m * self.l**2)
        )
        dz = theta - self.theta_ref  # error integration

        return np.array([dtheta, domega, dz])

    def simulate(self, dt_linearize=0.0):
        last_linearization = 0.0
        for _ in range(len(self.time) - 1):
            last_linearization += self.dt
            if last_linearization >= dt_linearize:
                last_linearization = 0.0
                self.linearize_pendulum()
                self.design_lqr_controller()
            xp1 = self.xs[-1] + dt * self.pendulum_dynamics_lqr()
            self.xs.append(xp1)
        return np.array(self.xs), self.u


# %%
# System parameters
l = 1.0  # length
m = 1.0  # mass
b = 0.1  # damping

# Define controller
Q = np.diag([5, 1])  # weight on angle and angular velocity
R = np.array([[0.1]])  # weight on control effort
theta_ref = np.pi * 3 / 4

# Initial state
theta0 = 0
omega0 = 0.0
z0 = 0.0
x0 = np.array([theta0, omega0, z0])

# Simulator parameters
T = 30
dt = 0.01

pendulum = Pendulum(l, m, b, Q, R, dt, T, x0, theta_ref)
xs, u = pendulum.simulate(dt_linearize=1.0)
time = pendulum.time

# Plot results
theta = xs[:, 0]
omega = xs[:, 1]
z = xs[:, 2]

plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
plt.plot(time, theta)
plt.axhline(
    theta_ref, color="r", linestyle="--", label=f"Target θ = {np.round(theta_ref,1)}"
)
plt.ylabel("Theta (rad)")
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(time, omega, color="orange")
plt.ylabel("Omega (rad/s)")
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(time, z, label="Integral of error", color="green")
plt.ylabel("Integral of Error")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(time, u, color="k")
plt.ylabel("Control action")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.suptitle("LQR + Integral Action for Pendulum Stabilization at θ = π")
plt.tight_layout()
plt.show()

# %%
