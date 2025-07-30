# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# System parameters
g = 9.81  # gravity
l = 1.0  # length
m = 1.0  # mass
b = 0.1  # damping

theta_ref = np.pi  # Upright target


# Linearization of the pendulum dynamics around theta = pi
def linearize_pendulum(theta_eq):
    A = np.array([[0, 1], [-g / l * np.cos(theta_eq), -b / (m * l**2)]])
    B = np.array([[0], [1 / (m * l**2)]])
    C = np.array([[1, 0]])  # output y = theta
    return A, B, C


# Design LQR controller
def design_lqr_controller(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


# Get linearized system at theta = pi
A, B, C = linearize_pendulum(theta_ref)

# Define LQR weights
Q = np.diag([10, 1])  # weight on angle and angular velocity
R = np.array([[0.1]])  # weight on control effort

# Compute LQR gain
K_lqr = design_lqr_controller(A, B, Q, R)


# Dynamics of the nonlinear pendulum with integral state
def pendulum_dynamics_lqr(t, state):
    theta, omega, z = state
    x = np.array([[theta - theta_ref], [omega]])

    # Add integral control: u = -Kx - Ki*z
    Ki = 0.5
    u = -K_lqr @ x - Ki * z

    dtheta = omega
    domega = -(g / l) * np.sin(theta) - b * omega + float(u) / (m * l**2)
    dz = theta - theta_ref  # error integration

    return [dtheta, domega, dz]


# %%
# Initial state: close to upright
theta0 = 0
omega0 = 0.0
z0 = 0.0
x0 = [theta0, omega0, z0]

# Time span
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Simulate the system
sol = solve_ivp(pendulum_dynamics_lqr, t_span, x0, t_eval=t_eval)

# Extract results
theta = sol.y[0]
omega = sol.y[1]
z = sol.y[2]
time = sol.t

# Plot results
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(time, theta, label="Theta (rad)")
plt.axhline(theta_ref, color="r", linestyle="--", label="Target θ = π")
plt.ylabel("Theta")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, omega, label="Omega (rad/s)", color="orange")
plt.axhline(0, color="k", linestyle="--")
plt.ylabel("Omega")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, z, label="Integral of error", color="green")
plt.axhline(0, color="k", linestyle="--")
plt.ylabel("Integral of Error")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.suptitle("LQR + Integral Action for Pendulum Stabilization at θ = π")
plt.tight_layout()
plt.show()
