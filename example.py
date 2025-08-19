# %%
from pinn import PINN

import numpy as np
import matplotlib.pyplot as plt

from utils import plot, AffineSystem

plt.rcParams["text.usetex"] = True
import tensorflow as tf

# %% System
A = np.array([[-5, 3], [0, -2]], dtype=np.float32)
B = np.array([[1], [2]], dtype=np.float32)
C = np.array([[1, 0]], dtype=np.float32)

f = lambda x: A @ x
g = lambda _: B
h = lambda x: C @ x

std_noise = 0.01
u = lambda t: tf.concat([tf.sin(t) + tf.cos(t)], 0)

# External simulator
ss = AffineSystem(f, g, h, n=2, std_noise=std_noise, seed=1234)
T = 6  # training interval
P = 3  # prediction interval
deltaT = 0.01

# External solution
x0 = np.array([[1], [0]], dtype=np.float32)
x = ss.simulate(x0, T + P, deltaT, u=u)
y = ss.y()

# Measurements
k = 10
max_T = int(np.floor(T / deltaT))
data = (ss.t[:, 0:max_T:k], y[:, 0:max_T:k])

# %% PINN Optimizer
pinn = PINN(
    [20, 20, 20], ss, N_phys=10, T=T + P, N_dual=10, forgetting_decay=1, seed=1234
)
pinn.set_data(data, u)
losses = []
weights = []
Ks = []

# %% Train
loss, weight, K = pinn.train(5000)
losses += loss
weights += weight
Ks += K

# %% Plot after training
plt.figure()
plt.plot(losses)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.grid()

plt.figure()
plt.plot(weights)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Weight value")
plt.grid()

plt.figure()
plt.plot(np.array(Ks))
plt.xlabel("Epoch")
plt.ylabel("Gains")
plt.yscale("log")
plt.legend(["$K_p$", "$K_i$"])
plt.grid()

plot(ss.t, x, pinn, T=T)
plot(ss.t, y, pinn.y, T=T, name="y")

plt.figure()
error = np.linalg.norm(x - pinn(ss.t).numpy(), axis=0).reshape((1, -1))
plt.plot(ss.t[0, :], error[0, :])
plt.yscale("log")
plt.xlabel("Time [s]")
plt.ylabel("$L_2$ error")
plt.grid()
plt.show()

normalized_error = np.mean(error) / np.mean(np.linalg.norm(x, axis=0))
print(f"Normalized error: {np.round(100*normalized_error, 3)}%")
# %%
