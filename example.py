# %%
from pinn import PINN

import numpy as np
import matplotlib.pyplot as plt

from utils import plot, AffineSystem
import csv
import os

plt.rcParams["text.usetex"] = True
import tensorflow as tf

# %% System
A = np.array([[-5, 3], [0, -2]], dtype=np.float32)
A2 = np.array([[-5, 2.7], [0, -2]], dtype=np.float32)
B = np.array([[1], [2]], dtype=np.float32)
C = np.array([[1, 0], [0, 1]], dtype=np.float32)

f = lambda x: A @ x
f2 = lambda x: A2 @ x
g = lambda _: B
h = lambda x: C @ x

std_noise = 0.01
u = lambda t: tf.concat([tf.sin(t) + tf.cos(t)], 0)

# External simulator
ss = AffineSystem(f, g, h, n=2, std_noise=std_noise, seed=1234)
ss2 = AffineSystem(f2, g, h, n=2, std_noise=std_noise, seed=1234)
T = 6  # training interval
P = 0  # prediction interval
deltaT = 0.01

# External solution
x0 = np.array([[1], [0]], dtype=np.float32)
x = ss.simulate(x0, T + P, deltaT, u=u)
y = ss.y()

# Measurements
k = 50
max_T = int(np.floor(T / deltaT))
data = (ss.t[:, 0:max_T:k], y[:, 0:max_T:k])

# %% PINN Optimizer
forgetting_factors = [1]
training = "vanilla"  # "vanilla" or "primal-dual"
seeds = np.arange(20)
total = len(forgetting_factors) * 2 * len(seeds)
num = 0
for forgetting_factor in forgetting_factors:
    for model_mismatch in [True, False]:
        for seed in seeds:
            print(f"Progress: {num}/{total}")
            pinn = PINN(
                [20, 20, 20],
                ss2 if model_mismatch else ss,
                N_phys=10,
                T=T + P,
                N_dual=10,
                forgetting_factor=forgetting_factor,
                seed=seed,
                training=training,
                initial_integral=(
                    0.0 if training == "primal-dual" else forgetting_factor
                ),
            )
            pinn.set_data(data, u)
            losses = []
            weights = []

            loss, weight = pinn.train(5000)
            losses += loss
            weights += weight

            error = np.linalg.norm(x - pinn(ss.t).numpy(), axis=0).reshape((1, -1))
            normalized_error = np.sqrt(
                np.mean(error**2) / np.mean(np.linalg.norm(x, axis=0) ** 2)
            )
            print(f"Normalized error: {np.round(100*normalized_error, 3)}%")

            directory = f"output/{training}/{"mismatch" if model_mismatch else "match"}"
            os.makedirs(directory, exist_ok=True)
            with open(
                f"{directory}/{str(forgetting_factor).replace(".", "_")}.csv",
                "a",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow([seed, normalized_error])
            num += 1

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

normalized_error = np.sqrt(np.mean(error**2) / np.mean(np.linalg.norm(x, axis=0) ** 2))
print(f"Normalized error: {np.round(100*normalized_error, 3)}%")

# %% Save results
directory = f"output/{"mismatch" if model_mismatch else "match"}"
os.makedirs(directory, exist_ok=True)
with open(
    f"{directory}/{forgetting_factor}.csv",
    "a",
    newline="",
) as f:
    writer = csv.writer(f)
    writer.writerow([seed, normalized_error])
# %%
