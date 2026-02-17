import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import NeuralNetwork


class PINN:
    def __init__(
        self,
        layers,
        ss,
        N_phys=10,
        N_dual=10,
        T=5,
        forgetting_factor=1e-3,
        training="primal-dual",
        initial_integral=0.0,
        seed=1234,
    ):
        self.x_hat = NeuralNetwork([1] + layers + [ss.n], seed=seed)
        self.n = ss.n
        self.optimizer_primal = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer_dual = tf.keras.optimizers.SGD(learning_rate=1e-3 * N_dual)
        self.delta_t_dual = self.optimizer_primal.learning_rate * N_dual
        self.N_dual = N_dual
        self.N_phys = N_phys
        self.T = T
        self.f = ss.f
        self.g = ss.g
        self.h = ss.h
        self.t_tf = tf.Variable(
            tf.zeros((int(self.N_phys * self.T), 1)), dtype=self.x_hat.dtype
        )
        self.data = None

        self.Kp = 1.0
        self.Ki = 1.0

        self.training = training
        self.forgetting_factor = forgetting_factor
        self.integral = tf.Variable(initial_integral, dtype=self.x_hat.dtype)
        self.weight = tf.Variable(0, dtype=self.x_hat.dtype)

        tf.random.set_seed(seed)
        self.resample()

    def set_data(self, data, u):
        self.data = data[0], data[1]
        self.u = u

    def resample(self):
        t_tf = tf.random.uniform(self.t_tf.shape, dtype=self.x_hat.dtype) * self.T
        self.t_tf.assign(t_tf)

    def __call__(self, t):
        return self.x_hat(tf.transpose(t))

    def y(self, t):
        return self.h(self(t))

    def get_residual(self):
        dx_hat_tf = []
        t = tf.transpose(self.t_tf)
        for i in range(self.n):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(t)
                x_hat_tf = self(t)[i]
            grads = tape.gradient(x_hat_tf, t)
            dx_hat_tf.append(tf.reshape(grads, (1, -1)))
        dx_hat_tf = tf.concat(dx_hat_tf, 0)
        return dx_hat_tf - self.f(self(t)) - self.g(self(t)) @ self.u(t)

    def get_mse_data(self):
        if self.data is None:
            return 0.0
        mse_data = tf.reduce_mean(
            tf.square(tf.norm(self.data[1] - self.y(self.data[0]), axis=0))
        )
        return mse_data

    def get_mse_residual(self):
        residuals = tf.square(self.get_residual())
        return tf.reduce_mean(residuals)

    @tf.function
    def get_cost(self):
        return self.get_mse_data() + self.weight * self.get_mse_residual()

    @tf.function
    def primal_update(self):
        with tf.GradientTape(watch_accessed_variables=False) as loss_tape:
            loss_tape.watch(self.x_hat.trainable_variables)
            loss = self.get_cost()
        grads = loss_tape.gradient(loss, self.x_hat.trainable_variables)
        self.optimizer_primal.apply_gradients(
            zip(grads, self.x_hat.trainable_variables)
        )
        return loss

    @tf.function
    def dual_update(self):
        self.integral.assign_add(
            self.delta_t_dual
            * (self.get_mse_residual() - self.forgetting_factor * self.integral)
        )
        return self.get_cost()

    @tf.function
    def compute_weight(self):
        self.weight.assign(self.Kp * self.get_mse_residual() + self.Ki * self.integral)

    def train(self, epochs=3000):
        losses = []
        weights = []
        self.resample()
        pbar = tqdm(range(epochs))
        for i in pbar:
            self.primal_update()
            if i % self.N_dual == 0 and i > 0:
                if self.training == "primal-dual":
                    self.dual_update()
                self.resample()
            self.compute_weight()
            loss = self.get_cost().numpy()
            pbar.set_description(f"Loss: {loss:.6f}")
            losses.append(loss)
            weights.append(self.weight.numpy())
        return losses, weights
