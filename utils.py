import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

import tensorflow as tf
import numpy as np

## Neural Network utilities


def flatten(vars):
    vs = []
    for v in vars:
        if v is not None:
            vs.append(tf.reshape(v, (-1, 1)))
    return tf.concat(vs, axis=0)


def init(layers, bias=True, seed=1234, dtype=tf.float32):
    Ws, bs = [], []
    default_rng = np.random.default_rng(seed)
    for i in range(len(layers) - 1):
        W = xavier_init([layers[i], layers[i + 1]], rng=default_rng)
        b = tf.zeros([1, layers[i + 1]])
        Ws.append(tf.Variable(W, dtype=dtype, name=f"W_{i}"))
        bs.append(tf.Variable(b, dtype=dtype, trainable=bias, name=f"b_{i}"))
    return Ws, bs


def xavier_init(size, rng=np.random.default_rng()):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return rng.normal(size=[in_dim, out_dim], scale=xavier_stddev)


class LinearLayer(tf.Module):
    def __init__(self, size, dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self.matrix = tf.Variable(tf.zeros(size), dtype=dtype)

    def __call__(self, x):
        return tf.matmul(x, self.matrix)


class NeuralNetwork(tf.Module):
    def __init__(
        self,
        layers,
        bias=True,
        in_phi=tf.tanh,
        phi=tf.identity,
        output_dim=None,
        seed=1234,
        dtype=tf.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = layers
        self.bias = bias
        self.dtype = dtype
        self.seed = seed
        self.reinit()
        self.in_phi = in_phi
        self.phi = phi
        self.output_dim = output_dim

    # @tf.function
    def __call__(self, input):
        num_layers = len(self.Ws)
        H = tf.cast(input, self.dtype)
        for layer in range(0, num_layers - 1):
            W = self.Ws[layer]
            b = self.bs[layer]
            H = self.in_phi(tf.add(tf.matmul(H, W), b))
        W = self.Ws[-1]
        b = self.bs[-1]
        output = self.phi(tf.add(tf.matmul(H, W), b))
        output = tf.transpose(output)
        if self.output_dim is not None:
            output = tf.reshape(output, (-1, self.output_dim[0], self.output_dim[1]))
        return output

    def reinit(self):
        self.Ws, self.bs = init(
            layers=self.layers, bias=self.bias, seed=self.seed, dtype=self.dtype
        )


## Hessian


def flatten_and_filter(grads_list):
    return tf.concat([tf.reshape(g, [-1]) for g in grads_list if g is not None], axis=0)


def gradient(f, p):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(p)
        loss = f()
    grads = tape.gradient(loss, p)
    return tf.reshape(flatten_and_filter(grads), (-1, 1))


def hessian(f, p):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(p)
        g = gradient(f, p)
    hessian = tape.jacobian(g, p)

    hessian_matrix = []
    for h in hessian:
        h_flat = tf.reshape(h, [g.shape[0], -1])
        hessian_matrix.append(h_flat)

    # Final full Hessian matrix
    return tf.concat(hessian_matrix, axis=1)


## System definition


class AffineSystem:

    def __init__(self, f, g, h, n, std_noise=0, seed=1234) -> None:
        self.f = f
        self.g = g
        self.h = h
        self.n = n
        self.std_noise = std_noise
        self.default_rng = np.random.default_rng(seed)

    def simulate(self, x0, T, deltaT, u=lambda t: np.array([[0]])):
        ts = np.arange(0, T, deltaT)
        self.t = ts.reshape((1, -1))
        self.deltaT = deltaT
        self.xs = [x0]
        self.ys = [self.h(x0)]
        us = u(self.t)
        for i in range(len(ts) - 1):
            x_plus = self.xs[-1] + deltaT * (
                self.f(self.xs[-1]) + self.g(self.xs[-1]) @ us[:, i : (i + 1)]
            )
            self.xs.append(x_plus)
            self.ys.append(self.h(x_plus))
        self.xs = np.hstack(self.xs)
        self.ys = np.hstack(self.ys)
        self.ys += self.std_noise * self.default_rng.normal(size=self.ys.shape)
        return self.xs

    def y(self):
        return self.ys


## Plotting functions


def imshow(image, logscale=False, **kwargs):
    image = np.log(1 + abs(image)) if logscale else image
    cmap = None if logscale else "seismic"
    if logscale:
        lims = tf.reduce_min(image), tf.reduce_max(image)
    else:
        lims = -tf.reduce_max(abs(image)), tf.reduce_max(abs(image))
    plt.imshow(image, vmin=lims[0], vmax=lims[1], cmap=cmap, **kwargs)
    plt.colorbar()


def plot(t, x, x_hat, T=None, subsampling=10, name="x"):
    colors = cm.rainbow(np.linspace(0, 1, x.shape[0]))
    fig, ax = plt.subplots()
    x_hat_values = x_hat(t).numpy()
    for i, c in enumerate(colors):
        plt.plot(t[0, :], x[i, :], color=c, label=rf"${name}_{i+1}$", alpha=0.3)
        plt.scatter(
            t[0, 0::subsampling],
            x_hat_values[i, 0::subsampling],
            s=5,
            color=c,
            label=rf"$\hat {name}_{i+1}$",
        )
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel(f"${name}$")
    plt.legend()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if T is not None:
        ax.fill_betweenx(ylim, 0, T, alpha=0.1, color="gray")
        ax.set_ylim(ylim)
    ax.set_xlim([0, xlim[1]])

    plt.show(block=False)
