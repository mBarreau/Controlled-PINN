import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

np.random.seed(100)
tf.random.set_seed(100)
# -----------------------------
# Utils
# -----------------------------
def now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def sample_interior(N):
    x = (2.0*np.random.rand(N,1).astype(np.float32))-1.0
    y = (2.0*np.random.rand(N,1).astype(np.float32))-1.0
    return tf.constant(x), tf.constant(y)

def sample_boundary(N):
    Nq = N//4
    s = (2.0*np.random.rand(Nq,1).astype(np.float32))-1.0
    ones = np.ones_like(s, dtype=np.float32)
    xL, yL = -ones, s
    xR, yR =  ones, s
    xB, yB = s, -ones
    xT, yT = s,  ones
    x = np.vstack([xL, xR, xB, xT])
    y = np.vstack([yL, yR, yB, yT])
    return tf.constant(x), tf.constant(y)

def u_exact_xy(x, y):
    return np.sin(np.pi*x) * np.sin(4*np.pi*y)

def q_xy_func(x, y, k):
    s = u_exact_xy(x, y)
    return (-(np.pi**2)*s - (4*np.pi)**2*s + (k**2)*s)

def q_tensor(x, y, k):
    x_np = x.numpy(); y_np = y.numpy()
    q_np = q_xy_func(x_np, y_np, k).astype(np.float32)
    return tf.constant(q_np)

def make_grid(Nx=121, Ny=121):
    xv = np.linspace(-1.0, 1.0, Nx, dtype=np.float32)
    yv = np.linspace(-1.0, 1.0, Ny, dtype=np.float32)
    X, Y = np.meshgrid(xv, yv, indexing='xy')
    return xv, yv, X, Y

def predict_on_grid(model, xv, yv):
    X, Y = np.meshgrid(xv, yv, indexing='xy')
    x_col = X.reshape(-1,1).astype(np.float32)
    y_col = Y.reshape(-1,1).astype(np.float32)
    u_pred = model.u(tf.constant(x_col), tf.constant(y_col)).numpy().reshape(Y.shape)
    return u_pred

@tf.function
def mse(a, b):
    return tf.reduce_mean(tf.square(a-b))

def validate(model, k, Nv_int=4096, Nv_bnd=2048, Nx=121, Ny=121):
    xi, yi = sample_interior(Nv_int)
    qv = q_tensor(xi, yi, k)
    f = model.pde_residual(xi, yi, qv)
    L_f = tf.reduce_mean(tf.square(f)).numpy()

    xb, yb = sample_boundary(Nv_bnd)
    u_true_b = u_exact_xy(xb.numpy(), yb.numpy()).astype(np.float32)
    u_pred_b = model.u(xb, yb).numpy()
    L_bc = float(np.mean((u_pred_b - u_true_b)**2))

    xv, yv, X, Y = make_grid(Nx, Ny)
    U_true = u_exact_xy(X, Y).astype(np.float32)
    U_pred = predict_on_grid(model, xv, yv)
    rel_l2 = float(np.linalg.norm(U_pred - U_true) / (np.linalg.norm(U_true)+1e-12))

    val_total = L_f + L_bc
    return val_total, rel_l2, (xv, yv, U_true, U_pred)

# -----------------------------
# Network
# -----------------------------
def xavier_init(shape, dtype=tf.float32):
    in_dim, out_dim = shape
    stddev = np.sqrt(2.0/(in_dim+out_dim))
    return tf.Variable(tf.random.truncated_normal(shape, stddev=stddev, dtype=dtype))

class MLP(tf.Module):
    def __init__(self, layers, name=None):
        super().__init__(name=name)
        self.sizes = layers
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            self.weights.append(xavier_init((layers[i], layers[i+1])))
            self.biases.append(tf.Variable(tf.zeros([layers[i+1]], dtype=tf.float32)))

    @tf.function
    def __call__(self, x):
        z = x
        for i in range(len(self.sizes)-2):
            z = tf.tanh(tf.matmul(z, self.weights[i]) + self.biases[i])
        return tf.matmul(z, self.weights[-1]) + self.biases[-1]

# -----------------------------
# Helmholtz 2D PINN
# -----------------------------
class Helmholtz2DPINN(tf.Module):
    """
    Δu + k^2 u - q(x,y) = 0 on [-1,1]^2.
    Inputs (x,y) are *physical*; internal net sees normalised coords.
    """
    def __init__(self, k=1.0, layers=(2, 64, 64, 64, 1)):
        super().__init__()
        self.net = MLP(layers, name='pinn')
        self.k2 = tf.constant(float(k*k), tf.float32)

        # normalisation stats (set later by Trainer)
        self.mu_x  = tf.constant(0.0, tf.float32)
        self.sx    = tf.constant(1.0, tf.float32)
        self.mu_y  = tf.constant(0.0, tf.float32)
        self.sy    = tf.constant(1.0, tf.float32)

    def set_normalisation(self, mu_x, sx, mu_y, sy):
        self.mu_x = tf.constant(mu_x, tf.float32)
        self.sx   = tf.constant(sx,   tf.float32)
        self.mu_y = tf.constant(mu_y, tf.float32)
        self.sy   = tf.constant(sy,   tf.float32)

    @tf.function
    def u(self, x, y):
        # x,y are physical; we normalise inside
        x_hat = (x - self.mu_x) / self.sx
        y_hat = (y - self.mu_y) / self.sy
        inp = tf.concat([x_hat, y_hat], axis=1)
        return self.net(inp)

    @tf.function
    def pde_residual(self, x, y, q_xy):
        # Δu + k^2 u - q(x,y)
        with tf.GradientTape(persistent=True) as g2:
            g2.watch([x, y])
            with tf.GradientTape(persistent=True) as g1:
                g1.watch([x, y])
                u = self.u(x, y)
            u_x = g1.gradient(u, x)
            u_y = g1.gradient(u, y)
        u_xx = g2.gradient(u_x, x)
        u_yy = g2.gradient(u_y, y)
        del g1, g2
        return u_xx + u_yy + self.k2 * u - q_xy

# -----------------------------
# Trainer with modes: "pi", "primal_dual", "primal"
# λ_f multiplies L_f (PDE), not boundary loss.
# -----------------------------
class Trainer:
    def __init__(self, model: Helmholtz2DPINN, mode="pi",
                 lr_primal=1e-3, lr_dual=5e-3,
                 epochs=10000, Nf=4096, Nb=2048, val_interval=250,
                 dual_period=50,
                 root_dir="runs", run_dir=None,
                 pi_Kp=0.0, pi_Ki=0.05, forgetting_rate=5e-1, lambda_clamp=(0.0, 1e3)):
        self.model = model
        self.mode = mode
        self.opt = tf.keras.optimizers.Adam(lr_primal)
        self.opt_dual = tf.keras.optimizers.Adam(lr_dual)
        self.dual_period = int(dual_period)

        # λ_f weighting L_f
        self.l_f = tf.Variable(0.0, dtype=tf.float32, trainable=True)

        # PI state (for PI mode)
        self.pi_Kp = tf.constant(float(pi_Kp), tf.float32)
        self.pi_Ki = tf.constant(float(pi_Ki), tf.float32)
        t = np.arange(0, float(5.0 / max(forgetting_rate,1e-6)), lr_primal)
        self.forgetting = tf.constant(np.exp(-forgetting_rate * t[::-1]).astype(np.float32))
        W = int(self.forgetting.shape[0])
        self.win_f = tf.Variable(tf.zeros([W], dtype=tf.float32), trainable=False)
        self.lam_min = tf.constant(lambda_clamp[0], tf.float32)
        self.lam_max = tf.constant(lambda_clamp[1], tf.float32)
        self.global_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)

        # schedule
        self.epochs = int(epochs)
        self.Nf = int(Nf)
        self.Nb = int(Nb)
        self.val_interval = int(val_interval)

        # --------- NEW: estimate normalisation and pass to model ---------
        xi_big, yi_big = sample_interior(100000)
        mu_x = float(tf.reduce_mean(xi_big).numpy())
        sx   = float(tf.math.reduce_std(xi_big).numpy() + 1e-6)
        mu_y = float(tf.reduce_mean(yi_big).numpy())
        sy   = float(tf.math.reduce_std(yi_big).numpy() + 1e-6)
        self.model.set_normalisation(mu_x, sx, mu_y, sy)

        # run dirs (unchanged structurally)
        self.run_dir = run_dir or os.path.join(root_dir, f"{mode}_{now_stamp()}")
        self.paths = {k: os.path.join(self.run_dir, k) for k in ["checkpoints","figs","logs","config"]}
        os.makedirs(self.run_dir, exist_ok=True)
        for p in self.paths.values(): os.makedirs(p, exist_ok=True)
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.paths["checkpoints"], max_to_keep=1)

        # histories
        self.train_loss_hist = []
        self.Lf_hist, self.Lbc_hist = [], []
        self.val_hist, self.relL2_hist, self.val_ep_hist = [], [], []
        self.lam_hist = []

    # ---------- common losses ----------
    @tf.function(reduce_retracing=True)
    def compute_losses(self, xi, yi, xb, yb, q_int, u_true_b):
        f = self.model.pde_residual(xi, yi, q_int)
        L_f = tf.reduce_mean(tf.square(f))
        u_b = self.model.u(xb, yb)
        L_bc = tf.reduce_mean(tf.square(u_b - u_true_b))
        return L_f, L_bc

    # ---------- primal step for PD ----------
    @tf.function(reduce_retracing=True)
    def step_primal_pd(self, xi, yi, xb, yb, q_int, u_true_b):
        with tf.GradientTape() as tape:
            L_f, L_bc = self.compute_losses(xi, yi, xb, yb, q_int, u_true_b)
            total = self.l_f * L_f + L_bc
        grads = tape.gradient(total, self.model.net.weights + self.model.net.biases)
        self.opt.apply_gradients(zip(grads, self.model.net.weights + self.model.net.biases))
        return L_f, L_bc, total

    # ---------- dual step for PD (Adam on λ_f, every dual_period steps) ----------
    @tf.function(reduce_retracing=True)
    def step_dual_pd(self, xi, yi, xb, yb, q_int, u_true_b):
        with tf.GradientTape() as tape:
            tape.watch(self.l_f)
            L_f, L_bc = self.compute_losses(xi, yi, xb, yb, q_int, u_true_b)
            total = self.l_f * L_f + L_bc
            loss_dual = -total
        grad_l = tape.gradient(loss_dual, self.l_f)
        self.opt_dual.apply_gradients([(grad_l, self.l_f)])
        return loss_dual

    # ---------- PI version (unchanged except for normalisation) ----------
    @tf.function(reduce_retracing=True)
    def _pi_update(self, e):
        self.win_f.assign(tf.concat([self.win_f[1:], tf.reshape(e, (1,))], axis=0))
        I = tf.reduce_sum( self.win_f)
        lam = tf.clip_by_value(self.pi_Kp * e + self.pi_Ki * I, self.lam_min, self.lam_max)
        return lam

    @tf.function(reduce_retracing=True)
    def step_pi(self, xi, yi, xb, yb, q_int, u_true_b):

        with tf.GradientTape() as tape:
            L_f2, L_bc2 = self.compute_losses(xi, yi, xb, yb, q_int, u_true_b)
            total = self.l_f * L_f2 + L_bc2
        grads = tape.gradient(total, self.model.net.weights + self.model.net.biases)
        self.opt.apply_gradients(zip(grads, self.model.net.weights + self.model.net.biases))

        def do_update():
            lam = self._pi_update(L_f2)
            self.l_f.assign(lam)
            return 0
        tf.cond(tf.equal(tf.math.floormod(self.global_epoch, 10), 0), do_update, lambda: 0)

        self.global_epoch.assign_add(1)
        return L_f2, L_bc2, total


    # ---------- training loop ----------
    def train(self, k, xv, yv):
        # pre-training error map
        Xg, Yg = np.meshgrid(xv, yv, indexing='xy')
        U_true = u_exact_xy(Xg, Yg).astype(np.float32)
        U_before = predict_on_grid(self.model, xv, yv)
        err_before = U_before - U_true

        best_val = np.inf; best_info = {}

        for ep in range(1, self.epochs+1):
            if ep == 1 or ep % 1 == 0:
                xi, yi = sample_interior(self.Nf)
                xb, yb = sample_boundary(self.Nb)
                q_int = q_tensor(xi, yi, k)
                u_true_b = u_exact_xy(xb.numpy(), yb.numpy()).astype(np.float32)
                u_true_b = tf.constant(u_true_b)

            if self.mode == "pi":
                L_f, L_bc, total = self.step_pi(xi, yi, xb, yb, q_int, u_true_b)
            elif self.mode == "primal_dual":
                L_f, L_bc, total = self.step_primal_pd(xi, yi, xb, yb, q_int, u_true_b)
                if ep % self.dual_period == 0:
                    self.step_dual_pd(xi, yi, xb, yb, q_int, u_true_b)
            elif self.mode == "primal":
                L_f, L_bc, total = self.step_primal_pd(xi, yi, xb, yb, q_int, u_true_b)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            self.train_loss_hist.append(float(total.numpy()))
            self.Lf_hist.append(float(L_f.numpy()))
            self.Lbc_hist.append(float(L_bc.numpy()))
            self.lam_hist.append(float(self.l_f.numpy()))

            if ep % self.val_interval == 0 or ep == 1:
                val_mse, rel_l2, _ = validate(self.model, k)
                self.val_hist.append(val_mse)
                self.relL2_hist.append(rel_l2)
                self.val_ep_hist.append(ep)
                improved = val_mse < best_val
                if improved:
                    best_val = val_mse
                    best_info = dict(epoch=ep, val_mse=val_mse, rel_l2=rel_l2)
                    self.manager.save(checkpoint_number=ep)
                print(f"[{self.mode}] ep {ep:5d} | Lf {self.Lf_hist[-1]:.3e} | Lbc {self.Lbc_hist[-1]:.3e} | "
                      f"λ_f {self.l_f.numpy():.3e} | val {val_mse:.3e} | relL2 {rel_l2:.3e}"
                      f"{'  <-- best' if improved else ''}")

        np.savez(os.path.join(self.paths["checkpoints"], "best_summary.npz"), **best_info)

        # post-training error map
        U_after = predict_on_grid(self.model, xv, yv)
        err_after = U_after - U_true

        # --------- Visual 1: error maps ---------
        v = np.max(np.abs(np.concatenate([err_before.ravel(), err_after.ravel()])))
        fig, ax = plt.subplots(1,2, figsize=(12,4), constrained_layout=True, sharey=True)
        im0 = ax[0].imshow(err_before, origin='lower',
                           extent=[xv[0], xv[-1], yv[0], yv[-1]], aspect='auto',cmap='jet')
        ax[0].set_title("Error BEFORE (u_pred - u_true)"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(err_after, origin='lower',
                           extent=[xv[0], xv[-1], yv[0], yv[-1]], aspect='auto',cmap='jet')
        ax[1].set_title("Error AFTER"); ax[1].set_xlabel("x")
        plt.colorbar(im1, ax=ax[1])
        plt.savefig(os.path.join(self.paths["figs"], "error_maps_before_after.png"), dpi=150)

        # --------- Visual 2: training & validation ---------
        fig2, ax2 = plt.subplots(1,2, figsize=(12,4), constrained_layout=True)
        ax2[0].plot(self.train_loss_hist, label="total")
        ax2[0].plot(self.Lf_hist, label="Lf (PDE)")
        ax2[0].plot(self.Lbc_hist, label="Lbc (BC)")
        ax2[0].set_yscale("log"); ax2[0].grid(True, alpha=0.3)
        ax2[0].set_title("Training losses"); ax2[0].set_xlabel("Epoch"); ax2[0].legend()

        ax2[1].plot(self.val_ep_hist, self.val_hist, marker="o", label="Val total")
        ax2[1].plot(self.val_ep_hist, self.relL2_hist, marker="s", label="Rel L2 (grid)")
        ax2[1].set_yscale("log"); ax2[1].grid(True, alpha=0.3)
        ax2[1].set_title("Validation"); ax2[1].set_xlabel("Epoch"); ax2[1].legend()
        plt.savefig(os.path.join(self.paths["figs"], "training_validation_curves.png"), dpi=150)

        # --------- Visual 3: λ_f history ---------
        fig3, ax3 = plt.subplots(1,1, figsize=(7,4), constrained_layout=True)
        ax3.plot(self.lam_hist)
        ax3.set_title("PDE lambda (λ_f) vs epochs"); ax3.set_xlabel("Epoch"); ax3.set_ylabel("λ_f")
        ax3.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.paths["figs"], "lambda_curve.png"), dpi=150)
        plt.show()

# -----------------------------
# Main
# -----------------------------
def main():
    k = 1.0
    mode = "primal_dual"   # "pi" or "primal_dual" or "primal"

    model = Helmholtz2DPINN(k=k, layers=(2,50, 50, 20,1))
    trainer = Trainer(model, mode=mode,
                      lr_primal=1e-3, lr_dual=1e-3,
                      epochs=15000, Nf=5000, Nb=4000, val_interval=250,
                      dual_period=10,
                      root_dir="runs",
                      pi_Kp=0.0, pi_Ki=0.001, forgetting_rate=5e-1, lambda_clamp=(0.0,1e3))

    xv, yv, _, _ = make_grid(Nx=100, Ny=100)

    print(f"Saving outputs to: {trainer.run_dir}")
    trainer.train(k, xv, yv)

if __name__ == "__main__":
    main()

