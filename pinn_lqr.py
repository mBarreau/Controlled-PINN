import tensorflow as tf
import numpy as np
from tqdm import tqdm
from scipy.linalg import solve_continuous_are

from utils import NeuralNetwork, gradient, hessian
from pinn import PINN

class PINN_LQR(PINN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   # reuse PINN init

    @tf.function
    def dual_update(self):
        """
        LQR-based dual update.
        Replace the PI update in PINN with an LQR-inspired update law.
        """
        # Integral action
        new_value = tf.reshape(tf.convert_to_tensor(self.get_mse_residual()), (1,))
        self.integral.assign(tf.concat([self.integral[1:], new_value], axis=0))
        integral = self.delta_t_dual * tf.reduce_sum(
            # self.forgetting_factor * self.integral
            self.integral
        )

        # Linearization: Compute gradients and Hessian
        A = -hessian(self.get_mse_data, self.x_hat.trainable_variables)
        B = -gradient(self.get_mse_residual, self.x_hat.trainable_variables)
        C = tf.transpose(
            gradient(self.get_mse_residual, self.x_hat.trainable_variables)
        )

        eigenvalues, eigenvectors = tf.linalg.eigh(A)
        explained_variance = tf.abs(eigenvalues) / tf.reduce_sum(tf.abs(eigenvalues))
        k = 2
        Q = eigenvectors[:, -k:]
        A_r = tf.transpose(Q) @ A @ Q
        B_r = tf.transpose(Q) @ B
        C_r = C @ Q

        # Check controllability and observability
        if (tf.abs(tf.linalg.det(tf.concat([B_r, A_r @ B_r], axis=1))) <= 0.01 or
                tf.abs(tf.linalg.det(tf.concat([C_r, C_r @ A_r], axis=0))) <= 0.01):

            # self.weight.assign(self.Kp * self.get_mse_residual() + self.Ki * integral)
            return self.Kp

        # ---- LQR design ----
        # Loss = y^T Qy y + u^T R u
        Qy = tf.eye(1, dtype=self.x_hat.dtype)   # tune this
        R = tf.eye(1, dtype=self.x_hat.dtype)   * 1      # scalar input weight

        # effective state weighting in reduced coordinates
        # Shift back to loss = x_r^T (C_r^T Q_y C_r) x_r + u^T R u
        Q_eff =  tf.transpose(C_r) @ Qy @ C_r
        # Q_eff = Qy

        # Discrete ZOH
        dt = tf.cast(self.delta_t_dual, A_r.dtype)  # your dual step size
        Ad, Bd = zoh_discretize(A_r, B_r, dt)

        # keep Cr unchanged (output matrix does not change under ZOH)
        Cd = C_r

        # Augmented Matrix LQI for integral action
        zero_k1 = tf.zeros((k, 1), dtype=Ad.dtype)
        one = tf.ones((1, 1), dtype=Ad.dtype)

        A_aug = tf.concat([
            tf.concat([Ad, zero_k1], axis=1),
            tf.concat([-self.delta_t_dual * C_r, one], axis=1)
        ], axis=0)  # (k+1) x (k+1)
        B_aug = tf.concat([Bd, tf.zeros((1, 1), dtype=Ad.dtype)], axis=0)  # (k+1) x 1
        C_aug = tf.concat([C_r, tf.zeros((1, 1), dtype=Ad.dtype)], axis=1)  # 1 x (k+1)

        Q_aug = tf.concat([
            tf.concat([Q_eff, tf.zeros((k, 1), dtype=self.x_hat.dtype)], axis=1),
            tf.concat([tf.zeros((1, k), dtype=self.x_hat.dtype),  tf.ones((1, 1), dtype=self.x_hat.dtype)], axis=1)
        ], axis=0)

        # solve DARE: A_r^T P + P A_r - P B_r R^-1 B_r^T P + Q_eff = 0
        P, K, flag = dare_iter(A_aug, B_aug, Q_aug, R)
        # K_p = -tf.squeeze(K[:, :k] @ tf.transpose(C_r) / (C_r @ tf.transpose(C_r))/ 100)
        # Compute Moore-Penrose pseudoinverse
        K_p = -tf.squeeze(K[:, :k] @ tf.linalg.pinv(C_r, rcond=1e-8) )
        Ki = -tf.squeeze(K[:, k:k+1])
        
        # feedback gain with smoothing
        alpha = 0.9
        self.Kp.assign(alpha * self.Kp + (1 - alpha) * K_p)
        self.Ki.assign(alpha * self.Ki + (1 - alpha) * Ki)
        # ---- update rule for weight ----
        residual = self.get_mse_residual()
        self.weight.assign(self.Kp * residual + self.Ki * integral)   # schematic: you might refine
        return self.Kp


@tf.function
def dare_iter(Ad, Bd, Qd, Rd, tol=1e-5, max_iter=2000):
    """Solve discrete ARE:  ."""
    P = tf.identity(Qd)
    def body(k, P, converged):
        BtP = tf.transpose(Bd) @ P
        S = Rd + BtP @ Bd
        K = tf.linalg.solve(S, BtP @ Ad)  # (m x n)
        P_next = (tf.transpose(Ad) @ P @ Ad
                  - tf.transpose(Ad) @ P @ Bd @ K
                  + Qd)
        diff = tf.norm(P_next - P) / (1.0 + tf.norm(P))
        return k+1, P_next, tf.logical_or(converged, diff < tol)

    def cond(k, P, converged):
        return tf.logical_and(tf.less(k, max_iter), tf.logical_not(converged))

    k0 = tf.constant(0)
    converged0 = tf.constant(False)
    _, P_star, flag = tf.while_loop(cond, body, [k0, P, converged0])
    # Return both P and the corresponding K
    BtP = tf.transpose(Bd) @ P_star
    S = Rd + BtP @ Bd
    K = tf.linalg.solve(S, BtP @ Ad)  # (m x n)
    return P_star, K, flag  # K is the discrete-time LQR gain

@tf.function
def zoh_discretize(A, B, dt):
    """
    Discretize continuous-time (A,B) using zero-order hold.
    x_{k+1} = Ad x_k + Bd u_k
    """
    n = tf.shape(A)[0]
    m = tf.shape(B)[1]
    Zmn = tf.zeros((m, n), dtype=A.dtype)
    Zmm = tf.zeros((m, m), dtype=A.dtype)

    # build block matrix
    M = tf.concat(
        [tf.concat([A, B], axis=1),
         tf.concat([Zmn, Zmm], axis=1)],
        axis=0
    ) * dt

    # matrix exponential
    E = tf.linalg.expm(M)

    # extract Ad, Bd
    Ad = E[:n, :n]
    Bd = E[:n, n:]
    return Ad, Bd