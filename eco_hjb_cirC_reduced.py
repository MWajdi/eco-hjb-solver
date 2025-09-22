#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-objective Optuna tuning for the 4-D HJB solver (Policy-Iteration-Aux).
Extended for CIR process where C_t follows dC_t = κ(β - C_t)dt + δ√C_t dB_t
(Simplified version with reduced hyperparameters)
"""
# ---------------------------------------------------------------------------

import os, gc, random, json, pickle, argparse, datetime, logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from tqdm import trange, tqdm

from DGM import DGMNet
from hammersley import tf_hammersley_sampler

logging.getLogger("tensorflow").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.INFO)

# ---------------------------------------------------------------------------
#  TensorFlow basic setup
# ---------------------------------------------------------------------------
tf.keras.backend.clear_session()
gc.collect()
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
tf.config.run_functions_eagerly(False)  # graph mode

# ---------------------------------------------------------------------------
#  Problem-specific primitives
# ---------------------------------------------------------------------------
def pi(P):  # running payoff
    return tf.sqrt(P)


# ---------------------------------------------------------------------------
#  P I A   trainer (4D version) - Simplified
# ---------------------------------------------------------------------------
class PIATrainer4D:
    def __init__(self, value_model, control_model, optimizer_f, optimizer_g,
                 domain_bounds, batch_size=2816, candidate_size=50000,
                 resample_every=1000, dtype="float32"):

        # accept str or tf.DType
        if isinstance(dtype, str):
            self.dtype = tf.float32 if dtype == "float32" else tf.float64
        else:
            self.dtype = tf.as_dtype(dtype)

        self.f_theta = value_model
        self.g_phi = control_model
        self.optimizer_f = optimizer_f
        self.optimizer_g = optimizer_g

        # domain_bounds now: [(t_min, t_max), (P_min, P_max), (Y_min, Y_max), (C_min, C_max)]
        self.domain_bounds = domain_bounds  
        self.batch_size = batch_size
        self.candidate_size = candidate_size
        self.resample_every = resample_every

        # Fixed k schedule
        self.k_start = 1.0
        self.k_end = 4.0
        self.k_schedule_steps = 5000

        self.T = domain_bounds[0][1]
        self.terminal_bounds = domain_bounds[1:]  # P, Y, C bounds for terminal condition
        self.candidates = tf_hammersley_sampler(self.candidate_size, self.domain_bounds, dtype=self.dtype)

        # Initialize normalization parameters
        self._setup_normalization()

        self.history = {
            "total_loss": [],
            "pde_loss": [],
            "terminal_loss": [],
            "VPP_penalty": [],
            "grad_norm_f": [],
            "control_loss": [],
            "mean_integrand": [],
            "grad_norm_g": []
        }

        self.resamples = 0
        self.best_total_loss = float('inf')
        self.best_snapshot_path = None  # (f_weights_path, g_weights_path)
        self.snapshot_dir = os.path.join("experiments", "snapshots",
                                         datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.snapshot_dir, exist_ok=True)

    # -----------------------------
    # NEW: small helpers
    # -----------------------------
    def _build_models(self):
        """Run a dummy forward pass to create variables before loading weights."""
        bs = 4
        t = tf.zeros((bs, 1), dtype=self.dtype)
        xyc = tf.zeros((bs, 3), dtype=self.dtype)  # P, Y, C
        _ = self.f_theta(t, xyc)
        _ = self.g_phi(t, xyc)

    def _safe_numpy_probs(self, p_tf):
        """Convert TF weights -> valid numpy prob vector."""
        p_tf = tf.where(tf.math.is_finite(p_tf), p_tf, tf.zeros_like(p_tf))  # drop NaN/Inf
        p_tf = tf.nn.relu(p_tf)                                              # clamp negatives
        s = tf.reduce_sum(p_tf)
        if s <= 0:
            N = int(p_tf.shape[0])
            return np.full(N, 1.0 / max(N, 1), dtype=np.float64)
        p_tf = p_tf / s
        p = p_tf.numpy().astype(np.float64).ravel()
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        s = p.sum()
        if s <= 0:
            p[:] = 1.0 / len(p)
        else:
            p /= s
        return p

    def _try_load_best_snapshot(self):
        """Reload best-known weights (weights-only snapshots)."""
        if self.best_snapshot_path:
            f_path, g_path = self.best_snapshot_path
            if os.path.exists(f_path) and os.path.exists(g_path):
                # Build variables first, then load weights
                self._build_models()
                self.f_theta.load_weights(f_path)
                self.g_phi.load_weights(g_path)
                return True
        return False
    # -----------------------------

    def _setup_normalization(self):
        t_bounds, P_bounds, Y_bounds, C_bounds = self.domain_bounds
        self.t_min, self.t_max = t_bounds
        self.P_min, self.P_max = P_bounds
        self.Y_min, self.Y_max = Y_bounds
        self.C_min, self.C_max = C_bounds

        self.t_mean = tf.constant((self.t_max + self.t_min) / 2, dtype=self.dtype)
        self.t_scale = tf.constant((self.t_max - self.t_min) / 2, dtype=self.dtype)
        self.P_mean = tf.constant((self.P_max + self.P_min) / 2, dtype=self.dtype)
        self.P_scale = tf.constant((self.P_max - self.P_min) / 2, dtype=self.dtype)
        self.Y_mean = tf.constant((self.Y_max + self.Y_min) / 2, dtype=self.dtype)
        self.Y_scale = tf.constant((self.Y_max - self.Y_min) / 2, dtype=self.dtype)
        self.C_mean = tf.constant((self.C_max + self.C_min) / 2, dtype=self.dtype)
        self.C_scale = tf.constant((self.C_max - self.C_min) / 2, dtype=self.dtype)

    def normalize_inputs(self, t, P, Y, C):
        t = tf.cast(t, self.dtype)
        P = tf.cast(P, self.dtype)
        Y = tf.cast(Y, self.dtype)
        C = tf.cast(C, self.dtype)
        t_norm = (t - self.t_mean) / self.t_scale
        P_norm = (P - self.P_mean) / self.P_scale
        Y_norm = (Y - self.Y_mean) / self.Y_scale
        C_norm = (C - self.C_mean) / self.C_scale
        return t_norm, P_norm, Y_norm, C_norm

    def get_V_and_alpha(self, t, P, Y, C):
        t_norm, P_norm, Y_norm, C_norm = self.normalize_inputs(t, P, Y, C)
        V = self.f_theta(t_norm, tf.concat([P_norm, Y_norm, C_norm], axis=1))
        alpha = self.g_phi(t_norm, tf.concat([P_norm, Y_norm, C_norm], axis=1))
        return V, alpha

    def get_scheduled_k(self, step):
        progress = min(step / self.k_schedule_steps, 1.0)
        return self.k_start + progress * (self.k_end - self.k_start)

    @tf.function
    def compute_value_loss(self, t, P, Y, C, t_term, P_term, Y_term, C_term):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([t, P, Y, C])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([t, P, Y, C])
                t_norm, P_norm, Y_norm, C_norm = self.normalize_inputs(t, P, Y, C)
                alpha = self.g_phi(t_norm, tf.concat([P_norm, Y_norm, C_norm], axis=1))
                V = self.f_theta(t_norm, tf.concat([P_norm, Y_norm, C_norm], axis=1))
            V_t = tape1.gradient(V, t)
            V_P = tape1.gradient(V, P)
            V_Y = tape1.gradient(V, Y)
            V_C = tape1.gradient(V, C)
            del tape1
        V_PP = tape2.gradient(V_P, P)
        V_CC = tape2.gradient(V_C, C)
        V_PC = tape2.gradient(V_P, C)  # Mixed derivative ∂²V/∂P∂C
        del tape2

        # Safe second derivatives
        V_PP_safe = self._safe_pp(V_PP, eps=1e-5)
        V_CC_safe = self._safe_cc(V_CC, eps=1e-5)

        # Convert to original units (no scaling needed now)
        alpha = tf.cast(alpha, self.dtype)

        # HJB equation terms (4D version with CIR dynamics)
        # Main drift and diffusion from P dynamics
        drift_P = mu * alpha * P * V_P
        diffusion_P = (sigma**2 * alpha**2 * P**2 * V_PP_safe) / tf.constant(2.0, dtype=self.dtype)
        
        # Mixed term from correlation between P and C processes
        mixed_PC = rho * sigma * delta * alpha * P * tf.sqrt(tf.maximum(C, tf.constant(1e-8, dtype=self.dtype))) * V_PC
        
        # Y dynamics (integral of C(a + bP))
        advec_Y = C * (a + b * P) * V_Y
        
        # CIR dynamics for C
        drift_C = kappa * (beta - C) * V_C
        diffusion_C = (delta**2 * C * V_CC_safe) / tf.constant(2.0, dtype=self.dtype)
        
        # Source term
        source_term = pi(P)

        V = tf.cast(V, self.dtype)
        
        # Complete HJB residual (primal form)
        residual = V_t + drift_P + diffusion_P + mixed_PC + advec_Y + drift_C + diffusion_C + source_term - r * V
        pde_loss = tf.reduce_mean(tf.square(residual))

        # Terminal condition
        t_term_norm, P_term_norm, Y_term_norm, C_term_norm = self.normalize_inputs(t_term, P_term, Y_term, C_term)
        V_terminal = self.f_theta(t_term_norm, tf.concat([P_term_norm, Y_term_norm, C_term_norm], axis=1))

        # Terminal penalty (no scaling needed)
        zero = tf.constant(0.0, dtype=self.dtype)
        penalty = - tf.maximum(zero, Y_term - L)

        V_terminal = tf.cast(V_terminal, self.dtype)
        terminal_errors = V_terminal - penalty
        terminal_loss = tf.reduce_mean(tf.square(terminal_errors))

        # Concavity penalty (V_PP should be negative)
        V_PP_pos = tf.nn.relu(V_PP)
        penalty_coeff = tf.constant(1e5, dtype=self.dtype)
        VPP_penalty = penalty_coeff * tf.reduce_mean(tf.square(V_PP_pos))
        VPP_penalty_vec = penalty_coeff * tf.square(V_PP_pos)

        return (pde_loss + terminal_loss + VPP_penalty,
                pde_loss, terminal_loss, VPP_penalty,
                residual, terminal_errors, VPP_penalty_vec)
    
    # inside class PIATrainer4D
    def _safe_pp(self, pp, eps=None):
        # ensure second derivative has magnitude at least eps and never equals 0
        if eps is None:
            eps = tf.cast(1e-5, self.dtype)
        else:
            eps = tf.cast(eps, self.dtype)
        sign = tf.where(pp >= 0, tf.ones_like(pp), -tf.ones_like(pp))  # +1 for >=0, -1 otherwise
        return tf.where(tf.abs(pp) < eps, eps * sign, pp)
    
    def _safe_cc(self, cc, eps=None):
        # ensure second derivative w.r.t. C has magnitude at least eps
        if eps is None:
            eps = tf.cast(1e-5, self.dtype)
        else:
            eps = tf.cast(eps, self.dtype)
        sign = tf.where(cc >= 0, tf.ones_like(cc), -tf.ones_like(cc))
        return tf.where(tf.abs(cc) < eps, eps * sign, cc)

    @tf.function
    def compute_control_loss(self, t, P, Y, C):
        # Compute V and its derivatives
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([t, P, Y, C])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([t, P, Y, C])
                t_norm, P_norm, Y_norm, C_norm = self.normalize_inputs(t, P, Y, C)
                V = self.f_theta(t_norm, tf.concat([P_norm, Y_norm, C_norm], axis=1))
            V_P = tape1.gradient(V, P)
            del tape1
        V_PP = tape2.gradient(V_P, P)
        V_PC = tape2.gradient(V_P, C)  # Mixed derivative
        del tape2

        # Make V_PP strictly away from zero
        V_PP_safe = self._safe_pp(V_PP, eps=1e-5)

        # Predicted control (unbounded)
        t_norm, P_norm, Y_norm, C_norm = self.normalize_inputs(t, P, Y, C)
        alpha = self.g_phi(t_norm, tf.concat([P_norm, Y_norm, C_norm], axis=1))

        # No scaling needed for derivatives
        V_P_orig  = V_P
        V_PP_orig = V_PP_safe
        V_PC_orig = V_PC
        alpha_orig = alpha

        # Optimal control with CIR correlation term
        # α* = -(μ V_P + ρσδ√C V_PC) / (σ²P V_PP)
        sqrt_C_safe = tf.sqrt(tf.maximum(C, tf.constant(1e-8, dtype=self.dtype)))
        numerator = mu * V_P_orig + rho * sigma * delta * sqrt_C_safe * V_PC_orig
        denominator = sigma**2 * P * V_PP_orig
        
        alpha_optimal_orig = -numerator / denominator
        alpha_optimal = tf.stop_gradient(alpha_optimal_orig)

        # Robust control loss (Huber)
        delta_huber = tf.cast(getattr(self, "huber_delta", 5.0), self.dtype)
        huber = tf.keras.losses.Huber(delta=delta_huber, reduction=tf.keras.losses.Reduction.NONE)
        loss_vec = huber(alpha_optimal, alpha)   # y_true, y_pred
        loss = tf.reduce_mean(loss_vec)

        # Hamiltonian (for monitoring)
        H = (mu * alpha_orig * P * V_P_orig + 
             0.5 * sigma**2 * alpha_orig**2 * P**2 * V_PP_orig +
             rho * sigma * delta * alpha_orig * P * sqrt_C_safe * V_PC_orig)
        mean_integrand = tf.reduce_mean(H)

        return loss, mean_integrand

    # -----------------------------
    # Robust resampling (NaN-safe)
    # -----------------------------
    def resample_from_residual(self, step, batch_size=4096):
        t_cand, P_cand, Y_cand, C_cand = tf.split(self.candidates, 4, axis=1)  # 4D split
        self.k = tf.constant(self.get_scheduled_k(step), dtype=self.dtype)

        t_term_cand, P_term_cand, Y_term_cand, C_term_cand = self.sample_terminal(self.candidate_size)

        num_points = self.candidate_size
        num_batches = int(np.ceil(num_points / batch_size))

        residuals, terminal_errors, VPP_penalties = [], [], []
        eps = tf.constant(1e-12, dtype=self.dtype)

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_points)

            t_b = t_cand[start:end]; P_b = P_cand[start:end]; Y_b = Y_cand[start:end]; C_b = C_cand[start:end]
            t_term_b = t_term_cand[start:end]; P_term_b = P_term_cand[start:end]
            Y_term_b = Y_term_cand[start:end]; C_term_b = C_term_cand[start:end]

            _, _, _, _, residual_b, terminal_error_b, VPP_penalty_vec_b = self.compute_value_loss(
                t_b, P_b, Y_b, C_b, t_term_b, P_term_b, Y_term_b, C_term_b
            )

            residuals.append(tf.abs(residual_b))
            terminal_errors.append(tf.abs(terminal_error_b))
            VPP_penalties.append(tf.abs(VPP_penalty_vec_b))

        residuals = tf.concat(residuals, axis=0)
        terminal_errors = tf.concat(terminal_errors, axis=0)
        VPP_penalties = tf.concat(VPP_penalties, axis=0)

        max_pde_residual      = tf.reduce_max(residuals)
        max_terminal_residual = tf.reduce_max(terminal_errors)
        max_VPP_penalty       = tf.reduce_max(VPP_penalties)
        max_total_loss        = tf.reduce_max(residuals + terminal_errors + VPP_penalties)

        residual_abs = residuals + terminal_errors + VPP_penalties + eps
        residual_k   = tf.pow(residual_abs, self.k)
        residual_k   = tf.where(tf.math.is_finite(residual_k), residual_k, tf.zeros_like(residual_k))

        p = self._safe_numpy_probs(residual_k)

        n_residuals = residuals.shape[0]
        replace_flag = self.batch_size > n_residuals

        try:
            indices = np.random.choice(n_residuals, size=self.batch_size, p=p, replace=replace_flag)
        except ValueError:
            indices = np.random.choice(n_residuals, size=self.batch_size, replace=replace_flag)

        selected = tf.gather(self.candidates, indices)

        return (tf.split(selected, 4, axis=1),  # 4D split
                float(max_pde_residual.numpy()),
                float(max_terminal_residual.numpy()),
                float(max_VPP_penalty.numpy()),
                float(max_total_loss.numpy()))

    def sample_terminal(self, n):
        points = tf_hammersley_sampler(n, self.terminal_bounds, dtype=self.dtype)  # Sample P, Y, C
        P, Y, C = tf.split(points, 3, axis=1)
        t = tf.ones_like(P, dtype=self.dtype) * self.T
        return t, P, Y, C

    # -----------------------------
    # Train with rollback
    # -----------------------------
    def train(self, steps=5000, terminal_batch_size=512, log_every=10):
        for step in trange(1, steps + 1, desc="Training"):
            if step % self.resample_every == 1:
                try:
                    (t_int, P_int, Y_int, C_int), max_pde_residual, max_terminal_residual, max_VPP_penalty, max_total_loss = \
                        self.resample_from_residual(step)
                except Exception as e:
                    tqdm.write(f"[{step:05d}] ERROR during resample: {e}")
                    if self._try_load_best_snapshot():
                        tqdm.write(f"[{step:05d}] Rolled back to best snapshot and continuing...")
                        continue
                    else:
                        raise

                tqdm.write(
                    f"[{step:05d}] Max PDE residual: {max_pde_residual:.2e}, "
                    f"Max terminal residual: {max_terminal_residual:.2e}, "
                    f"Max pos V_PP penalty: {max_VPP_penalty:.2e}, "
                    f"Max total loss: {max_total_loss:.2e}"
                )

                self.resamples += 1

                if max_total_loss < self.best_total_loss:
                    snapshot_name = f"step_{step:05d}"
                    f_path = os.path.join(self.snapshot_dir, f"{snapshot_name}_f_theta.weights.h5")
                    g_path = os.path.join(self.snapshot_dir, f"{snapshot_name}_g_phi.weights.h5")
                    # ensure variables are built before saving
                    self._build_models()
                    self.f_theta.save_weights(f_path)
                    self.g_phi.save_weights(g_path)
                    self.best_total_loss = max_total_loss
                    self.best_snapshot_path = (f_path, g_path)
                    tqdm.write(f"[{step:05d}] New best snapshot saved (weights) (max total loss = {max_total_loss:.2e})")

                if max_total_loss < 1e-2:
                    tqdm.write(f"[{step:05d}] Early stopping: max total loss < 1e-2 ({max_total_loss:.2e})")
                    break

            t_term, P_term, Y_term, C_term = self.sample_terminal(terminal_batch_size)

            # --- Update Value Network ---
            with tf.GradientTape() as tape_f:
                total_loss, pde_loss, term_loss, VPP_penalty, _, _, _ = self.compute_value_loss(
                    t_int, P_int, Y_int, C_int, t_term, P_term, Y_term, C_term
                )

            if tf.reduce_any(tf.math.is_nan(total_loss)) or tf.reduce_any(tf.math.is_inf(total_loss)):
                tqdm.write(f"[{step:05d}] WARNING: NaN/Inf detected in loss!")
                if self._try_load_best_snapshot():
                    tqdm.write(f"[{step:05d}] Rolled back to best snapshot and continuing...")
                    continue
                else:
                    raise RuntimeError("Loss became NaN/Inf and no snapshot to rollback to.")

            grads_f = tape_f.gradient(total_loss, self.f_theta.trainable_variables)
            grads_f_vars = [(g, v) for g, v in zip(grads_f, self.f_theta.trainable_variables) if g is not None]
            grads_f, vars_f = zip(*grads_f_vars)
            clipped_grads_f, _ = tf.clip_by_global_norm(grads_f, 1e3)
            self.optimizer_f.apply_gradients(zip(clipped_grads_f, vars_f))
            grad_norm_f = tf.linalg.global_norm(clipped_grads_f)

            # --- Update Control Network ---
            with tf.GradientTape() as tape_g:
                control_loss, mean_integrand = self.compute_control_loss(t_int, P_int, Y_int, C_int)
            grads_g = tape_g.gradient(control_loss, self.g_phi.trainable_variables)
            grads_g_vars = [(g, v) for g, v in zip(grads_g, self.g_phi.trainable_variables) if g is not None]
            grads_g, vars_g = zip(*grads_g_vars)
            clipped_grads_g, _ = tf.clip_by_global_norm(grads_g, 1e3)
            self.optimizer_g.apply_gradients(zip(clipped_grads_g, vars_g))
            grad_norm_g = tf.linalg.global_norm(clipped_grads_g)

            # --- Logging ---
            self.history["total_loss"].append(total_loss.numpy())
            self.history["pde_loss"].append(pde_loss.numpy())
            self.history["terminal_loss"].append(term_loss.numpy())
            self.history["VPP_penalty"].append(VPP_penalty.numpy())
            self.history["grad_norm_f"].append(grad_norm_f.numpy())
            self.history["control_loss"].append(control_loss.numpy())
            self.history["mean_integrand"].append(mean_integrand.numpy())
            self.history["grad_norm_g"].append(grad_norm_g.numpy())

            if step % log_every == 0:
                tqdm.write(
                    f"[{step:05d}] V Loss: {total_loss:.4e}, PDE: {pde_loss:.4e}, Term: {term_loss:.4e}, VPP Penalty: {VPP_penalty:.4e}, "
                    f"||∇V||: {grad_norm_f:.4e}, α Loss: {control_loss:.4e}, ||∇α||: {grad_norm_g:.4e}, -|H|: {mean_integrand:.4e}"
                )


# Set up logging for Optuna
optuna.logging.set_verbosity(optuna.logging.INFO)


def create_objective_function_4d(PIATrainer4D, DGMNet, domain_bounds, tf_hammersley_sampler,
                                 mu, sigma, a, b, pi, r, L, kappa, beta, delta, rho,
                                 seed=3, training_steps=1000, candidate_size=100_000, dtype=tf.float32):

    def objective(trial):
        # Reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # --- Simplified Hyperparams ---
        # Only optimize base learning rate
        base_lr = trial.suggest_float("lr_base", 1e-5, 1e-3, log=True)
        
        # Fixed parameters
        batch_size = 2816
        resample_every = trial.suggest_int('resample_every', 50, 500, step=50)  # Keep this tunable
        n_layers = trial.suggest_int('n_layers', 3, 13, step=2)  # Keep architecture tunable
        layer_width = trial.suggest_int('layer_width', 64, 256, step=32)  # Keep architecture tunable

        try:
            input_dim = 3  # P, Y, C (4D problem: t + 3 spatial dimensions)

            f_theta = DGMNet(layer_width=layer_width, n_layers=n_layers, input_dim=input_dim, dtype=dtype)
            g_phi   = DGMNet(layer_width=layer_width, n_layers=n_layers, input_dim=input_dim, final_trans=None, dtype=dtype)

            # Fixed decay schedules
            lr_schedule_f = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=base_lr, decay_steps=250, decay_rate=0.9, staircase=True
            )
            lr_schedule_g = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=base_lr, decay_steps=250, decay_rate=0.9, staircase=True
            )

            optimizer_f = tf.keras.optimizers.Adam(learning_rate=lr_schedule_f)
            optimizer_g = tf.keras.optimizers.Adam(learning_rate=lr_schedule_g)

            trainer = PIATrainer4D(
                value_model=f_theta, control_model=g_phi,
                optimizer_f=optimizer_f, optimizer_g=optimizer_g,
                domain_bounds=domain_bounds, batch_size=batch_size,
                candidate_size=candidate_size, resample_every=resample_every,
                dtype=dtype
            )

            trainer.train(steps=training_steps, log_every=10)

            final_pde_loss     = trainer.history["pde_loss"][-1]
            final_terminal_loss= trainer.history["terminal_loss"][-1]
            final_vpp_penalty  = trainer.history["VPP_penalty"][-1]
            final_control_loss = trainer.history["control_loss"][-1]

            del trainer, f_theta, g_phi, optimizer_f, optimizer_g
            tf.keras.backend.clear_session()

            return (float(final_pde_loss),
                    float(final_terminal_loss),
                    float(final_vpp_penalty),
                    float(final_control_loss))

        except Exception as e:
            print(f"Trial failed with error: {e}")
            tf.keras.backend.clear_session()
            return (float('inf'),) * 4

    return objective


def optimize_hyperparameters_4d(PIATrainer4D, DGMNet, domain_bounds, tf_hammersley_sampler,
                                mu, sigma, a, b, pi, r, L, kappa, beta, delta, rho,
                                n_trials=100, seed=3, training_steps=1000,
                                candidate_size=100_000, dtype=tf.float32):

    objective = create_objective_function_4d(
        PIATrainer4D, DGMNet, domain_bounds, tf_hammersley_sampler,
        mu, sigma, a, b, pi, r, L, kappa, beta, delta, rho,
        seed, training_steps=training_steps, candidate_size=candidate_size, dtype=dtype
    )

    study = optuna.create_study(
        directions=['minimize', 'minimize', 'minimize', 'minimize'],
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def pick_lexico_best(study: optuna.study.Study) -> optuna.trial.FrozenTrial:
    THRESHOLD = {  # For testing accept all trials
        "pde": float('inf'), "control": float('inf'), "terminal": float('inf'), "vpp": float('inf'),
    }
    acceptable = [
        t for t in study.best_trials
        if t.values[0] <= THRESHOLD["pde"]
        and t.values[1] <= THRESHOLD["terminal"]
        and t.values[2] <= THRESHOLD["vpp"]
        and t.values[3] <= THRESHOLD["control"]
    ]
    if not acceptable:
        raise RuntimeError("No trial meets all quality thresholds.")
    best = min(acceptable, key=lambda t: t.values)
    print("Best trial found:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    return best


def train_with_best_params_4d(study, PIATrainer4D, DGMNet, domain_bounds, tf_hammersley_sampler,
                              mu, sigma, a, b, pi, r, L, kappa, beta, delta, rho,
                              candidate_size=100_000, final_training_steps=5000,
                              seed=3, dtype=tf.float32):
    # Repro
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)

    best_trial = pick_lexico_best(study)
    best_params = best_trial.params
    print("Best trial parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    input_dim = 3  # P, Y, C
    f_theta = DGMNet(layer_width=best_params['layer_width'], n_layers=best_params['n_layers'],
                     input_dim=input_dim, dtype=dtype)
    g_phi   = DGMNet(layer_width=best_params['layer_width'], n_layers=best_params['n_layers'],
                     input_dim=input_dim, final_trans=None, dtype=dtype)

    base_lr = best_params['lr_base']

    # Fixed decay schedules
    lr_schedule_f = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr, decay_steps=250, decay_rate=0.9, staircase=True
    )
    lr_schedule_g = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr, decay_steps=250, decay_rate=0.9, staircase=True
    )

    optimizer_f = tf.keras.optimizers.Adam(learning_rate=lr_schedule_f)
    optimizer_g = tf.keras.optimizers.Adam(learning_rate=lr_schedule_g)

    trainer = PIATrainer4D(
        value_model=f_theta, control_model=g_phi,
        optimizer_f=optimizer_f, optimizer_g=optimizer_g,
        domain_bounds=domain_bounds,
        batch_size=2816,  # Fixed batch size
        candidate_size=candidate_size,
        resample_every=best_params['resample_every'],
        dtype=dtype,
    )

    try:
        trainer.train(steps=final_training_steps, log_every=100)
    except Exception as e:
        print(f"[final training] ERROR: {e}")
        if trainer._try_load_best_snapshot():
            print("[final training] Rolled back to best snapshot; continuing without further training.")
        else:
            print("[final training] No snapshot available; re-raising to signal failure.")
            raise

    return trainer


# === Pipeline ===
def run_optimization_pipeline_4d(PIATrainer4D, DGMNet, domain_bounds, tf_hammersley_sampler,
                                 mu, sigma, a, b, pi, r, L, kappa, beta, delta, rho,
                                 n_trials=50, final_training_steps=5000, steps_per_trial=1000,
                                 seed=3, candidate_size=100_000, dtype=tf.float32):

    print("Starting 4D hyperparameter optimization...")

    study = optimize_hyperparameters_4d(
        PIATrainer4D, DGMNet, domain_bounds, tf_hammersley_sampler,
        mu, sigma, a, b, pi, r, L, kappa, beta, delta, rho,
        n_trials=n_trials, training_steps=steps_per_trial, seed=seed,
        candidate_size=candidate_size, dtype=dtype
    )

    print("\nOptimization completed!")
    chosen_trial = pick_lexico_best(study)
    print("Chosen trial:", chosen_trial.number, chosen_trial.values)

    print("\nTraining with best parameters...")
    trainer = train_with_best_params_4d(
        study, PIATrainer4D, DGMNet, domain_bounds, tf_hammersley_sampler,
        mu, sigma, a, b, pi, r, L, kappa, beta, delta, rho,
        final_training_steps=final_training_steps, seed=seed,
        candidate_size=candidate_size, dtype=dtype
    )

    # Save study
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_path = f"optuna_study_4d_{timestamp}.pkl"
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"\nOptimization results saved to: {study_path}")
    print(f"Best snapshot weights: {trainer.best_snapshot_path}")

    return study, trainer


# === Visualization (unchanged) ===
def plot_optimization_results(study):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax1)
        ax1.set_title('Optimization History')
        optuna.visualization.matplotlib.plot_param_importances(study, ax=ax2)
        ax2.set_title('Parameter Importances')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Plotting failed: {e}")


# Final export helpers (simplified metadata)
def save_model(model, name, experiment_name, metadata=None, base_dir="experiments", history=None, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    model_path = os.path.join(experiment_dir, f"{name}.keras")
    model.save(model_path)  # full saved model for deployment
    print(f"Model '{name}' saved to: {model_path}")

    if metadata is not None:
        metadata_path = os.path.join(experiment_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}")

    if history is not None:
        history_path = os.path.join(experiment_dir, "training_history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"Training history saved to: {history_path}")


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Run 4D HJB PIA Solver with Optuna tuning (CIR process) - Simplified")
    
    # Original parameters
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--a", type=float, default=0.05)
    parser.add_argument("--b", type=float, default=0.2)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=0.5)
    
    # New CIR parameters
    parser.add_argument("--kappa", type=float, default=2.0, help="CIR mean reversion speed")
    parser.add_argument("--beta", type=float, default=1.5, help="CIR long-term mean")
    parser.add_argument("--delta", type=float, default=0.3, help="CIR volatility parameter")
    parser.add_argument("--rho", type=float, default=-0.5, help="Correlation between P and C processes")
    
    # Domain bounds for C
    parser.add_argument("--C_min", type=float, default=0.1, help="Minimum value for C")
    parser.add_argument("--C_max", type=float, default=5.0, help="Maximum value for C")
    
    # Training parameters
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--steps_per_trial", type=int, default=1000)
    parser.add_argument("--final_training_steps", type=int, default=5000)
    parser.add_argument("--experiment_name", type=str, default="cir_4d_experiment_simplified")
    parser.add_argument("--candidate_size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = tf.float64 if args.dtype == "float64" else tf.float32

    # Globals used inside losses
    global mu, sigma, a, b, r, T, L, kappa, beta, delta, rho, domain_bounds, seed
    mu = tf.constant(args.mu, dtype=dtype)
    sigma = tf.constant(args.sigma, dtype=dtype)
    a = tf.constant(args.a, dtype=dtype)
    b = tf.constant(args.b, dtype=dtype)
    r = tf.constant(args.r, dtype=dtype)
    T = tf.constant(args.T, dtype=dtype)
    L = tf.constant(args.L, dtype=dtype)
    
    # CIR parameters
    kappa = tf.constant(args.kappa, dtype=dtype)
    beta = tf.constant(args.beta, dtype=dtype)
    delta = tf.constant(args.delta, dtype=dtype)
    rho = tf.constant(args.rho, dtype=dtype)

    # 4D domain bounds: [t, P, Y, C]
    Y_max = args.C_max * (a + b * tf.constant(10.0, dtype=dtype)) * T  # upper bound for Y
    domain_bounds = [
        (0.0, T.numpy()),           # t bounds
        (0.1, 10.0),               # P bounds
        (0.0, Y_max.numpy()),      # Y bounds  
        (args.C_min, args.C_max)   # C bounds
    ]

    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)

    study, trainer = run_optimization_pipeline_4d(
        PIATrainer4D, DGMNet, domain_bounds, tf_hammersley_sampler,
        mu, sigma, a, b, pi, r, L, kappa, beta, delta, rho,
        n_trials=args.n_trials,
        final_training_steps=args.final_training_steps,
        steps_per_trial=args.steps_per_trial,
        seed=args.seed,
        candidate_size=args.candidate_size,
        dtype=dtype
    )

    best_trial = pick_lexico_best(study)
    best_params = best_trial.params

    metadata = {
        "model_parameters": {
            "mu": float(mu.numpy()),
            "sigma": float(sigma.numpy()),
            "a": float(a.numpy()),
            "b": float(b.numpy()),
            "r": float(r.numpy()),
            "T": float(T.numpy()),
            "L": float(L.numpy()),
            "kappa": float(kappa.numpy()),
            "beta": float(beta.numpy()),
            "delta": float(delta.numpy()),
            "rho": float(rho.numpy()),
            "pi": "sqrt(P)",
            "model_type": "CIR_4D_Simplified"
        },
        "network_parameters": {
            "n_layers": best_params['n_layers'],
            "layer_width": best_params['layer_width'],
            "input_dim": 3  # P, Y, C (time is separate)
        },
        "training_parameters": {
            "seed": seed,
            "dtype": args.dtype,
            "lr_base": best_params['lr_base'],
            "lr_schedule": {
                "initial_learning_rate": best_params['lr_base'],
                "decay_steps": 250,
                "decay_rate": 0.9
            },
            "batch_size": 2816,
            "candidate_size": args.candidate_size,
            "resample_every": best_params['resample_every'],
            "k_start": 1.0,
            "k_end": 4.0,
            "k_schedule_steps": 5000,
        },
        "domain_bounds": {
            "t": [float(domain_bounds[0][0]), float(domain_bounds[0][1])],
            "P": [float(domain_bounds[1][0]), float(domain_bounds[1][1])],
            "Y": [float(domain_bounds[2][0]), float(domain_bounds[2][1])],
            "C": [float(domain_bounds[3][0]), float(domain_bounds[3][1])]
        },
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(trainer.f_theta, "f_theta", args.experiment_name, metadata=metadata, history=trainer.history, timestamp=timestamp)
    save_model(trainer.g_phi, "g_phi", args.experiment_name, timestamp=timestamp)

if __name__ == "__main__":
    main()