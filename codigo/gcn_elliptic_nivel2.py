#!/usr/bin/env python3
"""Experimento Nivel 2 sobre Elliptic: óptimo global convexo (L_net=1) vs. GCNN
multiescala ego-separada (L_net=2, softplus_beta) con L-BFGS y Multi-start.

Implementa exactamente la arquitectura polinomial de la Definición 2.2.7 y
Proposición 2.2.1 del Capítulo 2 de ``PNL_Indice_Integrado.tex``:
  1) L_net = 1, K_1 = 2 (d_0 = 165, d_1 = 1 => p = 3 * 165 = 495):
     modelo FIR multiescala fuertemente convexo (con lambda > 0), donde L-BFGS
     alcanza el único minimizador global exacto independientemente de theta_0.
  2) L_net = 2, K_1 = 1, K_2 = 1 (d_0 = 165, d_1 = 32, d_2 = 1 => p = 10 624):
     GCNN ego-separada (X W_0 + S_t X W_1) con activación C^infty
     softplus_beta(z) = (1/beta) log(1 + exp(beta z)), evaluada en corrida base
     y con Multi-start (R = 5 semillas, tope 300 iteraciones) + ensamblado.

Estado de revisión: review_pending.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from gcn_elliptic_lbfgs import (  # noqa: E402
    EllipticDataset,
    _sigmoid_stable,
    build_normalized_shift_operator,
    choose_f1_threshold,
    chronological_split,
    evaluate_binary_metrics,
    load_elliptic_dataset,
    run_lbfgs_andrei,
)


def softplus_and_deriv(z: np.ndarray, beta: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Evalúa de forma estable softplus_beta(z) = (1/beta)*log(1 + exp(beta*z)) y su derivada."""
    bz = beta * z
    val = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(bz))) / beta
    deriv = _sigmoid_stable(bz)
    return val, deriv


def act_and_deriv(z: np.ndarray, kind: str, beta: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    if kind == "softplus":
        return softplus_and_deriv(z, beta=beta)
    if kind == "tanh":
        h = np.tanh(z)
        return h, 1.0 - h * h
    raise ValueError(f"Activación no soportada: {kind!r}")


class ConvexFIR1LayerObjective:
    """Caso L_net = 1, K_1 = 2 de la Definición 2.2.7 (p = 3 * d_0 = 495)."""

    def __init__(
        self,
        x_tilde_tr: np.ndarray,
        y_tr: np.ndarray,
        omega1: float = 1.0,
        omega0: float = 1.0,
        l2_reg: float = 5e-3,
    ) -> None:
        self.x_tr = np.ascontiguousarray(x_tilde_tr, dtype=np.float64)
        self.y_tr = np.ascontiguousarray(y_tr, dtype=np.float64)
        self.m_train, self.n_params = self.x_tr.shape
        self.omega1 = float(omega1)
        self.omega0 = float(omega0)
        self.l2_reg = float(l2_reg)
        self.sample_w = np.where(self.y_tr == 1.0, self.omega1, self.omega0)
        self.eval_count = 0

    def loss_and_grad(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        self.eval_count += 1
        z = self.x_tr @ theta
        # BCE numéricamente estable: y*log(1+e^{-z}) + (1-y)*log(1+e^z)
        pos_loss = np.maximum(-z, 0.0) + np.log1p(np.exp(-np.abs(z)))
        neg_loss = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
        data_loss = float(
            np.mean(self.omega1 * self.y_tr * pos_loss + self.omega0 * (1.0 - self.y_tr) * neg_loss)
        )
        reg_loss = 0.5 * self.l2_reg * float(np.dot(theta, theta))

        p = _sigmoid_stable(z)
        dl_dz = (self.sample_w * (p - self.y_tr)) / self.m_train
        grad = self.x_tr.T @ dl_dz + self.l2_reg * theta
        return data_loss + reg_loss, grad


class EgoSeparatedGCNN2LayerObjective:
    """Caso L_net = 2, K_1 = 1, K_2 = 1 de la Definición 2.2.7 (sin sesgos, p = 10 624):
    H^(1) = sigma_1(X W_0^(1) + S X W_1^(1)),
    z     = H^(1) W_0^(2) + S H^(1) W_1^(2).
    """

    def __init__(
        self,
        x_full: np.ndarray,
        s_train: sparse.csr_matrix,
        train_mask: np.ndarray,
        y_full: np.ndarray,
        hidden_dim: int = 32,
        activation: str = "softplus",
        beta: float = 2.0,
        omega1: float = 1.0,
        omega0: float = 1.0,
        l2_reg: float = 5e-3,
    ) -> None:
        self.d0 = int(x_full.shape[1])
        self.d1 = int(hidden_dim)
        self.activation = activation
        self.beta = float(beta)
        self.omega1 = float(omega1)
        self.omega0 = float(omega0)
        self.l2_reg = float(l2_reg)

        tr_idx = np.flatnonzero(train_mask)
        self.m_train = int(tr_idx.size)
        self.y_tr = np.ascontiguousarray(y_full[tr_idx], dtype=np.float64)
        self.sample_w = np.where(self.y_tr == 1.0, self.omega1, self.omega0)

        # Vecindad activa U = I_tr U vecinos de 1 salto en S_train
        s_rows = s_train[tr_idx, :].tocsc()
        has_col = np.diff(s_rows.indptr) > 0
        has_col[tr_idx] = True
        u_idx = np.flatnonzero(has_col)
        self.u_idx = u_idx
        self.tr_pos_in_u = np.searchsorted(u_idx, tr_idx)
        assert np.array_equal(u_idx[self.tr_pos_in_u], tr_idx)

        sx_full = s_train @ x_full
        self.x_u = np.ascontiguousarray(x_full[u_idx], dtype=np.float64)
        self.sx_u = np.ascontiguousarray(sx_full[u_idx], dtype=np.float64)
        self.s_tr_u = s_rows[:, u_idx].tocsr()
        self.s_tr_u_t = self.s_tr_u.T.tocsr()

        # Conteo exacto Proposición 2.2.1: 2*d0*d1 + 2*d1*1
        self.n_w1 = self.d0 * self.d1
        self.n_params = 2 * self.n_w1 + 2 * self.d1
        self.eval_count = 0

    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        i1 = self.n_w1
        i2 = 2 * self.n_w1
        i3 = i2 + self.d1
        w0_1 = theta[:i1].reshape(self.d0, self.d1)
        w1_1 = theta[i1:i2].reshape(self.d0, self.d1)
        w0_2 = theta[i2:i3]
        w1_2 = theta[i3:]
        return w0_1, w1_1, w0_2, w1_2

    def init_theta(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        b1 = math.sqrt(6.0 / (2 * self.d0 + self.d1))
        b2 = math.sqrt(6.0 / (2 * self.d1 + 1))
        w0_1 = rng.uniform(-b1, b1, size=(self.d0, self.d1))
        w1_1 = rng.uniform(-b1, b1, size=(self.d0, self.d1))
        w0_2 = rng.uniform(-b2, b2, size=(self.d1,))
        w1_2 = rng.uniform(-b2, b2, size=(self.d1,))
        return np.concatenate([w0_1.ravel(), w1_1.ravel(), w0_2, w1_2])

    def loss_and_grad(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        self.eval_count += 1
        w0_1, w1_1, w0_2, w1_2 = self.unpack(theta)

        z1_u = self.x_u @ w0_1 + self.sx_u @ w1_1
        h1_u, d_sig_u = act_and_deriv(z1_u, self.activation, self.beta)

        h1_tr = h1_u[self.tr_pos_in_u]
        sh1_tr = self.s_tr_u @ h1_u
        logits = h1_tr @ w0_2 + sh1_tr @ w1_2

        pos_loss = np.maximum(-logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
        neg_loss = np.maximum(logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
        data_loss = float(
            np.mean(self.omega1 * self.y_tr * pos_loss + self.omega0 * (1.0 - self.y_tr) * neg_loss)
        )
        reg_loss = 0.5 * self.l2_reg * float(np.dot(theta, theta))

        p = _sigmoid_stable(logits)
        dl_dz = (self.sample_w * (p - self.y_tr)) / self.m_train  # (m_train,)

        grad_w0_2 = h1_tr.T @ dl_dz
        grad_w1_2 = sh1_tr.T @ dl_dz

        # Adjunto hacia H^(1)_U
        dh1_u = (self.s_tr_u_t @ dl_dz)[:, None] * w1_2[None, :]
        dh1_u[self.tr_pos_in_u] += dl_dz[:, None] * w0_2[None, :]
        dz1_u = dh1_u * d_sig_u

        grad_w0_1 = self.x_u.T @ dz1_u
        grad_w1_1 = self.sx_u.T @ dz1_u

        grad = np.concatenate([grad_w0_1.ravel(), grad_w1_1.ravel(), grad_w0_2, grad_w1_2])
        if self.l2_reg > 0.0:
            grad = grad + self.l2_reg * theta
        return data_loss + reg_loss, grad


def check_gradient_fd(loss_grad_fn, theta: np.ndarray, n_coords: int = 24, eps: float = 1e-6) -> float:
    _, grad = loss_grad_fn(theta)
    rng = np.random.default_rng(12345)
    idxs = rng.choice(theta.size, size=min(n_coords, theta.size), replace=False)
    max_rel = 0.0
    for idx in idxs:
        tp = theta.copy()
        tm = theta.copy()
        tp[idx] += eps
        tm[idx] -= eps
        fp, _ = loss_grad_fn(tp)
        fm, _ = loss_grad_fn(tm)
        fd = (fp - fm) / (2.0 * eps)
        rel = abs(fd - grad[idx]) / max(1e-12, abs(fd), abs(grad[idx]))
        max_rel = max(max_rel, float(rel))
    return max_rel


def predict_convex_temporal(
    ds: EllipticDataset,
    mask: np.ndarray,
    ops: Dict[int, sparse.csr_matrix],
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    d0 = ds.n_features
    w0 = theta[:d0]
    w1 = theta[d0 : 2 * d0]
    w2 = theta[2 * d0 :]
    ys, ps = [], []
    for step in sorted(np.unique(ds.time_steps[mask]).tolist()):
        idx = np.flatnonzero(mask & (ds.time_steps == step))
        s_op = ops[step]
        sx = s_op @ ds.x
        s2x_sub = s_op[idx, :] @ sx
        logits = ds.x[idx] @ w0 + sx[idx] @ w1 + s2x_sub @ w2
        ys.append(ds.y[idx])
        ps.append(_sigmoid_stable(logits))
    return np.concatenate(ys), np.concatenate(ps)


def build_ego_eval_cache(
    ds: EllipticDataset,
    mask: np.ndarray,
    ops: Dict[int, sparse.csr_matrix],
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, sparse.csr_matrix, np.ndarray]]:
    cache = {}
    for step in sorted(np.unique(ds.time_steps[mask]).tolist()):
        idx = np.flatnonzero(mask & (ds.time_steps == step))
        s_op = ops[step]
        s_rows = s_op[idx, :].tocsc()
        has_col = np.diff(s_rows.indptr) > 0
        has_col[idx] = True
        u_idx = np.flatnonzero(has_col)
        pos_in_u = np.searchsorted(u_idx, idx)
        sx_full = s_op @ ds.x
        x_u = np.ascontiguousarray(ds.x[u_idx], dtype=np.float64)
        sx_u = np.ascontiguousarray(sx_full[u_idx], dtype=np.float64)
        s_sub = s_rows[:, u_idx].tocsr()
        cache[step] = (x_u, sx_u, pos_in_u, s_sub, ds.y[idx])
    return cache


def predict_ego_gcnn_temporal(
    obj: EgoSeparatedGCNN2LayerObjective,
    theta: np.ndarray,
    cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, sparse.csr_matrix, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    w0_1, w1_1, w0_2, w1_2 = obj.unpack(theta)
    ys, ps = [], []
    for step in sorted(cache.keys()):
        x_u, sx_u, pos_in_u, s_sub, y_st = cache[step]
        h1_u, _ = act_and_deriv(x_u @ w0_1 + sx_u @ w1_1, obj.activation, obj.beta)
        logits = h1_u[pos_in_u] @ w0_2 + (s_sub @ h1_u) @ w1_2
        ys.append(y_st)
        ps.append(_sigmoid_stable(logits))
    return np.concatenate(ys), np.concatenate(ps)


def main() -> None:
    base_dir = CODE_DIR.parent
    zip_path = base_dir / "Fundamentos_Matematicos_GNN.zip"
    out_json = base_dir / "resultados" / "resultados_elliptic_nivel2.json"

    ds = load_elliptic_dataset(zip_path=zip_path)
    tr_mask, val_mask, te_mask = chronological_split(ds.time_steps, ds.y, 34, 39)
    tr_idx = np.flatnonzero(tr_mask)

    n_pos = int(np.sum(ds.y[tr_mask] == 1.0))
    n_neg = int(np.sum(ds.y[tr_mask] == 0.0))
    omega1_bal = n_neg / n_pos
    l2_reg = 5e-3
    max_iter = 300
    tol_grad = 1e-4
    tol_rel = 1e-5

    print("[INFO] Construyendo operadores temporales sin fuga...")
    s_train = build_normalized_shift_operator(ds.edge_src, ds.edge_dst, ds.time_steps, ds.n_nodes, 34)
    needed_steps = sorted(set(np.unique(ds.time_steps[val_mask]).tolist() + np.unique(ds.time_steps[te_mask]).tolist()))
    ops = {st: build_normalized_shift_operator(ds.edge_src, ds.edge_dst, ds.time_steps, ds.n_nodes, st) for st in needed_steps}

    # =========================================================================
    # PARTE A: Modelo convexo L_net = 1, K_1 = 2 (óptimo global exacto, p = 495)
    # =========================================================================
    print("\n=== PARTE A: Modelo FIR Convexo L_net=1, K_1=2 (p = 495) ===")
    sx_34 = s_train @ ds.x
    s2x_34_tr = s_train[tr_idx, :] @ sx_34
    x_tilde_tr = np.hstack([ds.x[tr_idx], sx_34[tr_idx], s2x_34_tr])
    assert x_tilde_tr.shape == (tr_idx.size, 3 * ds.n_features)

    # Constante Lipschitz global exacta de \nabla F_{\omega,\lambda}
    s_max = float(svds(x_tilde_tr, k=1, return_singular_vectors=False)[0])
    L_global_bal = (omega1_bal / (4.0 * tr_idx.size)) * (s_max ** 2) + l2_reg
    print(f"[INFO] ||X_tilde_tr||_2 = {s_max:.4f} => L_global = {L_global_bal:.4f}, kappa = L/lambda = {L_global_bal/l2_reg:.1f}")

    obj_conv = ConvexFIR1LayerObjective(x_tilde_tr, ds.y[tr_idx], omega1=omega1_bal, omega0=1.0, l2_reg=l2_reg)
    rng0 = np.random.default_rng(42)
    theta0_conv_rand = rng0.normal(0.0, 0.05, size=(obj_conv.n_params,))
    theta0_conv_zero = np.zeros(obj_conv.n_params, dtype=np.float64)

    fd_err_conv = check_gradient_fd(obj_conv.loss_and_grad, theta0_conv_rand)
    print(f"[CHECK] Error rel. máx. gradiente (L_net=1 convexo): {fd_err_conv:.3e}")

    res_conv_rand = run_lbfgs_andrei(
        obj_conv, theta0_conv_rand, "conv_l1_k2_cost_l2_rand",
        memory=10, max_iter=max_iter, tol_grad=tol_grad, tol_rel_step=tol_rel,
    )
    res_conv_zero = run_lbfgs_andrei(
        obj_conv, theta0_conv_zero, "conv_l1_k2_cost_l2_zero",
        memory=10, max_iter=max_iter, tol_grad=tol_grad, tol_rel_step=tol_rel,
    )
    dist_opt = float(np.linalg.norm(res_conv_rand.theta - res_conv_zero.theta))
    diff_f_opt = abs(float(res_conv_rand.history["loss"][-1] - res_conv_zero.history["loss"][-1]))
    print(
        f"[UNICIDAD GLOBAL] Semilla aleatoria vs theta_0=0: ||theta_A - theta_B||_2 = {dist_opt:.3e}, "
        f"|F_A - F_B| = {diff_f_opt:.3e}"
    )

    yv_c, pv_c = predict_convex_temporal(ds, val_mask, ops, res_conv_rand.theta)
    thr_c = choose_f1_threshold(yv_c, pv_c)
    val_m_conv = evaluate_binary_metrics(yv_c, pv_c, thr_c)
    yt_c, pt_c = predict_convex_temporal(ds, te_mask, ops, res_conv_rand.theta)
    test_m_conv = evaluate_binary_metrics(yt_c, pt_c, thr_c)
    print(
        f"[CONVEXO L_net=1] it={res_conv_rand.iterations}, t={res_conv_rand.elapsed_sec:.2f}s, "
        f"F={res_conv_rand.history['loss'][-1]:.6f}, ||g||={res_conv_rand.history['grad_norm'][-1]:.3e}, "
        f"Val PR-AUC={val_m_conv['pr_auc']:.4f}, Test PR-AUC={test_m_conv['pr_auc']:.4f}, Test F1={test_m_conv['f1']:.4f}"
    )

    # =========================================================================
    # PARTE B: GCNN ego-separada L_net=2, K_1=K_2=1 (p = 10 624)
    # =========================================================================
    print("\n=== PARTE B: GCNN Ego-Separada L_net=2, K_1=K_2=1 (p = 10 624) ===")
    val_cache = build_ego_eval_cache(ds, val_mask, ops)
    te_cache = build_ego_eval_cache(ds, te_mask, ops)

    obj_sp = EgoSeparatedGCNN2LayerObjective(
        ds.x, s_train, tr_mask, ds.y, hidden_dim=32, activation="softplus", beta=2.0,
        omega1=omega1_bal, omega0=1.0, l2_reg=l2_reg,
    )
    obj_th = EgoSeparatedGCNN2LayerObjective(
        ds.x, s_train, tr_mask, ds.y, hidden_dim=32, activation="tanh", beta=2.0,
        omega1=omega1_bal, omega0=1.0, l2_reg=l2_reg,
    )

    th0_42 = obj_sp.init_theta(42)
    fd_err_sp = check_gradient_fd(obj_sp.loss_and_grad, th0_42)
    fd_err_th = check_gradient_fd(obj_th.loss_and_grad, th0_42)
    print(f"[CHECK] Error rel. máx. gradiente (L_net=2 softplus_2): {fd_err_sp:.3e}")
    print(f"[CHECK] Error rel. máx. gradiente (L_net=2 tanh):       {fd_err_th:.3e}")

    # Comparativa con tanh en la misma arquitectura ego-separada (semilla 42)
    res_ego_th = run_lbfgs_andrei(
        obj_th, th0_42, "ego_gcnn_tanh_s42",
        memory=10, max_iter=max_iter, tol_grad=tol_grad, tol_rel_step=tol_rel,
    )
    yv_th, pv_th = predict_ego_gcnn_temporal(obj_th, res_ego_th.theta, val_cache)
    thr_th = choose_f1_threshold(yv_th, pv_th)
    val_m_th = evaluate_binary_metrics(yv_th, pv_th, thr_th)
    yt_th, pt_th = predict_ego_gcnn_temporal(obj_th, res_ego_th.theta, te_cache)
    test_m_th = evaluate_binary_metrics(yt_th, pt_th, thr_th)
    print(
        f"[EGO tanh s42] it={res_ego_th.iterations}, t={res_ego_th.elapsed_sec:.2f}s, "
        f"F={res_ego_th.history['loss'][-1]:.6f}, ||g||={res_ego_th.history['grad_norm'][-1]:.3e}, "
        f"Val PR-AUC={val_m_th['pr_auc']:.4f}, Test PR-AUC={test_m_th['pr_auc']:.4f}, Test F1={test_m_th['f1']:.4f}"
    )

    # Multi-start R = 5 semillas con softplus_2 (tope 300 iteraciones)
    seeds = [42, 43, 44, 45, 46]
    ms_runs: List[Dict] = []
    val_probs_list: List[np.ndarray] = []
    test_probs_list: List[np.ndarray] = []
    thetas_ms: List[np.ndarray] = []
    yv_ref = None
    yt_ref = None

    for sd in seeds:
        th0 = obj_sp.init_theta(sd)
        res = run_lbfgs_andrei(
            obj_sp, th0, f"ego_gcnn_softplus2_s{sd}",
            memory=10, max_iter=max_iter, tol_grad=tol_grad, tol_rel_step=tol_rel,
        )
        yv, pv = predict_ego_gcnn_temporal(obj_sp, res.theta, val_cache)
        thr = choose_f1_threshold(yv, pv)
        vm = evaluate_binary_metrics(yv, pv, thr)
        yt, pt = predict_ego_gcnn_temporal(obj_sp, res.theta, te_cache)
        tm = evaluate_binary_metrics(yt, pt, thr)
        yv_ref, yt_ref = yv, yt
        val_probs_list.append(pv)
        test_probs_list.append(pt)
        thetas_ms.append(res.theta)

        cpsi_arr = np.asarray(res.history["cos_psi"][1:], dtype=float)
        run_info = {
            "seed": sd,
            "iterations": res.iterations,
            "total_evals": res.total_evals,
            "elapsed_sec": res.elapsed_sec,
            "stop_reason": res.stop_reason,
            "f_init": float(res.history["loss"][0]),
            "f_final": float(res.history["loss"][-1]),
            "grad_norm_final": float(res.history["grad_norm"][-1]),
            "cos_psi_min": float(np.min(cpsi_arr)) if cpsi_arr.size else 1.0,
            "zoutendijk_sum_final": float(res.history["zoutendijk_sum"][-1]),
            "val_metrics": vm,
            "test_metrics": tm,
        }
        ms_runs.append(run_info)
        print(
            f"[EGO softplus2 s{sd}] it={res.iterations}, evals={res.total_evals}, t={res.elapsed_sec:.2f}s, "
            f"F={run_info['f_final']:.6f}, ||g||={run_info['grad_norm_final']:.3e}, "
            f"Val PR-AUC={vm['pr_auc']:.4f}, Test PR-AUC={tm['pr_auc']:.4f}, Test F1={tm['f1']:.4f}"
        )

    # Distancias entre los distintos puntos estacionarios alcanzados en Multi-start
    pairwise_dists = [
        float(np.linalg.norm(thetas_ms[i] - thetas_ms[j]))
        for i in range(len(thetas_ms))
        for j in range(i + 1, len(thetas_ms))
    ]

    # Selección de mejor cuenca por validación y ensamblado promedio (Deep Ensemble R=5)
    best_idx = int(np.argmax([r["val_metrics"]["pr_auc"] for r in ms_runs]))
    best_single = ms_runs[best_idx]

    pv_ens = np.mean(np.vstack(val_probs_list), axis=0)
    pt_ens = np.mean(np.vstack(test_probs_list), axis=0)
    assert yv_ref is not None and yt_ref is not None
    thr_ens = choose_f1_threshold(yv_ref, pv_ens)
    val_m_ens = evaluate_binary_metrics(yv_ref, pv_ens, thr_ens)
    test_m_ens = evaluate_binary_metrics(yt_ref, pt_ens, thr_ens)

    print(
        f"\n[MULTI-START RESUMEN R=5] F_fin en [{min(r['f_final'] for r in ms_runs):.6f}, "
        f"{max(r['f_final'] for r in ms_runs):.6f}], "
        f"distancias ||theta^(i)-theta^(j)||_2 en [{min(pairwise_dists):.3f}, {max(pairwise_dists):.3f}]"
    )
    print(
        f"[MEJOR CUENCA VAL (s{best_single['seed']})] Val PR-AUC={best_single['val_metrics']['pr_auc']:.4f}, "
        f"Test PR-AUC={best_single['test_metrics']['pr_auc']:.4f}, Test F1={best_single['test_metrics']['f1']:.4f}"
    )
    print(
        f"[ENSAMBLE R=5] Val PR-AUC={val_m_ens['pr_auc']:.4f}, "
        f"Test PR-AUC={test_m_ens['pr_auc']:.4f}, Test F1={test_m_ens['f1']:.4f}, "
        f"Prec={test_m_ens['precision']:.4f}, Rec={test_m_ens['recall']:.4f}"
    )

    summary = {
        "hyperparameters": {
            "memory_M": 10,
            "max_iter": max_iter,
            "tol_grad": tol_grad,
            "tol_rel_step": tol_rel,
            "l2_reg": l2_reg,
            "omega1": omega1_bal,
            "omega0": 1.0,
            "softplus_beta": 2.0,
        },
        "gradient_checks_max_rel_err": {
            "convex_l1_k2": fd_err_conv,
            "ego_gcnn_softplus2": fd_err_sp,
            "ego_gcnn_tanh": fd_err_th,
        },
        "convex_l1_k2": {
            "n_params": obj_conv.n_params,
            "spectral_norm_x_tilde": s_max,
            "L_global": L_global_bal,
            "condition_upper_bound": L_global_bal / l2_reg,
            "uniqueness_theta_dist": dist_opt,
            "uniqueness_f_diff": diff_f_opt,
            "run_seed42": {
                "iterations": res_conv_rand.iterations,
                "total_evals": res_conv_rand.total_evals,
                "elapsed_sec": res_conv_rand.elapsed_sec,
                "stop_reason": res_conv_rand.stop_reason,
                "f_init": float(res_conv_rand.history["loss"][0]),
                "f_final": float(res_conv_rand.history["loss"][-1]),
                "grad_norm_final": float(res_conv_rand.history["grad_norm"][-1]),
                "cos_psi_min": float(np.min(res_conv_rand.history["cos_psi"][1:])),
                "val_metrics": val_m_conv,
                "test_metrics": test_m_conv,
            },
            "run_zero": {
                "iterations": res_conv_zero.iterations,
                "total_evals": res_conv_zero.total_evals,
                "elapsed_sec": res_conv_zero.elapsed_sec,
                "f_final": float(res_conv_zero.history["loss"][-1]),
                "grad_norm_final": float(res_conv_zero.history["grad_norm"][-1]),
            },
        },
        "ego_gcnn_tanh_seed42": {
            "n_params": obj_th.n_params,
            "iterations": res_ego_th.iterations,
            "total_evals": res_ego_th.total_evals,
            "elapsed_sec": res_ego_th.elapsed_sec,
            "stop_reason": res_ego_th.stop_reason,
            "f_init": float(res_ego_th.history["loss"][0]),
            "f_final": float(res_ego_th.history["loss"][-1]),
            "grad_norm_final": float(res_ego_th.history["grad_norm"][-1]),
            "cos_psi_min": float(np.min(res_ego_th.history["cos_psi"][1:])),
            "val_metrics": val_m_th,
            "test_metrics": test_m_th,
        },
        "ego_gcnn_softplus2_multistart": {
            "n_params": obj_sp.n_params,
            "seeds": seeds,
            "runs": ms_runs,
            "pairwise_theta_l2_min": min(pairwise_dists),
            "pairwise_theta_l2_max": max(pairwise_dists),
            "best_val_seed": best_single["seed"],
            "ensemble_val_metrics": val_m_ens,
            "ensemble_test_metrics": test_m_ens,
        },
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Resultados guardados en {out_json}")


if __name__ == "__main__":
    main()
