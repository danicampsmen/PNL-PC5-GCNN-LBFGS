#!/usr/bin/env python3
"""Entrenamiento de ExplicitGCN sobre el dataset real Elliptic con L-BFGS.

Implementa de forma explícita el Algoritmo 1.6 (recursión de dos bucles L-BFGS)
de N. Andrei (2020), *Nonlinear Conjugate Gradient Methods for Unconstrained
Optimization*, Springer, pp. 38–39, acoplado con búsqueda lineal de Wolfe fuerte
(que implica las condiciones estándar de Wolfe del Teorema 3.1, p. 98) y el
criterio de parada relativo solicitado en el curso.

Lee directamente los tres CSV originales dentro de
``Fundamentos_Matematicos_GNN.zip`` (sin modificar el archivo original) o desde
un directorio ``--data-dir`` ya extraído.

Autoría técnica: código generado para la PC5 de Programación No Lineal (CM4E2).
Estado de revisión: review_pending.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.optimize import minimize


# =============================================================================
# 1. Carga fiel del dataset Elliptic y operadores normalizados temporales
# =============================================================================


@dataclass
class EllipticDataset:
    x: np.ndarray              # (n, d0) float64
    y: np.ndarray              # (n,) float64 en {1.0, 0.0, -1.0}
    time_steps: np.ndarray     # (n,) int32 en 1..49
    edge_src: np.ndarray       # (E,) int64
    edge_dst: np.ndarray       # (E,) int64
    n_nodes: int
    n_edges: int
    n_features: int


def _open_text_source(
    zip_path: Optional[Path], data_dir: Optional[Path], filename: str
):
    if data_dir is not None:
        fpath = data_dir / filename
        return open(fpath, "r", encoding="utf-8", newline="")
    assert zip_path is not None
    zf = zipfile.ZipFile(zip_path, "r")
    member = f"data/{filename}"
    raw = zf.open(member, "r")
    wrapper = io.TextIOWrapper(raw, encoding="utf-8", newline="")

    class _ManagedText:
        def __enter__(self):
            return wrapper

        def __exit__(self, exc_type, exc, tb):
            wrapper.close()
            raw.close()
            zf.close()

    return _ManagedText()


def load_elliptic_dataset(
    zip_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> EllipticDataset:
    """Carga los 3 CSV originales de Elliptic preservando el orden de clases."""
    t0 = time.perf_counter()

    # 1) Clases (orden canónico de nodos, igual que data_loader.py de Max)
    tx_ids: List[str] = []
    labels_list: List[float] = []
    label_map = {"1": 1.0, "1.0": 1.0, "2": 0.0, "2.0": 0.0, "unknown": -1.0}

    with _open_text_source(zip_path, data_dir, "elliptic_txs_classes.csv") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tid = row["txId"].strip()
            cls = row["class"].strip().lower()
            if cls not in label_map:
                raise ValueError(f"Etiqueta desconocida en Elliptic: {cls!r}")
            tx_ids.append(tid)
            labels_list.append(label_map[cls])

    n_nodes = len(tx_ids)
    id_map: Dict[str, int] = {tid: idx for idx, tid in enumerate(tx_ids)}
    if len(id_map) != n_nodes:
        raise ValueError("Existen txId duplicados en elliptic_txs_classes.csv.")
    y = np.asarray(labels_list, dtype=np.float64)

    # 2) Características y paso temporal (alineados por txId)
    time_steps = np.empty(n_nodes, dtype=np.int32)
    seen = np.zeros(n_nodes, dtype=bool)
    x: Optional[np.ndarray] = None
    n_features = 0

    with _open_text_source(zip_path, data_dir, "elliptic_txs_features.csv") as fh:
        for line in fh:
            p1 = line.find(",")
            p2 = line.find(",", p1 + 1)
            if p1 < 0 or p2 < 0:
                continue
            tid = line[:p1].strip()
            idx = id_map.get(tid)
            if idx is None:
                raise ValueError(f"txId {tid!r} sin clase correspondiente.")
            ts = int(float(line[p1 + 1 : p2]))
            vals = np.fromstring(line[p2 + 1 :], dtype=np.float64, sep=",")
            if x is None:
                n_features = vals.size
                x = np.empty((n_nodes, n_features), dtype=np.float64)
            x[idx] = vals
            time_steps[idx] = ts
            seen[idx] = True

    if x is None or not seen.all():
        missing = int((~seen).sum())
        raise ValueError(f"Faltan características para {missing} nodos.")
    if not np.isfinite(x).all():
        raise ValueError("La matriz de características contiene valores no finitos.")

    # 3) Aristas
    src_list: List[int] = []
    dst_list: List[int] = []
    dropped = 0
    with _open_text_source(zip_path, data_dir, "elliptic_txs_edgelist.csv") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            u = id_map.get(row["txId1"].strip())
            v = id_map.get(row["txId2"].strip())
            if u is None or v is None:
                dropped += 1
                continue
            src_list.append(u)
            dst_list.append(v)

    edge_src = np.asarray(src_list, dtype=np.int64)
    edge_dst = np.asarray(dst_list, dtype=np.int64)
    elapsed = time.perf_counter() - t0
    print(
        f"[INFO] Elliptic cargado en {elapsed:.2f} s: {n_nodes} nodos, "
        f"{edge_src.size} aristas ({dropped} descartadas), {n_features} atributos."
    )
    return EllipticDataset(
        x=x,
        y=y,
        time_steps=time_steps,
        edge_src=edge_src,
        edge_dst=edge_dst,
        n_nodes=n_nodes,
        n_edges=int(edge_src.size),
        n_features=n_features,
    )


def build_normalized_shift_operator(
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    time_steps: np.ndarray,
    num_nodes: int,
    max_step: int,
) -> sparse.csr_matrix:
    """Construye S = D^{-1/2} (A_undir + I) D^{-1/2} usando solo aristas <= max_step.

    Replica exactamente ``compute_normalized_shift_operator`` +
    ``_operator_through_step`` del código original de Max (simetrización sin
    duplicados y lazos propios restantes con peso 1).
    """
    active = time_steps <= max_step
    vis = active[edge_src] & active[edge_dst]
    u = edge_src[vis]
    v = edge_dst[vis]

    diag = np.arange(num_nodes, dtype=np.int64)
    rows = np.concatenate([u, v, diag])
    cols = np.concatenate([v, u, diag])
    data = np.ones(rows.size, dtype=np.float64)

    # Eliminar duplicados (to_undirected + add_remaining_self_loops)
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0

    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.where(deg > 0.0, np.power(deg, -0.5), 0.0)
    d_mat = sparse.diags(inv_sqrt, format="csr")
    return (d_mat @ adj @ d_mat).tocsr()


def chronological_split(
    time_steps: np.ndarray,
    y: np.ndarray,
    train_step_max: int = 34,
    val_step_max: int = 39,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labeled = (y == 0.0) | (y == 1.0)
    train_mask = labeled & (time_steps <= train_step_max)
    val_mask = labeled & (time_steps > train_step_max) & (time_steps <= val_step_max)
    test_mask = labeled & (time_steps > val_step_max)
    return train_mask, val_mask, test_mask


# =============================================================================
# 2. Modelo ExplicitGCN (2 capas) con gradiente analítico exacto en float64
# =============================================================================


def _activation_and_deriv(z: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    if name == "tanh":
        h = np.tanh(z)
        return h, 1.0 - h * h
    if name == "softplus":
        # softplus(z) = log(1 + exp(z)) estable; derivada = sigmoid(z)
        h = np.logaddexp(0.0, z)
        sig = np.where(
            z >= 0.0,
            1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0))),
            np.exp(np.clip(z, -500.0, 500.0)) / (1.0 + np.exp(np.clip(z, -500.0, 500.0))),
        )
        return h, sig
    if name == "relu":
        mask = z > 0.0
        return np.where(mask, z, 0.0), mask.astype(np.float64)
    raise ValueError(f"Activación no soportada: {name!r}")


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -500.0, 500.0)
    return np.where(
        z_clip >= 0.0,
        1.0 / (1.0 + np.exp(-z_clip)),
        np.exp(z_clip) / (1.0 + np.exp(z_clip)),
    )


class ExplicitGCNObjective:
    """Evalúa F(theta) y nabla F(theta) exactamente sobre el subgrafo activo.

    Algebraicamente idéntico a evaluar la GCN de 2 capas sobre los 203 769 nodos:
    primero se calcula X_agg = S_tr @ X sobre el grafo completo (incluyendo
    vecinos no etiquetados) y luego se restringe la segunda capa a las filas
    I_tr y sus columnas activas U (vecindad de 1 salto de I_tr).
    """

    def __init__(
        self,
        x_full: np.ndarray,
        s_train: sparse.csr_matrix,
        y_full: np.ndarray,
        train_mask: np.ndarray,
        hidden_dim: int = 32,
        activation: str = "tanh",
        c1: float = 1.0,
        c0: float = 1.0,
        l2_reg: float = 0.0,
    ) -> None:
        self.d_in = x_full.shape[1]
        self.h = hidden_dim
        self.activation = activation
        self.c1 = float(c1)
        self.c0 = float(c0)
        self.l2_reg = float(l2_reg)

        self.n_params = self.d_in * self.h + self.h + self.h + 1
        train_idx = np.flatnonzero(train_mask)
        self.m_train = int(train_idx.size)
        self.y_tr = y_full[train_idx].astype(np.float64)

        # Producto S_tr @ X sobre todo el grafo (constante en todo el entrenamiento)
        sx_full = s_train @ x_full
        s_tr_rows = s_train[train_idx, :].tocsc()
        active_cols = np.flatnonzero(np.diff(s_tr_rows.indptr) > 0)
        self.n_active_u = int(active_cols.size)
        self.sx_u = np.ascontiguousarray(sx_full[active_cols, :], dtype=np.float64)
        self.s_sub = s_tr_rows[:, active_cols].tocsr()
        self.s_sub_T = self.s_sub.T.tocsr()

        self.eval_count = 0

    def unpack(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        i1 = self.d_in * self.h
        i2 = i1 + self.h
        i3 = i2 + self.h
        w1 = theta[:i1].reshape(self.d_in, self.h)
        b1 = theta[i1:i2]
        w2 = theta[i2:i3]
        b2 = float(theta[i3])
        return w1, b1, w2, b2

    def init_theta(self, seed: int = 42) -> np.ndarray:
        """Inicialización determinista equivalente a model.py de Max."""
        rng = np.random.default_rng(seed)
        bound1 = 1.0 / math.sqrt(self.d_in)
        w1 = rng.uniform(-bound1, bound1, size=(self.d_in, self.h))
        b1 = rng.uniform(-bound1, bound1, size=(self.h,))
        bound2 = math.sqrt(6.0 / (self.h + 1.0))
        w2 = rng.uniform(-bound2, bound2, size=(self.h,))
        b2 = np.zeros(1, dtype=np.float64)
        return np.concatenate([w1.ravel(), b1, w2, b2])

    def loss_and_grad(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        self.eval_count += 1
        w1, b1, w2, b2 = self.unpack(theta)

        # Capa 1 sobre nodos activos U
        z1 = self.sx_u @ w1 + b1
        h1, dh1 = _activation_and_deriv(z1, self.activation)

        # Capa 2 sobre I_tr
        sh1 = self.s_sub @ h1
        logits = sh1 @ w2 + b2
        p = _sigmoid_stable(logits)

        # Pérdida BCE (exacta y numéricamente estable)
        loss_pos = np.logaddexp(0.0, -logits)
        loss_neg = np.logaddexp(0.0, logits)
        data_loss = float(
            np.mean(self.c1 * self.y_tr * loss_pos + self.c0 * (1.0 - self.y_tr) * loss_neg)
        )
        reg_loss = 0.5 * self.l2_reg * float(np.dot(theta, theta)) if self.l2_reg > 0.0 else 0.0
        total_loss = data_loss + reg_loss

        # Retropropagación exacta
        dl_dz = ((self.c0 + (self.c1 - self.c0) * self.y_tr) * p - self.c1 * self.y_tr) / self.m_train
        grad_w2 = sh1.T @ dl_dz
        grad_b2 = np.array([float(np.sum(dl_dz))], dtype=np.float64)

        sT_dl = self.s_sub_T @ dl_dz  # (|U|,)
        g1 = np.outer(sT_dl, w2) * dh1  # (|U|, h)
        grad_w1 = self.sx_u.T @ g1
        grad_b1 = np.sum(g1, axis=0)

        grad = np.concatenate([grad_w1.ravel(), grad_b1, grad_w2, grad_b2])
        if self.l2_reg > 0.0:
            grad = grad + self.l2_reg * theta
        return total_loss, grad

    def predict_logits_precomputed(
        self, theta: np.ndarray, sx_full: np.ndarray, s_rows: sparse.csr_matrix
    ) -> np.ndarray:
        w1, b1, w2, b2 = self.unpack(theta)
        h1, _ = _activation_and_deriv(sx_full @ w1 + b1, self.activation)
        sh1 = s_rows @ h1
        return sh1 @ w2 + b2


def verify_analytical_gradient(
    obj: ExplicitGCNObjective,
    theta: np.ndarray,
    n_checks: int = 12,
    eps: float = 1e-6,
    seed: int = 123,
) -> float:
    """Comprueba el gradiente analítico contra diferencias finitas centrales."""
    _, grad = obj.loss_and_grad(theta)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(theta.size, size=min(n_checks, theta.size), replace=False)
    max_rel_err = 0.0
    for idx in idxs:
        t_plus = theta.copy()
        t_minus = theta.copy()
        t_plus[idx] += eps
        t_minus[idx] -= eps
        f_plus, _ = obj.loss_and_grad(t_plus)
        f_minus, _ = obj.loss_and_grad(t_minus)
        num_g = (f_plus - f_minus) / (2.0 * eps)
        ana_g = grad[idx]
        denom = max(1.0, abs(num_g), abs(ana_g))
        max_rel_err = max(max_rel_err, abs(num_g - ana_g) / denom)
    return max_rel_err


# =============================================================================
# 3. Búsqueda lineal de Wolfe fuerte y Algoritmo 1.6 (L-BFGS de dos bucles)
# =============================================================================


def strong_wolfe_line_search(
    func_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    theta: np.ndarray,
    f0: float,
    g0: np.ndarray,
    d: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha_init: float = 1.0,
    alpha_max: float = 50.0,
    max_bracket_iter: int = 15,
    max_zoom_iter: int = 20,
) -> Tuple[float, float, np.ndarray, bool, int]:
    """Búsqueda lineal de Wolfe fuerte (Nocedal & Wright, Algs. 3.5–3.6).

    Retorna: (alpha, f_new, g_new, wolfe_ok, n_evals).
    """
    phi0 = f0
    dphi0 = float(np.dot(g0, d))
    if dphi0 >= 0.0:
        return 0.0, f0, g0, False, 0

    evals = 0
    best_armijo: Optional[Tuple[float, float, np.ndarray]] = None

    def eval_phi(a: float) -> Tuple[float, float, np.ndarray]:
        nonlocal evals, best_armijo
        f_a, g_a = func_grad(theta + a * d)
        evals += 1
        dphi_a = float(np.dot(g_a, d))
        if np.isfinite(f_a) and f_a <= phi0 + c1 * a * dphi0:
            if best_armijo is None or f_a < best_armijo[1]:
                best_armijo = (a, f_a, g_a)
        return f_a, dphi_a, g_a

    def zoom(
        a_lo: float, f_lo: float, dphi_lo: float, g_lo: np.ndarray,
        a_hi: float, f_hi: float, dphi_hi: float,
    ) -> Tuple[float, float, np.ndarray, bool]:
        for _ in range(max_zoom_iter):
            if abs(a_hi - a_lo) <= 1e-14 * max(1.0, abs(a_lo)):
                break
            # Interpolación cúbica salvaguardada con respaldo a bisección
            a_min, a_max_b = (a_lo, a_hi) if a_lo < a_hi else (a_hi, a_lo)
            margin = 0.1 * (a_max_b - a_min)
            a_j = 0.5 * (a_lo + a_hi)
            denom = a_hi - a_lo
            if abs(denom) > 1e-15 and np.isfinite(f_lo) and np.isfinite(f_hi):
                d1 = dphi_lo + dphi_hi - 3.0 * (f_lo - f_hi) / (a_lo - a_hi)
                disc = d1 * d1 - dphi_lo * dphi_hi
                if disc >= 0.0:
                    d2 = math.copysign(math.sqrt(disc), a_hi - a_lo)
                    q_den = dphi_hi - dphi_lo + 2.0 * d2
                    if abs(q_den) > 1e-15:
                        cand = a_hi - (a_hi - a_lo) * (dphi_hi + d2 - d1) / q_den
                        if a_min + margin <= cand <= a_max_b - margin:
                            a_j = cand

            f_j, dphi_j, g_j = eval_phi(a_j)
            if (not np.isfinite(f_j)) or (f_j > phi0 + c1 * a_j * dphi0) or (f_j >= f_lo):
                a_hi, f_hi, dphi_hi = a_j, f_j, dphi_j
            else:
                if abs(dphi_j) <= -c2 * dphi0:
                    return a_j, f_j, g_j, True
                if dphi_j * (a_hi - a_lo) >= 0.0:
                    a_hi, f_hi, dphi_hi = a_lo, f_lo, dphi_lo
                a_lo, f_lo, dphi_lo, g_lo = a_j, f_j, dphi_j, g_j
        if best_armijo is not None:
            return best_armijo[0], best_armijo[1], best_armijo[2], False
        return a_lo, f_lo, g_lo, False

    a_prev = 0.0
    f_prev = phi0
    dphi_prev = dphi0
    g_prev = g0
    a_curr = alpha_init

    for i in range(1, max_bracket_iter + 1):
        f_curr, dphi_curr, g_curr = eval_phi(a_curr)
        if (not np.isfinite(f_curr)) or (f_curr > phi0 + c1 * a_curr * dphi0) or (i > 1 and f_curr >= f_prev):
            a_star, f_star, g_star, ok = zoom(
                a_prev, f_prev, dphi_prev, g_prev,
                a_curr, f_curr, dphi_curr,
            )
            return a_star, f_star, g_star, ok, evals
        if abs(dphi_curr) <= -c2 * dphi0:
            return a_curr, f_curr, g_curr, True, evals
        if dphi_curr >= 0.0:
            a_star, f_star, g_star, ok = zoom(
                a_curr, f_curr, dphi_curr, g_curr,
                a_prev, f_prev, dphi_prev,
            )
            return a_star, f_star, g_star, ok, evals
        a_prev, f_prev, dphi_prev, g_prev = a_curr, f_curr, dphi_curr, g_curr
        a_curr = min(2.0 * a_curr, alpha_max)
        if a_curr <= a_prev:
            break

    if best_armijo is not None:
        return best_armijo[0], best_armijo[1], best_armijo[2], False, evals
    return a_prev, f_prev, g_prev, False, evals


@dataclass
class OptimizationResult:
    name: str
    theta: np.ndarray
    stop_reason: str
    iterations: int
    total_evals: int
    elapsed_sec: float
    memory_resets: int = 0
    wolfe_fallbacks: int = 0
    history: Dict[str, List[float]] = field(default_factory=dict)


def run_lbfgs_andrei(
    obj: ExplicitGCNObjective,
    theta0: np.ndarray,
    name: str,
    memory: int = 10,
    max_iter: int = 150,
    tol_grad: float = 1e-5,
    tol_rel_step: float = 1e-6,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> OptimizationResult:
    """Algoritmo 1.6 de Andrei (2020) (L-BFGS dos bucles) + Wolfe fuerte.

    Registra además cos(psi_k), la suma parcial de Zoutendijk (Teorema 3.1) y
    el paso relativo ||theta_k - theta_{k-1}|| / max(1, ||theta_k||).
    """
    t_start = time.perf_counter()
    obj.eval_count = 0

    theta = theta0.copy()
    f_val, g_val = obj.loss_and_grad(theta)
    g_norm = float(np.linalg.norm(g_val))

    s_hist: List[np.ndarray] = []
    y_hist: List[np.ndarray] = []
    rho_hist: List[float] = []

    hist: Dict[str, List[float]] = {
        "iter": [0],
        "loss": [f_val],
        "grad_norm": [g_norm],
        "rel_step": [float("nan")],
        "abs_step": [float("nan")],
        "alpha": [float("nan")],
        "cos_psi": [1.0],
        "zoutendijk_sum": [0.0],
        "sy": [float("nan")],
        "time_sec": [0.0],
    }

    zout_sum = 0.0
    memory_resets = 0
    wolfe_fallbacks = 0
    stop_reason = "max_iter"
    rel_step_streak = 0

    for k in range(1, max_iter + 1):
        if g_norm <= tol_grad:
            stop_reason = f"grad_norm <= {tol_grad:.1e}"
            break

        # Recursión de dos bucles (Andrei 2020, Algoritmo 1.6, p. 39)
        q = g_val.copy()
        m_cur = len(s_hist)
        alphas_loop = [0.0] * m_cur
        for i in range(m_cur - 1, -1, -1):
            a_i = rho_hist[i] * float(np.dot(s_hist[i], q))
            alphas_loop[i] = a_i
            q = q - a_i * y_hist[i]

        if m_cur > 0:
            yy = float(np.dot(y_hist[-1], y_hist[-1]))
            sy_last = float(np.dot(s_hist[-1], y_hist[-1]))
            gamma_k = sy_last / yy if yy > 1e-30 else 1.0
            if not np.isfinite(gamma_k) or gamma_k <= 0.0:
                gamma_k = 1.0
        else:
            gamma_k = 1.0 / max(1.0, g_norm)

        r = gamma_k * q
        for i in range(m_cur):
            b_i = rho_hist[i] * float(np.dot(y_hist[i], r))
            r = r + s_hist[i] * (alphas_loop[i] - b_i)

        d = -r
        gd = float(np.dot(g_val, d))
        d_norm = float(np.linalg.norm(d))

        # Salvaguarda de descenso (puede activarse con activaciones no suaves)
        if gd >= -1e-14 * g_norm * max(1.0, d_norm) or d_norm == 0.0:
            s_hist.clear()
            y_hist.clear()
            rho_hist.clear()
            d = -g_val
            gd = -g_norm * g_norm
            d_norm = g_norm
            memory_resets += 1

        cos_psi = float(-gd / (g_norm * d_norm))
        alpha_init = 1.0 if m_cur > 0 else min(1.0, 1.0 / max(1.0, g_norm))

        alpha, f_new, g_new, wolfe_ok, _ = strong_wolfe_line_search(
            obj.loss_and_grad,
            theta,
            f_val,
            g_val,
            d,
            c1=c1,
            c2=c2,
            alpha_init=alpha_init,
        )
        if not wolfe_ok:
            wolfe_fallbacks += 1

        # Si la búsqueda lineal no logra descenso con dirección L-BFGS, reiniciar a -g
        if alpha <= 0.0 or f_new > f_val:
            if m_cur > 0:
                s_hist.clear()
                y_hist.clear()
                rho_hist.clear()
                memory_resets += 1
                d = -g_val
                gd = -g_norm * g_norm
                d_norm = g_norm
                cos_psi = 1.0
                alpha, f_new, g_new, wolfe_ok, _ = strong_wolfe_line_search(
                    obj.loss_and_grad,
                    theta,
                    f_val,
                    g_val,
                    d,
                    c1=c1,
                    c2=c2,
                    alpha_init=min(1.0, 1.0 / max(1.0, g_norm)),
                )
            if alpha <= 0.0:
                stop_reason = "line_search_stalled"
                break

        theta_new = theta + alpha * d
        s_k = theta_new - theta
        y_k = g_new - g_val
        sy = float(np.dot(s_k, y_k))
        s_norm = float(np.linalg.norm(s_k))
        y_norm = float(np.linalg.norm(y_k))

        # Suma parcial de Zoutendijk (Andrei 2020, Teorema 3.1, ec. 3.30)
        zout_sum += (cos_psi ** 2) * (g_norm ** 2)

        # Actualizar memoria L-BFGS si se satisface curvatura positiva s_k^T y_k > 0
        if sy > 1e-12 * s_norm * y_norm:
            if len(s_hist) >= memory:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s_k)
            y_hist.append(y_k)
            rho_hist.append(1.0 / sy)

        rel_step = s_norm / max(1.0, float(np.linalg.norm(theta_new)))
        theta = theta_new
        f_val = f_new
        g_val = g_new
        g_norm = float(np.linalg.norm(g_val))

        hist["iter"].append(k)
        hist["loss"].append(f_val)
        hist["grad_norm"].append(g_norm)
        hist["rel_step"].append(rel_step)
        hist["abs_step"].append(s_norm)
        hist["alpha"].append(alpha)
        hist["cos_psi"].append(cos_psi)
        hist["zoutendijk_sum"].append(zout_sum)
        hist["sy"].append(sy)
        hist["time_sec"].append(time.perf_counter() - t_start)

        if k == 1 or k % 25 == 0:
            print(
                f"  [{name}] Iter {k:03d} | F={f_val:.6f} | ||g||={g_norm:.3e} | "
                f"rel_step={rel_step:.3e} | cos_psi={cos_psi:.3e} | alpha={alpha:.3g}"
            )

        if g_norm <= tol_grad:
            stop_reason = f"grad_norm <= {tol_grad:.1e}"
            break
        if rel_step <= tol_rel_step:
            rel_step_streak += 1
            if rel_step_streak >= 2:
                stop_reason = f"rel_step <= {tol_rel_step:.1e} (2 pasos consecutivos)"
                break
        else:
            rel_step_streak = 0

    elapsed = time.perf_counter() - t_start
    return OptimizationResult(
        name=name,
        theta=theta,
        stop_reason=stop_reason,
        iterations=int(hist["iter"][-1]),
        total_evals=obj.eval_count,
        elapsed_sec=elapsed,
        memory_resets=memory_resets,
        wolfe_fallbacks=wolfe_fallbacks,
        history=hist,
    )


def run_steepest_descent_wolfe(
    obj: ExplicitGCNObjective,
    theta0: np.ndarray,
    name: str,
    max_iter: int = 150,
    tol_grad: float = 1e-5,
    tol_rel_step: float = 1e-6,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> OptimizationResult:
    """Línea base: método del gradiente con búsqueda lineal de Wolfe fuerte."""
    t_start = time.perf_counter()
    obj.eval_count = 0

    theta = theta0.copy()
    f_val, g_val = obj.loss_and_grad(theta)
    g_norm = float(np.linalg.norm(g_val))
    zout_sum = 0.0
    stop_reason = "max_iter"
    rel_step_streak = 0
    alpha_guess = 1.0

    hist: Dict[str, List[float]] = {
        "iter": [0],
        "loss": [f_val],
        "grad_norm": [g_norm],
        "rel_step": [float("nan")],
        "abs_step": [float("nan")],
        "alpha": [float("nan")],
        "cos_psi": [1.0],
        "zoutendijk_sum": [0.0],
        "sy": [float("nan")],
        "time_sec": [0.0],
    }

    for k in range(1, max_iter + 1):
        if g_norm <= tol_grad:
            stop_reason = f"grad_norm <= {tol_grad:.1e}"
            break
        d = -g_val
        alpha, f_new, g_new, _, _ = strong_wolfe_line_search(
            obj.loss_and_grad,
            theta,
            f_val,
            g_val,
            d,
            c1=c1,
            c2=c2,
            alpha_init=alpha_guess,
        )
        if alpha <= 0.0:
            stop_reason = "line_search_stalled"
            break
        theta_new = theta + alpha * d
        s_k = theta_new - theta
        y_k = g_new - g_val
        s_norm = float(np.linalg.norm(s_k))
        sy = float(np.dot(s_k, y_k))
        zout_sum += g_norm ** 2
        rel_step = s_norm / max(1.0, float(np.linalg.norm(theta_new)))

        # Estimación Barzilai-Borwein solo como semilla inicial de Wolfe
        yy = float(np.dot(y_k, y_k))
        alpha_guess = min(50.0, max(1e-3, sy / yy)) if sy > 1e-14 and yy > 1e-20 else 1.0

        theta = theta_new
        f_val = f_new
        g_val = g_new
        g_norm = float(np.linalg.norm(g_val))

        hist["iter"].append(k)
        hist["loss"].append(f_val)
        hist["grad_norm"].append(g_norm)
        hist["rel_step"].append(rel_step)
        hist["abs_step"].append(s_norm)
        hist["alpha"].append(alpha)
        hist["cos_psi"].append(1.0)
        hist["zoutendijk_sum"].append(zout_sum)
        hist["sy"].append(sy)
        hist["time_sec"].append(time.perf_counter() - t_start)

        if g_norm <= tol_grad:
            stop_reason = f"grad_norm <= {tol_grad:.1e}"
            break
        if rel_step <= tol_rel_step:
            rel_step_streak += 1
            if rel_step_streak >= 2:
                stop_reason = f"rel_step <= {tol_rel_step:.1e} (2 pasos consecutivos)"
                break
        else:
            rel_step_streak = 0

    return OptimizationResult(
        name=name,
        theta=theta,
        stop_reason=stop_reason,
        iterations=int(hist["iter"][-1]),
        total_evals=obj.eval_count,
        elapsed_sec=time.perf_counter() - t_start,
        history=hist,
    )


# =============================================================================
# 4. Evaluación temporal sin fuga y métricas exactas (sin sklearn)
# =============================================================================


def precision_recall_curve_exact(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equivalente exacto de sklearn.metrics.precision_recall_curve."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    desc = np.argsort(y_prob, kind="mergesort")[::-1]
    y_prob_s = y_prob[desc]
    y_true_s = y_true[desc]

    distinct = np.where(np.diff(y_prob_s))[0]
    thr_idxs = np.r_[distinct, y_true_s.size - 1]
    tps = np.cumsum(y_true_s)[thr_idxs].astype(np.float64)
    fps = (1 + thr_idxs).astype(np.float64) - tps
    thresholds = y_prob_s[thr_idxs]

    precision = tps / (tps + fps)
    total_pos = float(tps[-1])
    recall = tps / total_pos if total_pos > 0.0 else np.zeros_like(tps)

    # Orden estándar de sklearn (umbrales crecientes, recall decreciente hasta 0)
    last_ind = int(tps.searchsorted(tps[-1]))
    sl = slice(last_ind, None, -1)
    return (
        np.r_[precision[sl], 1.0],
        np.r_[recall[sl], 0.0],
        thresholds[sl],
    )


def choose_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve_exact(y_true, y_prob)
    if thr.size == 0:
        return 0.5
    denom = prec[:-1] + rec[:-1]
    f1 = np.where(denom > 0.0, 2.0 * prec[:-1] * rec[:-1] / np.where(denom > 0.0, denom, 1.0), 0.0)
    return float(thr[int(np.argmax(f1))])


def evaluate_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    p_curve, r_curve, _ = precision_recall_curve_exact(y_true, y_prob)
    pr_auc = float(-np.trapezoid(p_curve, r_curve))

    return {
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def build_eval_cache(
    ds: EllipticDataset,
    mask: np.ndarray,
    operators_by_step: Dict[int, sparse.csr_matrix],
) -> Dict[int, Tuple[np.ndarray, sparse.csr_matrix, np.ndarray]]:
    """Precalcula (SX_U, S_rows_U, y_step) por paso temporal sin fuga."""
    cache: Dict[int, Tuple[np.ndarray, sparse.csr_matrix, np.ndarray]] = {}
    steps = sorted(np.unique(ds.time_steps[mask]).tolist())
    for step in steps:
        sm = np.flatnonzero(mask & (ds.time_steps == step))
        s_op = operators_by_step[step]
        s_rows = s_op[sm, :].tocsc()
        active_u = np.flatnonzero(np.diff(s_rows.indptr) > 0)
        sx_u = np.ascontiguousarray((s_op[active_u, :] @ ds.x), dtype=np.float64)
        s_sub = s_rows[:, active_u].tocsr()
        cache[step] = (sx_u, s_sub, ds.y[sm])
    return cache


def predict_chronological(
    obj: ExplicitGCNObjective,
    theta: np.ndarray,
    eval_cache: Dict[int, Tuple[np.ndarray, sparse.csr_matrix, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    y_parts: List[np.ndarray] = []
    p_parts: List[np.ndarray] = []
    for step in sorted(eval_cache.keys()):
        sx_u, s_sub, y_st = eval_cache[step]
        logits = obj.predict_logits_precomputed(theta, sx_u, s_sub)
        probs = _sigmoid_stable(logits)
        y_parts.append(y_st)
        p_parts.append(probs)
    return np.concatenate(y_parts), np.concatenate(p_parts)


# =============================================================================
# 5. Figuras a color para informe LaTeX y presentación Beamer
# =============================================================================


def generate_figures(
    results: Dict[str, OptimizationResult],
    pr_curves_test: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    palette = {
        "lbfgs_tanh_bce": ("#1f77b4", "-", "L-BFGS (tanh, BCE informe, λ=0)"),
        "lbfgs_tanh_bce_l2": ("#2ca02c", "-", "L-BFGS (tanh, BCE + L2)"),
        "lbfgs_tanh_cost_l2": ("#d62728", "-", "L-BFGS (tanh, costo + L2)"),
        "lbfgs_relu_cost_l2": ("#9467bd", "--", "L-BFGS (ReLU, costo + L2)"),
        "gd_tanh_bce_l2": ("#ff7f0e", ":", "Gradiente Wolfe (tanh, BCE + L2)"),
    }

    # Figura 1: Convergencia (2x2): F(theta_k), ||grad F||, paso relativo, tiempo
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 7.8))
    for key, res in results.items():
        color, ls, label = palette[key]
        it = np.asarray(res.history["iter"], dtype=float)
        loss = np.asarray(res.history["loss"], dtype=float)
        gnorm = np.asarray(res.history["grad_norm"], dtype=float)
        rstep = np.asarray(res.history["rel_step"], dtype=float)
        tsec = np.asarray(res.history["time_sec"], dtype=float)

        axs[0, 0].plot(it, loss, color=color, linestyle=ls, linewidth=2.0, label=label)
        axs[0, 1].semilogy(it, gnorm, color=color, linestyle=ls, linewidth=2.0, label=label)
        valid = np.isfinite(rstep) & (rstep > 0)
        axs[1, 0].semilogy(it[valid], rstep[valid], color=color, linestyle=ls, linewidth=2.0, label=label)
        axs[1, 1].semilogy(tsec, gnorm, color=color, linestyle=ls, linewidth=2.0, label=label)

    axs[0, 0].set_title(r"Evolución de la función de pérdida $F(\theta_k)$")
    axs[0, 0].set_xlabel(r"Iteración $k$")
    axs[0, 0].set_ylabel(r"$F(\theta_k)$")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].set_title(r"Norma euclidiana del gradiente $\|\nabla F(\theta_k)\|_2$")
    axs[0, 1].axhline(1e-4, color="#555555", linestyle="-.", linewidth=1.1)
    axs[0, 1].set_xlabel(r"Iteración $k$")
    axs[0, 1].set_ylabel(r"$\|\nabla F(\theta_k)\|_2$")
    axs[0, 1].grid(True, which="both", alpha=0.3)

    axs[1, 0].set_title(r"Criterio de parada relativo $\|\theta_k - \theta_{k-1}\|_2 / \max(1,\|\theta_k\|_2)$")
    axs[1, 0].axhline(1e-5, color="#555555", linestyle="-.", linewidth=1.1, label=r"Tolerancia $\varepsilon_\theta = 10^{-5}$")
    axs[1, 0].set_xlabel(r"Iteración $k$")
    axs[1, 0].set_ylabel("Cambio relativo")
    axs[1, 0].grid(True, which="both", alpha=0.3)

    axs[1, 1].set_title(r"Decaimiento de $\|\nabla F(\theta_k)\|_2$ frente al tiempo de CPU")
    axs[1, 1].axhline(1e-4, color="#555555", linestyle="-.", linewidth=1.1)
    axs[1, 1].set_xlabel("Tiempo transcurrido (s)")
    axs[1, 1].set_ylabel(r"$\|\nabla F(\theta_k)\|_2$")
    axs[1, 1].grid(True, which="both", alpha=0.3)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"lbfgs_elliptic_convergencia.{ext}", dpi=220)
    plt.close(fig)

    # Figura 2: Verificación numérica del Teorema 3.1 (Zoutendijk) y cos(psi_k)
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for key in ("lbfgs_tanh_bce", "lbfgs_tanh_bce_l2", "lbfgs_tanh_cost_l2", "lbfgs_relu_cost_l2"):
        res = results[key]
        color, ls, label = palette[key]
        it = np.asarray(res.history["iter"][1:], dtype=float)
        cpsi = np.asarray(res.history["cos_psi"][1:], dtype=float)
        zsum = np.asarray(res.history["zoutendijk_sum"][1:], dtype=float)
        zterm = np.diff(np.asarray(res.history["zoutendijk_sum"], dtype=float))
        axs[0].semilogy(it, np.maximum(cpsi, 1e-16), color=color, linestyle=ls, linewidth=1.8, label=label)
        axs[1].semilogy(it, np.maximum(zterm, 1e-16), color=color, linestyle=ls, linewidth=1.6, label=label)
        axs[2].plot(it, zsum, color=color, linestyle=ls, linewidth=2.0, label=label)

    axs[0].set_title(r"Alineación $\cos\psi_k = -\frac{g_k^\top d_k}{\|g_k\|_2 \|d_k\|_2}$")
    axs[0].set_xlabel(r"Iteración $k$")
    axs[0].set_ylabel(r"$\cos\psi_k$")
    axs[0].grid(True, which="both", alpha=0.3)

    axs[1].set_title(r"Término de Zoutendijk $\cos^2\psi_k \|\nabla F(\theta_k)\|_2^2$")
    axs[1].set_xlabel(r"Iteración $k$")
    axs[1].set_ylabel(r"$\cos^2\psi_k \|\nabla F(\theta_k)\|_2^2$")
    axs[1].grid(True, which="both", alpha=0.3)

    axs[2].set_title(r"Suma parcial $\sum_{j=0}^{k-1} \cos^2\psi_j \|\nabla F(\theta_j)\|_2^2$")
    axs[2].set_xlabel(r"Iteración $k$")
    axs[2].set_ylabel("Suma parcial acumulada")
    axs[2].grid(True, alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"lbfgs_elliptic_zoutendijk.{ext}", dpi=220)
    plt.close(fig)

    # Figura 3: Curvas Precisión–Exhaustividad en el conjunto de prueba temporal (pasos 40–49)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    for key, (prec, rec, pr_auc) in pr_curves_test.items():
        color, ls, base_label = palette[key]
        ax.plot(
            rec,
            prec,
            color=color,
            linestyle=ls,
            linewidth=2.2,
            label=f"{base_label} (PR-AUC = {pr_auc:.4f})",
        )
    ax.set_title("Curvas Precisión–Exhaustividad en Elliptic (prueba temporal, pasos 40–49)")
    ax.set_xlabel("Exhaustividad (Recall)")
    ax.set_ylabel("Precisión (Precision)")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"lbfgs_elliptic_pr_curvas.{ext}", dpi=220)
    plt.close(fig)


# =============================================================================
# 6. Ejecución principal
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="L-BFGS (Andrei 2020) sobre GCNN en Elliptic.")
    base_dir = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=base_dir / "Fundamentos_Matematicos_GNN.zip",
        help="Ruta al archivo Fundamentos_Matematicos_GNN.zip.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directorio opcional con los 3 CSV ya extraídos.",
    )
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--memory", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sufijo",
        default="",
        help="Sufijo para no sobrescribir salidas (p. ej. _1000): JSON, historial y figuras.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("GCNN (ExplicitGCN) + L-BFGS (Andrei 2020, Alg. 1.6) SOBRE ELLIPTIC REAL")
    print("=" * 72)

    ds = load_elliptic_dataset(
        zip_path=None if args.data_dir is not None else args.zip_path,
        data_dir=args.data_dir,
    )
    train_mask, val_mask, test_mask = chronological_split(ds.time_steps, ds.y, 34, 39)

    def _split_info(mask: np.ndarray) -> Dict[str, float]:
        yl = ds.y[mask]
        pos = int(np.sum(yl == 1.0))
        neg = int(np.sum(yl == 0.0))
        tot = pos + neg
        return {
            "total": tot,
            "illicit": pos,
            "licit": neg,
            "prevalence": pos / tot if tot else 0.0,
        }

    tr_info = _split_info(train_mask)
    va_info = _split_info(val_mask)
    te_info = _split_info(test_mask)
    print(f"[INFO] Entrenamiento (t<=34): {tr_info}")
    print(f"[INFO] Validación (35<=t<=39): {va_info}")
    print(f"[INFO] Prueba (40<=t<=49):     {te_info}")

    # Operadores temporales sin fuga del futuro
    t_op = time.perf_counter()
    s_train = build_normalized_shift_operator(ds.edge_src, ds.edge_dst, ds.time_steps, ds.n_nodes, 34)
    val_steps = sorted(np.unique(ds.time_steps[val_mask]).tolist())
    test_steps = sorted(np.unique(ds.time_steps[test_mask]).tolist())
    val_ops = {
        st: build_normalized_shift_operator(ds.edge_src, ds.edge_dst, ds.time_steps, ds.n_nodes, st)
        for st in val_steps
    }
    test_ops = {
        st: build_normalized_shift_operator(ds.edge_src, ds.edge_dst, ds.time_steps, ds.n_nodes, st)
        for st in test_steps
    }
    val_cache = build_eval_cache(ds, val_mask, val_ops)
    test_cache = build_eval_cache(ds, test_mask, test_ops)
    del val_ops, test_ops
    print(
        f"[INFO] Operadores temporales construidos en {time.perf_counter()-t_op:.2f} s "
        f"(nnz(S_tr)={s_train.nnz}, densidad={s_train.nnz/(ds.n_nodes**2):.3e})."
    )

    c1_max = tr_info["licit"] / tr_info["illicit"]
    print(f"[INFO] Peso de clase ilícita (Max, main.py): c1 = {c1_max:.4f}, c0 = 1.0")

    l2_reg_main = 5e-3
    tol_grad = 1e-4
    tol_rel_step = 1e-5

    # Definir configuraciones experimentales
    configs = [
        ("lbfgs_tanh_bce", "tanh", 1.0, 1.0, 0.0, "lbfgs"),
        ("lbfgs_tanh_bce_l2", "tanh", 1.0, 1.0, l2_reg_main, "lbfgs"),
        ("lbfgs_tanh_cost_l2", "tanh", c1_max, 1.0, l2_reg_main, "lbfgs"),
        ("lbfgs_relu_cost_l2", "relu", c1_max, 1.0, l2_reg_main, "lbfgs"),
        ("gd_tanh_bce_l2", "tanh", 1.0, 1.0, l2_reg_main, "gd"),
    ]

    # Verificación analítica del gradiente antes de optimizar
    grad_checks: Dict[str, float] = {}
    for act in ("tanh", "relu"):
        obj_chk = ExplicitGCNObjective(
            ds.x, s_train, ds.y, train_mask,
            hidden_dim=args.hidden_dim, activation=act, c1=c1_max, c0=1.0, l2_reg=l2_reg_main,
        )
        t0_chk = obj_chk.init_theta(seed=args.seed)
        err = verify_analytical_gradient(obj_chk, t0_chk, n_checks=12, eps=1e-6)
        grad_checks[act] = err
        print(f"[CHECK] Error relativo máximo gradiente ({act}): {err:.3e}")
        if act == "tanh" and err > 1e-6:
            raise AssertionError(f"Error de gradiente demasiado alto en {act}: {err}")

    results: Dict[str, OptimizationResult] = {}
    pr_curves_test: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    summary_json: Dict[str, object] = {
        "dataset": {
            "n_nodes": ds.n_nodes,
            "n_edges": ds.n_edges,
            "n_features": ds.n_features,
            "nnz_s_train": int(s_train.nnz),
            "splits": {"train": tr_info, "val": va_info, "test": te_info},
            "c1_balanced": c1_max,
        },
        "model": {
            "architecture": f"ExplicitGCN({ds.n_features} -> {args.hidden_dim} -> 1)",
            "n_params": ds.n_features * args.hidden_dim + args.hidden_dim + args.hidden_dim + 1,
        },
        "hyperparameters": {
            "memory_M": args.memory,
            "max_iter": args.max_iter,
            "l2_reg": l2_reg_main,
            "tol_grad": tol_grad,
            "tol_rel_step": tol_rel_step,
            "wolfe_c1": 1e-4,
            "wolfe_c2": 0.9,
            "seed": args.seed,
        },
        "gradient_checks_max_rel_err": grad_checks,
        "runs": {},
    }

    theta0_shared: Optional[np.ndarray] = None

    for key, act, c1_w, c0_w, l2_w, method in configs:
        print(f"\n--- Ejecutando {key} (act={act}, c1={c1_w:.3f}, l2={l2_w}, method={method}) ---")
        obj = ExplicitGCNObjective(
            ds.x, s_train, ds.y, train_mask,
            hidden_dim=args.hidden_dim, activation=act, c1=c1_w, c0=c0_w, l2_reg=l2_w,
        )
        if theta0_shared is None:
            theta0_shared = obj.init_theta(seed=args.seed)
            summary_json["model"]["n_active_u_train"] = obj.n_active_u  # type: ignore[index]

        if method == "lbfgs":
            res = run_lbfgs_andrei(
                obj,
                theta0_shared,
                name=key,
                memory=args.memory,
                max_iter=args.max_iter,
                tol_grad=tol_grad,
                tol_rel_step=tol_rel_step,
            )
        else:
            res = run_steepest_descent_wolfe(
                obj,
                theta0_shared,
                name=key,
                max_iter=args.max_iter,
                tol_grad=tol_grad,
                tol_rel_step=tol_rel_step,
            )
        results[key] = res

        y_val, p_val = predict_chronological(obj, res.theta, val_cache)
        thr = choose_f1_threshold(y_val, p_val)
        val_m = evaluate_binary_metrics(y_val, p_val, threshold=thr)

        y_test, p_test = predict_chronological(obj, res.theta, test_cache)
        test_m = evaluate_binary_metrics(y_test, p_test, threshold=thr)
        p_c, r_c, _ = precision_recall_curve_exact(y_test, p_test)
        pr_curves_test[key] = (p_c, r_c, test_m["pr_auc"])

        cos_vals = [c for c in res.history["cos_psi"][1:] if np.isfinite(c)]
        rel_vals = [r for r in res.history["rel_step"][1:] if np.isfinite(r)]

        run_summary = {
            "activation": act,
            "c1": c1_w,
            "c0": c0_w,
            "l2_reg": l2_w,
            "method": method,
            "stop_reason": res.stop_reason,
            "iterations": res.iterations,
            "total_evals": res.total_evals,
            "elapsed_sec": round(res.elapsed_sec, 3),
            "ms_per_iter": round(1000.0 * res.elapsed_sec / max(1, res.iterations), 2),
            "memory_resets": res.memory_resets,
            "wolfe_fallbacks": res.wolfe_fallbacks,
            "f_init": res.history["loss"][0],
            "f_final": res.history["loss"][-1],
            "grad_norm_init": res.history["grad_norm"][0],
            "grad_norm_final": res.history["grad_norm"][-1],
            "rel_step_final": rel_vals[-1] if rel_vals else None,
            "cos_psi_min": min(cos_vals) if cos_vals else None,
            "cos_psi_median": float(np.median(cos_vals)) if cos_vals else None,
            "zoutendijk_sum_final": res.history["zoutendijk_sum"][-1],
            "val_metrics": val_m,
            "test_metrics": test_m,
        }
        summary_json["runs"][key] = run_summary  # type: ignore[index]
        print(
            f"  => {key}: stop={res.stop_reason} | iters={res.iterations} | "
            f"evals={res.total_evals} | t={res.elapsed_sec:.2f}s | "
            f"F={run_summary['f_final']:.6f} | ||g||={run_summary['grad_norm_final']:.3e} | "
            f"Val PR-AUC={val_m['pr_auc']:.4f} | Test PR-AUC={test_m['pr_auc']:.4f} | "
            f"Test F1={test_m['f1']:.4f} (Rec={test_m['recall']:.4f}, Prec={test_m['precision']:.4f})"
        )

    # Validación cruzada con scipy.optimize.minimize(..., method="L-BFGS-B")
    print("\n--- Validación cruzada con SciPy L-BFGS-B (tanh, BCE + L2) ---")
    obj_sp = ExplicitGCNObjective(
        ds.x, s_train, ds.y, train_mask,
        hidden_dim=args.hidden_dim, activation="tanh", c1=1.0, c0=1.0, l2_reg=l2_reg_main,
    )
    assert theta0_shared is not None
    t_sp = time.perf_counter()
    sp_res = minimize(
        obj_sp.loss_and_grad,
        theta0_shared,
        method="L-BFGS-B",
        jac=True,
        options={"maxcor": args.memory, "maxiter": args.max_iter, "gtol": tol_grad, "ftol": 0.0},
    )
    sp_elapsed = time.perf_counter() - t_sp
    sp_gnorm = float(np.linalg.norm(sp_res.jac))
    y_val_sp, p_val_sp = predict_chronological(obj_sp, sp_res.x, val_cache)
    thr_sp = choose_f1_threshold(y_val_sp, p_val_sp)
    y_test_sp, p_test_sp = predict_chronological(obj_sp, sp_res.x, test_cache)
    test_sp = evaluate_binary_metrics(y_test_sp, p_test_sp, threshold=thr_sp)
    summary_json["scipy_crosscheck_tanh_bce_l2"] = {
        "success": bool(sp_res.success),
        "message": str(sp_res.message),
        "iterations": int(sp_res.nit),
        "total_evals": int(sp_res.nfev),
        "elapsed_sec": round(sp_elapsed, 3),
        "f_final": float(sp_res.fun),
        "grad_norm_final": sp_gnorm,
        "test_metrics": test_sp,
    }
    print(
        f"  => SciPy L-BFGS-B: iters={sp_res.nit} | evals={sp_res.nfev} | "
        f"t={sp_elapsed:.2f}s | F={sp_res.fun:.6f} | ||g||={sp_gnorm:.3e} | "
        f"Test PR-AUC={test_sp['pr_auc']:.4f} | Test F1={test_sp['f1']:.4f}"
    )

    # Guardar resultados JSON y figuras
    res_dir = base_dir / "resultados"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_json = res_dir / f"resultados_elliptic_lbfgs{args.sufijo}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Resumen guardado en {out_json}")

    # Historial completo por iteración (una fila por iterado θ_k)
    hist_dir = res_dir / f"historial_elliptic{args.sufijo}"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for key, res in results.items():
        cols = list(res.history.keys())
        with open(hist_dir / f"{key}.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            for row in zip(*(res.history[c] for c in cols)):
                writer.writerow(row)
    print(f"[INFO] Historial por iteración guardado en {hist_dir}")

    fig_dir = base_dir / "figuras" / f"elliptic{args.sufijo}" if args.sufijo else base_dir / "figuras"
    generate_figures(results, pr_curves_test, fig_dir)
    print(f"[INFO] Figuras guardadas en {fig_dir}")


if __name__ == "__main__":
    main()
