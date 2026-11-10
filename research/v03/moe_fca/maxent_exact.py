"""Exact finite-state pairwise MaxEnt oracle for small expert counts.

This is a validation oracle, NOT the scalable E=32 sampler. It enumerates all
2^E binary states and fits a pairwise log-linear distribution. Use E <= 16.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import minimize

def enumerate_states(n_experts: int, max_experts: int = 16):
    if n_experts > max_experts:
        raise ValueError(f"exact oracle limited to E <= {max_experts}; got {n_experts}")
    return np.array(list(itertools.product([0., 1.], repeat=n_experts)), dtype=float)

def sufficient_statistics(states: np.ndarray):
    _, e = states.shape
    pairs = [(i, j) for i in range(e) for j in range(i+1, e)]
    feats = [states[:, i] for i in range(e)]
    feats += [states[:, i] * states[:, j] for i, j in pairs]
    return np.column_stack(feats), pairs

def fit_pairwise_maxent(binary_context: np.ndarray, max_experts: int = 16,
                        max_iter: int = 2000, tol: float = 1e-7):
    X = np.asarray(binary_context, dtype=float)
    if X.ndim != 2 or not np.isin(X, [0, 1]).all():
        raise ValueError("binary_context must be binary")
    _, e = X.shape
    states = enumerate_states(e, max_experts=max_experts)
    F, pairs = sufficient_statistics(states)
    target = np.concatenate([X.mean(axis=0),
        np.array([(X[:, i] * X[:, j]).mean() for i, j in pairs])])
    def objective(theta):
        z = F @ theta
        z = z - z.max()
        logz = np.log(np.exp(z).sum())
        prob = np.exp(z - logz)
        return logz - target @ theta, prob @ F - target
    result = minimize(lambda t: objective(t)[0], np.zeros(F.shape[1]),
                      jac=lambda t: objective(t)[1], method="L-BFGS-B",
                      options={"maxiter": max_iter, "ftol": tol})
    z = F @ result.x
    z = z - z.max()
    prob = np.exp(z)
    prob /= prob.sum()
    achieved = prob @ F
    return {
        "states": states, "probabilities": prob, "parameters": result.x,
        "target_moments": target, "achieved_moments": achieved,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "moment_max_abs_error": float(np.max(np.abs(achieved-target))),
        "pairs": pairs,
    }

def sample_exact(fit: dict, n_samples: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fit["states"]), size=n_samples, p=fit["probabilities"])
    return fit["states"][idx].astype(np.uint8)
