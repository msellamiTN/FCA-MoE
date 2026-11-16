#!/usr/bin/env python3
"""FCA-MoE F2-E: nested parametric bootstrap for temporal pairwise-MaxEnt FCA.

The procedure is deliberately explicit: fit the fixed-cardinality pairwise
MaxEnt model, select the temporal Gibbs mixture by the same lag-profile screen,
then compare FCA higher-order counts to an inner null. Outer bootstrap samples
repeat fitting, lambda selection, and temporal-weight selection.

This is a research implementation, not a claim that the experiment has run.
"""
from __future__ import annotations
import argparse, itertools, json, math, os, time, hashlib, platform, sys
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp

E_DEFAULT, K_DEFAULT = 60, 4
THRESHOLDS = (0.01, 0.02, 0.05)
TAGS = ("1pct", "2pct", "5pct")
LAGS = 10
FAMILY = [f"{prefix}_{tag}" for tag in TAGS for prefix in
          ("pooled_higher_order", "mean_sequence_higher_order")]


def all_states(E=60, K=4):
    return np.asarray(list(itertools.combinations(range(E), K)), dtype=np.int16)


def pair_index_table(E):
    pairs = list(itertools.combinations(range(E), 2))
    lut = np.full((E, E), -1, dtype=np.int32)
    for q, (i, j) in enumerate(pairs):
        lut[i, j] = lut[j, i] = q
    return pairs, lut


def state_pair_indices(states, lut):
    K = states.shape[1]
    cols = [lut[states[:, i], states[:, j]]
            for i, j in itertools.combinations(range(K), 2)]
    return np.column_stack(cols).astype(np.int32, copy=False)


@lru_cache(maxsize=4)
def shared_state_and_transition_structure(E, K):
    """Cache immutable state/pair/conditional-group structures across model fits."""
    states = all_states(E, K)
    pairs, lut = pair_index_table(E)
    state_pair = state_pair_indices(states, lut)
    base = np.arange(len(states), dtype=np.int32)
    transition = {}
    for k in (1, 2, 3):
        positions = list(itertools.combinations(range(K), k))
        key_parts, id_parts = [], []
        for pos in positions:
            vals = states[:, pos]
            masks = np.zeros(len(vals), dtype=np.uint64)
            for j in range(k):
                masks |= np.left_shift(np.uint64(1), vals[:, j].astype(np.uint64))
            key_parts.append(masks)
            id_parts.append(base)
        keys = np.concatenate(key_parts)
        ids = np.concatenate(id_parts)
        order = np.argsort(keys, kind="stable")
        keys, ids = keys[order], ids[order]
        uniq, starts, counts = np.unique(
            keys, return_index=True, return_counts=True
        )
        ends = starts + counts
        state_group_ids = np.empty((len(states), len(positions)), dtype=np.int32)
        for col, pos in enumerate(positions):
            vals = states[:, pos]
            masks = np.zeros(len(vals), dtype=np.uint64)
            for j in range(k):
                masks |= np.left_shift(np.uint64(1), vals[:, j].astype(np.uint64))
            group_ids = np.searchsorted(uniq, masks)
            if not np.array_equal(uniq[group_ids], masks):
                raise AssertionError("Missing retained-subset group")
            state_group_ids[:, col] = group_ids.astype(np.int32)
        lookup = {int(value): i for i, value in enumerate(uniq)}
        for array in (keys, ids, uniq, starts, ends, state_group_ids):
            array.setflags(write=False)
        transition[k] = {
            "keys": uniq, "starts": starts, "ends": ends, "ids": ids,
            "lookup": lookup, "state_group_ids": state_group_ids,
        }
    for array in (states, state_pair, lut):
        array.setflags(write=False)
    return states, pairs, lut, state_pair, transition


class ConditionalPairwiseMaxEnt:
    """Pairwise exponential family conditioned on exactly K active experts."""
    def __init__(self, E=60, K=4, states=None):
        self.E, self.K = int(E), int(K)
        if states is None:
            (self.states, self.pairs, self.lut, self.state_pair,
             self._transition_structure) = shared_state_and_transition_structure(
                self.E, self.K
            )
        else:
            self.states = np.asarray(states, np.int16)
            self.pairs, self.lut = pair_index_table(self.E)
            self.state_pair = state_pair_indices(self.states, self.lut)
            self._transition_structure = None
        self.n_params = len(self.pairs)
        self.subset_group_ids = None
        self.theta = None
        self.prob = None
        self.cdf = None
        self.fitted_pair_moments = None
        self.fit_info = None

    def empirical_pair_moments(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.E or not np.isin(X, [0, 1]).all():
            raise ValueError("X must be a binary N x E context")
        if not np.all(X.sum(axis=1) == self.K):
            raise ValueError(f"Every context row must contain exactly K={self.K} ones")
        out = np.zeros(self.n_params, dtype=np.float64)
        for i, j in self.pairs:
            out[self.lut[i, j]] = np.mean(X[:, i] * X[:, j])
        if not np.isclose(out.sum(), math.comb(self.K, 2), atol=1e-9):
            raise AssertionError("Fixed-cardinality pair moments do not sum to C(K,2)")
        return out

    def fit(self, X, lam=1e-3, maxiter=600, gtol=1e-7, initial=None):
        target = self.empirical_pair_moments(X)
        sp = self.state_pair
        def fg(theta):
            energies = theta[sp].sum(axis=1)
            lz = logsumexp(energies)
            p = np.exp(energies - lz)
            pred = np.bincount(sp.ravel(), weights=np.repeat(p, sp.shape[1]),
                               minlength=self.n_params)
            diff = pred - target
            value = float(lz - target @ theta + 0.5 * lam * (theta @ theta))
            grad = diff + lam * theta
            return value, grad
        x0 = np.zeros(self.n_params, dtype=np.float64) if initial is None else np.asarray(initial, np.float64).copy()
        result = minimize(fg, x0, jac=True, method="L-BFGS-B",
                          options={"maxiter": int(maxiter), "gtol": float(gtol),
                                   "ftol": 1e-12, "maxls": 40, "maxcor": 20})
        theta = result.x
        # The ridge penalty fixes the additive gauge; center numerically to sum zero.
        theta = theta - theta.mean()
        energies = theta[sp].sum(axis=1)
        logz = float(logsumexp(energies))
        prob = np.exp(energies - logz)
        pred = np.bincount(sp.ravel(), weights=np.repeat(prob, sp.shape[1]),
                           minlength=self.n_params)
        grad_inf = float(np.max(np.abs((pred-target) + lam*theta)))
        self.theta, self.prob, self.cdf = theta, prob, np.cumsum(prob)
        self.cdf[-1] = 1.0
        self.fitted_pair_moments = pred
        self.fit_info = {
            "success": bool(result.success or grad_inf <= max(gtol*10, 1e-6)),
            "optimizer_success": bool(result.success),
            "message": str(result.message), "iterations": int(result.nit),
            "objective": float(result.fun), "gradient_inf_norm": grad_inf,
            "lambda": float(lam), "log_partition": logz,
            "gauge_sum": float(theta.sum()),
            "pair_moment_rmse": float(np.sqrt(np.mean((pred-target)**2))),
            "pair_moment_max_abs_error": float(np.max(np.abs(pred-target))),
            "probability_sum": float(prob.sum()),
        }
        if not self.fit_info["success"]:
            raise RuntimeError(f"MaxEnt fit failed: {self.fit_info}")
        return self

    def nll(self, X):
        X = np.asarray(X)
        ids = self.state_ids(X)
        energies = self.theta[self.state_pair[ids]].sum(axis=1)
        return float(self.logz - energies.mean())

    @property
    def logz(self):
        return float(logsumexp(self.theta[self.state_pair].sum(axis=1)))

    def state_ids(self, X):
        # Exact lookup of sorted Top-K tuples in the lexicographic combinations table.
        rows = np.stack([np.flatnonzero(row) for row in np.asarray(X)], axis=0)
        rows = np.sort(rows, axis=1)
        # For E=60,K=4, use combinatorial rank (lexicographic order).
        return np.asarray([self.rank_combination(row) for row in rows], dtype=np.int64)

    def rank_combination(self, row):
        # Rank in itertools.combinations(range(E), K), zero-based.
        rank = 0; prev = -1
        for pos, val in enumerate(map(int, row)):
            for x in range(prev + 1, val):
                rank += math.comb(self.E - x - 1, self.K - pos - 1)
            prev = val
        return rank

    def draw_global(self, rng):
        return min(int(np.searchsorted(self.cdf, rng.random(), side="right")), len(self.cdf)-1)

    def build_conditional_indexes(self):
        """Build model-specific CDFs over cached conditional group structures."""
        self.cond = {}
        self.subset_group_ids = {}
        for k in (1, 2, 3):
            if self._transition_structure is not None:
                structure = self._transition_structure[k]
                keys, starts, ends = structure["keys"], structure["starts"], structure["ends"]
                ids, lookup = structure["ids"], structure["lookup"]
                state_groups = structure["state_group_ids"]
            else:
                positions = list(itertools.combinations(range(self.K), k))
                key_parts, id_parts = [], []
                base = np.arange(len(self.states), dtype=np.int32)
                for pos in positions:
                    vals = self.states[:, pos]
                    masks = np.zeros(len(vals), dtype=np.uint64)
                    for j in range(k):
                        masks |= np.left_shift(np.uint64(1), vals[:, j].astype(np.uint64))
                    key_parts.append(masks)
                    id_parts.append(base)
                keys, ids = np.concatenate(key_parts), np.concatenate(id_parts)
                order = np.argsort(keys, kind="stable")
                keys, ids = keys[order], ids[order]
                keys, starts, counts = np.unique(keys, return_index=True, return_counts=True)
                ends = starts + counts
                lookup = {int(v): i for i, v in enumerate(keys)}
                state_groups = np.empty((len(self.states), len(positions)), dtype=np.int32)
                for col, pos in enumerate(positions):
                    vals = self.states[:, pos]
                    masks = np.zeros(len(vals), dtype=np.uint64)
                    for j in range(k):
                        masks |= np.left_shift(np.uint64(1), vals[:, j].astype(np.uint64))
                    state_groups[:, col] = np.searchsorted(keys, masks)
            cdf = np.empty(len(ids), dtype=np.float64)
            weights = self.prob[ids]
            for a, b in zip(starts, ends):
                q = weights[a:b]
                total = q.sum()
                if total <= 0 or not np.isfinite(total):
                    raise RuntimeError("Invalid conditional mass")
                cdf[a:b] = np.cumsum(q / total)
                cdf[b - 1] = 1.0
            self.cond[k] = {
                "keys": keys, "starts": starts, "ends": ends,
                "ids": ids, "cdf": cdf, "lookup": lookup,
            }
            self.subset_group_ids[k] = state_groups
        return self

    @staticmethod
    def _mask(row):
        v = 0
        for x in row: v |= 1 << int(x)
        return v

    def draw_conditional(self, retained, rng):
        k = len(retained); q = self.cond[k]
        g = q["lookup"].get(self._mask(retained))
        if g is None: raise KeyError(f"Missing retained subset {retained}")
        a,b = int(q["starts"][g]), int(q["ends"][g])
        p = min(int(np.searchsorted(q["cdf"][a:b], rng.random(), side="right")), b-a-1)
        return int(q["ids"][a+p])

    def step(self, current, weights, rng):
        k = int(rng.choice(4, p=weights))
        if k == 0:
            return self.draw_global(rng)
        if self.subset_group_ids is not None:
            # A uniform subset-column draw is exactly uniform over C(K,k) subsets.
            col = int(rng.integers(self.subset_group_ids[k].shape[1]))
            group = int(self.subset_group_ids[k][int(current), col])
            q = self.cond[k]
            a, b = int(q["starts"][group]), int(q["ends"][group])
            p = min(
                int(np.searchsorted(q["cdf"][a:b], rng.random(), side="right")),
                b - a - 1,
            )
            return int(q["ids"][a + p])
        row = self.states[current]
        keep = np.sort(rng.choice(self.K, size=k, replace=False))
        return self.draw_conditional(row[keep], rng)


def select_lambda_cv(blocks, lambdas, maxiter=400):
    """Leave-one-sequence-out CV; returns selected lambda and full score table."""
    E, K = blocks[0].shape[1], int(blocks[0].sum(axis=1)[0])
    scores=[]
    for lam in lambdas:
        fold_nll=[]
        for held in range(len(blocks)):
            train=np.vstack([b for i,b in enumerate(blocks) if i != held])
            val=blocks[held]
            model=ConditionalPairwiseMaxEnt(E,K).fit(train,lam,maxiter=maxiter)
            fold_nll.append(model.nll(val))
        scores.append({"lambda":float(lam),"mean_validation_nll":float(np.mean(fold_nll)),
                       "fold_nll":fold_nll})
    best=min(scores,key=lambda r:r["mean_validation_nll"])
    return float(best["lambda"]), scores


def lag_profile(chain_ids, states, max_lag=LAGS):
    x=states[np.asarray(chain_ids,dtype=np.int64)]; out=[]
    for lag in range(1,max_lag+1):
        inter=np.sum(x[:-lag,:,None]==x[lag:,None,:],axis=(1,2))
        out.append(float(inter.mean()))
    return np.asarray(out)


def generate_dataset(model, weights, seed, chains=4, length=256):
    rng=np.random.default_rng(seed); ids=[]; contexts=[]; profiles=[]
    for _ in range(chains):
        chain=np.empty(length,dtype=np.int32); chain[0]=model.draw_global(rng)
        for t in range(1,length): chain[t]=model.step(int(chain[t-1]),weights,rng)
        ids.append(chain)
        selected=model.states[chain]
        X=np.zeros((length,model.E),dtype=np.uint8)
        X[np.arange(length)[:,None],selected]=1
        contexts.append(X); profiles.append(lag_profile(chain,model.states))
    return contexts, np.mean(profiles,axis=0)


def grid_weights(step):
    n=round(1/step)
    if abs(n*step-1)>1e-8: raise ValueError("grid step must divide 1")
    for a in range(n+1):
      for b in range(n-a+1):
       for c in range(n-a-b+1):
        d=n-a-b-c
        yield np.array([a,b,c,d],float)/n

class FCA:
    @staticmethod
    def mine_multi_threshold(X, minsups):
        """Count closed concepts at multiple supports in one CbO traversal."""
        X = np.asarray(X)
        if X.ndim != 2 or not np.isin(X, [0, 1]).all():
            raise ValueError("FCA context must be binary")
        supports = sorted(set(int(s) for s in minsups))
        if not supports or supports[0] < 1:
            raise ValueError("minsups must contain positive support counts")
        n, m = X.shape
        cols = []
        for j in range(m):
            mask = 0
            for i in np.flatnonzero(X[:, j]):
                mask |= 1 << int(i)
            cols.append(mask)
        full = (1 << n) - 1
        counts = {
            support: {"higher": 0, "triplets": 0, "quadruplets": 0}
            for support in supports
        }

        def closure(extent):
            return tuple(
                j for j, col in enumerate(cols)
                if (extent & col) == extent
            )

        def record(intent, extent):
            size = len(intent)
            if size < 3:
                return
            extent_size = extent.bit_count()
            for support in supports:
                if extent_size >= support:
                    counts[support]["higher"] += 1
                    if size == 3:
                        counts[support]["triplets"] += 1
                    elif size == 4:
                        counts[support]["quadruplets"] += 1

        def visit(intent, extent, start):
            record(intent, extent)
            current = set(intent)
            for j in range(start, m):
                if j in current:
                    continue
                new_extent = extent & cols[j]
                if new_extent.bit_count() < supports[0]:
                    continue
                new_intent = closure(new_extent)
                if any(k < j and k not in current for k in new_intent):
                    continue
                visit(new_intent, new_extent, j + 1)

        visit(closure(full), full, 0)
        return counts

    @staticmethod
    def mine(X,minsup):
        X=np.asarray(X)
        if X.ndim!=2 or not np.isin(X,[0,1]).all(): raise ValueError("FCA context must be binary")
        n,m=X.shape; cols=[]
        for j in range(m):
            mask=0
            for i in np.flatnonzero(X[:,j]): mask |= 1 << int(i)
            cols.append(mask)
        full=(1<<n)-1; by=np.zeros(m+1,dtype=np.int64); total=0
        def closure(ext): return tuple(j for j,col in enumerate(cols) if (ext&col)==ext)
        root=closure(full)
        def visit(intent,extent,start):
            nonlocal total
            by[len(intent)]+=1; total+=1; current=set(intent)
            for j in range(start,m):
                if j in current: continue
                newext=extent&cols[j]
                if newext.bit_count()<minsup: continue
                newintent=closure(newext)
                if any(k<j and k not in current for k in newintent): continue
                visit(newintent,newext,j+1)
        visit(root,full,0)
        return {"total":int(total),"higher":int(by[3:].sum()),"triplets":int(by[3]),
                "quadruplets":int(by[4]) if m>=4 else 0,"by_size":by}


def mine_dataset(blocks):
    """Compute all support-threshold metrics with one traversal per context."""
    pooled = np.vstack(blocks)
    pooled_supports = [
        max(1, int(math.ceil(frac * len(pooled) - 1e-12)))
        for frac in THRESHOLDS
    ]
    pooled_results = FCA.mine_multi_threshold(pooled, pooled_supports)
    sequence_supports = [
        [
            max(1, int(math.ceil(frac * len(block) - 1e-12)))
            for frac in THRESHOLDS
        ]
        for block in blocks
    ]
    sequence_results = [
        FCA.mine_multi_threshold(block, supports)
        for block, supports in zip(blocks, sequence_supports)
    ]
    row = {}
    for i, tag in enumerate(TAGS):
        pooled_min = pooled_supports[i]
        row[f"pooled_higher_order_{tag}"] = pooled_results[pooled_min]["higher"]
        row[f"pooled_triplets_{tag}"] = pooled_results[pooled_min]["triplets"]
        row[f"pooled_quadruplets_{tag}"] = pooled_results[pooled_min]["quadruplets"]
        row[f"mean_sequence_higher_order_{tag}"] = float(np.mean([
            result[supports[i]]["higher"]
            for result, supports in zip(sequence_results, sequence_supports)
        ]))
    return row


def _array_sha256(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).view(np.uint8)
    ).hexdigest()


def select_weights(model, target_profile, target_pair, grid_step=.1, screen_replicates=8,
                   chains=4, length=256, seed_base=70000000,
                   progress_path=None, resume=False, progress_every=25,
                   progress_label="temporal weight screen"):
    rows = []
    candidates = list(grid_weights(grid_step))
    signature = {
        "grid_step": float(grid_step),
        "screen_replicates": int(screen_replicates),
        "chains": int(chains), "length": int(length),
        "seed_base": int(seed_base),
        "theta_sha256": _array_sha256(model.theta),
        "target_profile_sha256": _array_sha256(np.asarray(target_profile)),
        "target_pair_sha256": _array_sha256(np.asarray(target_pair)),
    }
    meta_path = Path(str(progress_path) + ".json") if progress_path is not None else None
    start_candidate = 0
    if progress_path is not None:
        progress_path = Path(progress_path)
        if meta_path.exists():
            if not resume:
                raise RuntimeError(
                    f"Weight-screen checkpoint exists: {progress_path}; "
                    "use --resume or a new output directory."
                )
            saved = json.loads(meta_path.read_text())
            if saved.get("signature") != signature:
                raise RuntimeError(
                    f"Weight-screen checkpoint signature mismatch: {progress_path}"
                )
            if progress_path.exists():
                partial = pd.read_csv(progress_path)
                if len(partial) > len(candidates) or not np.array_equal(
                    partial["candidate"].to_numpy(dtype=int),
                    np.arange(len(partial)),
                ):
                    raise RuntimeError("Invalid weight-screen checkpoint ordering")
                for item in partial.to_dict("records"):
                    rows.append({
                        "weights": np.array([
                            item["w0"], item["w1"], item["w2"], item["w3"]
                        ], dtype=float),
                        "lag_rmse": float(item["lag_rmse"]),
                        "pair_rmse": float(item["pair_rmse"]),
                    })
                start_candidate = len(rows)
        else:
            if progress_path.exists():
                raise RuntimeError("Partial screen exists without its metadata sidecar")
            meta_path.write_text(json.dumps({
                "status": "running", "signature": signature,
                "candidate_count": len(candidates), "completed_candidates": 0,
            }, indent=2))

    for ci in range(start_candidate, len(candidates)):
        weights = candidates[ci]
        profiles, pair_errors = [], []
        for rep in range(screen_replicates):
            blocks, profile = generate_dataset(
                model, weights, seed_base + ci * 1000 + rep, chains, length
            )
            profiles.append(profile)
            pair_moments = model.empirical_pair_moments(np.vstack(blocks))
            pair_errors.append(float(np.sqrt(np.mean(
                (pair_moments - target_pair) ** 2
            ))))
        mean_profile = np.mean(profiles, axis=0)
        rows.append({
            "weights": weights,
            "lag_rmse": float(np.sqrt(np.mean(
                (mean_profile - target_profile) ** 2
            ))),
            "pair_rmse": float(np.mean(pair_errors)),
        })
        completed = ci + 1
        if progress_path is not None and (
            completed % max(1, progress_every) == 0
            or completed == len(candidates)
        ):
            partial_rows = [{
                "candidate": i,
                "w0": float(row["weights"][0]),
                "w1": float(row["weights"][1]),
                "w2": float(row["weights"][2]),
                "w3": float(row["weights"][3]),
                "lag_rmse": float(row["lag_rmse"]),
                "pair_rmse": float(row["pair_rmse"]),
            } for i, row in enumerate(rows)]
            temporary = Path(str(progress_path) + ".tmp")
            pd.DataFrame(partial_rows).to_csv(temporary, index=False)
            temporary.replace(progress_path)
            meta = json.loads(meta_path.read_text())
            meta["completed_candidates"] = completed
            meta_path.write_text(json.dumps(meta, indent=2))
            print(
                f"{progress_label}: {completed}/{len(candidates)} candidates",
                flush=True,
            )

    rows.sort(key=lambda row: row["lag_rmse"])
    if meta_path is not None and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["status"] = "completed"
        meta["completed_candidates"] = len(candidates)
        meta_path.write_text(json.dumps(meta, indent=2))
    return rows[0]["weights"], rows


def max_t_from_inner(observed_metrics, inner_metrics, active_family=None):
    """Compute a selection-aware max empirical-tail score over all six metrics.

    A zero-variance inner sample is not grounds for dropping a metric. For each
    predeclared metric j, estimate its upper-tail probability with the
    plus-one Monte Carlo estimator:
        p_j = (1 + #{inner replicate: T_j^null >= T_j^obs}) / (M + 1).
    Transform to a common extremeness score -log(p_j), then take the maximum.
    This remains defined when the sampled null is constant and the observation
    lies above its support. The outer bootstrap calibrates this complete
    statistic, including finite-inner-sample discreteness.

    Ties count as exceedances (conservative). All six metrics remain in the
    family in every outer replicate; active_family is accepted only as a
    consistency check for older callers.
    """
    if active_family is not None and list(active_family) != list(FAMILY):
        raise RuntimeError(
            "F2-E empirical-tail max statistic requires the complete fixed "
            f"six-metric family; received {list(active_family)}"
        )
    if not inner_metrics:
        raise ValueError("inner_metrics must contain at least one null replicate")
    means = {n: float(np.mean([r[n] for r in inner_metrics])) for n in FAMILY}
    sds = {n: float(np.std([r[n] for r in inner_metrics], ddof=1))
           if len(inner_metrics) > 1 else 0.0 for n in FAMILY}
    tail = {}
    constant_checks = {}
    for n in FAMILY:
        obs = float(observed_metrics[n])
        null_values = np.asarray([float(r[n]) for r in inner_metrics])
        exceedances = int(np.sum(null_values >= obs))
        p_plus_one = float((1 + exceedances) / (len(null_values) + 1))
        tail[n] = {
            "observed": obs,
            "null_mean": means[n],
            "null_sd": sds[n],
            "null_min": float(null_values.min()),
            "null_max": float(null_values.max()),
            "exceedances_ge": exceedances,
            "inner_replicates": int(len(null_values)),
            "p_plus_one": p_plus_one,
            "extremeness_score": float(-math.log(p_plus_one)),
        }
        if sds[n] <= 1e-12:
            constant_checks[n] = {
                "null_constant": means[n],
                "observed": obs,
                "observed_exceeds": bool(obs > means[n]),
                "handled_by_empirical_tail": True,
            }
    scores = {n: tail[n]["extremeness_score"] for n in FAMILY}
    return {
        "max_t": max(scores.values()),
        "z": scores,  # compatibility key; values are -log empirical tail p, not z-scores
        "tail": tail,
        "means": means,
        "sds": sds,
        "active": list(FAMILY),
        "constant": list(constant_checks),
        "constant_checks": constant_checks,
        "statistic": "maximum_negative_log_plus_one_empirical_upper_tail_probability",
    }

def load_observed(root):
    npz=root/"qwen_contexts_by_layer.npz"; align=root/"sequence_alignment.csv"
    with np.load(npz) as z: X=z["layer_00"].astype(np.uint8)
    a=pd.read_csv(align)
    order=a.sort_values(["sequence_id","token_position"],kind="stable").index.to_numpy()
    aa=a.iloc[order].reset_index(drop=True); ids=list(pd.unique(aa.sequence_id))
    blocks=[]
    for sid in ids:
        pos=np.flatnonzero(aa.sequence_id.to_numpy()==sid); block=X[order[pos]]
        if len(block)!=256 or not np.all(block.sum(1)==4): raise ValueError("Expected 4 x 256 Top-4 sequences")
        blocks.append(block)
    return blocks


def evaluate_context(blocks):
    return mine_dataset(blocks)


def fit_model(blocks,lambdas,maxiter):
    chosen,cv=select_lambda_cv(blocks,lambdas,maxiter=maxiter)
    model=ConditionalPairwiseMaxEnt(60,4).fit(np.vstack(blocks),chosen,maxiter=maxiter)
    model.build_conditional_indexes()
    return model,chosen,cv


def run_inner(model, weights, n, seed_base, chains=4, length=256,
              progress_path=None, resume=False, checkpoint_every=10,
              progress_label="inner null"):
    rows, profiles, pair_rmse = [], [], []
    target = model.fitted_pair_moments
    signature = {
        "n": int(n), "seed_base": int(seed_base),
        "chains": int(chains), "length": int(length),
        "theta_sha256": _array_sha256(model.theta),
        "weights": np.asarray(weights, dtype=float).tolist(),
    }
    meta_path = (
        Path(str(progress_path) + ".progress.json")
        if progress_path is not None else None
    )
    diagnostics_path = (
        Path(str(progress_path) + ".diagnostics.npz")
        if progress_path is not None else None
    )
    start = 0
    if progress_path is not None:
        progress_path = Path(progress_path)
        if meta_path.exists():
            if not resume:
                raise RuntimeError(
                    f"Inner-null checkpoint exists: {progress_path}; "
                    "use --resume or a new output directory."
                )
            meta = json.loads(meta_path.read_text())
            if meta.get("signature") != signature:
                raise RuntimeError(
                    f"Inner-null checkpoint signature mismatch: {progress_path}"
                )
            if progress_path.exists() and diagnostics_path.exists():
                frame = pd.read_csv(progress_path)
                saved = np.load(diagnostics_path)
                rows = [{
                    key: float(value)
                    for key, value in item.items()
                    if key != "inner_replicate"
                } for item in frame.to_dict("records")]
                profiles = list(saved["lag_profiles"])
                pair_rmse = list(saved["pair_rmse"])
                if not (len(rows) == len(profiles) == len(pair_rmse)
                        == int(meta.get("completed", -1))):
                    raise RuntimeError("Inner-null checkpoint files disagree")
                start = len(rows)
                if meta.get("status") == "completed" and start == n:
                    print(f"{progress_label}: reusing completed {n}/{n}", flush=True)
                    return rows, np.asarray(profiles), np.asarray(pair_rmse)
        else:
            if progress_path.exists() or diagnostics_path.exists():
                raise RuntimeError("Inner-null checkpoint files lack metadata")
            meta_path.write_text(json.dumps({
                "status": "running", "signature": signature, "completed": 0,
            }, indent=2))

    for rep in range(start, n):
        blocks, profile = generate_dataset(
            model, weights, seed_base + rep, chains, length
        )
        rows.append(mine_dataset(blocks))
        profiles.append(profile)
        pair_moments = model.empirical_pair_moments(np.vstack(blocks))
        pair_rmse.append(float(np.sqrt(np.mean(
            (pair_moments - target) ** 2
        ))))
        completed = rep + 1
        if progress_path is not None and (
            completed % max(1, checkpoint_every) == 0 or completed == n
        ):
            temporary = Path(str(progress_path) + ".tmp")
            metric_rows_frame(rows).to_csv(temporary, index=False)
            temporary.replace(progress_path)
            diagnostic_tmp = Path(str(diagnostics_path) + ".tmp.npz")
            np.savez_compressed(
                diagnostic_tmp, lag_profiles=np.asarray(profiles),
                pair_rmse=np.asarray(pair_rmse),
            )
            diagnostic_tmp.replace(diagnostics_path)
            meta_path.write_text(json.dumps({
                "status": "completed" if completed == n else "running",
                "signature": signature, "completed": completed,
            }, indent=2))
            print(
                f"{progress_label}: {completed}/{n} inner simulations",
                flush=True,
            )
    return rows, np.asarray(profiles), np.asarray(pair_rmse)


def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def weight_screen_frame(screen):
    return pd.DataFrame([
        {"candidate":i,"w0":float(r["weights"][0]),
         "w1":float(r["weights"][1]),"w2":float(r["weights"][2]),
         "w3":float(r["weights"][3]),"lag_rmse":float(r["lag_rmse"]),
         "pair_rmse":float(r["pair_rmse"])}
        for i,r in enumerate(screen)
    ])


def metric_rows_frame(rows):
    return pd.DataFrame([
        {"inner_replicate":i,**{k:float(v) for k,v in row.items()}}
        for i,row in enumerate(rows)
    ])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",default="/content/drive/MyDrive/FCA_MoE_v03/qwen15_moe_a27_wikitext_test/contexts_v1")
    ap.add_argument("--output",default=None)
    ap.add_argument("--outer-replicates",type=int,default=20)
    ap.add_argument("--inner-replicates",type=int,default=30)
    ap.add_argument("--screen-replicates",type=int,default=2)
    ap.add_argument("--weight-grid-step",type=float,default=.5)
    ap.add_argument("--lambdas",nargs="+",type=float,default=[.0003,.001,.003])
    ap.add_argument("--maxiter",type=int,default=400)
    ap.add_argument("--seed",type=int,default=20261120)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--checkpoint-every",type=int,default=1)
    ap.add_argument("--screen-checkpoint-every", type=int, default=25)
    ap.add_argument("--inner-checkpoint-every", type=int, default=10)
    ap.add_argument("--outer-chunk-size", type=int, default=25)
    ap.add_argument("--temporal-loo-summary",default=None,
                    help="Completed leave-one-sequence-out temporal validation JSON; required for confirmatory runs.")
    args=ap.parse_args()
    if args.outer_replicates < 2: raise ValueError("Use at least two outer replicates; 999 recommended for final inference")
    if args.inner_replicates < 10: raise ValueError("Use at least 10 inner replicates; 100-300 recommended for final inference")
    if args.screen_replicates < 1: raise ValueError("screen-replicates must be >= 1")
    if not args.lambdas or any(x <= 0 for x in args.lambdas): raise ValueError("All lambda candidates must be positive")
    root=Path(args.root); out=Path(args.output) if args.output else root/"sequence_replication/layer_00/temporal/f2e_nested_bootstrap"
    out.mkdir(parents=True,exist_ok=True)
    contexts_path=root/"qwen_contexts_by_layer.npz"
    alignment_path=root/"sequence_alignment.csv"
    if not contexts_path.exists() or not alignment_path.exists():
        raise FileNotFoundError("Missing qwen_contexts_by_layer.npz or sequence_alignment.csv")
    script_sha=sha256_file(Path(__file__))
    contexts_sha=sha256_file(contexts_path)
    alignment_sha=sha256_file(alignment_path)
    loo_summary=None
    loo_summary_sha=None
    loo_pass=False
    if args.temporal_loo_summary:
        loo_path=Path(args.temporal_loo_summary)
        if not loo_path.exists():
            raise FileNotFoundError(f"Temporal LOO summary not found: {loo_path}")
        loo_summary=json.loads(loo_path.read_text())
        loo_summary_sha=sha256_file(loo_path)
        loo_pass=bool(
            loo_summary.get("status")=="completed"
            and loo_summary.get("predeclared_loo_diagnostic_pass") is True
        )
    if args.outer_replicates>=999 and not loo_pass:
        raise RuntimeError(
            "Confirmatory bootstrap blocked: completed temporal LOO validation "
            "must pass the predeclared criterion (mean held-out transition-NLL "
            "gain > 0 and selected temporal kernel better than IID in >=3/4 folds)."
        )
    run_config={"root":str(root.resolve()),"outer_replicates":args.outer_replicates,
                "inner_replicates":args.inner_replicates,"screen_replicates":args.screen_replicates,
                "weight_grid_step":args.weight_grid_step,"lambdas":[float(x) for x in args.lambdas],
                "maxiter":args.maxiter,"seed":args.seed,
                "script_sha256":script_sha,"contexts_sha256":contexts_sha,
                "alignment_sha256":alignment_sha,
                "temporal_loo_summary_sha256":loo_summary_sha}
    fingerprint=hashlib.sha256(json.dumps(run_config,sort_keys=True).encode()).hexdigest()
    save_json(out/"frozen_manifest.json",{
        "model":"Qwen/Qwen1.5-MoE-A2.7B",
        "model_revision":"1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
        "dataset":"Salesforce/wikitext",
        "dataset_config":"wikitext-2-raw-v1",
        "dataset_split":"test",
        "dataset_revision":"main",
        "routing_context":"layer_00",
        "experts":60,"top_k":4,"sequences":4,"tokens_per_sequence":256,
        "contexts_file":{"name":contexts_path.name,"sha256":contexts_sha},
        "alignment_file":{"name":alignment_path.name,"sha256":alignment_sha},
        "analysis_script_sha256":script_sha,
        "temporal_loo_summary":loo_summary,
        "environment":{
            "python":platform.python_version(),
            "numpy":np.__version__,"pandas":pd.__version__,
            "scipy":scipy.__version__
        },
        "configuration":run_config
    })
    config_path=out/"run_config.json"
    if config_path.exists():
        prior=json.loads(config_path.read_text())
        if prior.get("fingerprint")!=fingerprint:
            raise RuntimeError("Run configuration differs from the checkpoint. Use a new --output directory; never resume with changed statistical settings.")
        if not args.resume and (out/"outer_bootstrap_replicates.csv").exists():
            raise RuntimeError("Checkpoint exists. Use --resume with identical arguments or choose a new --output directory.")
    else:
        if args.resume and (out/"outer_bootstrap_replicates.csv").exists():
            raise RuntimeError("Cannot resume: run_config.json is missing. Start a new output directory.")
        save_json(config_path,{"fingerprint":fingerprint,"configuration":run_config})
    blocks=load_observed(root); obs_metrics=evaluate_context(blocks)
    # Empirical lag profile from observed binary Top-4 incidence matrices.
    prof=[]
    for b in blocks:
        p=[]
        for lag in range(1,LAGS+1): p.append(float((b[:-lag]*b[lag:]).sum(axis=1).mean()))
        prof.append(p)
    obs_profile=np.mean(prof,axis=0)

    t0 = time.time()
    print("Fitting observed pairwise MaxEnt model and selecting temporal kernel.", flush=True)
    model, lam, cv = fit_model(blocks, args.lambdas, args.maxiter)
    target_pair = model.fitted_pair_moments
    observed_screen_path = out / "observed_weight_screen.partial.csv"
    observed_final_screen = out / "observed_weight_screen.csv"
    observed_fit_path = out / "observed_fit_and_selection.json"
    theta_hash = _array_sha256(model.theta)
    reuse_observed_screen = False
    if args.resume and observed_final_screen.exists() and observed_fit_path.exists():
        cached_observed = json.loads(observed_fit_path.read_text())
        if (
            cached_observed.get("theta_sha256") == theta_hash
            and cached_observed.get("target_profile_sha256")
            == _array_sha256(obs_profile)
            and cached_observed.get("target_pair_sha256")
            == _array_sha256(target_pair)
        ):
            weights = np.asarray(cached_observed["weights_selected"], dtype=float)
            frame = pd.read_csv(observed_final_screen)
            screen = [{
                "weights": np.array([
                    item["w0"], item["w1"], item["w2"], item["w3"]
                ], dtype=float),
                "lag_rmse": float(item["lag_rmse"]),
                "pair_rmse": float(item["pair_rmse"]),
            } for item in frame.to_dict("records")]
            reuse_observed_screen = True
            print("Reusing completed observed temporal screen.", flush=True)
    if not reuse_observed_screen:
        weights, screen = select_weights(
            model, obs_profile, target_pair, args.weight_grid_step,
            args.screen_replicates, seed_base=args.seed + 100000,
            progress_path=observed_screen_path, resume=args.resume,
            progress_every=args.screen_checkpoint_every,
            progress_label="Observed temporal screen",
        )
    np.savez_compressed(
        out / "observed_model_parameters.npz", theta=model.theta,
        fitted_pair_moments=model.fitted_pair_moments,
        state_probabilities=model.prob, lambda_selected=np.asarray(lam)
    )
    pd.DataFrame([
        {"lambda": r["lambda"],
         "mean_validation_nll": r["mean_validation_nll"],
         **{f"fold_{i}_nll": v for i, v in enumerate(r["fold_nll"])}}
        for r in cv
    ]).to_csv(out / "observed_lambda_cv.csv", index=False)
    pd.DataFrame([
        {"w0": r["weights"][0], "w1": r["weights"][1],
         "w2": r["weights"][2], "w3": r["weights"][3],
         "lag_rmse": r["lag_rmse"], "pair_rmse": r["pair_rmse"]}
        for r in screen
    ]).to_csv(observed_final_screen, index=False)
    save_json(observed_fit_path, {
        "lambda_selected": lam, "lambda_cv": cv, "fit": model.fit_info,
        "theta_sha256": theta_hash,
        "target_profile_sha256": _array_sha256(obs_profile),
        "target_pair_sha256": _array_sha256(target_pair),
        "weights_selected": weights.tolist(),
        "weight_screen_top10": [
            {"weights": x["weights"].tolist(), "lag_rmse": x["lag_rmse"],
             "pair_rmse": x["pair_rmse"]} for x in screen[:10]
        ],
        "observed_lag_profile": obs_profile.tolist(),
        "observed_metrics": obs_metrics,
        "weight_grid_step": args.weight_grid_step,
        "screen_replicates": args.screen_replicates,
    })
    # Remove the partial files only after the complete selection is durably saved.
    if observed_screen_path.exists():
        observed_screen_path.unlink()
    observed_sidecar = Path(str(observed_screen_path) + ".json")
    if observed_sidecar.exists():
        observed_sidecar.unlink()

    # Observed statistic uses an independent, deterministic inner null sample.
    obs_inner, obs_profiles, obs_pair = run_inner(
        model, weights, args.inner_replicates, args.seed + 200000,
        progress_path=out / "observed_inner_metrics.csv",
        resume=args.resume,
        checkpoint_every=args.inner_checkpoint_every,
        progress_label="Observed inner null",
    )
    np.savez_compressed(
        out / "observed_inner_diagnostics.npz",
        lag_profiles=obs_profiles, pair_rmse=obs_pair
    )
    obs_stat=max_t_from_inner(obs_metrics,obs_inner)
    save_json(out/"observed_inner_calibration.json",{
        "observed_max_t":obs_stat["max_t"],"observed_tail_scores":obs_stat["z"],
        "observed_tail_diagnostics":obs_stat["tail"],
        "statistic":obs_stat["statistic"],
        "inner_means":obs_stat["means"],"inner_sds":obs_stat["sds"],
        "active_metrics":obs_stat["active"],"constant_metrics":obs_stat["constant"],
        "constant_metric_checks":obs_stat["constant_checks"],
        "inner_replicates":args.inner_replicates,
        "pair_rmse_mean":float(obs_pair.mean()),"lag_profile_mean":obs_profiles.mean(0).tolist()})

    records=[]; csv=out/"outer_bootstrap_replicates.csv"
    if args.resume and csv.exists():
        old=pd.read_csv(csv)
        # Successful replicates are immutable checkpoints; failed replicates are retried.
        records=[r for r in old.to_dict("records") if r.get("status")=="ok"]
        done={int(r["replicate"]) for r in records}
        old_fail=old[old["status"]!="ok"] if "status" in old.columns else pd.DataFrame()
        if len(old_fail): old_fail.to_csv(out/"outer_bootstrap_prior_failures.csv",index=False)
    else: done=set()
    for b in range(args.outer_replicates):
        if b in done: continue
        try:
            print(f"Starting outer replicate {b + 1}/{args.outer_replicates}", flush=True)
            # Outer sample from the original fitted null.
            outer_blocks, _ = generate_dataset(model, weights, args.seed + 300000 + b)
            mb, lambdab, cvb = fit_model(outer_blocks, args.lambdas, args.maxiter)
            prof_b = [[
                float((block[:-lag] * block[lag:]).sum(axis=1).mean())
                for lag in range(1, LAGS + 1)
            ] for block in outer_blocks]
            target_prof_b = np.mean(prof_b, axis=0)

            screen_dir = out / "outer_weight_screens"
            screen_dir.mkdir(exist_ok=True)
            partial_screen = screen_dir / f"replicate_{b:04d}.partial.csv"
            final_screen = screen_dir / f"replicate_{b:04d}.csv"
            outer_fit_path = out / f"outer_fit_{b:04d}.json"
            theta_hash = _array_sha256(mb.theta)
            reuse_screen = False
            if args.resume and final_screen.exists() and outer_fit_path.exists():
                cached_fit = json.loads(outer_fit_path.read_text())
                if (
                    cached_fit.get("theta_sha256") == theta_hash
                    and cached_fit.get("target_profile_sha256")
                    == _array_sha256(target_prof_b)
                ):
                    wb = np.asarray(cached_fit["selected_weights"], dtype=float)
                    frame = pd.read_csv(final_screen)
                    screenb = [{
                        "weights": np.array([
                            item["w0"], item["w1"], item["w2"], item["w3"]
                        ], dtype=float),
                        "lag_rmse": float(item["lag_rmse"]),
                        "pair_rmse": float(item["pair_rmse"]),
                    } for item in frame.to_dict("records")]
                    reuse_screen = True
                    print(f"Outer {b + 1}: reusing completed weight screen.", flush=True)
            if not reuse_screen:
                wb, screenb = select_weights(
                    mb, target_prof_b, mb.fitted_pair_moments,
                    args.weight_grid_step, args.screen_replicates,
                    seed_base=args.seed + 400000 + b * 10000,
                    progress_path=partial_screen, resume=args.resume,
                    progress_every=args.screen_checkpoint_every,
                    progress_label=f"Outer {b + 1} temporal screen",
                )
                weight_screen_frame(screenb).to_csv(final_screen, index=False)
                if partial_screen.exists():
                    partial_screen.unlink()
                sidecar = Path(str(partial_screen) + ".json")
                if sidecar.exists():
                    sidecar.unlink()

            # Persist the completed fit and temporal screen before inner simulations,
            # so an interrupted inner batch can resume without repeating screening.
            save_json(outer_fit_path, {
                "lambda_cv": cvb, "fit": mb.fit_info,
                "theta_sha256": theta_hash,
                "target_profile_sha256": _array_sha256(target_prof_b),
                "selected_weights": wb.tolist(),
                "weight_screen_file": f"outer_weight_screens/replicate_{b:04d}.csv",
            })
            outer_metrics = mine_dataset(outer_blocks)
            inner_dir = out / "outer_inner_metrics"
            inner_dir.mkdir(exist_ok=True)
            inner_path = inner_dir / f"replicate_{b:04d}.csv"
            inner_b, profiles_b, pair_b = run_inner(
                mb, wb, args.inner_replicates, args.seed + 500000 + b * 1000,
                progress_path=inner_path, resume=args.resume,
                checkpoint_every=args.inner_checkpoint_every,
                progress_label=f"Outer {b + 1} inner null",
            )
            stat_b = max_t_from_inner(
                outer_metrics, inner_b, active_family=obs_stat["active"]
            )
            param_dir = out / "outer_model_parameters"
            param_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                param_dir / f"replicate_{b:04d}.npz", theta=mb.theta,
                fitted_pair_moments=mb.fitted_pair_moments,
                lambda_selected=np.asarray(lambdab), weights=wb
            )
            np.savez_compressed(
                inner_dir / f"replicate_{b:04d}_diagnostics.npz",
                lag_profiles=profiles_b, pair_rmse=pair_b
            )
            row={"replicate":b,"status":"ok","lambda":lambdab,
                 "weights":json.dumps(wb.tolist()),"max_t":stat_b["max_t"],
                 "active_metrics":json.dumps(stat_b["active"]),
                 "pair_rmse_mean":float(pair_b.mean()),
                 "lag_profile_rmse":float(np.sqrt(np.mean((profiles_b.mean(0)-target_prof_b)**2))),
                 "fit_grad_inf":mb.fit_info["gradient_inf_norm"]}
            for name,value in outer_metrics.items():
                row[f"outer_{name}"]=value
            for name in FAMILY:
                row[f"observed_{name}"]=outer_metrics[name]
                row[f"inner_mean_{name}"]=stat_b["means"][name]
                row[f"inner_sd_{name}"]=stat_b["sds"][name]
            records.append(row)
        except Exception as e:
            records.append({"replicate":b,"status":"failed","error":repr(e)})
        if (b + 1) % args.checkpoint_every == 0 or b + 1 == args.outer_replicates:
            checkpoint_frame = pd.DataFrame(records).sort_values("replicate")
            checkpoint_frame.to_csv(csv, index=False)
            chunk_size = max(1, int(args.outer_chunk_size))
            if (b + 1) % chunk_size == 0 or b + 1 == args.outer_replicates:
                chunk_dir = out / "outer_chunks"
                chunk_dir.mkdir(exist_ok=True)
                first = (b // chunk_size) * chunk_size
                chunk = checkpoint_frame[
                    (checkpoint_frame["replicate"] >= first)
                    & (checkpoint_frame["replicate"] <= b)
                ]
                chunk.to_csv(
                    chunk_dir / f"outer_{first:04d}_{b:04d}.csv", index=False
                )
            save_json(out / "progress.json", {
                "stage": "outer_bootstrap",
                "last_replicate_completed": b,
                "outer_requested": args.outer_replicates,
                "successful": int(sum(r.get("status") == "ok" for r in records)),
                "failed": int(sum(r.get("status") == "failed" for r in records)),
                "updated_unix_time": time.time(),
            })
            print(
                f"Outer bootstrap {b+1}/{args.outer_replicates}; "
                f"successful={sum(r.get('status')=='ok' for r in records)}; "
                f"failed={sum(r.get('status')=='failed' for r in records)}",
                flush=True,
            )

    ok=[r for r in records if r.get("status")=="ok"]
    tvals=np.asarray([r["max_t"] for r in ok if np.isfinite(r["max_t"])],float)
    if len(tvals):
        exceed=int(np.sum(tvals>=obs_stat["max_t"]))
        critical=float(np.quantile(tvals,.95,method="higher"))
        all_success=(len(ok)==args.outer_replicates and len(tvals)==args.outer_replicates)
        publication_config=(args.outer_replicates>=999 and args.inner_replicates>=100
                            and args.screen_replicates>=8 and args.weight_grid_step<=0.1+1e-12
                            and len(args.lambdas)>=3)
        ready=bool(all_success and publication_config)
        result={"status":"completed" if all_success else "incomplete_or_failed_replicates",
                "inferential_status":"ready_for_review" if ready else ("pilot_only_not_inferential" if all_success else "not_valid_for_confirmatory_inference"),
                "outer_requested":args.outer_replicates,
                "outer_success":len(ok),"outer_failed":len(records)-len(ok),
                "observed_max_t":obs_stat["max_t"],"bootstrap_critical_95":critical,
                "global_p_plus_one":float((1+exceed)/(len(tvals)+1)) if ready else None,
                "diagnostic_p_plus_one_successful_only":float((1+exceed)/(len(tvals)+1)),
                "exceedances":exceed,
                "global_excess_detected":bool(obs_stat["max_t"]>critical) if ready else None,
                "null_envelope_pass":bool(obs_stat["max_t"]<=critical) if ready else None,
                "publication_configuration_gate":publication_config,
                "outer_max_t_mean":float(tvals.mean()),"outer_max_t_sd":float(tvals.std(ddof=1)) if len(tvals)>1 else None,
                "family":FAMILY,
                "temporal_loo_validation":loo_summary,
                "temporal_loo_precondition_passed":loo_pass,
                "interpretation":"Nested parametric bootstrap conditional on the fitted pairwise-temporal null family; validity depends on adequacy of that family. Any failed outer replicate invalidates confirmatory reporting until resolved.",
                "configuration":run_config,"configuration_fingerprint":fingerprint,
                "elapsed_seconds":time.time()-t0}
    else:
        result={"status":"no_successful_outer_replicates","inferential_status":"not_valid_for_confirmatory_inference","outer_requested":args.outer_replicates,"outer_failed":len(records)}
    save_json(out/"f2e_nested_bootstrap_summary.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()

