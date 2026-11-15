#!/usr/bin/env python3
"""FCA-MoE F2-E: nested parametric bootstrap for temporal pairwise-MaxEnt FCA.

The procedure is deliberately explicit: fit the fixed-cardinality pairwise
MaxEnt model, select the temporal Gibbs mixture by the same lag-profile screen,
then compare FCA higher-order counts to an inner null. Outer bootstrap samples
repeat fitting, lambda selection, and temporal-weight selection.

This is a research implementation, not a claim that the experiment has run.
"""
from __future__ import annotations
import argparse, itertools, json, math, os, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
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


class ConditionalPairwiseMaxEnt:
    """Pairwise exponential family conditioned on exactly K active experts."""
    def __init__(self, E=60, K=4, states=None):
        self.E, self.K = int(E), int(K)
        self.states = all_states(E, K) if states is None else np.asarray(states, np.int16)
        self.pairs, self.lut = pair_index_table(E)
        self.state_pair = state_pair_indices(self.states, self.lut)
        self.n_params = len(self.pairs)
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
        """Build exact conditional CDFs for retaining 1,2,3 experts."""
        self.cond = {}
        for k in (1, 2, 3):
            positions = list(itertools.combinations(range(self.K), k))
            key_parts = []
            id_parts = []
            base = np.arange(len(self.states), dtype=np.int32)
            for pos in positions:
                vals = self.states[:, pos]
                masks = np.zeros(len(vals), dtype=np.uint64)
                for j in range(k): masks |= np.left_shift(np.uint64(1), vals[:, j].astype(np.uint64))
                key_parts.append(masks); id_parts.append(base)
            keys = np.concatenate(key_parts); ids = np.concatenate(id_parts)
            order = np.argsort(keys, kind="stable"); keys, ids = keys[order], ids[order]
            uniq, starts, counts = np.unique(keys, return_index=True, return_counts=True)
            ends = starts + counts
            cdf = np.empty(len(ids), dtype=np.float64)
            weights = self.prob[ids]
            for a, b in zip(starts, ends):
                q = weights[a:b]; total = q.sum()
                if total <= 0 or not np.isfinite(total): raise RuntimeError("Invalid conditional mass")
                cdf[a:b] = np.cumsum(q / total); cdf[b-1] = 1.0
            self.cond[k] = {"keys": uniq, "starts": starts, "ends": ends,
                            "ids": ids, "cdf": cdf,
                            "lookup": {int(v):i for i,v in enumerate(uniq)}}
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
        if k == 0: return self.draw_global(rng)
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
    pooled=np.vstack(blocks); row={}
    for tag,frac in zip(TAGS,THRESHOLDS):
        minp=max(1,int(math.ceil(frac*len(pooled)-1e-12)))
        pooled_mined=FCA.mine(pooled,minp)
        row[f"pooled_higher_order_{tag}"]=pooled_mined["higher"]
        seq=[FCA.mine(b,max(1,int(math.ceil(frac*len(b)-1e-12))))["higher"] for b in blocks]
        row[f"mean_sequence_higher_order_{tag}"]=float(np.mean(seq))
        row[f"pooled_triplets_{tag}"]=pooled_mined["triplets"]
        row[f"pooled_quadruplets_{tag}"]=pooled_mined["quadruplets"]
    return row


def select_weights(model, target_profile, target_pair, grid_step=.1, screen_replicates=8,
                   chains=4,length=256,seed_base=70000000):
    rows=[]; candidates=list(grid_weights(grid_step))
    for ci,w in enumerate(candidates):
        profiles=[]; pair_errors=[]
        for r in range(screen_replicates):
            blocks,prof=generate_dataset(model,w,seed_base+ci*1000+r,chains,length)
            profiles.append(prof)
            X=np.vstack(blocks)
            pm=model.empirical_pair_moments(X)
            pair_errors.append(float(np.sqrt(np.mean((pm-target_pair)**2))))
        mean=np.mean(profiles,axis=0)
        rows.append({"weights":w,"lag_rmse":float(np.sqrt(np.mean((mean-target_profile)**2))),
                     "pair_rmse":float(np.mean(pair_errors))})
    rows.sort(key=lambda r:r["lag_rmse"])
    return rows[0]["weights"], rows


def max_t_from_inner(observed_metrics, inner_metrics, active_family=None):
    """Studentize observed FCA counts using an independent inner-null sample.

    The inner sample estimates each metric's null mean and SD. It is NOT reused
    as a reference distribution for its own standardized residuals; the outer
    bootstrap supplies the reference distribution for the global statistic.
    """
    means={n:float(np.mean([r[n] for r in inner_metrics])) for n in FAMILY}
    sds={n:float(np.std([r[n] for r in inner_metrics],ddof=1)) for n in FAMILY}
    if active_family is None:
        active=[n for n in FAMILY if sds[n] > 1e-12]
    else:
        active=list(active_family)
        bad=[n for n in active if sds[n] <= 1e-12]
        if bad: raise RuntimeError(f"Inner null is degenerate for predeclared active metrics: {bad}")
    constants=[n for n in FAMILY if n not in active]
    if not active: raise RuntimeError("All predeclared FCA metrics are degenerate in the inner null sample")
    zobs={n:(float(observed_metrics[n])-means[n])/sds[n] for n in active}
    constant_checks={n:{"null_constant":means[n],"observed":float(observed_metrics[n]),
                        "observed_exceeds":bool(observed_metrics[n]>means[n])}
                     for n in constants}
    if active_family is None and any(v["observed_exceeds"] for v in constant_checks.values()):
        raise RuntimeError(f"Observed metric exceeds a degenerate null support; cannot exclude it from max-T: {constant_checks}")
    return {"max_t":max(zobs.values()) if zobs else None,"z":zobs,
            "means":means,"sds":sds,"active":active,"constant":constants,
            "constant_checks":constant_checks}


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


def run_inner(model,weights,n,seed_base,chains=4,length=256):
    rows=[]; profiles=[]; pair_rmse=[]
    target=model.fitted_pair_moments
    for r in range(n):
        blocks,prof=generate_dataset(model,weights,seed_base+r,chains,length)
        row=mine_dataset(blocks); rows.append(row); profiles.append(prof)
        pm=model.empirical_pair_moments(np.vstack(blocks))
        pair_rmse.append(float(np.sqrt(np.mean((pm-target)**2))))
    return rows,np.asarray(profiles),np.asarray(pair_rmse)


def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))


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
    args=ap.parse_args()
    if args.outer_replicates < 2: raise ValueError("Use at least two outer replicates; 999 recommended for final inference")
    if args.inner_replicates < 10: raise ValueError("Use at least 10 inner replicates; 100-300 recommended for final inference")
    if args.screen_replicates < 1: raise ValueError("screen-replicates must be >= 1")
    if not args.lambdas or any(x <= 0 for x in args.lambdas): raise ValueError("All lambda candidates must be positive")
    root=Path(args.root); out=Path(args.output) if args.output else root/"sequence_replication/layer_00/temporal/f2e_nested_bootstrap"
    out.mkdir(parents=True,exist_ok=True)
    run_config={"root":str(root.resolve()),"outer_replicates":args.outer_replicates,
                "inner_replicates":args.inner_replicates,"screen_replicates":args.screen_replicates,
                "weight_grid_step":args.weight_grid_step,"lambdas":[float(x) for x in args.lambdas],
                "maxiter":args.maxiter,"seed":args.seed}
    fingerprint=hashlib.sha256(json.dumps(run_config,sort_keys=True).encode()).hexdigest()
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

    t0=time.time(); model,lam,cv=fit_model(blocks,args.lambdas,args.maxiter)
    target_pair=model.fitted_pair_moments
    weights,screen=select_weights(model,obs_profile,target_pair,args.weight_grid_step,
                                  args.screen_replicates,seed_base=args.seed+100000)
    np.savez_compressed(out/"observed_model_parameters.npz", theta=model.theta,
                        fitted_pair_moments=model.fitted_pair_moments,
                        state_probabilities=model.prob, lambda_selected=np.asarray(lam))
    pd.DataFrame([{**{"lambda":r["lambda"],"mean_validation_nll":r["mean_validation_nll"]},
                   **{f"fold_{i}_nll":v for i,v in enumerate(r["fold_nll"])} } for r in cv]).to_csv(out/"observed_lambda_cv.csv",index=False)
    pd.DataFrame([{"w0":r["weights"][0],"w1":r["weights"][1],"w2":r["weights"][2],"w3":r["weights"][3],
                   "lag_rmse":r["lag_rmse"],"pair_rmse":r["pair_rmse"]} for r in screen]).to_csv(out/"observed_weight_screen.csv",index=False)
    save_json(out/"observed_fit_and_selection.json",{
        "lambda_selected":lam,"lambda_cv":cv,"fit":model.fit_info,
        "weights_selected":weights.tolist(),"weight_screen_top10":[{"weights":x["weights"].tolist(),"lag_rmse":x["lag_rmse"],"pair_rmse":x["pair_rmse"]} for x in screen[:10]],
        "observed_lag_profile":obs_profile.tolist(),"observed_metrics":obs_metrics,
        "weight_grid_step":args.weight_grid_step,"screen_replicates":args.screen_replicates})

    # Observed statistic uses an independent inner null sample.
    obs_inner,obs_profiles,obs_pair=run_inner(model,weights,args.inner_replicates,args.seed+200000)
    obs_stat=max_t_from_inner(obs_metrics,obs_inner)
    save_json(out/"observed_inner_calibration.json",{
        "observed_max_t":obs_stat["max_t"],"observed_z":obs_stat["z"],
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
            # Outer sample from the original fitted null.
            outer_blocks,_=generate_dataset(model,weights,args.seed+300000+b)
            # Refit pairwise MaxEnt and reselect regularization/temporal weights.
            mb,lambdab,cvb=fit_model(outer_blocks,args.lambdas,args.maxiter)
            prof_b=[]
            for block in outer_blocks:
                p=[float((block[:-lag]*block[lag:]).sum(axis=1).mean()) for lag in range(1,LAGS+1)]
                prof_b.append(p)
            target_prof_b=np.mean(prof_b,axis=0)
            wb,screenb=select_weights(mb,target_prof_b,mb.fitted_pair_moments,
                                      args.weight_grid_step,args.screen_replicates,
                                      seed_base=args.seed+400000+b*10000)
            outer_metrics=mine_dataset(outer_blocks)
            inner_b,profiles_b,pair_b=run_inner(mb,wb,args.inner_replicates,
                                                 args.seed+500000+b*1000)
            stat_b=max_t_from_inner(outer_metrics,inner_b,active_family=obs_stat["active"])
            param_dir=out/"outer_model_parameters"; param_dir.mkdir(exist_ok=True)
            np.savez_compressed(param_dir/f"replicate_{b:04d}.npz",theta=mb.theta,
                                fitted_pair_moments=mb.fitted_pair_moments,
                                lambda_selected=np.asarray(lambdab),weights=wb)
            save_json(out/f"outer_fit_{b:04d}.json",{"lambda_cv":cvb,"fit":mb.fit_info,
                      "selected_weights":wb.tolist(),"weight_screen_top10":[{"weights":x["weights"].tolist(),"lag_rmse":x["lag_rmse"],"pair_rmse":x["pair_rmse"]} for x in screenb[:10]]})
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
        if (b+1)%args.checkpoint_every==0 or b+1==args.outer_replicates:
            pd.DataFrame(records).sort_values("replicate").to_csv(csv,index=False)
            print(f"Outer bootstrap {b+1}/{args.outer_replicates}; successful={sum(r.get('status')=='ok' for r in records)}; failed={sum(r.get('status')=='failed' for r in records)}",flush=True)

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
                "family":FAMILY,"interpretation":"Nested parametric bootstrap conditional on the fitted pairwise-temporal null family; validity depends on adequacy of that family. Any failed outer replicate invalidates confirmatory reporting until resolved.",
                "configuration":run_config,"configuration_fingerprint":fingerprint,
                "elapsed_seconds":time.time()-t0}
    else:
        result={"status":"no_successful_outer_replicates","inferential_status":"not_valid_for_confirmatory_inference","outer_requested":args.outer_replicates,"outer_failed":len(records)}
    save_json(out/"f2e_nested_bootstrap_summary.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()

