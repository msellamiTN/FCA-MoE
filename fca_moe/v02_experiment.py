"""FCA-MoE v0.2: held-out positive/negative controls and sensitivity."""
from __future__ import annotations
import json, platform, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from .affinity import concept_derived_affinity, pairwise_coactivation
from .context import enumerate_concepts
from .model import TinyTopKMoE
from .synthetic import EXPECTED_COMMUNITY_INTENTS, make_correlated_routing_data
from .telemetry import RoutingTrace

def _seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _window_context(trace, threshold, attrs):
    try:
        return trace.to_window_context(layer_id="layer0", activation_rate_threshold=threshold, attributes=attrs)
    except TypeError:
        return trace.to_window_context("layer0", threshold, attrs)

def _token_context(trace, attrs):
    if hasattr(trace, "to_token_context"):
        try:
            return trace.to_token_context("layer0", attrs)
        except TypeError:
            return trace.to_token_context(layer_id="layer0", attributes=attrs)
    return trace.to_context(layer_id="layer0", attributes=attrs)

def _placement(attrs, fca_weights, pair_weights):
    costs = [[0.0, 1.0], [1.0, 0.0]]
    try:
        from .placement import greedy_place
        sizes = {e: 64 * 1024 * 1024 for e in attrs}
        caps = [4 * 64 * 1024 * 1024] * 2
        def place(w):
            p = greedy_place(attrs, w, costs, sizes, caps, balance_weight=0.01)
            return p.assignment, p.objective
    except ImportError:
        from .placement import greedy_virtual_placement
        sizes = {e: 1.0 for e in attrs}
        caps = [len(attrs) / 2] * 2
        def place(w):
            p = greedy_virtual_placement(w, sizes, caps, costs)
            return p["placement"], p["objective"]
    def cut(weights, assignment):
        total = sum(max(float(v), 0.0) for v in weights.values())
        if total <= 0:
            return None
        cross = sum(max(float(v), 0.0) for (a,b),v in weights.items() if assignment.get(a) != assignment.get(b))
        return cross / total
    fa, fo = place(fca_weights)
    pa, po = place(pair_weights)
    return {
        "fca": {"assignment": fa, "objective": fo, "normalized_weighted_cut": cut(fca_weights, fa)},
        "pairwise": {"assignment": pa, "objective": po, "normalized_weighted_cut": cut(pair_weights, pa)},
        "is_simulated": True,
        "warning": "Virtual topology simulation only; no physical multi-GPU communication measured."
    }

def run_controlled_case(
    *, mode, n_samples=4096, d_model=32, n_experts=8, window_size=64,
    activation_rate_threshold=0.10, epochs=120, train_ratio=0.80,
    seed=17, noise=0.20
):
    if not 0.5 <= train_ratio < 1.0:
        raise ValueError("train_ratio must be in [0.5,1)")
    if n_samples % window_size:
        raise ValueError("n_samples must be divisible by window_size")
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, _patterns = make_correlated_routing_data(
        n_samples=n_samples, d_model=d_model, n_experts=n_experts,
        seed=seed, noise=noise, window_size=window_size, mode=mode
    )
    total_windows = n_samples // window_size
    train_windows = int(total_windows * train_ratio)
    split = train_windows * window_size
    x_train, y_train = x[:split].to(device), y[:split].to(device)
    x_val, y_val = x[split:].to(device), y[split:].to(device)
    if not len(x_val):
        raise ValueError("validation split is empty")
    model = TinyTopKMoE(d_model=d_model, n_experts=n_experts, top_k=2).to(device)
    opt = torch.optim.AdamW(model.router.parameters(), lr=3e-2)
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model.router(x_train), y_train)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        logits = model.router(x_val)
        selected = logits.topk(2, dim=-1).indices
        expected = y_val.topk(2, dim=-1).indices
        exact = float((selected.sort(-1).values == expected.sort(-1).values).all(-1).float().mean().item())
    t0 = time.perf_counter()
    trace = RoutingTrace()
    for i, row in enumerate(selected.detach().cpu().tolist()):
        trace.add("layer0", row, workload=f"{mode}-heldout", window=train_windows + i // window_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    trace_s = time.perf_counter() - t0
    attrs = [f"E{i}" for i in range(n_experts)]
    token_ctx = _token_context(trace, attrs)
    ctx = _window_context(trace, activation_rate_threshold, attrs)
    t0 = time.perf_counter()
    enum = enumerate_concepts(ctx, min_support=0.0, min_intent_size=0)
    fca_s = time.perf_counter() - t0
    concepts = enum.concepts
    order4 = [c for c in concepts if len(c.intent) == 4 and len(c.extent) > 0]
    intents = {frozenset(c.intent) for c in concepts}
    recovered = [g for g in EXPECTED_COMMUNITY_INTENTS if g in intents]
    precision = len(recovered) / len(order4) if order4 else None
    unexpected = [list(c.intent) for c in order4 if frozenset(c.intent) not in EXPECTED_COMMUNITY_INTENTS]
    fca_w = concept_derived_affinity(ctx, enum)
    pair_w = pairwise_coactivation(ctx)
    placement = _placement(attrs, fca_w, pair_w)
    support = lambda c: getattr(c, "support", len(c.extent) / max(len(ctx.objects), 1))
    rows = getattr(ctx, "rows", getattr(ctx, "incidence", ()))
    result = {
        "experiment_version": "0.2.0", "mode": mode,
        "control_role": "positive" if mode == "window_correlated" else "negative_iid",
        "seed": seed, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "n_samples_total": n_samples, "n_train_samples": len(x_train),
        "n_validation_samples": len(x_val), "n_experts": n_experts, "top_k": 2,
        "window_size_tokens": window_size, "n_validation_windows": len(ctx.objects),
        "activation_rate_threshold": activation_rate_threshold, "epochs": epochs,
        "noise_std": noise, "training_seconds": train_s, "trace_seconds": trace_s,
        "fca_enumeration_seconds": fca_s, "route_exact_match_validation": exact,
        "token_context_objects": len(token_ctx.objects), "window_context_objects": len(ctx.objects),
        "window_context_rows": [sorted(r) for r in rows],
        "concepts_total_unfiltered": len(concepts), "concept_enumeration_complete": enum.complete,
        "concept_candidates_examined": enum.candidates_examined, "concept_stop_reason": enum.stop_reason,
        "concepts_order4": len(order4), "expected_order4_intent_count": len(EXPECTED_COMMUNITY_INTENTS),
        "expected_order4_recovered_count": len(recovered),
        "expected_order4_recall": len(recovered) / len(EXPECTED_COMMUNITY_INTENTS),
        "order4_precision": precision, "unexpected_order4_intents": unexpected,
        "concepts": [{"intent": list(c.intent), "extent_size": len(c.extent), "support": support(c)} for c in concepts],
        "expected_order4_intents": [sorted(g) for g in EXPECTED_COMMUNITY_INTENTS],
        "fca_affinity": {f"{a}|{b}":v for (a,b),v in fca_w.items()},
        "pairwise_affinity": {f"{a}|{b}":v for (a,b),v in pair_w.items()},
        "placement_comparison": placement, "virtual_placement_is_simulated": True,
        "warning": "Synthetic workload and held-out router traces; not pretrained-LLM or physical multi-GPU evidence."
    }
    if torch.cuda.is_available():
        result["gpu_memory_allocated_peak_bytes"] = torch.cuda.max_memory_allocated(0)
        result["gpu_memory_reserved_peak_bytes"] = torch.cuda.max_memory_reserved(0)
    return result

def run_sensitivity_grid(
    *, modes=("window_correlated","iid"), window_sizes=(32,64,128),
    thresholds=(0.10,0.20,0.30), n_samples=4096, epochs=120, seed=17, noise=0.20
):
    rows = []
    for mode in modes:
        for win in window_sizes:
            for tau in thresholds:
                r = run_controlled_case(
                    mode=mode, n_samples=n_samples, window_size=win,
                    activation_rate_threshold=tau, epochs=epochs, seed=seed, noise=noise
                )
                rows.append({
                    "mode": mode, "window_size": win, "threshold": tau,
                    "validation_top2_exact_match": r["route_exact_match_validation"],
                    "concepts_total": r["concepts_total_unfiltered"],
                    "concepts_order4": r["concepts_order4"],
                    "expected_order4_recall": r["expected_order4_recall"],
                    "order4_precision": r["order4_precision"],
                    "unexpected_order4_intents": len(r["unexpected_order4_intents"]),
                    "complete": r["concept_enumeration_complete"]
                })
    return rows

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
