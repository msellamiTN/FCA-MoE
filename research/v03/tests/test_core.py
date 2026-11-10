import numpy as np
from moe_fca.contexts import token_context, window_context, chronological_split
from moe_fca.fca import next_closure
from moe_fca.maxent_exact import fit_pairwise_maxent, sample_exact

def test_contexts_and_split():
    routes = np.array([[0,1],[1,2],[0,2],[2,3]] * 8)
    T = token_context(routes, 4)
    assert T.shape == (32, 4)
    assert np.all(T.sum(axis=1) == 2)
    W = window_context(routes, 4, .25, 4)
    assert W.shape == (8, 4)
    a, b, c = chronological_split(T)
    assert len(a) + len(b) + len(c) == len(T)

def test_fca_returns_closed_intents():
    C = np.array([[1,0],[1,0],[0,1]], dtype=np.uint8)
    intents = next_closure(C)
    assert len(intents) >= 1
    assert len(set(intents)) == len(intents)

def test_exact_maxent_oracle():
    X = np.array([[0,0],[0,1],[1,0],[1,1]] * 10, dtype=np.uint8)
    fit = fit_pairwise_maxent(X)
    Y = sample_exact(fit, 1000, seed=1)
    assert Y.shape == (1000, 2)
    assert fit["moment_max_abs_error"] < 1e-4
