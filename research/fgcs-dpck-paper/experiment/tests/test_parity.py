"""Exhaustive executable check for the paper's Top-3 parity proposition."""
from itertools import product

EXPERTS = ("A0", "A1", "B0", "B1", "C0", "C1")
EVEN = (
    ("A0", "B0", "C0"),
    ("A0", "B1", "C1"),
    ("A1", "B0", "C1"),
    ("A1", "B1", "C0"),
)
ODD = (
    ("A0", "B0", "C1"),
    ("A0", "B1", "C0"),
    ("A1", "B0", "C0"),
    ("A1", "B1", "C1"),
)
CAPACITIES = (3, 2, 1)
SOURCE_DEVICE = 1


def feasible_placements():
    for assignment in product(range(len(CAPACITIES)), repeat=len(EXPERTS)):
        if tuple(assignment.count(d) for d in range(len(CAPACITIES))) == CAPACITIES:
            yield assignment


def mean_destination_cost(assignment, routes):
    index = {expert: i for i, expert in enumerate(EXPERTS)}
    costs = []
    for route in routes:
        contacted = {assignment[index[e]] for e in route} - {SOURCE_DEVICE}
        costs.append(len(contacted))
    return sum(costs) / len(costs)


def marginal_signature(routes):
    expert_counts = {e: sum(e in r for r in routes) for e in EXPERTS}
    pair_counts = {}
    for i, e1 in enumerate(EXPERTS):
        for e2 in EXPERTS[i + 1 :]:
            pair_counts[(e1, e2)] = sum(e1 in r and e2 in r for r in routes)
    return expert_counts, pair_counts


def test_parity_counterexample():
    placements = list(feasible_placements())
    assert len(placements) == 60
    assert marginal_signature(EVEN) == marginal_signature(ODD)

    even_costs = {p: mean_destination_cost(p, EVEN) for p in placements}
    odd_costs = {p: mean_destination_cost(p, ODD) for p in placements}
    even_opt = min(even_costs.values())
    odd_opt = min(odd_costs.values())
    even_optima = {p for p, value in even_costs.items() if value == even_opt}
    odd_optima = {p for p, value in odd_costs.items() if value == odd_opt}

    assert even_opt == 1.25
    assert odd_opt == 1.25
    assert len(even_optima) == 12
    assert len(odd_optima) == 12
    assert not (even_optima & odd_optima)
    assert {odd_costs[p] for p in even_optima} == {1.50}
    assert {even_costs[p] for p in odd_optima} == {1.50}


if __name__ == "__main__":
    test_parity_counterexample()
    print("PASS: 60 placements; pairwise signatures equal; 12 disjoint optima each; cross-cost 1.50.")