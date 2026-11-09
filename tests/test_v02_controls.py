import unittest
import torch
from fca_moe.context import enumerate_concepts
from fca_moe.synthetic import EXPECTED_COMMUNITY_INTENTS, make_correlated_routing_data
from fca_moe.telemetry import RoutingTrace
from fca_moe.v02_experiment import run_controlled_case

class TestV02Controls(unittest.TestCase):
    def test_positive_oracle_context_has_ten_concepts_and_four_order4_intents(self):
        n, win = 4096, 64
        _, y, _ = make_correlated_routing_data(n_samples=n,d_model=32,n_experts=8,seed=17,window_size=win,mode="window_correlated")
        trace=RoutingTrace()
        for i,row in enumerate(y.tolist()):
            trace.add("layer0",[j for j,v in enumerate(row) if v],workload="oracle",window=i//win)
        try:
            ctx=trace.to_window_context(layer_id="layer0",activation_rate_threshold=.10,attributes=[f"E{i}" for i in range(8)])
        except TypeError:
            ctx=trace.to_window_context("layer0",.10,[f"E{i}" for i in range(8)])
        result=enumerate_concepts(ctx,min_support=0.0,min_intent_size=0)
        intents={frozenset(c.intent) for c in result.concepts}
        self.assertTrue(result.complete)
        self.assertEqual(len(result.concepts),10)
        self.assertTrue(all(g in intents for g in EXPECTED_COMMUNITY_INTENTS))

    def test_iid_control_is_exact_route_pair_marginal_shuffle(self):
        kwargs=dict(n_samples=4096,d_model=32,n_experts=8,seed=17,window_size=64)
        positive=make_correlated_routing_data(**kwargs,mode="window_correlated")[1]
        iid=make_correlated_routing_data(**kwargs,mode="iid")[1]
        p=sorted(tuple(torch.where(r>0)[0].tolist()) for r in positive)
        q=sorted(tuple(torch.where(r>0)[0].tolist()) for r in iid)
        self.assertEqual(p,q)

    def test_positive_control_heldout_router_and_fca(self):
        r=run_controlled_case(mode="window_correlated",n_samples=4096,window_size=64,activation_rate_threshold=.10,epochs=120,seed=17)
        self.assertGreaterEqual(r["route_exact_match_validation"],.98)
        self.assertTrue(r["concept_enumeration_complete"])
        self.assertEqual(r["concepts_total_unfiltered"],10)
        self.assertEqual(r["concepts_order4"],4)
        self.assertEqual(r["expected_order4_recall"],1.0)
        self.assertEqual(r["order4_precision"],1.0)

    def test_iid_negative_control_rejects_persistent_groups_at_baseline(self):
        r=run_controlled_case(mode="iid",n_samples=4096,window_size=64,activation_rate_threshold=.10,epochs=120,seed=17)
        self.assertGreaterEqual(r["route_exact_match_validation"],.98)
        self.assertTrue(r["concept_enumeration_complete"])
        self.assertEqual(r["concepts_total_unfiltered"],1)
        self.assertEqual(r["concepts_order4"],0)
        self.assertEqual(r["expected_order4_recall"],0.0)

if __name__=="__main__": unittest.main()
