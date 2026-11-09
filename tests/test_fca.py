import unittest
from fca_moe.context import FormalContext,enumerate_concepts
from fca_moe.telemetry import RoutingTrace
class TestFCA(unittest.TestCase):
 def setUp(self):
  tx=[{"E1","E2"}]*3+[{"E1","E3"},{"E2","E4"},{"E2","E4"},{"E3","E5"},{"E3","E5"}]
  self.ctx=FormalContext.from_transactions(tx,[f"E{i}" for i in range(1,6)],"T")
 def test_exact_concepts(self):
  r=enumerate_concepts(self.ctx);self.assertTrue(r.complete)
  self.assertEqual(len(r.concepts),9)
 def test_extents(self):
  self.assertEqual(len(self.ctx.extent({"E1","E2"})),3);self.assertEqual(len(self.ctx.extent({"E2","E4"})),2)
 def test_higher_order_window(self):
  t=RoutingTrace()
  for i in range(8):
   t.add("L0",[0,1] if i%2==0 else [2,3],window=0);t.add("L0",[2,3] if i%2==0 else [4,5],window=1)
  c=t.to_window_context("L0",.20,[f"E{i}" for i in range(6)]);r=enumerate_concepts(c); intents={x.intent for x in r.concepts}
  self.assertIn(frozenset({"E0","E1","E2","E3"}),intents);self.assertIn(frozenset({"E2","E3","E4","E5"}),intents)
 def test_truncation(self):
  r=enumerate_concepts(self.ctx,max_concepts=2);self.assertFalse(r.complete);self.assertIn("truncated",r.stop_reason)
if __name__=="__main__":unittest.main()
