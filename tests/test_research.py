import unittest
from forgerl.research_tasks import suite
from forgerl.research import variants,valid_action
from forgerl.research_eval import evaluate
class ResearchTests(unittest.TestCase):
    def test_suite(self):
        ts=list(suite())
        self.assertEqual(len(ts),6)
        self.assertTrue(all(len(t['cases'])>=4 for t in ts))
        self.assertEqual(len({t['id'] for t in ts}),6)
    def test_variants(self):
        for source in variants().values(): compile(source,'candidate','exec')
        self.assertEqual(len(set(variants().values())),5)
    def test_actions(self):
        self.assertFalse(valid_action(None))
        self.assertFalse(valid_action({'tool':'patch'}))
        self.assertTrue(valid_action({'tool':'read'}))
    def test_failure_retained(self):
        r=evaluate(dict(events=[dict(action={'tool':'invalid'},observation={'error':'bad'})],checks=[],signature=False,update_delivered=False,elapsed_ms=1,tokens=0,status='error'))
        self.assertEqual(r['failed'],1)
        self.assertEqual(r['correctness'],0)
        self.assertEqual(r['action_selection'],0)
