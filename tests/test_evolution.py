import unittest
from forgerl.evolution import tasks, episode, evaluate, diagnose, Harness

class EvolutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_fixture_intervention(self):
        t=next(tasks())
        a=evaluate(await episode(t,Harness(),'fixture','unused',0))
        b=evaluate(await episode(t,diagnose([a]),'fixture','unused',0))
        self.assertEqual(a['correctness'],b['correctness'])
        self.assertGreater(b['intent_preservation'],a['intent_preservation'])
        self.assertGreater(b['action_selection'],a['action_selection'])
        self.assertLess(b['tool_operations'],a['tool_operations'])
    def test_split(self):
        a={t['id'].split('-')[0] for t in tasks() if t['split']=='discovery'}
        b={t['id'].split('-')[0] for t in tasks() if t['split']=='heldout'}
        self.assertFalse(a & b)
