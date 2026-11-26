import unittest
from forgerl.pipeline import normalize
from forgerl.export import export
class Tests(unittest.TestCase):
    def test_advantages(self):
        rows=normalize([{'task_id':'a','policy':'p','reward':x} for x in [0,1]])
        self.assertAlmostEqual(rows[0]['advantage'],-1); self.assertAlmostEqual(rows[1]['advantage'],1)
    def test_errors_excluded(self):
        rows=normalize([{'task_id':'a','error':'timeout'},{'task_id':'a','policy':'p','reward':1}]); self.assertNotIn('advantage',rows[0]); self.assertEqual(rows[1]['advantage'],0)
    def test_export(self): self.assertEqual(list(export([{'reward':0},{'reward':1,'schema_version':1}])),[])
