import unittest,sqlite3,json
from forgerl.benchmark_stats import paired_report
class StatsTests(unittest.TestCase):
    def test_pairs(self):
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE episodes(phase,task,sample,evaluation)')
        a=dict(correctness=0,intent_preservation=0,action_selection=0,execution_ms=10,tool_operations=3,tokens=20)
        b=dict(a,correctness=1)
        for phase,row in [('heldout_baseline',a),('heldout_candidate',b)]:
            db.execute('INSERT INTO episodes VALUES (?,?,?,?)',(phase,'task',0,json.dumps(row)))
        r=paired_report(db,100)
        self.assertEqual(r['pairs'],1)
        self.assertEqual(r['metrics']['correctness']['delta'],1)
        db.close()
