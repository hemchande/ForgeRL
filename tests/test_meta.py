import unittest, os
from forgerl.meta import BASELINE, REFERENCE, Program, propose

class MetaTests(unittest.IsolatedAsyncioTestCase):
    async def test_reference(self):
        self.assertEqual(await propose(None,[]),REFERENCE)
        self.assertNotEqual(Program(BASELINE,'unused').sha,Program(REFERENCE,'unused').sha)
    @unittest.skipUnless(os.getenv('FORGERL_DOCKER_INTEGRATION')=='1','requires Docker')
    async def test_candidate_hooks(self):
        m=[{'role':'user','content':'task'}]
        a=await Program(BASELINE,'forgerl-world:local').hooks(m,{})
        b=await Program(REFERENCE,'forgerl-world:local').hooks(m,{})
        self.assertIn('execute',a[1])
        self.assertNotIn('execute',b[1])
        self.assertTrue(b[2])
        self.assertGreater(len(b[0]),len(a[0]))
