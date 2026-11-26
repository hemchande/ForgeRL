import asyncio, unittest
from forge import TASKS, episode
class TestRollouts(unittest.TestCase):
    def test_success_and_replay(self):
        async def check():
            r=await episode(TASKS[0],0,False)
            self.assertEqual(r['reward'],1)
            s=await episode(TASKS[0],0,False,[x['action'] for x in r['trace']])
            self.assertEqual(r['trace'],s['trace'])
            self.assertEqual(s['reward'],1)
        asyncio.run(check())
    def test_failure(self):
        self.assertEqual(asyncio.run(episode(TASKS[0],1,False))['reward'],0)
