"""Opt-in actual sandbox integration: FORGERL_DOCKER_INTEGRATION=1."""
import asyncio
import os
import unittest
from forge import TASKS
from forgerl.world import World

@unittest.skipUnless(os.getenv("FORGERL_DOCKER_INTEGRATION") == "1",
                     "set FORGERL_DOCKER_INTEGRATION=1 after building forgerl-world:local")
class DockerWorldIntegration(unittest.TestCase):
    def test_isolated_world_lifecycle(self):
        async def run():
            w = World()
            try:
                await w.start(TASKS[0][2])
                source = await w.read()
                self.assertEqual(source["exit"], 0)
                self.assertEqual(source["stdout"], TASKS[0][2])
                result = await w.execute("from solution import solve; print(solve(5,2))")
                self.assertEqual(result["exit"], 0)
                self.assertEqual(result["stdout"].strip(), "2")
            finally:
                await w.close()
        asyncio.run(run())
