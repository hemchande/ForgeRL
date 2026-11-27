import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request

from forge import TASKS
from forgerl.mock_inference import Handler
from forgerl.policy import ChatPolicy

class MockInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
    def test_health(self):
        with urlopen(f"http://127.0.0.1:{self.server.server_port}/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")
    def test_agent_actions_over_http(self):
        import asyncio
        async def run():
            policy = ChatPolicy("mock", f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions")
            messages = [{"role": "user", "content": TASKS[0][1]}]
            actions = []
            for _ in range(3):
                action, raw, logprobs = await policy.act(messages)
                actions.append(action["tool"])
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "{}"})
            return actions
        self.assertEqual(asyncio.run(run()), ["read", "patch", "finish"])
