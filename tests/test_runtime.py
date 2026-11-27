import unittest
from unittest.mock import patch
from forgerl.runtime import wait_for_inference
class TestRuntime(unittest.TestCase):
    def test_healthcheck_failure(self):
        with self.assertRaises(RuntimeError):
            wait_for_inference("http://127.0.0.1:1/v1/chat/completions", 0)
