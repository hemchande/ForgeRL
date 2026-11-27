"""Deterministic HTTP inference stub for exercising the *real* rollout transport.

This is NOT an LLM or a benchmark. It emits fixed actions for bundled tasks.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from forge import TASKS, digest

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self.respond({"status": "ok", "kind": "deterministic-test-stub"})
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 262144:
                self.send_error(413)
                return
            body = json.loads(self.rfile.read(size))
            messages = body["messages"]
            user_messages = [m["content"] for m in messages if m["role"] == "user"]
            prompt = user_messages[0]
            task = next(t for t in TASKS if t[1] == prompt)
            _, _, initial, correct, _ = task
            # The mock intentionally chooses its action from the transcript.
            # It doesn't need the model's hidden verification tests.
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            if not assistant_messages:
                action = {"tool": "read"}
            elif len(assistant_messages) == 1:
                action = {"tool": "patch", "before": digest(initial), "text": correct}
            else:
                action = {"tool": "finish"}
            self.respond({"id": "mock-" + task[0], "object": "chat.completion",
                          "choices": [{"index": 0, "message": {"role": "assistant",
                           "content": json.dumps(action)}, "finish_reason": "stop",
                           "logprobs": None}]})
        except (ValueError, KeyError, StopIteration, TypeError) as exc:
            self.send_error(400, str(exc))
    def respond(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, format, *args):
        pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    print(f"Mock inference listening at http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
