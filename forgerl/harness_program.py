"""Editable harness program. Data-only JSON output; no evaluator access."""
def build_context(messages, state):
    return messages

def allowed_actions(state):
    return ["read", "patch", "execute", "finish"]

def cache_reads(state):
    return False
