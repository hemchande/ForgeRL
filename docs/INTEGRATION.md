# End-to-end integration validation

## Level 1: HTTP inference transport (no Docker/GPU)

```bash
python -m unittest discover -s tests -v
python -m forgerl.mock_inference --port 8000
```

In another terminal, inspect `curl http://127.0.0.1:8000/health`.
This deterministic test stub returns read → patch → finish and is **not a real
model**. It is used to isolate orchestration bugs from inference quality.

## Level 2: Multi-process workers + real Docker sandboxes

```bash
docker build -f Dockerfile.world -t forgerl-world:local .
FORGERL_DOCKER_INTEGRATION=1 python -m unittest discover -s tests -v
python -m forgerl.runtime --model mock --workers 3 --samples 2 \
  --model-url http://127.0.0.1:8000/v1/chat/completions \
  --output integration-rollouts.jsonl
```

Expected: 6 rows across three coding tasks; each successful row has an
independent `episode_id`, `worker_id`, tool traces and reward 1.
Inspect errors rather than assuming success:

```bash
python - <<'PY'
import json
rows=[json.loads(line) for line in open("integration-rollouts.jsonl")]
for row in rows:
    print(row["task_id"], row["worker_id"], row.get("reward"), row.get("error"))
assert len(rows)==6
assert all(row.get("reward")==1 for row in rows), rows
PY
```

## Level 3: Real policy inference

Stop the mock server, start vLLM using `infra/compose.inference.yaml`, then
run `python -m forgerl.runtime --model YOUR_MODEL --workers 3 --samples 2`.
Unlike the deterministic mock, actual LLM completions may be invalid or fail
verification. Do not interpret this as a guarantee of success.

## Security and limitations

The mock returns known fixture solutions, so it is an infrastructure smoke
test, not a measure of coding ability. Docker sandboxing is not a security
boundary against hostile code. Use disposable hosts and never mount secrets
or the Docker socket into sandboxes. This test does not validate Kubernetes,
Temporal, throughput, model training or production readiness.
