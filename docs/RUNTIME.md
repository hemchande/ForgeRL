# Runtime separation

## Topology
```text
                  +----------------------------+
                  | Local driver / task queue  |
                  +----------------------------+
                      |                 |
                OS process 0       OS process 1
                rollout worker     rollout worker
                      |                 |
                      +---- HTTP -------+----> vLLM inference service (GPU)
                      |                 |
                 Docker exec        Docker exec
                      |                 |
                episode sandbox A  episode sandbox B
                      |                 |
                 verifier A         verifier B
                      +------ result queue ------> JSONL / grouped advantages
```

The local driver launches actual OS processes using Python multiprocessing
(`spawn`). Each worker calls the LLM endpoint for actions, then creates a
persistent Docker sandbox per episode and a separate sandbox for verification.
The inference server is a distinct long-running process/container. Each
sandbox is a distinct Docker container, not a worker process.

## Launch
Requires Docker, Python 3.10+, NVIDIA GPU + compatible Docker runtime for
the included vLLM Compose example. CPU-only users can point at an existing
OpenAI-compatible inference server instead.

```bash
bash scripts/run-local-stack.sh
# or bring your own inference server:
docker build -f Dockerfile.world -t forgerl-world:local .
MODEL_URL=http://localhost:8000/v1/chat/completions \
  python -m forgerl.runtime --model Qwen/Qwen2.5-Coder-1.5B-Instruct --workers 2 --samples 4
```

## Request path
1. Driver queues (task_id, sample) jobs.
2. Each rollout worker is a real OS process and owns its agent control loop.
3. Worker calls inference service over HTTP for the next JSON action.
4. Worker dispatches read/patch/execute to its own Docker sandbox.
5. Tool observations feed the next inference request.
6. A fresh sandbox runs hidden verification cases.
7. Worker emits trace and reward; driver computes grouped advantages.

## Caveats
This is a proof of concept. Docker CLI access on the host is privileged.
Do not grant untrusted model code access to the Docker socket. The rollout
worker invokes Docker, but the model's code runs inside the sandbox.
The world container has no network, drops capabilities and uses resource
limits, but Docker is not a high-assurance adversarial isolation boundary.
Use disposable dedicated hosts, a microVM executor for hostile workloads,
and stronger verification for production. No load benchmark or end-to-end
GPU/Compose integration test has been run yet. The inference adapter expects
JSON text actions; it does not consume provider-native tool-call objects.
