# ForgeRL

A proof of concept for reward-labeled coding-agent trajectories, harness experimentation and RL post-training data generation.

## Quick start
Python 3.10+.
```sh
python forge.py run --trusted-local --samples 8 --workers 4
python forge.py replay rollouts.jsonl --trusted-local
python -m unittest discover -s tests -v
```
Trusted-local mode is ONLY for bundled synthetic fixtures. Never execute arbitrary model output on the host.

## Architecture
Task registry → rollout scheduler → agent policy → persistent execution world → fresh verifier → reward → grouped advantages → JSONL.

The scripted fixture policy alternates correct and unchanged solutions to exercise the pipeline; its results are not model benchmarks. Model trajectories retain raw completions and provider log probabilities, not tokenizer-aligned IDs or action masks. This is not a GRPO trainer.

## Model-driven rollouts
```sh
docker build -f Dockerfile.world -t forgerl-world:local .
export MODEL_URL=http://localhost:8000/v1/chat/completions
python -m forgerl.local --model YOUR_MODEL --samples 8 --workers 4
python -m forgerl.export model-rollouts.jsonl sft.jsonl
```
The endpoint must support chat completions and logprobs. Set MODEL_API_KEY if needed. Use --system-file to compare tool prompts.

## Distributed execution
```sh
pip install -r requirements.txt
temporal server start-dev
# In separate terminals:
python -m forgerl.distributed worker
python -m forgerl.distributed submit --model YOUR_MODEL --samples 8
```
For Kubernetes, push the world image to a registry, apply infra/network-policy.yaml in a dedicated namespace with a network-policy-capable CNI, and submit with --backend kubernetes --image REGISTRY/IMAGE:TAG --namespace NAMESPACE.

## Limitations
Docker, Kubernetes, Temporal and live inference require integration testing. Containers are not hardened adversarial sandboxes; use dedicated disposable workers without production credentials. Verification is not robust to reward hacking. Patches are hash-checked full-file replacements. Temporal retries restart episodes; orphan reconciliation, durable external artifact storage, token-level training integration and load testing remain future work. No proprietary code or data is used.
