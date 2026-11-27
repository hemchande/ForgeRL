#!/usr/bin/env bash
set -euo pipefail
MODEL="${FORGERL_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
WORKERS="${FORGERL_WORKERS:-2}"
SAMPLES="${FORGERL_SAMPLES:-4}"
docker build -f Dockerfile.world -t forgerl-world:local .
docker compose -f infra/compose.inference.yaml up -d inference
export MODEL_URL="http://localhost:8000/v1/chat/completions"
python -m forgerl.runtime --model "$MODEL" --workers "$WORKERS" --samples "$SAMPLES" --model-url "$MODEL_URL"
