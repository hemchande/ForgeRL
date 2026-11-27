"""Local multi-process rollout runtime.

One OS process per rollout worker. Each episode creates a separate Docker
sandbox via World. Inference is served by an independent OpenAI-compatible
server (e.g. vLLM), never by the rollout workers.
"""
import argparse
import asyncio
import json
import multiprocessing as mp
import os
import queue
import time
import urllib.request
import uuid
from pathlib import Path

from forge import TASKS
from .pipeline import rollout, normalize

def worker_main(worker_id, jobs, results, config):
    """Independent process: async agent loop -> inference RPC -> Docker tool RPC."""
    os.environ["MODEL_URL"] = config["model_url"]
    while True:
        job = jobs.get()
        if job is None:
            break
        try:
            row = asyncio.run(rollout(
                task_id=job["task_id"], sample=job["sample"],
                model=config["model"], backend="docker",
                image=config["image"], max_steps=config["max_steps"],
            ))
        except Exception as exc:
            row = {"task_id": job["task_id"], "sample": job["sample"],
                   "error": f"{type(exc).__name__}: {exc}"}
        row["worker_id"] = worker_id
        results.put(row)

def wait_for_inference(url, seconds):
    """Wait for HTTP server readiness without sending an inference request."""
    base = url.rsplit("/v1/", 1)[0]
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Inference server not healthy at {base}/health")

def main():
    p = argparse.ArgumentParser(description="Run ForgeRL with separate worker processes and Docker sandboxes")
    p.add_argument("--model", required=True)
    p.add_argument("--model-url", default=os.getenv("MODEL_URL", "http://localhost:8000/v1/chat/completions"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--image", default="forgerl-world:local")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--output", default="runtime-rollouts.jsonl")
    p.add_argument("--skip-healthcheck", action="store_true")
    args = p.parse_args()
    if min(args.workers, args.samples, args.max_steps) < 1:
        p.error("workers, samples, max-steps must be positive")
    if not args.skip_healthcheck:
        wait_for_inference(args.model_url, 60)
    config = dict(model=args.model, model_url=args.model_url, image=args.image, max_steps=args.max_steps)
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    planned = [(task[0], sample) for task in TASKS for sample in range(args.samples)]
    for task_id, sample in planned:
        jobs.put({"task_id": task_id, "sample": sample})
    for _ in range(args.workers):
        jobs.put(None)
    workers = [
        ctx.Process(target=worker_main, args=(i, jobs, results, config), name=f"rollout-worker-{i}")
        for i in range(args.workers)
    ]
    for proc in workers:
        proc.start()
    rows = []
    try:
        while len(rows) < len(planned):
            try:
                rows.append(results.get(timeout=2))
            except queue.Empty:
                if all(not proc.is_alive() for proc in workers):
                    raise RuntimeError(f"All rollout workers exited after {len(rows)}/{len(planned)} results")
    except KeyboardInterrupt:
        for proc in workers:
            if proc.is_alive(): proc.terminate()
        raise
    finally:
        for proc in workers: proc.join(timeout=5)
        for proc in workers:
            if proc.is_alive(): proc.terminate(); proc.join(timeout=5)
    normalize(rows)
    Path(args.output).write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"episodes": len(rows), "successes": sum(row.get("reward", 0) == 1 for row in rows),
                      "workers": args.workers, "output": args.output}))

if __name__ == "__main__":
    mp.freeze_support()
    main()
