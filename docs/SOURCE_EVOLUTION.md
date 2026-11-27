# Source-level harness evolution

The editable surface is harness_program.py: context construction, action availability and read-cache policy. A meta-model receives discovery evaluation records and baseline source and proposes replacement Python source. The candidate is never imported on the host; hooks run in a separate restricted Docker sandbox and return validated JSON. Task execution and verification use separate worlds. This is a constrained source-level prototype, not a general repository editing agent or hardened hostile-code sandbox.

Build the world image and set MODEL_URL and optionally MODEL_API_KEY. Run:

    python -m forgerl.source_experiment --model POLICY --meta-model META --samples 2 --workers 4 --output source-run

The output directory must not exist. It contains baseline and candidate source, unified diff, SQLite raw events and evaluations, and held-out report. No automatic merge or deployment. Omit meta-model for a reference source intervention. The task model checkpoint must be pinned externally. The experiment uses fresh rollouts, not recorded-action replay. Hook sandboxes are created per step, adding substantial latency; this is isolation-first, not throughput optimized. Real model results are not yet established. User sentiment, robust load tests, statistical uncertainty, and adversarial verification remain future work. The existing action metric is heuristic and may miss unnecessary execution. The candidate cannot edit host evaluators through the supported interface, but Docker is not a guarantee against hostile code.
