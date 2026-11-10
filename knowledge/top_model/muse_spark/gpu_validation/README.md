# Historical GPU validation attempt — 2026-11-09

**No GPU execution occurred and no GPU trace was collected.** The selected pair was [1020 → 1027, TEP8 c1](../pairs/tep8_conc1_off_2026-10-22_to_2026-10-30.md). This directory is an allocation receipt, not a validated runbook or a kernel benchmark.

Burst jobs 276043 and 276044 were cancelled without an allocation. The first deadline was too short to serve as a capacity test. The second retained scheduler reason `Priority`; the cancellation mechanism was not independently established. AgentX rejected submission with `AssocMaxSubmitJobLimit`. Both accepted requests asked for eight GPUs and one CPU. [Status and workload scope](status.json) · [Burst receipt](scheduler_attempt_276044.txt) · [AgentX rejection](submission_agentx.json).

The prepared [endpoint recipes](../../../../tools/inferencex/muse_spark/gpu_endpoints.json) and [probe runner](../../../../tools/inferencex/muse_spark/gpu_probe.py) were not exercised on a GPU. They propose a synthetic concurrency-one shape probe with separate warmup, unprofiled and profiled requests, not the historical AgentX replay. Model-cache targets, allocation ownership, device count and current launch settings must be rechecked before future use.

The [recorded submission script](job.sh) and commands are historical evidence with environment-specific paths. Prepare a fresh task-owned allocation using the [current workflow](../../../../tools/inferencex/skills/inferencex-kernel-evolution/SKILL.md); do not interpret these receipts or the server profiling API as proof of GPU kernel coverage.
