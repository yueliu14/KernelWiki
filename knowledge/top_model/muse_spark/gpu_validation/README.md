# Bounded GPU validation attempt: 2026-11-09

**No GPU execution occurred and no new GPU trace was collected.** The selected pair was [1020 → 1027, TEP8 c1](../pairs/tep8_conc1_off_2026-10-22_to_2026-10-30.md), which adds the target monokernel and draft stage-2 fusion. This directory records an unsuccessful allocation attempt, not measured kernel evidence.

Node `crsuse2-m2m-v3-107` reported eight idle MI355X devices, but idle inventory did not translate into an allocation for this task. Burst jobs `276043` and `276044` were cancelled without a node or runtime; the retained scheduler reason is `Priority`. The first attempt's deadline was too short and is excluded as a capacity test. The second used a deadline 95 minutes after submission for a 90-minute job; its cancellation mechanism was not independently established. The known permitted AgentX account rejected submission with `AssocMaxSubmitJobLimit`. Other tasks' allocations were not used. No task job remains queued or running.

Both accepted requests used **one node, one task, one CPU, eight GPUs, 90 minutes**, verified in `ReqTRES=cpu=1,node=1,gres/gpu=8`. See [status](status.json), [Burst scheduler receipt](scheduler_attempt_276044.txt), and [AgentX rejection](submission_agentx.json). No automatic renewal was submitted.

## Prepared reproduction

[Endpoint recipes](../../../../tools/inferencex/muse_spark/gpu_endpoints.json) preserve the two immutable image references, CI jobs, commands and simulation policy. [Probe runner](../../../../tools/inferencex/muse_spark/gpu_probe.py) and [batch script](job.sh) are prepared but **have not been exercised on a GPU**. They require a fresh task-owned eight-GPU allocation and complete offline model-cache symlinks on that host. Run the script only after that check; its paths reference this investigation's workspace.

The intended bounded experiment uses one warmup, three unprofiled requests, and one separate profiled request per image, sequentially. The prompt is a fixed repeated public text, temperature zero, concurrency one, 256 output tokens, with EOS ignored. Radix cache is flushed outside each measured interval. This is a synthetic shape probe, not the historical variable-length AgentX replay, and its latency must not replace the dashboard's p90 interactivity or throughput. Both endpoints retain expected DFLASH acceptance length 4.05, `match-expected`, and `real-draft-token`.

Planned deviations are the prompt/workload, reduced request count, local host/ports, output directories, current node/driver/clocks, and profiling instrumentation. The historical recipes themselves differ in admission and graph settings, so this image pair alone does not isolate the monokernel's causal contribution. A later causal test would need a controlled fusion on/off experiment as well.

The cached main-model ref is `c7bc74fd64371315a091bc15f9723882eead0dc1`; the draft ref is `46c90f412549c472941a68382d91a99e5e6372a0`. Readable shared model directories contain 65 main shards and one draft shard, but the HF cache's node-local targets could not be verified on an allocated GPU host. Historical CI model revisions remain incompletely established. No model download was performed.

The planned profiler uses SGLang's `/start_profile` and `/stop_profile` with CPU/GPU activities, shapes and detailed annotations, leaving graphs enabled. **Trace contents must still be checked** for actual device kernels inside graph execution, all eight ranks, prefill, target verify and draft coverage. A successful HTTP profiling response alone would not validate kernel coverage. Full traces belong outside Git under the external directory in [status.json](status.json); only verified summaries and hashes should be added here.
