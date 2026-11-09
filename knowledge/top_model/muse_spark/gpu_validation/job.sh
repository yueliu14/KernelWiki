#!/bin/bash
set -euo pipefail
OUT=/shared_nfs/yueliu14/inferencex_muse_spark_research/gpu_validation/20261109_c1
TOOLS=/shared_nfs/yueliu14/KernelWiki_wt_muse_spark_verification/tools/inferencex/muse_spark
cleanup() {
  for side in before after; do
    name=muse_verify_${SLURM_JOB_ID}_${side}_yueliu14
    if docker inspect "$name" >/dev/null 2>&1; then
      mkdir -p "$OUT/$side"
      docker logs "$name" > "$OUT/$side/server_final.log" 2>&1 || true
      docker rm -f "$name" || true
    fi
  done
}
trap cleanup EXIT TERM INT
scontrol show job "$SLURM_JOB_ID" > "$OUT/scheduler_receipt.txt"
timeout --signal=TERM 82m python3 "$TOOLS/gpu_probe.py" --endpoints "$TOOLS/gpu_endpoints.json" --output "$OUT"
