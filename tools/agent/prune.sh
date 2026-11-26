#!/bin/bash
# Derive the kernel-level tree from a main checkout: remove everything above the kernel (model histories,
# workload/serving chains, library integration, agent-process notes, pre-CDNA3/client GPUs) and the generated corpus
# pages that cards.md replaces. Idempotent; run from the repository root, then build_cards.py, slim.py, lint.py.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
G=knowledge/general
rmrf() { for p in "$@"; do [ -n "$(git ls-files -- "$p")" ] && git rm -rqf -- "$p"; done; return 0; }

# not kernel-level
rmrf knowledge/top_model knowledge/workload knowledge/_sources \
     $G/landscape $G/reference $G/libraries $G/case_studies/by_model \
     $G/quantization/deployment_recipes.md $G/quantization/calibration_and_quark.md \
     $G/optimization/geak_self_monitoring.md \
     $G/expert_recipes/skills/apply_flydsl_moe_to_vllm \
     $G/expert_recipes/tuning/tuning-in-sglang $G/expert_recipes/tuning/tuning-in-vllm \
     $G/expert_recipes/tuning/tuning-hipblaslt $G/expert_recipes/tuning/env-setup \
     $G/expert_recipes/tuning/validate $G/expert_recipes/tuning/docs $G/expert_recipes/tuning/tools \
     tools/inferencex tools/workload tests/inferencex tests/workload
# workflows: keep only the single-kernel procedures
for f in $(git ls-files $G/workflows); do
  case $f in $G/workflows/README.md|$G/workflows/optimize_single_kernel.md|$G/workflows/choosing_a_backend.md|$G/workflows/geak/analysis_skills/roofline/*) ;;
    *) rmrf "$f";; esac
done
# GPUs outside the AMD Instinct inference targets
rmrf $G/hardware/cdna1_mi100 $G/hardware/cdna2_mi200 $G/hardware/rdna4_gfx1201 \
     $G/hardware/geak_amd_rdna4.md $G/hardware/geak_amd_ryzen.md
# generated corpus pages (replaced by cards.md); the corpus maintenance toolchain and its tests need the full main
# tree (_sources PR records, generated pages) and stay on main, where the YAML is edited
rmrf $(git ls-files 'knowledge/operator/*/corpus/decisions.md' 'knowledge/operator/*/corpus/source_evidence.md' \
       'knowledge/operator/*/corpus/seeds.md') $G/corpus/decisions.md knowledge/operator/CORPUS.md \
     tools/corpus tests
# corpus extraction bookkeeping: only decisions.yaml (cards) and tuned_configs.yaml (shipped config seeds) are read here
rmrf $(git ls-files 'knowledge/*/corpus/source_evidence.yaml' 'knowledge/*/corpus/scope.yaml' \
       'knowledge/*/corpus/performance_axes.yaml' 'knowledge/*/*/corpus/source_evidence.yaml' \
       'knowledge/*/*/corpus/scope.yaml' 'knowledge/*/*/corpus/performance_axes.yaml')
echo "pruned: $(git ls-files | wc -l) files left"
