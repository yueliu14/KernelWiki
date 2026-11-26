# Workflows

Kernel-level procedures imported from GEAK. Their source dates and historical measurements remain attached; check the
installed stack before applying them.

| Workflow | Use when | Gate |
|---|---|---|
| [optimize_single_kernel.md](optimize_single_kernel.md) | Make one kernel faster in isolation | unittest correctness + isolated time |
| [choosing_a_backend.md](choosing_a_backend.md) | "Which language/backend for this operator family?" | per-family decision table |
| [roofline skill](geak/analysis_skills/roofline/SKILL.md) | How far is the kernel from the hardware limit? | achieved vs peak ([peaks](geak/analysis_skills/roofline/peaks.md)) |

Shared checks:
1. **Engagement proof, not inference** — prove your kernel is what runs (a load banner, a counter, a profiler
   kernel name); a timing wiggle is not proof.
2. **Measure the noise floor** — run an A/A comparison before trusting an A/B difference; report the protocol.
3. **Immutable oracle** — never edit the test, reference outputs or tolerances; re-hash before trusting a result.
