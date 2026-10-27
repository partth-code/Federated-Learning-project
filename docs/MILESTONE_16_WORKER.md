# Milestone 16 follow-up: DP within Flower

Execute only this milestone in this fresh repository. Read AGENTS.md and
SUPERVISOR_LOG.md first. OpenClaw orchestrates; Gemini/Antigravity codes;
the supervisor verifies. Preserve existing uncommitted work.

## Required implementation

- Connect Opacus DP-SGD to the actual three-client lightweight Flower workload,
  not to YOLO. Keep the existing non-private workflow working.
- Retain each client's accountant and optimizer across federation rounds.
  Report cumulative per-client epsilon at a configured delta, actual step count,
  sample rate, clipping norm, noise multiplier, accountant and secure RNG state.
- Do not reseed DP noise to a public fixed seed. Model initialization must not
  reset the random generator used for privacy noise or Poisson sampling.
- Never send raw training loss, local accuracy, images, labels or samples to the
  server. Evaluate only on the independent synthetic held-out server dataset.
- Reject invalid or nonfinite configuration and malformed parameter arrays.
- Run a bounded two-round, three-client CPU experiment. Ensure all client and
  server processes exit; reject incomplete rounds and missing checkpoints.
- Record actual results, local checkpoint and registry version, SQLite records
  and scoped Privacy Center status. Do not count round zero as a training round.
- Clearly label example-level DP-SGD within each simulated client; not
  client-level DP, not PPE DP, not a combined SecAgg+DP experiment. Non-secure
  RNG means experimental only. Separate runs are not automatically composed.
- Tests must cover cumulative accounting, no raw metric release, parameter and
  configuration validation, honest status labels, and the completed live run.
- Update README, test report and supervisor log with actual commands/results,
  failures and fixes. Never fabricate evidence or launch full PPE training.

## Primary reference

https://opacus.ai/api/privacy_engine.html documents accountant state,
get_epsilon(delta), Poisson sampling and secure_mode limitations.

## Existing code

- privacy/opacus_demo.py: standalone toy demonstration, not federated.
- federated/lightweight_fl.py: actual Flower toy workload, currently non-private.
- privacy/status.py: currently distinguishes PPE privacy from toy DP availability.

Report exact files changed and verification evidence. Stop after this milestone.
