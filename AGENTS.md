# SafeSight-FL workspace rules

## Mission

SafeSight-FL is a college prototype for privacy-preserving workplace-safety monitoring. It is not a production surveillance system.

## Execution hierarchy

- The supervisor is responsible for scope, truthfulness, review, and milestone verification.
- OpenClaw is the orchestrator and sends exactly one milestone prompt at a time.
- Google Antigravity/Gemini CLI is the coding worker when available.
- A milestone is not complete until its tests and verification are recorded in `docs/SUPERVISOR_LOG.md`.

## Non-negotiable rules

1. This is a fresh repository. Do not copy or patch the previous SafeSight-FL project.
2. Treat any pre-existing SafeSight-FL checkout as an old reference only; never edit it. Work only in this fresh repository.
3. Never fabricate detection metrics, privacy metrics, federation results, dataset counts, or benchmarks.
4. Label simulated, experimental, disabled, and reused functionality explicitly.
5. Raw CCTV and training images remain local to simulated clients; federation shares model information only.
6. Factory A/B/C are simulated logical sites on one laptop unless independent data is later supplied.
7. CV/rules determine whether an incident exists. An LLM may only summarize structured incidents.
8. Keep secrets in `.env`; never commit keys or absolute machine paths.
9. Prefer Python 3.11, CPU-compatible code, small understandable modules, and SQLite.
10. Inspect before editing, preserve attribution and licenses, and keep `README.md` current.
11. Do not call the tamper-evident log a blockchain.
12. Do not claim production readiness, complete security, anonymity, or unsupported novelty.
13. Federation results must say whether they are lightweight integration tests or PPE experiments.

## Evidence standard

Every displayed value must come from a real local source: a database query, an actual artifact, an actual dataset manifest, or an explicitly labeled demo state. Missing experiments must say `Experiment not run yet`.
