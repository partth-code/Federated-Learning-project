# Test report — Milestone 21 system verification

## Latest: experimental vertical zone transitions — 2026-10-27

- 20 focused video, geometry, incident, analytics, and integration tests passed. The available regression suite passed 58 tests when the five Flower/Opacus-dependent test modules were excluded. Changed Python files compiled, and `git diff --check` passed.
- Tests cover per-track entry/return confirmation, independent IDs, boundary hysteresis, temporary detection loss, first-observed-in-danger suppression, video timestamps, transition idempotency, exit duration, persistent lifecycle updates, local evidence, offline metadata queueing, audit-chain integrity, and zone analytics.
- Streamlit AppTest rendered Overview, Incidents (Camera/Event type/Severity filters), and Live Detection without exceptions. The workspace has no active local model artifact, so Live Detection showed its disabled state; actual model-backed dashboard recording was not exercised.
- Full-suite collection was blocked because the configured Python 3.13 environment lacks `flwr` and `opacus`; no unrelated dependency installation was attempted.

## Latest: user-supplied YOLO26 integration — 2026-10-27

- 74 tests passed; compilation and whitespace checks passed. New checks cover supplied class aliases, no false negative-class conversion, retained generic-model rejection, and manifest scoping to registry version/workspace.
- Actual imported best.pt CPU inference returned 5 detections on existing local image1.jpeg (gloves, helmet, person, vest); no incidents written. This is compatibility evidence only.
- Live Detection, Training, Evaluation and registry AppTests passed, including active version, missing-PPE limitation, imported metric scope and enabled rollback. Loopback server health `ok`, HTTP 200. Weights copied with SHA-256 verification; source files unchanged.
- ONNX runtime not tested; no independent dataset/test evaluation for the supplied 8-class model. CUDA unavailable in this session; inference verified on CPU.

## Latest: Milestone 20 integrated demo

- 71 tests passed, including two new temporary-fixture integration checks; compilation passed.
- All 14 dashboard routes passed Streamlit AppTest with no exceptions. Loopback server health returned `ok`, dashboard HTTP 200.
- Offline event → review → local report → analytics → authenticated loopback sync → audit verified. This uses synthetic test fixtures, not real camera-detection evidence. No project demo records were seeded.

## Latest: Milestone 19 analytics follow-up

- 69 regression tests passed; compilation passed. Analytics AppTest rendered date filters and seven metrics without exceptions.
- Verified UTC offset boundary, empty periods, date validation, site/camera filtering, query-parameter handling and latest-review counting. Synthetic test fixtures were confined to temporary databases; no invented project records added.

## Latest: Milestone 18 Copilot follow-up

- 64 regression tests passed; compilation passed; Copilot AppTest confirmed local-only default, UTC date selection and zero exceptions.
- Focused provider tests verified outbound field filtering, resource cleanup, local-only behavior, feedback preservation, malformed values, error/empty-response fallback and full-day counts beyond 20 events.
- Live Gemini request with synthetic metadata returned nonempty text. No real incident/media sent and no credentials printed. This verifies connectivity, not factual reliability of generated prose.

## Latest: Milestone 17 Privacy Center follow-up

- Full regression: 59 tests passed. Compile check passed. Streamlit AppTest verified three summary metrics, separate experiment and DP accounting tables, and zero exceptions.
- New tests: separate-run controls never merged, multiple rounds deduplicated, unknown bytes remain absent, malformed/partial DP records cannot enable accounting, empty-state handling and privacy source scope retained.
- Existing Opacus non-secure RNG/backward-hook warnings remain expected. These checks validate a local evidence display, not production privacy or live network health.

Date: 2026-10-09  
Python: 3.13.12 shared environment; Python 3.11 remains the preferred project runtime.

## Verified

- `py -m pytest -q` — 19 tests passed.
- `py -m compileall -q .` — passed.
- Core imports — passed for audit, config, database, detector, edge pipeline, feedback, federated client, privacy, Copilot, analytics, and training-artifact modules.
- Isolated SQLite initialization — passed; exactly three sites were seeded and all are marked simulated. No project database reset was performed.
- Streamlit startup — health endpoint returned `ok`; dashboard root returned HTTP 200.
- `.env` ignore/tracking check — ignored and not tracked. Tracked-source scan for known API-key patterns returned no matches.
- `git diff --check` — passed.
- Restricted-zone entry and loitering unit behavior — passed.
- Incident persistence and local feedback decisions — passed.
- Offline queue persistence — passed.
- Audit-chain corruption detection — passed.
- Model registry rejects missing artifacts — passed.

## Verified with explicit scope

- PPE dataset inspection: real Construction-PPE YOLO labels, class distribution, split counts, and deterministic simulated Factory A/B/C manifests.
- PPE demo training/evaluation: one CPU epoch on a 2% training fraction, with held-out test metrics written to `artifacts/ppe_demo_summary.json`. The metrics are weak and are not a production claim.
- Ultralytics adapter import/model loading: verified against the locally generated demo artifact; first held-out image returned zero detections at the tested threshold.
- Flower rounds: real lightweight PyTorch workload, three simulated clients, three rounds, and stored metrics.
- Flower SecAgg+: real `SecAggPlusWorkflow` + `secaggplus_mod` run, three simulated clients, two rounds, stored held-out metrics, and Privacy Center state verified from SQLite.
- PPE Flower SecAgg+: one actual `SecAggPlusWorkflow` + `secaggplus_mod` round on locally trained PPE detection-head updates, three simulated clients, 3 fit results / 0 failures, changed validated aggregate, reloaded/hash-checked PPE checkpoint, held-out test metrics, and a completed SQLite model/privacy record. Raw images were not sent; DP was disabled.
- Opacus DP: real toy DP-SGD workload with computed epsilon/delta; not a PPE or SecAgg privacy result.
- Gemini provider: Google GenAI SDK call succeeded with a synthetic structured incident; no CCTV/image content was sent.

## Not verified yet

- Continuous webcam/live video, RTSP, and a complete live-tracking incident loop. Uploaded video is sampled at bounded intervals, not streamed continuously.
- Full-model PPE federation and a real multi-machine SuperLink/SuperNode deployment. The head-only FedAvg and SecAgg+ rounds are too small for efficacy claims.
- DP-SGD on the PPE detector or a combined DP + SecAgg run.
- A new all-page browser pass after the PPE SecAgg+ follow-up; Federated Learning and Privacy Center were checked with Streamlit AppTest and had zero exceptions.
- Automatic or independently deployed federation-queue delivery. A manual authenticated sender and loopback demo receiver are now implemented; no remote endpoint is configured.

## Warnings and known limits

- Opacus emits a warning that secure RNG is disabled for the experimental DP test, plus a PyTorch backward-hook warning. The test passes; this is not a production DP configuration.
- PPE demo and one-round federated training were intentionally tiny CPU experiments; the detector's held-out precision remains very weak. They must not be presented as a reliable deployed detector.
- Milestone 20 remains partial: no legal prerecorded media is bundled and no continuous live/RTSP tracker is available. Uploaded-video frame review and sampled experimental zone tracking do not complete those missing paths.

## Milestone 22 UI verification

- All 14 navigation destinations rendered through both Streamlit AppTest and a headless Chrome browser; each page's visible heading was checked.
- Overview and Settings were visually inspected in Chrome. Settings displayed only workspace-relative paths and did not expose machine-specific paths.
- No backend behavior was changed in the UI pass. Camera permission and image-upload interactions were not exercised.
- Streamlit minimum is now `>=1.49` to support the current `width="stretch"` layout API.

## Milestones 23–24 documentation verification

- README, attribution/reuse ledger, viva guide, and 5–7 minute demo script cross-checked against saved run evidence and implementation boundaries.
- All README documentation targets exist; no code or model behavior changed in these documentation milestones.
- `py -m pytest -q` — 19 tests passed. Opacus emitted its expected secure-RNG-disabled experimental warning and a PyTorch backward-hook warning; neither failed the suite.
- `py -m compileall` over project modules — passed.
- `git diff --check` — passed after correcting Markdown trailing whitespace.

## Milestone 25 release audit

- Streamlit app smoke test — `/_stcore/health` returned HTTP 200 `ok`; `/` returned HTTP 200.
- Required UI references — all 14 expected filenames exist.
- Tracked-content scan — no `.env`, model/video/media payloads, known API-key patterns, machine-specific absolute paths, or listed unsupported claims found.
- PPE checkpoint and SecAgg+ evidence artifacts were present locally at audit time; both are ignored and not distributed through Git.
- `python scripts/init_demo_db.py --help` — passed without modifying or resetting the project database. The destructive `--reset` option was not run.
- Readiness remains PARTIALLY READY: the model is weak, and continuous video/production-validated tracking, automatic federation queue synchronization, full-model PPE federation, and PPE privacy composition are not implemented or verified.

## Milestone 15 follow-up — PPE detection-head SecAgg+

- Run `ppe-secagg-2bab95dfed` completed one Flower SecAgg+ round with three simulated clients (23/28/7 local images), three fit results, and zero fit failures. Flower logged `Secure aggregation completed`.
- The recorded clipping bound was derived from the actual tensors (10.52); quantization range 4,194,304; three shares with reconstruction threshold two. Only the 432,801-parameter PPE head was shared; the full YOLO model and raw training images were not sent. Differential Privacy was disabled.
- The artifact/checkpoint passed aggregate schema/finiteness/change checks, model reload, SHA-256 verification, and held-out evaluation. SHA-256: `5d7adeb3765456d8d7b5fb25ecc1238a91c6e1228040dc40f00b0b0d802166bd`. mAP@0.5 changed 0.021211 → 0.049006; final precision was 0.003761. Results remain too weak for safety use.
- Flower's WSL launcher returned exit code 1 although the server task finished and wrote its post-SecAgg aggregate. The runner preserves that status and accepts it only after the aggregate file exists and all downstream model checks pass. Same-machine temporary updates remain visible to a local administrator/orchestrator.
- The model version was registered but not activated. Federated Learning and Privacy Center AppTests showed zero exceptions and reported PPE SecAgg+ true / PPE DP false from stored evidence.
- Final suite after this follow-up: 48 tests passed; `compileall` and `git diff --check` passed. The two expected Opacus experimental warnings were shown.

## Milestone 20 follow-up — uploaded video frame review

- Added a local decoder that samples at one frame per second, caps processing at 30 frames / approximately the first 30 seconds, scales large frames, and retains JPEG previews/stills only in session state. Upload bytes remain in the local Streamlit session until cleared/closed; the decoder's temporary disk copy is removed after analysis.
- UI displays actual model outputs and requires explicit reviewer selection before creating incidents from supported violation labels. Evidence is a local sampled still; offline queue records metadata only. No person tracking, cross-frame deduplication, or restricted-zone inference is claimed.
- `py -m pytest tests/test_video_inference.py tests/test_incident_pipeline.py -q` — 4 passed using a mocked decoder/model and real isolated SQLite/evidence/audit/queue services.
- Full suite `py -m pytest -q` — 21 passed. Compile check and `git diff --check` passed; Opacus warnings unchanged.
- Streamlit AppTest switched to Upload video and found the uploader with zero exceptions; health endpoint returned HTTP 200 `ok` and dashboard root HTTP 200. Actual webcam/video permissions and a legal external video were not exercised.

## Milestone 4 follow-up — sampled-video restricted-zone tracking

- Added normalized polygon text parsing/validation and optional Ultralytics person tracking during the existing sampled video analysis. Entries require an observed outside-to-inside transition with a real model tracker ID; a person first observed inside the polygon does not trigger an entry.
- Zone entries carry the configured zone ID, sampled-video time, tracker ID, and confidence into the existing incident/audit/offline-metadata services after explicit selection in the UI.
- Tests cover polygon validation, transition versus initial occupancy, and local zone-incident/evidence/audit/metadata queue persistence. This path is experimental at one frame per second and has not been validated against a real video clip or stable tracker identity.
- Full suite after this follow-up: 25 tests passed; compile check and `git diff --check` passed. Streamlit AppTest enabled zone mode and selected Upload video; uploader appeared with zero exceptions.

## Milestone 11 follow-up — authenticated manual metadata delivery

- Added a manually invoked HTTP sender for allowlisted `incident_metadata`; payloads exclude evidence paths/media, use bearer authentication and an idempotency key, and require exact queue-ID acknowledgement before local status changes. Remote URLs require HTTPS; HTTP is loopback-only.
- Added a loopback-only receiver with authentication, request-size/schema validation, idempotent SQLite persistence, and acknowledgement response. No automatic delivery or default endpoint/token is configured.
- Tests cover authenticated delivery, metadata-only receipt, offline/unconfigured no-send, failed-send retry state, unsupported queue types, remote endpoint validation, receiver authentication, and hash-chain audit.
- Settings AppTest showed sync disabled when endpoint/token are unconfigured, with zero exceptions. Test count and full compile result are recorded after the full regression run.

## Milestone 11 follow-up — explicit authenticated metadata delivery

- Added a manually invoked HTTP sender for allowlisted `incident_metadata`; the payload excludes evidence paths and media. Each request uses a bearer token and idempotency key. Remote destinations require HTTPS; unencrypted HTTP is accepted only for loopback.
- Added a loopback-only demo receiver with request-size limits, strict schema validation, bearer authentication, idempotent SQLite persistence, and exact queue-ID acknowledgements. The sender marks records synced and appends a local audit entry only after a matching acknowledgement. No remote endpoint or token is configured by default.
- Tests cover successful local delivery, metadata-only payload, receiver authentication, endpoint validation, offline/unconfigured no-send behavior, and audit-chain integrity.
- Full suite after this follow-up: 31 tests passed; compile check and `git diff --check` passed. Opacus warnings are unchanged.

## Milestone 13 follow-up — frozen-backbone PPE head exchange boundary

- Added a Flower `NumPyClient` boundary for exchanging trainable final-head parameters, keeping the backbone and fixed DFL projection local. A `FedAvg` guard rejects missing, mixed, or unexpected model schema hashes before aggregation.
- The local compatibility script loaded the existing 11-class PPE checkpoint and reported 432,801 head parameters / 1,731,204 raw bytes per update direction. It performed no training, network call, or aggregation; metadata records zero PPE rounds and null validation metrics.
- Tests cover head-only exchange, frozen-parameter preservation, schema guard, malformed/non-finite arrays, and PPE-label validation. Focused suite: 5 passed.
- Historical boundary at that point: this was an interface/compatibility check only. The follow-up below supersedes the zero-round statement; PPE SecAgg and DP remain unimplemented.
- Final verification for this milestone: `py -m pytest -q` — all 36 tests passed; `py -m compileall -q federated scripts dashboard tests` passed; Federated Learning AppTest showed the matched-checkpoint status with zero exceptions; `git diff --check` passed. The only full-suite warnings were the already documented Opacus experimental secure-RNG and PyTorch backward-hook warnings.

## Milestone 13 follow-up — bounded PPE detection-head federation

- Prompt: advance from a schema adapter to one real, tightly bounded local YOLO training + Flower FedAvg round, then verify the client and global weights changed before reporting success.
- Partition correction: regenerated the local manifest with deterministic dominant-class Dirichlet allocation (alpha 0.3, seed 20261009). Counts are based on actual local label files; Factory A/B/C remain simulated one-machine clients.
- Failure and fixes: the first Flower launch raced server startup while the server evaluated the baseline; added a local server-readiness barrier. A later run completed protocol exchange but client weights were unchanged because the default nominal batch size (64) prevented optimizer steps with tiny batches. That artifact/DB round is marked invalid and its registry note warns against use. Set nominal batch size to the demo batch and added per-client nonzero-update plus aggregate-change checks.
- Actual valid run: one round, one CPU epoch/client, 320px images, nominal 5% sample rounded up to whole images; clients used 23/28/7 images. All reported nonzero head updates; the final checkpoint changed 65 shared arrays and its SHA-256 matched the run artifact. Head payload is 1,731,204 raw bytes per client per direction. Raw images were not sent; PPE Secure Aggregation and DP were disabled.
- Held-out test evidence (initial → after): precision 0.005289 → 0.003777; recall 0.101778 → 0.101644; mAP@0.5 0.021211 → 0.049003; mAP@0.5:0.95 0.007979 → 0.023927. This single tiny result is weak integration evidence, not a detector efficacy or safety result.
- Verification: focused federation/dataset/privacy suite — 16 passed; full suite — 44 collected and passed (exit 0); compileall and `git diff --check` passed. Federated Learning page AppTest showed the actual metrics and no exceptions. Opacus emitted its documented experimental secure-RNG-disabled and PyTorch hook warnings.
- Status: COMPLETE for one experimental PPE detection-head FedAvg integration round. Full-model federation, PPE SecAgg/DP, repeated/multi-round PPE validation, and useful safety performance remain open.

## Milestone 16 follow-up — federated DP integration

- Final verification: full suite 56 passed; compileall passed; `git diff --check` passed. Privacy Center Streamlit AppTest: no exceptions, experimental DP warning visible. Checkpoint SHA, two SQLite round records, three privacy records and PPE DP still false independently verified.

- Actual run `dp-fl-37f8627e0a`: 3/3 clients completed both rounds, zero failures, clean process exits. Saved checkpoint reloaded and parameters validated before registration.
- RDP accounting: per-client epsilon 4.544477 after 5 steps and 5.753689 after 10 steps, delta 1e-5, Poisson rate 0.2, clipping 1, noise 1. Secure RNG off; experimental synthetic classifier only. No PPE/combined SecAgg claim.
- Tests cover growing cumulative accounting, round replay rejection, disabled private evaluation, raw metric omission, RNG isolation, malformed arrays/configs and excluding failed/baseline records from privacy status.
- Real held-out synthetic accuracy 0.733333 → 0.820833 → 0.833333. Separate runs are not composed. Flower legacy API deprecation and Opacus experimental RNG/hook warnings remain explicit limitations.

## Milestone 14 — model version activation/rollback

- SQLite schema migration adds model kind, client IDs, and a current/previous version pointer without resetting existing records. Legacy PPE head metadata is recovered from recorded Flower run metrics; toy Flower classifier versions remain ineligible for PPE activation.
- Tests cover two-version activation/switch/rollback toggling, active checkpoint resolution, missing/out-of-workspace artifacts, non-PPE model rejection, and migration from the prior registry schema.
- Manual model-load check activated the actual local baseline and PPE head checkpoint, switched between them, and rolled back using a disposable registry. The project database's active pointer remains unset. Model Versions and Live Detection AppTest routes both rendered with zero exceptions.
- Full post-Milestone-14 suite: 47 tests passed. Compile and `git diff --check` results are recorded after final verification. Registry switching changes the local Live Detection checkpoint; it does not improve or certify detector quality.
