# Final release audit — 2026-10-09

## Update — 2026-10-27 imported model

Active Live Detection model is user-supplied YOLO26n `model-6aabf259c7`, with eight
PPE/person presence classes. Old GPU YOLO11 remains the rollback version. Imported
validation metrics are shown with provenance limits; earlier local test/FL/privacy
results belong to earlier models. No missing-PPE classes are available in this
checkpoint, and no new local held-out evaluation was performed. ONNX is retained
as an unverified export. Actual CPU inference, 74 tests, four affected dashboard
routes and loopback server health passed. Overall project readiness remains partial.

## Verdict: PARTIALLY READY

SafeSight-FL is ready to present as a local college prototype with reproducible evidence and clearly scoped demonstrations. It is **not ready for operational workplace monitoring, independent-site deployment, or safety decisions**. The PPE model's recorded held-out performance is weak, and several end-to-end paths are incomplete.

## Verified release checks

- Milestone 20 local demo: 71 tests passed, compilation passed, all 14 dashboard routes rendered in AppTest. Loopback HTTP health and root verified. Launcher and guided Overview added. Full live-story readiness remains partial: permitted operator media, detector quality and continuous streaming limitations still apply.

- Milestone 19 analytics follow-up: 69 regression tests passed, compile check passed and Analytics AppTest passed. Date/site/camera filters, alert charts and current-review counts are implemented; repeated reviews no longer inflate incident decision totals.

- Milestone 18 Copilot follow-up: 64 regression tests passed, compilation passed and Copilot AppTest passed. Gemini synthetic-event connectivity verified; UI defaults local, outbound fields filtered, daily summaries cover the selected UTC day. AI prose is not an incident decision or verified factual report.

- Milestone 17 Privacy Center follow-up: 59 regression tests passed; compile check passed; dashboard AppTest verified the per-run evidence view without exceptions. Historical experiments, unknown fields, demo connectivity and separate privacy scopes are explicitly labeled.

- Fresh Git repository is in the configured OpenClaw workspace. The earlier SafeSight-FL folder was not copied or modified.
- Milestones 23–25 documentation and audit records are present; the README, attribution ledger, reuse plan, viva guide, and 5–7 minute demo script have been cross-checked against current behavior and saved evidence, including the PPE SecAgg+ follow-up.
- All 14 required visual-only UI reference PNGs exist under `ui_refs/`; no illustrative values are used as application data.
- Latest full regression run after the Milestone 16 Flower/Opacus follow-up: 56 tests passed. Compile check and `git diff --check` passed. Privacy Center AppTest rendered without exceptions and displayed the experimental DP warning and per-round client accounting. Opacus emitted expected experimental secure-RNG-disabled and PyTorch backward-hook warnings.
- Federated Learning and Privacy Center Streamlit AppTests both rendered with zero exceptions and displayed PPE SecAgg+ enabled, PPE Differential Privacy disabled, and only the recorded experiment values.
- Streamlit smoke test: health endpoint returned HTTP 200 with `ok`; dashboard root returned HTTP 200.
- Release scan found no tracked `.env`, model/video/image payloads, known API-key patterns, machine-specific absolute paths in project docs/source, or listed unsupported security/novelty claims.
- Database initialization help path was checked without resetting or changing the existing project database. `--reset` is destructive by design and must be used only when intentionally deleting that selected demo database.
- Local PPE checkpoint and Flower SecAgg+ evidence artifact were present at audit time, but are ignored local artifacts and are not distributed in Git.
- Recorded evidence remains scoped: lightweight Flower FL, Flower SecAgg+ on a synthetic classifier, standalone Opacus DP-SGD, Flower + Opacus DP-SGD (three synthetic clients/two rounds, cumulative per-client accounting), one short Construction-PPE training/evaluation run, one PPE detection-head FedAvg round, and one separate PPE detection-head SecAgg+ round. PPE DP was not enabled; no combined PPE DP+SecAgg result exists. DP runs are not composed across invocations; secure RNG remains disabled.

## Current implementation status

- **Dataset Explorer:** actual local Construction-PPE labels/counts and deterministic simulated-site manifests are available. Dataset images stay outside Git.
- **PPE model:** YOLO11n one-epoch CPU demo trained on a 2% fraction and evaluated on held-out test data. Actual metrics are P=0.00595, R=0.10178, mAP@0.5=0.02079, mAP@0.5:0.95=0.00804. These results are weak and unsuitable for safety decisions.
- **Detection workflow:** local webcam snapshot, still-image upload, and bounded uploaded-video sampling at a configurable 0.25–5 second interval (one second by default), capped to 30 seconds. A reviewer explicitly selects model-returned PPE violations from sampled frames before incidents are written. An optional experimental vertical split automatically persists confirmed person transitions; the configured normalized-polygon entry rule remains available separately. Upload bytes remain in local Streamlit session memory until cleared/closed; the decoder's temporary disk copy is deleted; event snapshots are local. There is no continuous video/RTSP, pause/resume control, or production-validated tracking.
- **Restricted zones:** normalized polygon parsing and tracked-person entry persistence remain tested. The vertical transition tracker adds boundary hysteresis, consecutive-sample confirmation, temporary-loss tolerance, entry/return event rows, active/resolved lifecycle, local annotated evidence, and dashboard filters. This remains an experimental sampled-video feature; tracking is not validated for live safety use.
- **Feedback, evidence, audit:** local incident/review persistence, local evidence handling, and SHA-256 hash-chain verification are implemented. The chain is not blockchain or immutable storage.
- **Offline behavior:** local SQLite works without federation connectivity. A manual authenticated sender and loopback-only demo receiver deliver allowlisted incident metadata with matching queue-ID acknowledgements, retry-safe receiver storage, and local audit entries. No automatic delivery occurs; no remote receiver is configured.
- **Federation and privacy:** a bounded Flower FedAvg run and a separate Flower SecAgg+ run each shared only the PPE detection head across three simulated clients. The SecAgg+ workflow completed one round with 3 fit results and 0 failures; the server aggregate was shape/finiteness-checked, converted to the model dtype, reloaded, hashed, and evaluated. On 23/28/7 training images, held-out mAP@0.5 changed 0.021211 → 0.049006 and final precision was 0.003761. This remains weak single-run integration evidence, not a useful safety detector. PPE DP was disabled; no DP+SecAgg composition is claimed. Factory A/B/C are simulated on one laptop. An earlier unchanged diagnostic remains explicitly invalid.
- **Copilot:** deterministic local fallback is available; Gemini can summarize structured event fields. It does not classify incidents; configured requests send structured text to Google's service.
- **Model versions:** registration includes artifact kind and simulated-client provenance. Only existing, workspace-contained PPE checkpoints that load with supported classes can be activated. Active/previous IDs support switching and rollback; Live Detection resolves the active model when present and otherwise keeps using the configured demo checkpoint. No active version was selected during this verification.
- **UI/docs:** all 14 routes were browser-checked at Milestone 22; Model Versions and Live Detection were rechecked with AppTest after the registry change. Current documentation includes attribution, limitations, viva answers, and safe live-demo fallbacks.

## Not implemented or not release-verified

- A sufficiently accurate PPE detector validated for real safety use.
- Continuous dashboard video/RTSP, Start/Pause/Resume/Stop controls, dwell alerts, and production-validated cross-frame tracking/deduplication. The sampled-video zone adapter is not suitable for operational monitoring.
- Legally sourced prerecorded PPE demo clips bundled with provenance.
- Automatic delivery and independently deployed/secured remote receiver for cross-site queue synchronization.
- Full-model PPE federation, PPE DP-SGD, or a composed PPE SecAgg+DP experiment.
- Multi-machine/independent-factory deployment, authentication, roles, hardened service operations, and production security validation.
- Full dependency lock and clean-room installation verification on the preferred Python 3.11 target.

## Distribution and presentation gates

- Review software, dataset, and checkpoint terms separately before redistribution; see `THIRD_PARTY.md` and `docs/open_source_reuse_plan.md`.
- The Construction-PPE dataset and trained checkpoints are not included in Git. They remain local external/cache artifacts.
- Keep API credentials in ignored `.env`; do not include them in screenshots, logs, or a release archive.
- Do not reset the presentation database unless its records are intentionally disposable. Do not seed fabricated incidents or metrics for a more populated demo.
- Present recorded toy-run metrics only with their workload labels. Describe the PPE run as a weak demo experiment, not model readiness.

## Final status

**PARTIALLY READY — suitable for an honest college viva/demo; not production-ready.** The local prototype and several integration demonstrations are verified. PPE quality, continuous video/tracking, automatic or independently deployed cross-site queue delivery, and privacy mechanisms on the PPE workload remain open research and implementation work. The dashboard supports bounded uploaded-clip frame sampling and experimental automatic transition logging; it does not provide continuous streaming, camera controls, or automatic remote delivery.
