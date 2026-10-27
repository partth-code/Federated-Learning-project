# Viva guide — SafeSight-FL

## 30-second pitch

SafeSight-FL is a final-year prototype for exploring workplace PPE monitoring without centralizing raw training footage. It combines local PPE inference, structured incidents, human review, a tamper-evident audit log, and experimental one-round Flower FedAvg and SecAgg+ runs over only the PPE detection head. Factory A/B/C are simulated clients on one laptop. The PPE model is weak; Differential Privacy was tested separately on a toy workload, not on this detector. This is an integration prototype, not a production safety system.

## One-minute pitch

Manual CCTV review can miss safety events, while collecting camera footage centrally creates data-governance, storage, and bandwidth concerns. SafeSight-FL explores a local-first workflow: a PPE-trained model processes images/video samples locally; a supported violation may become a structured incident; a human can review it; and local evidence/actions can be audited. The uploaded-video demo now has an experimental vertical left/danger split that automatically records confirmed sampled-frame transitions and resolves an entry when a tracked person is observed returning left. One bounded Flower FedAvg round and one separate Flower SecAgg+ round each exchanged only the PPE detection head across three simulated clients; toy classifier-FL/SecAgg+ and Opacus DP-SGD runs are separate. A local PPE version can be activated or rolled back from the registry. The current detector's held-out metrics remain very weak; there is no continuous dashboard stream, and offline delivery requires a manually configured authenticated receiver. Those gaps are visible in our documentation and UI.

## Problem and scope

- Problem: manual monitoring can miss PPE violations or restricted-zone entry; central footage collection can increase privacy, bandwidth, storage, and governance burden.
- Proposed direction: keep raw training images at each client and exchange model information through federation, with privacy controls applied to the actual training workload.
- Built scope: local PPE dataset inspection and a short training/evaluation path; bounded head-only PPE FedAvg and SecAgg+ rounds; image/snapshot and bounded uploaded-video review; feedback, local evidence, SQLite, audit chain, and dashboard; experimental sampled-video person/zone tracking; separate toy Flower and Opacus demonstrations.
- Out of scope for current claims: production surveillance, independent factory deployment, reliable safety decisions, full-model PPE federation, PPE DP-SGD or combined PPE DP+SecAgg, continuous video/RTSP, reliable zone tracking, pause/resume processing, and automatic or remote-site queue synchronization.

## Why these methods?

**Federated Learning (FL):** clients train locally and communicate model updates/parameters for aggregation instead of sending raw training samples in the intended workflow. FL alone is not a guarantee of privacy. We completed one small Flower round over only the PPE detector head; a separate lightweight classifier run is synthetic. Neither is an independent-factory deployment.

**Secure Aggregation (SecAgg+):** a protocol can limit what the server learns about an individual client update by aggregating protected updates. It does not make the whole system invulnerable, and it is not equivalent to DP. Flower's SecAgg+ completed on both a synthetic NumPy classifier and one bounded PPE detection-head round (three simulated clients, one round). The Flower server strategy received the aggregate; because all clients and staging files were on one laptop, a same-machine administrator/orchestrator can inspect local updates. The PPE run was not combined with DP.

**Differential Privacy (DP):** a formally accounted noise mechanism can bound the influence of an individual training example, under specific assumptions and parameters. Epsilon is not a universal privacy score; it depends on the mechanism, sampling, steps, delta, and workload. Our Opacus DP-SGD run computed epsilon 4.99460264989912 at delta 1e-5 for a toy local-client workload. The experiment warned that secure RNG was disabled, so this is not a production configuration or a PPE privacy guarantee.

**Human review and audit:** CV output can be wrong. Review labels preserve supervisor feedback, while the local hash-chain log makes subsequent record edits detectable if the chain is checked. It is not a blockchain, immutable storage, or protection from every privileged attacker.

**AI Incident Copilot:** the LLM summarizes structured event fields and may suggest review actions. It does not create or adjudicate incidents. With Gemini configured, structured text is sent to Google's API; otherwise a deterministic local fallback is available. No image is sent by the Copilot integration.

## Real vs simulated vs reused

- **REAL / exercised:** local Construction-PPE annotations and non-IID partition counts; one short CPU YOLO11n run and held-out evaluation; one non-secure PPE detection-head FedAvg round and one Flower SecAgg+ PPE detection-head round with three simulated clients each; changed local/global weights, a schema- and SHA-256-verified saved SecAgg+ checkpoint; local PPE checkpoint registration, validated activation/rollback, and Live Detection fallback resolution; separate toy Flower and Opacus DP runs; a synthetic structured-event Gemini call; and the currently recorded automated test suite.
- **EXPERIMENTAL / weak:** each PPE federation experiment used one CPU epoch/client and a nominal 5% of each client split, rounded up to whole images (23/28/7 images). For the SecAgg+ run, held-out mAP@0.5 changed 0.02121 → 0.04901, while test precision was only 0.00376 afterward. These tiny runs are not efficacy results; PPE DP was not run.
- **SIMULATED:** Factory A/B/C are deterministic logical partitions/clients on one machine. They are not independent factories or separate networked organizations.
- **EXPERIMENTAL / weak:** the initial PPE run used one epoch, 2% of training data, and 320px input; a separate one-round federated head-only update used a nominal 5% per client, rounded up to whole images, at 320px. Its test mAP@0.5:0.95 changed 0.00798 → 0.02393, but final precision was 0.00378 and results are not safety-performance evidence.
- **REUSED:** third-party packages and documented Flower APIs/examples are attributed in `THIRD_PARTY.md`; the Construction-PPE dataset is external and is not redistributed here. The two community repositories in the reuse plan remain reference-only; no code or weights were copied from them.
- **EXPERIMENTAL / NOT PRODUCTION-VERIFIED:** PPE head-only SecAgg+ was exercised on one machine; vertical and polygon zone rules use uploaded-video samples at a configurable interval (one second by default), so IDs may be unstable and transitions may be missed. **NOT IMPLEMENTED / NOT VERIFIED:** full-model PPE federation, PPE DP-SGD, DP+SecAgg composition, continuous dashboard video/RTSP, pause/resume controls, dwell alerts, automatic queue synchronization, production security, and deployment at independent sites.

## Project contribution and limitations

Our contribution is the integration and honest scope separation: dataset manifests and simulated non-IID partitions; a supported-label PPE adapter with bounded detection-head Flower FedAvg and SecAgg+ experiments; local model version activation/rollback; incident/review persistence; local evidence and hash-chain audit; local dashboard and privacy-state reporting; and a separate Opacus toy workload. We did not invent those upstream algorithms or claim a tiny PPE run proves detector efficacy or complete privacy.

Important limitations to volunteer before being asked:

1. The PPE model is far too weak for a real safety workflow; it may miss events or return false alarms.
2. Dashboard detection accepts a webcam still snapshot, uploaded still image, or uploaded clip sampled at a configurable interval for up to 30 seconds. PPE predictions are frame-level. The experimental vertical tracker automatically records confirmed transitions when it returns person IDs; sparse sampling may miss transitions and this is not reliable monitoring.
3. The sampled-video tracker is experimental and low-rate; its transition log is automatic within the analyzed clip, but it is not continuous or production-validated.
4. No legal prerecorded PPE demo clip is bundled.
5. The PPE head-only FedAvg and SecAgg+ runs are small simulated integration tests. The SecAgg+ run protects aggregation from the Flower ServerApp's view, but local staging is on one shared machine; it is not DP and is not an independent-factory deployment. The Opacus DP run is a separate toy workload.
6. Offline incident metadata remains queued until a user explicitly invokes the authenticated sender. The included receiver is loopback-only; no remote receiver is configured, and no background sync runs.
7. There is no authentication, role-based access control, hardened deployment, or production validation.
8. Dataset software and data licenses are distinct; verify terms before redistribution.

## Likely examiner questions and defensible answers

**Does SafeSight-FL detect every helmet/vest violation?**
No. The demo model's held-out metrics are weak. It is an experiment, not a reliable safety monitor.

**Does COCO detect helmets and vests?**
We do not assume that. The PPE adapter checks for supported PPE labels and rejects an unsuitable generic detector.

**Are Factory A/B/C real factories?**
No. They are simulated, deterministic partitions running on one computer because independent factory datasets and hardware were not available.

**Does FL guarantee privacy?**
No. FL avoids sending raw samples in the intended design, but updates can leak information. Additional mechanisms and threat analysis are needed.

**Did you use SecAgg+ and DP together?**
No. We ran SecAgg+ on a bounded PPE head-only round and on a separate synthetic toy workload; Opacus DP-SGD ran on a different toy workload. We do not claim a composed PPE privacy guarantee.

**What does the epsilon value mean?**
It is the accountant's result for one specific Opacus toy run at the recorded delta and configuration. It cannot be transferred to another model or workload. Secure RNG was disabled in this experiment.

**Is the audit log blockchain?**
No. It is a local SHA-256 hash chain. Verification can detect changes that break the chain; it does not provide distributed consensus or immutable storage.

**Can the AI decide whether an incident is real?**
No. Detection/rules create the structured event. The Copilot only explains or summarizes those fields and can offer suggested supervisor actions.

**What happens offline?**
Local image/video-sample processing and SQLite persistence do not depend on federation connectivity. Incident metadata is queued. After reconnect, a user may manually send allowlisted metadata to a configured authenticated receiver; images are never sent. The sample receiver runs only on loopback, and no remote receiver is configured by default.

**What is your main contribution?**
The prototype integration, local review/audit workflow, dataset/training evidence, and clearly scoped privacy/federation demonstrations—not a new FL, DP, or SecAgg algorithm.

**What would you do next?**
Improve and validate the PPE model on legally usable held-out data; repeat the head-only federation experiment over more clients/rounds; evaluate PPE-specific privacy mechanisms; validate continuous tracking/video and sampled zone events; and validate deployment and sync with independent sites.

## Seven-minute live demo plan

Use `docs/DEMO_SCRIPT.md` for the click-by-click flow. The safe fallback is to use existing local artifacts/records and explain missing camera or dataset inputs; never create dummy counts, artificial incidents, or illustrative metric bars to fill an unavailable page. Do not run a full training job during viva. If the local short demo checkpoint is absent, show the training page's actual state and use the saved test report for the recorded historical experiment, clearly distinguishing report evidence from a current local artifact.
