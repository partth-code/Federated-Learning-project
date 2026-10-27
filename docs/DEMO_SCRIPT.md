# SafeSight-FL — 5–7 minute viva demo

This script presents the current working baseline, including its limitations. It must not depend on a predicted violation appearing at a particular time. Use only media and datasets you are authorized to use, and describe simulated sites and toy experiments exactly as such.

## Before the viva

Start the local dashboard with `powershell -ExecutionPolicy Bypass -File scripts/launch_demo.ps1`.
Open http://127.0.0.1:8501 on the same laptop. The launcher binds only to loopback;
it does not publish the app or reset data. The Overview contains a guided tour.
For a short live federation proof, run `python scripts/run_dp_fl_demo.py --rounds 2`
separately; this trains only the toy classifier, not PPE. The saved two-round run
has cumulative per-client epsilon 5.753689 at delta 1e-5; do not conflate it with
the standalone Opacus or PPE SecAgg experiments. Copilot defaults local-only.

Integrated workflow tests use temporary synthetic fixtures, never project demo
records. The interactive demo still needs operator-provided permitted media and
actual detections to create incidents. Sync needs a configured local receiver;
no receiver or automatic delivery is enabled by the dashboard launcher.

1. Use the target machine and the project's virtual environment. Run `python -m pytest -q` and check the dashboard starts before the presentation.
2. Run `python scripts/init_demo_db.py` only if a local database needs initialization. Do **not** add `--reset` during setup; it deletes the selected local database.
3. If showing Dataset Explorer samples or inference, configure `SAFESIGHT_DATASET_ROOT` to the permitted local Construction-PPE dataset. No dataset media or weights are bundled in Git.
4. If showing PPE inference, confirm the ignored local checkpoint exists at `artifacts/runs/construction_ppe_demo/weights/best.pt`. The recorded checkpoint is weak; do not imply a reliable result. Webcam access is optional and must be granted by the operator.
5. For Gemini Copilot, configure the key in the local ignored `.env` beforehand. Do not show, paste, or narrate the secret. If unavailable, demonstrate the local deterministic fallback.
6. The Flower demos are optional in a short viva and may require their prepared runtime; show recorded evidence if not already run. Do not launch a new long training job during the talk.
7. Never create sample incidents, client counts, predictions, or metrics just to make a page look populated. If a live input produces no detections, say so and continue with the actual run record/report.

## 0:00–0:40 — Problem and honest scope

Say: “Manual camera review can miss PPE events, but centralizing raw footage raises governance and storage concerns. SafeSight-FL explores local processing with human review and federated-learning concepts. This is a college prototype, not a production safety system. Factory A/B/C are simulated clients on one laptop.”

## 0:40–1:25 — Dataset Explorer

Open **Model Lab → Dataset Explorer**. Show the dataset name/source/license notes, actual configured classes and split counts, class-instance distribution, annotated samples if the local dataset path is available, and the generated simulated-client manifest.

Say: “These counts come from the local Construction-PPE labels. The partition is a deterministic non-IID simulation, not data from three real factories. The raw images are outside this repository.” If samples are unavailable, show the manifest/report and state that the local dataset cache is not configured on this machine.

## 1:25–2:10 — Training and Evaluation

Open **Model Training** and **Model Evaluation**. Show recorded configuration/artifacts and held-out metrics that are actually present.

Say: “The recorded PPE run was YOLO11n, CPU, 320 pixels, one epoch, using 2% of the training split. Held-out precision was 0.00595, recall 0.10178, mAP@0.5 0.02079, and mAP@0.5:0.95 0.00804. These weak results are not adequate for safety decisions. We do not start full training during this demo.”

The evaluation page is the source of current local state. If a metric or artifact is absent on the presentation machine, show “not run” and use `artifacts/ppe_demo_summary.json` only if it exists; explain that it is a saved run summary, not a newly generated result.

## 2:10–3:05 — Detection and review workflow

Open **Operations → Live Detection**. If permissions and a local model are ready, capture a webcam still, upload an authorized still image, or upload an authorized clip. Video sampling is local at one frame per second for up to 30 seconds. Show only boxes/labels returned by the actual model; the upload remains in local session memory until cleared/closed, the decoder's temporary disk copy is removed, and only sampled still evidence is retained when an explicitly selected violation is saved.

If a supported violation is really produced, inspect it and save only the sampled-frame event you intend to review; then record a review decision only if you can genuinely assess it. PPE events are not deduplicated across time. For a permitted uploaded clip, optionally enable the experimental vertical left/danger-zone transition rule. Confirmed transitions are recorded automatically after sampling; inspect their clip offsets, annotated snapshots, and active/resolved lifecycle on Incidents. Sampling defaults to one second and is configurable from 0.25–5 seconds, but only the first 30 seconds are processed; sparse sampling can miss events and is not reliable monitoring. The prior normalized-polygon option remains available for an explicitly selected entry. Otherwise, do not manufacture an incident: explain that the returned prediction did not meet the event rule or that input/model is unavailable. Continuous video/RTSP processing and pause/resume controls are not implemented.

If vertical left/danger mode is enabled, explain that it samples at the configured interval, uses a configurable vertical boundary, hysteresis, two consecutive observations, and model-provided anonymous track IDs. An occupant first seen in the danger zone does not trigger an entry; a confirmed return resolves a matching entry record. These are experimental sampled-clip rules, not a live safety system. If polygon mode is enabled instead, an entry is flagged only after an observed outside-to-inside transition; a person first seen inside does not trigger an entry.

Open **Human Feedback** and describe Confirmed / False Positive / Uncertain as reviewer feedback. If the database has no review records, show the empty state; do not seed fake rows.

## 3:05–4:05 — Federated learning and privacy

Open **Federation & Privacy → Federated Learning** and **Privacy Center**. Show stored run state only.

Say: “We ran two separate one-round experiments on the PPE detection head, each with three simulated clients and 23, 28, and 7 local images. The Flower SecAgg+ round completed its secure aggregation workflow; it did not send raw images and the server received the aggregate. The resulting held-out mAP@0.5 changed from 0.02121 to 0.04901, while precision remained very weak at 0.00376. This is only a tiny integration check, not a reliable detector. Differential Privacy was not enabled; the Opacus epsilon 4.9946 at delta 1e-5 belongs to a separate toy workload. The lightweight classifier FL and two-round classifier SecAgg+ are also separate toy runs.”

Point out that raw CCTV/training images are not sent. SecAgg+ limits what the Flower ServerApp receives, but the temporary client update files and all three simulated clients are on the same laptop, so this is not an independent-site privacy boundary. Do not describe the experiment as proof of detector quality, Differential Privacy, or a real multi-factory deployment. If this machine's database does not have a recorded run, state that rather than enabling a status by hand.

## 4:05–4:45 — Copilot and audit

Open **AI Copilot** and explain that it summarizes structured event fields, not images, and never decides whether an incident happened. If an actual incident is available and provider access is configured, generate a report; otherwise show the deterministic fallback or explain the unconfigured state. Note that Gemini transmits structured event text to Google when used.

Open **Audit Integrity**. Run/show verification against the actual local chain and explain it is a tamper-evident SHA-256 hash chain—not blockchain, immutable storage, or a guarantee against every attacker.

## 4:45–5:30 — Offline state and close

Show the federation connectivity setting and explain: “Local processing and SQLite continue offline. Incident metadata is queued; after reconnect, a user can manually send allowlisted metadata to a configured authenticated receiver. There is no automatic sender.” If the loopback receiver is configured and running, click the explicit sync button and show the actual acknowledgement/count. Otherwise, show the pending count and say no receiver is configured. Do not imply the loopback demo is a real factory deployment.

Close with: “The next research steps are improving PPE model quality, replicating the head-only federation experiment at larger scale, evaluating PPE-specific privacy mechanisms, validating continuous tracking/video, and testing with independent sites. The sampled left/danger-zone workflow is experimental and can miss transitions; we report the gaps rather than substituting mock values.”

## Optional extra minute

If time allows, open **Model Versions** to show the recorded model kind/clients and current active version. Explain that activation validates a local PPE checkpoint and rollback switches to the previously active version; only exercise the buttons if the operator intends to change which model Live Detection uses. **Analytics** and **Settings** show actual SQL aggregates and local configuration. Avoid displaying machine-specific paths or secrets. If there is no real data for a view, point out the empty/not-run state.

## Fallback order

1. Webcam still snapshot, only if operator permission and camera work.
2. An authorized local image or video. The dashboard supports bounded sampled-frame video review and an experimental automatic zone-transition mode; continuous streaming remains unavailable. The CLI adapter accepts a local video source.
3. Saved manifest/run summaries and honest empty UI states. Static imagery is not a substitute for a model prediction; do not draw mock boxes or claim an incident.

No legal prerecorded PPE video is currently bundled. If a suitable clip is later added, record its source and license in `THIRD_PARTY.md` before demo use.
