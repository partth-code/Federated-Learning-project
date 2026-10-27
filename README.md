# SafeSight-FL

**Privacy-preserving, self-improving and agent-assisted workplace-safety monitoring using Federated Learning**

SafeSight-FL is a final-year college prototype exploring local PPE detection, restricted-zone rules, human review, and privacy-aware collaboration across sites. It is a research/demo workspace—not a production surveillance system.

> **Status: PARTIALLY READY.** The project includes local detection/review services, a dashboard, and recorded YOLO11 head-only FedAvg/SecAgg+ and toy DP experiments. The development machine now uses user-trained YOLO26 weights with eight PPE/person classes. These weights are not bundled; their validation metrics are imported rather than independently evaluated. Continuous video monitoring and missing-PPE inference for this eight-class model remain incomplete.

### GitHub checkout versus the local demo

This repository contains source, tests, documentation, visual-only UI references
and historical experiment summaries. It excludes API keys, `.env`, databases,
raw images/videos, model weights and virtual environments. A fresh checkout starts
with an empty local database; Live Detection requires importing your own permitted
PPE weights. Saved experiment JSON files describe past runs and do not recreate
their checkpoints or populated database records. Follow Setup below, then
[import the trained bundle](#imported-yolo26-ppe-weights-2026-10-27).

## Problem and proposed solution

Manual CCTV monitoring can miss missing-PPE events or restricted-area entry. Moving all camera footage to a central training system also creates privacy, bandwidth, storage, and governance concerns.

The project investigates a local-first design: each site processes its own images locally; a human reviews resulting structured incidents; and federated-learning protocols aggregate model information without sending raw training images. Factory A/B/C are **simulated logical clients on one laptop**, not independent real factories. Bounded Flower experiments now federate only the PPE detection head, with both ordinary FedAvg and a separate official SecAgg+ run; the lightweight classifier and its SecAgg+ demonstration remain toy workloads.

## What is implemented

| Area | Current implementation | Important boundary |
|---|---|---|
| Dataset Explorer | Reads a local YOLO dataset manifest, class labels, split counts, and deterministic non-IID simulated-site partitions; displays annotated samples when the local root is configured. | Images remain outside Git. Factory partitions are simulated. |
| PPE model | Validated adapter with YOLO11 training history and imported YOLO26 support, class normalization and local activation/rollback. | The imported 8-class model detects PPE presence; it has no explicit missing-PPE classes. Its imported validation metrics are not local test results. Weights are local-only. |
| Local detection | Dashboard accepts a webcam **snapshot**, still image, or uploaded video clip. Video sampling defaults to one second and is configurable from 0.25–5 seconds, capped to the first 30 seconds. A reviewer explicitly selects qualifying sampled-frame PPE results before they are recorded. An optional vertical left/danger-zone mode automatically records confirmed person transitions from sampled frames. The CLI adapter also accepts a local image or video source. | No continuous webcam/RTSP stream, pause/resume controls, or production-validated tracking. Sparse experimental person tracking can miss zone transitions. PPE video incidents remain frame-level and are not deduplicated. Upload bytes remain in local Streamlit session memory until cleared/closed; the decoder's temporary disk copy is removed after analysis. |
| Restricted zones | Experimental vertical split defaults to 50% frame width, with configurable boundary/hysteresis and two-sample transition confirmation. It stores `DANGER_ZONE_ENTRY`, records an observed return as `LEFT_ZONE_ENTRY`, and resolves the matching local entry record with video offset/dwell duration and an annotated still. The existing normalized-polygon option remains available. | Initial observations in the danger zone do not create an entry; temporary missed detections do not by themselves create a transition. Sparse samples, short clips, weak detection, and unstable/reused tracker IDs can still miss or misattribute transitions. This is not a continuous tracker or operational alerting. |
| Incidents and feedback | Supported model-returned PPE violations can be stored as typed incidents. Human review supports Confirmed, False Positive, or Uncertain. | A detection can be missed; a review action is not ground truth unless a reviewer actually verifies it. |
| Evidence and audit | Qualifying image evidence is stored locally; incident/review actions are appended to a SHA-256 hash-chain audit log. | The audit chain is tamper-evident, not a blockchain or proof against every attacker. |
| Offline operation | Local detection and SQLite recording continue without federation connectivity. Incident metadata is queued locally; a manually triggered authenticated sender can deliver allowlisted metadata to a configured receiver and requires an acknowledgement before marking items synced. | No automatic/background send. Remote delivery requires a separately configured HTTPS receiver; the included demo receiver binds only to loopback. Images are never queued or sent. |
| Federated learning | Flower lightweight PyTorch integration test plus one bounded YOLO11n PPE detection-head FedAvg round and a separate one-round PPE head-only SecAgg+ run, each with three simulated clients. | One CPU epoch/client using a nominal 5% sample, rounded up to whole images; only the head is shared. Both are weak integration evidence, not safety-grade results. Sites are simulated on one laptop. DP was not enabled. |
| Secure Aggregation | Flower SecAgg+ (`SecAggPlusWorkflow` and `secaggplus_mod`) completed on both a synthetic NumPy classifier (two rounds) and the PPE detector head (one round). | The PPE run does not federate the full YOLO model and is not combined with DP or evidence of production security. |
| Differential Privacy | Standalone Opacus demo and three-client Flower DP-SGD with cumulative per-client accounting. | Synthetic toy workloads; secure RNG disabled and separate runs not composed. No combined PPE DP+SecAgg result. |
| Privacy Center | Reads local processing, federation, verified SecAgg state, DP workload scope, and available recorded privacy metrics. | It reports only stored run state; it does not imply PPE model privacy guarantees. |
| AI Incident Copilot | Optional Gemini provider summarizes structured incident fields; deterministic local fallback is available. | It never decides whether an incident happened. If Gemini is enabled, structured incident text is sent to Google; no image or raw video is sent by this integration. |
| Analytics and model registry | SQL-backed incident summaries and registered model versions with local activation and rollback for PPE checkpoints. | Missing experiments are shown as not run; switches require an existing, validated PPE artifact. |

### Why this prototype is useful

- Keeps raw CCTV and training images local in the intended multi-client architecture.
- Makes simulation boundaries explicit: the three sites share one machine and are not real factories.
- Separates the bounded PPE head-only FedAvg and SecAgg+ results from the toy classifier/Opacus workloads, so unrelated protocol plumbing is not misrepresented as PPE privacy or performance.
- Preserves human review and local audit records while keeping the LLM outside incident classification.
- Provides a viva-friendly path through dataset provenance, model artifacts, federated-learning concepts, privacy controls, and known gaps.

These are project design choices, not claims of novelty, complete privacy, or production security.

## Architecture at a glance

```text
Local webcam snapshot, still image, or sampled video frame
                 │
                 ▼
     Verified PPE model adapter ───────► actual model detections
                 │                                  │
                 │                                  ▼
                 │                        PPE incident pipeline
                 │                          │       │       │
                 │                          ▼       ▼       ▼
                 │                       SQLite  local    audit
                 │                       metadata evidence chain
                 │                          │
                 │                          ├── human feedback
                 │                          ├── local analytics / Copilot summary
                 │                          └── metadata-only pending queue
                 │                               └── explicit authenticated sync (optional)
                 │
                 └── optional sampled-video tracker
                     ├── vertical left/danger transition rule (experimental)
                     └── normalized polygon rule (experimental)
                         (no continuous stream)

PPE head-only experiments (not full YOLO federation):
  Flower FedAvg ───────────► 3 simulated clients, one round
  Flower SecAgg+ ──────────► 3 simulated clients, one protected aggregate

Separate toy workloads:
  Flower classifier FL/SecAgg+ ──► synthetic clients
  Opacus ─────────────────────────► DP-SGD with recorded epsilon/delta
```

## Data and model evidence

The checked-in dataset manifest records the locally inspected Ultralytics Construction-PPE dataset: **1,132 train images, 143 validation images, 141 test images, 11 classes**. The generated manifest records relative paths and class labels—not image bytes. Factory A/B/C training partitions are deterministic, non-IID **simulations**.

The initial historical PPE demo used YOLO11n, CPU, image size 320, one epoch, and 2% of the training split, then evaluated the held-out test split. Its actual results were:

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.00595 | 0.10178 | 0.02079 | 0.00804 |

These are weak demo-scale results, not a safety-performance claim. The trained weights and run outputs are local/ignored artifacts; the repository does not redistribute the dataset or weights. Review the upstream software and data terms before downloading, training, or redistributing anything. See [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Setup

Python 3.11 is preferred. The test environment used Python 3.13.12. CPU use is supported for the small demo paths, though ML libraries can take time and disk space to install.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/init_demo_db.py
python -m streamlit run dashboard/app.py
```

Open the local address printed by Streamlit. If `.env` already exists, do not overwrite it; update only the settings you intend to change. `.env` is ignored by Git.

For other platforms, create/activate a Python 3.11 virtual environment, install `requirements.txt`, copy `.env.example` to `.env`, and run the same `python` commands.

### Local configuration

The standard defaults use `data/safesight.sqlite3` for local metadata and `data/evidence/` for local incident evidence. To show real annotated dataset samples in the Dashboard, set `SAFESIGHT_DATASET_ROOT` in `.env` to a dataset root you are entitled to use. The dataset files themselves are not copied into this repository.

Optional Gemini configuration uses `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`. Never paste keys into source, docs, issues, screenshots, or Git. When the key is present, the Copilot may transmit structured incident fields to Google's API. Leave it empty to use the deterministic local fallback.

## Running checks and demos

Run the regression suite:

```powershell
python -m pytest -q
```

Initialize the database without deleting existing local records:

```powershell
python scripts/init_demo_db.py
```

The optional `--reset` flag deletes and recreates the selected local database. Use it only when you intentionally want to discard that database's records.

### Dataset, training, and inference

```powershell
python scripts/inspect_dataset.py <local-construction-ppe-root>
python scripts/train_ppe_demo.py
python scripts/run_ppe_inference.py <local-image-or-video> --model artifacts/runs/construction_ppe_demo/weights/best.pt
```

Dataset inspection reads the path supplied on the command line. The demo training script is fixed to a short one-epoch, 2%-fraction CPU run; it is not a full-training launcher and should not be presented as one. The Ultralytics dataset configuration includes an upstream download URL; review its terms before allowing any download. Inference output contains only detections returned by the configured PPE model; a generic COCO person detector is not accepted as a helmet/vest detector.

### Experimental sampled-video zone demo

Launch the existing dashboard with `python -m streamlit run dashboard/app.py`, choose **Live Detection → Upload video**, select **Experimental automatic left/danger transitions**, then click **Analyze video locally**. The frame-sampling interval defaults to one second; the boundary defaults to 50% and can be adjusted. Confirmed tracked-person transitions are stored locally without manually selecting each transition. The **Incidents** page shows entry/return events, active/resolved state, clip offsets, elapsed danger-zone duration, filters, and saved local snapshots.

Use only a clip you are permitted to process. This experiment analyzes at most the first 30 seconds; it does not provide live webcam/CCTV streaming, Start/Pause/Resume/Stop controls, dwell alerts, or reliable safety monitoring. Initial observations already in the danger zone are not treated as entries. No sample video is bundled, and no movement event is created from a still image.

### Federated learning and privacy demonstrations

```powershell
python scripts/run_fl_demo.py --rounds 3
python scripts/run_ppe_fl_demo.py
python scripts/run_ppe_fl_demo.py --secure-aggregation
python scripts/run_dp_demo.py
python scripts/run_secagg_demo.py
```

- The Flower command runs a lightweight PyTorch classifier integration test over three simulated, deterministic non-IID clients.
- The PPE command is a deliberately bounded YOLO11n head-only experiment: one CPU epoch and one round, requesting a nominal 5% of each local simulated partition (rounded up to a whole image). By default it uses FedAvg; add `--secure-aggregation` to use Flower's SecAgg+ workflow for that same PPE head workload. It requires the local Construction-PPE dataset and existing `best.pt`; it never starts full training or downloads replacement data. The global artifact is hash-checked, registered, and evaluated on the held-out test split. Factory A/B/C run on one machine; no raw images are sent. DP is disabled in both modes.
- The DP command runs Opacus DP-SGD on a toy local workload and stores its computed privacy accounting result in SQLite and a local artifact. Its epsilon is workload/configuration specific; it does not describe the PPE model.
- The SecAgg+ command runs a separate Flower protocol integration test. On Windows it expects WSL/Ubuntu and the Flower demo runtime. Its workload is a synthetic NumPy classifier and is not combined with the DP-SGD run.

The lightweight classifier and Opacus DP demonstrations remain separate from PPE training; the PPE `--secure-aggregation` option is the sole SecAgg+ run on the PPE head. Do not compare toy workload results as PPE metrics or describe Factory A/B/C as real independent organizations.

### Optional offline metadata sync demo

The sender is disabled unless both `SAFESIGHT_SYNC_ENDPOINT` and `SAFESIGHT_SYNC_TOKEN` are configured. It sends only allowlisted incident metadata, never evidence paths, images, or video. Delivery is manual from **Settings → Send pending metadata now**; items are marked synced only after the receiver acknowledges the exact queue ID. Remote destinations must use HTTPS. Plain HTTP is accepted only for loopback demo addresses.

For a same-laptop test, generate a token locally with `python -c "import secrets; print(secrets.token_urlsafe(32))"` and put the same value in `SAFESIGHT_SYNC_TOKEN` in `.env` (do not commit it). Set `SAFESIGHT_SYNC_ENDPOINT=http://127.0.0.1:8767/v1/incidents`, then start the included loopback-only receiver in one terminal:

```powershell
python scripts/run_sync_receiver.py
```

Restart the dashboard, create/retain pending demo metadata, switch the sidebar federation toggle back online, then explicitly click **Send pending metadata now** on Settings. The receiver stores metadata in the separate local `data/sync_receiver.sqlite3`; it is a demo receiver, not a Flower server or a real factory deployment. No remote receiver is configured by default.

## Dashboard tour

The navigation is grouped into Operations, Model Lab, Federation & Privacy, and System. The dashboard uses local database records and actual local artifacts; missing experiments are labeled rather than populated with mockup values.

- **Overview:** current local site, incident, pending-queue, and audit counts.
- **Live Detection:** webcam snapshot, still-image upload, or a local video clip sampled at one frame per second (up to 30 seconds); selected real model outputs can be saved as incidents. Optional polygon/person tracking is experimental and sparse. Continuous video/RTSP is not implemented.
- **Incidents / Human Feedback:** view stored incidents and record Confirmed, False Positive, or Uncertain review decisions.
- **Dataset Explorer / Model Training / Model Evaluation:** inspect real manifests, run artifacts, and held-out test metrics where they exist.
- **Federated Learning / Privacy Center / Model Versions:** inspect the recorded one-round PPE head-only experiment, separate toy integration runs, and actual registry/privacy state; activate or roll back among validated local PPE checkpoints.
- **Audit Integrity / AI Copilot / Analytics / Settings:** verify the local hash chain, summarize structured events, query actual incident counts, and view local configuration.

Fourteen UI reference images live under [`ui_refs/`](ui_refs/). They are visual references only; any illustrative values in them are not project data and must never be copied into the dashboard.

## Privacy, safety, and threat boundaries

- Raw camera images and training images are not sent to the Flower federation examples and are not stored in SQLite. Qualifying still-image incident evidence is saved locally. An uploaded clip remains in local Streamlit session memory until cleared/closed; only selected sampled stills may be saved as incident evidence, and the decoder's temporary clip file is removed after analysis.
- The offline queue contains incident metadata only. A manual HTTPS/loopback sender is implemented, but no remote receiver is configured by default and no automatic delivery occurs.
- One bounded PPE head-only round used Flower SecAgg+; the server strategy received the aggregate, not individual updates. The simulated clients and temporary update files still reside on the same laptop, so a same-machine administrator/orchestrator can inspect them. This is not a multi-organization privacy boundary, does not cover the full YOLO model, and is not Differential Privacy.
- Opacus DP-SGD now also runs within three real Flower toy clients: `python scripts/run_dp_fl_demo.py --rounds 2`. Each client's accountant persists across rounds; only noised model parameters and accounting metadata are sent. This is experimental example-level DP on synthetic data, not client-level DP, PPE DP, or combined DP+SecAgg. Separate runs are not composed.
- The DP demo's test warning states secure RNG is disabled. Do not use its result as a production privacy guarantee.
- The hash-chain audit log can expose local modification; it is not immutable storage and is not called a blockchain.
- The Copilot uses structured incident fields only. With Gemini configured, those fields leave the machine for the provider; use the local fallback when external processing is not allowed.
- This prototype has no authentication, role-based access control, hardened deployment, independent-site orchestration, or production safety validation.

## Limitations and next work

1. Train and evaluate a substantially stronger PPE model on legally usable data; report full held-out metrics and artifacts.
2. Validate tracker identity across real consecutive frames, then connect a continuous stream to configurable restricted-zone polygons and test combined PPE/zone behavior. The current sampled-video tracker is only an experimental demo path.
3. Add continuous local video/live tracking and a tested tracker-to-zone path; handle webcam/RTSP permissions safely. The current uploaded-clip path is bounded sampled-frame review only.
4. Deploy and independently secure a remote receiver before claiming cross-site synchronization; the included transport/receiver test is a local prototype only.
5. Repeat and strengthen PPE federation experiments, assess whether broader model federation is practical, and separately validate whether technically valid DP can be composed with SecAgg+ for the PPE workload.
6. Complete installation testing on the target Python version and document exact hardware/runtime constraints.
7. Keep security, privacy, licensing, and human-review limitations explicit before any real-site pilot.

## Project documentation

- [Supervisor milestone log](docs/SUPERVISOR_LOG.md)
- [System test report](docs/test_report.md)
- [Final audit and open release gates](docs/FINAL_AUDIT.md)
- [Open-source reuse plan](docs/open_source_reuse_plan.md)
- [Third-party software/data ledger](THIRD_PARTY.md)
- [Viva guide](docs/VIVA_GUIDE.md)
- [Demo script](docs/DEMO_SCRIPT.md)

## AI Incident Copilot

The Copilot page defaults to local templates for incident reports, supervisor
actions and daily counts. Select a UTC date to summarize all recorded alerts that
day (not only the latest 20). Reports preserve human feedback and do not confirm
violations. Downloadable reports are deterministic local text.

Gemini explanations require the page's explicit opt-in. Only bounded allowlisted
event metadata is sent: no images, evidence paths, person IDs or free-form notes.
Provider output is unverified assistance and must be checked against the event.
Missing credentials, failures and empty responses use local fallback; source and
outbound-attempt status are displayed. SDK requests use a 15-second HTTP timeout
and one attempt; exception details are not exposed. Credentials remain in `.env`.

## Privacy Center evidence view

The Privacy Center separates completed experiments by run, with the latest recorded
round, simulated client count, SecAgg/DP flags and raw model-array bytes where
recorded. Missing fields remain blank. Historical results do not describe a live
server, and separate SecAgg and DP experiments do not imply combined protection.
The ONLINE/OFFLINE toggle is a demo control, not a network probe. DP accounting
includes its workload scope; malformed/partial DP records cannot enable its panel.
Raw imagery stays out of federation; optional Gemini structured-incident requests
are a separate outbound path. All three sites share one laptop.

## Optional GPU training

On this machine, the ignored `.venv-gpu` contains CUDA-enabled PyTorch separately
from the dashboard's CPU runtime. User-approved full training is launched with
`.venv-gpu\Scripts\python.exe scripts/train_ppe_gpu.py --epochs 50 --batch 4 --imgsz 512`.
Do not launch duplicate jobs. Model Training shows saved epoch progress after
refresh; local logs are in `artifacts/runs/gpu-training.stdout.log` and
`gpu-training.stderr.log` for the current supervised launch. The run uses the
full training split, early stopping, and a fresh output directory. Held-out test
evaluation and registration follow training; existing detection weights are
not replaced automatically. Keep the laptop powered and awake.

## Open the local demo

Run `powershell -ExecutionPolicy Bypass -File scripts/launch_demo.ps1` from the
repository, then open http://127.0.0.1:8501 on that laptop. The server is loopback-only,
with no database reset, full training or remote publication. Overview contains a
guided demo route. Use `-Port 8502` if another app already occupies the default port.
The supported demo uses operator-supplied media and saved real experiments;
continuous monitoring and bundled prerecorded PPE video remain unavailable.

## Daily safety analytics

Analytics supports inclusive UTC date ranges (up to 366 days), site and camera
filters, daily trends, and event/site/camera charts. Counts include recorded PPE
alerts, zone events and current Confirm/False Positive/Uncertain decisions.
Repeated reviews do not duplicate incidents. Dates refer to incident time, not
review time; counts can therefore change after a later review. Empty periods
show zero, ties for most common alert are preserved, and invalid timestamps are
reported rather than silently assigned a date. These are alert counts, not
detector accuracy or proof of violations. No LLM or external service is involved.

## Imported YOLO26 PPE weights (2026-10-27)

Live Detection now uses the user's trained `best.pt`, registered as
`model-6aabf259c7`. The original YOLO11 version `model-98f8352724` is available
through Model Versions → Roll back. All four supplied files are kept locally
under `artifacts/weights/user-0397d07f009b/` with SHA-256 checks and an import
manifest. The originals are untouched. `last.pt` is retained as the final
checkpoint; `best.onnx` is a retained export (ONNX runtime has not been verified);
`yolo26n.pt` is retained as the base checkpoint. These binaries are ignored by Git.

The trained detector exposes Dust Mask, Eye Wear, Shield, boots, glove, helmet,
person and vest. Adapter aliases are Dust Mask → dust_mask, Eye Wear → goggles,
glove → gloves, with lowercase normalization. New accessory labels are displayed
only and do not add incident rules. There are no explicit missing-PPE classes,
so this checkpoint alone does not create no-helmet/no-vest alerts. Tracked person
zone rules can still use its person class.

Training/Evaluation pages distinguish imported checkpoint validation metrics
from prior local test results. The imported metrics are precision 0.85045,
recall 0.83311, mAP50 0.87723 and mAP50-95 0.59548. Their dataset provenance and
split have not been independently verified; local held-out evaluation is not
run. The old 11-class dataset cannot be evaluated against this 8-class model
without an explicit dataset/label mapping. Prior FL, SecAgg and DP experiments
remain historical evidence for their own workloads.

For repeatable import, use `python scripts/import_ppe_weights.py --source-dir <local-bundle-directory> --sample <authorized-local-image>`.
The importer validates classes and actual inference before activation, never
creates incidents from its smoke test, and checks hashes before copying/reusing files.

## Attribution

SafeSight-FL uses and studies third-party packages, Flower examples/protocols, and an externally sourced PPE dataset under their respective terms. This repository does not claim upstream work as original. Exact roles, sources, license notes, and reference-only repositories are tracked in [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`docs/open_source_reuse_plan.md`](docs/open_source_reuse_plan.md).
