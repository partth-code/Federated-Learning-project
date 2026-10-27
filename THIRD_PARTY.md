# Third-party software and data ledger

This project contains original integration and application code plus third-party dependencies and data references. The ledger distinguishes software dependencies, datasets, protocol examples, and repositories that are reference-only. It is not legal advice; re-check the exact version's notices and the applicable data terms before redistribution or deployment.

## Software and protocol dependencies

| Component | Project use | Verified status/version | License/source |
|---|---|---|---|
| Python standard library | Runtime, paths, SQLite binding, JSON, hashing | Python 3.13.12 test environment; Python 3.11 preferred | [Python license](https://docs.python.org/3/license.html) |
| SQLite | Local metadata and incident database through Python `sqlite3` | Used by schema, service, and dashboard tests | [SQLite is public domain](https://www.sqlite.org/copyright.html) |
| PyTorch | Toy federated classifier, Opacus demo, Ultralytics runtime | Installed and exercised by Flower/DP/PPE test paths | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Streamlit | Local dashboard | 1.65.0 verified; project requires `>=1.49,<2` for current layout API | [Streamlit license](https://github.com/streamlit/streamlit/blob/develop/LICENSE) |
| Flower | Lightweight federated-learning test and Flower App integration | 1.39.0; local simulated-client runs verified | [Flower Apache-2.0 license](https://github.com/flwrlabs/flower/blob/main/LICENSE) |
| Flower SecAgg+ | `SecAggPlusWorkflow` and `secaggplus_mod` in a two-round synthetic run and a one-round PPE detection-head run | Both local simulations completed with three simulated clients; PPE run shares only the head and does not use DP | [Official Flower repository](https://github.com/flwrlabs/flower); preserve upstream license/notices |
| Opacus | DP-SGD `PrivacyEngine` on standalone and three-client Flower toy workloads | 1.6.0; real per-client cumulative RDP accounting recorded; secure RNG disabled, experimental only | [Opacus Apache-2.0 license](https://github.com/meta-pytorch/opacus/blob/main/LICENSE) |
| Ultralytics | PPE-model training and inference adapter | 8.4.174 used in the recorded short experiment | [Ultralytics license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); upstream project is AGPL-3.0. Review obligations before redistribution/use beyond this prototype. |
| OpenCV | Local video decoding and frame sampling for the optional uploaded-video demo path | Declared direct dependency; used only for local file decoding | [OpenCV license](https://github.com/opencv/opencv/blob/4.x/LICENSE); bundled codec availability may vary by platform. |
| Google GenAI Python SDK (`google-genai`) | Optional structured-incident summary provider | 1.75.0 installed; a synthetic structured-event request was verified | [SDK source and license](https://github.com/googleapis/python-genai); a configured request transmits structured event text to Google's service. |
| NumPy | Synthetic workload data and model arrays | Used by Flower integration tests | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |

License labels above refer to upstream software repositories and do not grant rights to unrelated model weights, dataset images, video, or generated artifacts. Transitive dependencies have their own licenses; consult the installed package metadata when preparing a redistributable release.

## Dataset, weights, and media

| Asset | Project use | Status and terms |
|---|---|---|
| Ultralytics Construction-PPE | Local dataset exploration and one demo training/evaluation experiment | The inspected external dataset contained 1,416 images across 11 classes (1,132 train / 143 val / 141 test). Raw images are not in this repository. Upstream configuration/documentation carries an AGPL-3.0 notice; verify the exact dataset and download terms independently before use or redistribution. |
| YOLO11n base/trained checkpoint | One-epoch demo training and local inference | Downloaded/generated artifacts are local and ignored by Git. No checkpoint is distributed here; review Ultralytics model/software terms and weight provenance before redistribution. |
| CCTV/demo video or images | Optional future viva input | No prerecorded video is bundled. Use only media with documented permission/license and record its source before adding it locally. |
| UI reference PNGs in `ui_refs/` | Visual-only layout guidance | They are project UI references. Any example labels/numbers in them are not real measurements and must not seed dashboard state. |

The checked-in `artifacts/dataset_manifest.json` contains relative filenames, classes, counts, and simulated partition metadata—not image bytes or an absolute local dataset root. Dataset terms are separate from the software-package license.

## Repositories retained as reference-only

| Repository | Review note | Decision |
|---|---|---|
| [achuajays/PPE_Detection](https://github.com/achuajays/PPE_Detection) | Repository metadata and README have been observed to state different licenses; checkpoint provenance is not established. | No code or weights copied. Do not reuse until the conflict and weight terms are resolved. |
| [AGI-RIZEN/restricted-zone-intrusion-detection](https://github.com/AGI-RIZEN/restricted-zone-intrusion-detection) | No clear license file was identified in the reviewed repository snapshot. | Architecture reference only; no code copied. |
| [Flower secure aggregation example](https://github.com/flwrlabs/flower/tree/main/examples/flower-secure-aggregation) | Official protocol example used to understand the Flower SecAgg+ API. | SafeSight-FL invokes Flower's library APIs on both a synthetic workload and a bounded PPE detection-head workload; neither represents an independent-factory deployment. |
| [Flower `fl-dp-sa` example](https://github.com/flwrlabs/flower/tree/main/examples/fl-dp-sa) | Reference for how Flower describes DP plus secure aggregation in an example. | Not implemented as a combined SafeSight-FL PPE workload. |

## Attribution and redistribution checklist

### User-supplied YOLO26 model bundle — 2026-10-27

The user supplied trained `best.pt`, `last.pt`, `best.onnx` and base `yolo26n.pt`.
The selected checkpoint reports Ultralytics 8.4.177 and `yolo26n.yaml`, with eight
PPE/person classes. This is user-provided training work using Ultralytics, not an
original SafeSight architecture. Original files and checkpoint metadata are
preserved in local ignored storage with SHA-256 provenance. Source dataset/license
documentation was not supplied; no public redistribution is performed or licensed
dataset ownership claimed. Upstream Ultralytics terms still apply.

- Keep original license and notice files for any third-party code actually copied or modified.
- Mark adapted files and retain a source URL. The local dataset YAML identifies its upstream Ultralytics source in its header.
- Do not commit `.env`, API keys, raw CCTV, training images, unreviewed media, or model checkpoints.
- Before publishing a bundled installer/model/data release, audit direct and transitive licenses, dataset/model terms, notices, and any copyleft obligations for that exact distribution.
- Do not claim suggested reference repositories, dependencies, model weights, or datasets as SafeSight-FL original work.
