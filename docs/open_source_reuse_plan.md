# Open-source reuse and verification plan

## Scope and method

This record distinguishes installed software, actually exercised integrations, datasets/weights, and reference-only projects. A dependency's software license does not automatically grant rights to a separately distributed dataset or model checkpoint. Before redistribution, verify the exact release, license, notices, and data terms; see [`THIRD_PARTY.md`](../THIRD_PARTY.md). The project uses upstream libraries through their documented APIs and does not copy source from the community reference repositories below.

## Components actually used

| Component | Use verified in this repository | Boundary / attribution |
| --- | --- | --- |
| [Flower](https://github.com/flwrlabs/flower) | A three-client lightweight PyTorch integration test, a two-round synthetic SecAgg+ run, one bounded YOLO11n PPE detection-head FedAvg round, and one PPE head-only SecAgg+ round ran locally. | PPE runs are one CPU epoch and one round over tiny simulated clients; only the detection head is shared, and the evidence is weak. Sites are not independent factories; PPE DP was not enabled. Flower is Apache-2.0; retain upstream notices. |
| [Opacus](https://github.com/meta-pytorch/opacus) | A toy local-client DP-SGD demonstration ran through `PrivacyEngine`; the actual recorded run computed epsilon 4.99460264989912 at delta 1e-5, with clipping norm 1.0 and noise multiplier 1.0. | Toy workload only, not PPE training and not composed with either SecAgg+ run. The experiment warned that secure RNG was disabled; do not treat it as production DP configuration. Opacus is Apache-2.0. |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO11n trained and evaluated on the Construction-PPE dataset; the local inference adapter loaded the resulting demo checkpoint. | Upstream Ultralytics software is AGPL-3.0. The dataset and any weights have separate terms; review before redistribution. The short demo metrics are weak and are not a release-quality claim. |
| [Construction-PPE](https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/detect/construction-ppe.md) | Actual local dataset config and YOLO labels were parsed: 1,132 train, 143 validation, 141 test images; 11 configured classes. Deterministic manifests partitioned training samples into three simulated non-IID sites. | Dataset images are in an external local cache, not this repository. Dataset terms must be checked separately; no factory footage or independent factory datasets are claimed. Class names/counts and site partitions are recorded from local data, not UI mockups. |
| [Google GenAI Python SDK](https://github.com/googleapis/python-genai) | A structured synthetic incident was sent through the configured provider during setup verification; the app retains a deterministic local fallback. | Gemini receives structured incident fields only when configured; no images or CCTV are sent by this feature. Keep credentials in ignored `.env`; never commit keys. |
| [Streamlit](https://github.com/streamlit/streamlit) | Dashboard and all 14 navigation pages were rendered and browser-checked. | UI framework only; its current API use is reflected in the dependency minimum and project notices. |

## Reference-only projects and reuse decisions

- [`achuajays/PPE_Detection`](https://github.com/achuajays/PPE_Detection): reference only. Its licensing/provenance requires further review; no source or weights were copied into this project.
- [`AGI-RIZEN/restricted-zone-intrusion-detection`](https://github.com/AGI-RIZEN/restricted-zone-intrusion-detection): reference only. No clear license was found during the initial review; no source or weights were copied.
- [Flower secure aggregation example](https://github.com/flwrlabs/flower/tree/main/examples/flower-secure-aggregation) and [Flower fl-dp-sa example](https://github.com/flwrlabs/flower/tree/main/examples/fl-dp-sa): official API/examples used as implementation references. PPE SecAgg+ and the Opacus DP-SGD demonstration are separate runs; no combined PPE DP+SecAgg claim is made.
- [Frigate](https://github.com/blakeblackshear/frigate) and NVIDIA DeepStream were optional architecture references only and are not required dependencies.

No pretrained COCO model is represented as detecting helmets or vests. PPE results require the configured PPE-trained model and its actual classes. No third-party video clips or checkpoint weights are bundled in Git. The 14 UI reference images are visual-only design assets; their illustrative content is never used as application data.

## Reproducibility and follow-up gates

- Preserve this file and `THIRD_PARTY.md` when dependencies or data sources change; record exact package versions/releases and notices before distribution.
- Recheck upstream software licenses, dataset terms, checkpoint terms, and redistribution permissions before publishing a packaged release.
- Do not redistribute the external dataset, generated checkpoints, or evidence unless their applicable terms and consent/privacy requirements permit it.
- Treat Factory A/B/C as simulated logical partitions on one machine, not separate organizations.
- Keep measured outputs, privacy scope, and implementation limits explicit in the README, Privacy Center, test report, and viva materials.

## Primary sources

- Flower license: https://github.com/flwrlabs/flower/blob/main/LICENSE
- Flower SecAgg+ example: https://github.com/flwrlabs/flower/blob/main/examples/flower-secure-aggregation/README.md
- Flower DP + SecAgg example: https://github.com/flwrlabs/flower/blob/main/examples/fl-dp-sa/README.md
- Opacus license: https://github.com/meta-pytorch/opacus/blob/main/LICENSE
- Ultralytics license: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Ultralytics Construction-PPE dataset notes/config: https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/detect/construction-ppe.md
- Google GenAI SDK: https://github.com/googleapis/python-genai
