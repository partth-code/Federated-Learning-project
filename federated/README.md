# Federated learning

`lightweight_fl.py` runs a real three-client Flower integration test using a tiny PyTorch classifier and deterministic non-IID synthetic data. It is explicitly not a PPE detector. The run records real round accuracy and loss in SQLite and writes a JSON artifact.

`secagg_demo/` is a Flower 1.39 App using the official `SecAggPlusWorkflow` and `secaggplus_mod`. It retains a deterministic NumPy classifier demo and also accepts locally trained PPE detection-head updates. Factory A/B/C are simulated clients on one laptop; neither workload includes Differential Privacy. On Windows, launch through WSL/Ubuntu. A PPE SecAgg+ completion is accepted only after the server-side aggregate file exists and the caller validates its schema, finite values, nonzero change, saved checkpoint, and held-out metrics.

The separate Opacus demo records DP metrics for one toy workload and is not a SecAgg-enabled or PPE privacy experiment.

## Experimental PPE detection-head federation

Run `python scripts/run_ppe_fl_demo.py` for exactly one Flower FedAvg round and one local CPU epoch per simulated client, requesting a nominal 5% of each deterministic non-IID partition (rounded up to a whole image). It requires the existing local Construction-PPE dataset and demo checkpoint; it does not download data or start full training. Factory A/B/C run on one laptop. The callback sets Ultralytics' nominal batch size to the small demo batch so each tiny client cohort performs optimizer steps, rejects unchanged head weights, and checks that the final global head changed. The run records per-client head-update magnitudes, held-out test metrics before/after, and a SHA-256 hash for the saved global checkpoint.

The ordinary FedAvg run used 23, 28, and 7 images respectively (5% requested, rounded up). The shared head contains 432,801 parameters (1,731,204 raw bytes per update direction); only the final detection head is federated, not the complete YOLO model. Its held-out mAP@0.5 changed from 0.02121 to 0.04900, while final precision remained 0.00378.

For the secure PPE run, add `--secure-aggregation` to `scripts/run_ppe_fl_demo.py`. Flower's official SecAgg+ workflow completed one round with the same simulated-client partition/sample sizes. The server strategy received a quantized aggregate; the temporary locally trained client head files are still visible to an administrator on this single machine. The exact clipping bound is measured from the transmitted model tensors; in the recorded run it was 10.52. The resulting held-out mAP@0.5 changed from 0.02121 to 0.04901; final precision was 0.00376. This remains weak integration evidence, not an efficacy result or safety-ready model. DP was disabled, and no combined PPE DP+SecAgg guarantee is claimed. The earlier unchanged attempt remains invalid diagnostic evidence.

Run `python scripts/check_ppe_federation_adapter.py` for the separate offline schema/payload check. It does not train, federate, or send data.

Run it with:

```powershell
python -m federated.lightweight_fl
```
