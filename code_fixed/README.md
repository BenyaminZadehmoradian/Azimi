# Fixed notebooks (INFO01)

- `INFO01_TIMING_FIXED.ipynb` — INFO00 with corrected timing instruments. With default switches the training path is INFO00's (warm-up uses a separate batch and a bit-exact, verified restore). Experiments: `EXP_ENDPOINT2_PROTOCOL`, `EXP_S10_RELATIVE_SCALE`, `EXP_S10_HEAD_EPOCHS`, `EXP_RECOVERY_MAX_EPOCHS`, `EXP_LOCAL_MAX_EPOCHS`, existing `BN_DIRECT_BIAS`. Change one per run.
- `BASELINE_INFO01_TIMING.ipynb` — baseline with the same timing instruments; `PHASES_MODE='phax_endpoint2_matched'` is control E0.
- Rebuild from the originals: `python patch_phax.py ../code/INFO00_BN05_CONTROL_DIAGNOSTICS.ipynb INFO01_TIMING_FIXED.ipynb` and `python patch_base.py ../code/PHAX_DIGITAL_BASELINE_VGG9_CIFAR10_MATCHED_FP32.ipynb BASELINE_INFO01_TIMING.ipynb`.
- CPU smoke tests (synthetic data, mechanics only): `tests/smoke_info01.py`, `tests/smoke_base.py`.

Report (Persian): `analysis/REPORT_FA.md`.
