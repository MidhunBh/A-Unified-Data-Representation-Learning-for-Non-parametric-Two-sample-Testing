# A Unified Data Representation Learning for Non-parametric Two-sample Testing

This repository is the official implementation of [RL-TST](https://arxiv.org/abs/2030.12345). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Model

All file name that contain "semi" will be our proposed paradigm to conduct the two-sample testing.

## Praktikum closeout — October 2026

The planned reproduction and sensitivity phase for *A Unified Data Representation Learning for Non-parametric Two-sample Testing* is complete. This repository retains implementations, results, execution logs, and documented discrepancies.

Beyond reproduction, the completed sensitivity studies examine encoder freezing versus joint training, latent dimension, fixed-budget train/test allocation, and controlled addition of independent nuisance dimensions. Preliminary pretrained-representation experiments and implementation checks are recorded separately from reproduced results.

### Interpretation and limitations

- Results apply to the recorded configurations; exact agreement with every published value is not claimed.
- Some historical runners repeat an identical deterministic test decision. These repetitions do not constitute additional independent trials. The corrected split and nuisance runners use one decision per outer trial.
- Tian-style autoencoder experiments use pooled unlabeled training and held-out inputs. This protocol should be distinguished from representation learning restricted to training data.
- Historical DINOv2 preprocessing and null-sampling issues were addressed in a separate exploratory implementation; older outputs remain historical diagnostics.
- Sensitivity results are exploratory. Small null runs and adapter smoke tests do not establish precise calibration or comparative performance gains.

Historical results are retained for provenance. Corrections should remain distinguishable from the implementations that produced earlier outputs.

### Retained experiment documentation

- [`Experiments/factorial/README.md`](Experiments/factorial/README.md) — controlled sensitivity studies and provenance notes.
- [`Experiments/result/README.md`](Experiments/result/README.md) — retained result-file organization and provenance policy.
- [`Experiments/factorial/factorial_split_ratio_HDGM_d10_cell.py`](Experiments/factorial/factorial_split_ratio_HDGM_d10_cell.py) — corrected fixed-budget split-ratio study.
- [`Experiments/factorial/factorial_ambient_nuisance_HDGM.py`](Experiments/factorial/factorial_ambient_nuisance_HDGM.py) — controlled independent-nuisance study.
- [`Experiments/extensions/dinov2/frozen_small_m_pilot_v2.py`](Experiments/extensions/dinov2/frozen_small_m_pilot_v2.py) — corrected frozen-DINOv2 exploratory validation.
- [`Experiments/extensions/dinov2/lora_mnist_smoke.py`](Experiments/extensions/dinov2/lora_mnist_smoke.py) — small-scale adapter engineering validation.

**Closeout status:** planned reproduction and sensitivity experiments completed; documented discrepancies and limitations remain.

