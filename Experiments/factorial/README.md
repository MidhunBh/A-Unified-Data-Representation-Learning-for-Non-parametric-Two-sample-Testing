# Factorial and sensitivity experiments

This directory contains controlled sensitivity studies, their supporting
runners, and a small number of pilot/diagnostic scripts.

Files are intentionally kept at their historical paths because experiment
commands and provenance notes may refer to them directly.

## Main sensitivity studies

- `factorial_freeze_vs_joint_HDGM_d10.py`
  - HDGM d=10 frozen-encoder versus joint-training comparison.

- `freeze_vs_joint_mnist_M100.py`
  - MNIST M=100 companion freeze-versus-joint comparison.

- `factorial_latent_dim_HDGM_d10.py`
  - HDGM d=10 latent-representation-dimension sensitivity.

- `factorial_split_ratio_HDGM_d10.py`
  - Original fixed-budget train/test split sensitivity driver.
  - Retained for historical provenance.

## Completed split-ratio implementation

The completed fixed-budget split study used the corrected cell-based
implementation:

- `factorial_split_ratio_HDGM_d10_cell.py`
- `merge_split_ratio_HDGM_d10.py`
- `run_split_ratio_HDGM_d10_parallel.sh`

The cell runner removes the redundant repeated deterministic testing loop
found during the split-ratio audit while preserving the intended
permutation budget.

## Diagnostics and pilots

- `diagnose_joint_train_vs_test.py`
  - Diagnostic comparison of apparent training discrimination and held-out
    behaviour under joint adaptation.

- `pilot_freeze_vs_joint.py`
  - Early pilot used before the controlled freeze-versus-joint study.

These files are retained because pilots and negative/diagnostic results
form part of the methodological provenance of the reproduction.
