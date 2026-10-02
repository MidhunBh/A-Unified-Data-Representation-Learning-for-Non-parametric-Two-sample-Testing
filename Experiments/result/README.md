# Experiment results

This directory contains structured outputs from reproduction runs,
diagnostics, provenance reconstruction, sensitivity studies, and
extensions.

The directory is intentionally kept mostly flat at this stage. Several
historical experiment scripts use hard-coded result paths, so reorganizing
existing files into new subdirectories could break reproducibility.

## Interpretation of result files

- Final or matched reproduction outputs are retained.
- Intermediate runs that contributed to identifying the final
  configuration are retained.
- `audit_*` files record methodological or implementation audits.
- `author*` files preserve reconstruction of runs associated with
  recovered author-job provenance.
- `pilot*` and diagnostic outputs are retained when they explain how a
  configuration or discrepancy was resolved.
- `split_ratio_cells/` contains the independently computed cells used to
  construct the merged fixed-budget split-ratio result.

Where both JSON and PKL versions exist, JSON should normally be preferred
for human inspection and provenance review; PKL files are retained for
compatibility with the experiment code.

## Cleanup policy

Research results are not deleted simply because a later run superseded
them. A result may be removed only when it is demonstrably accidental,
empty, byte-identical redundant output, or safely represented elsewhere
with no loss of provenance.

Physical reorganization of result paths should wait until hard-coded paths
in experiment scripts have been systematically refactored.
