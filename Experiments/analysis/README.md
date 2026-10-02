# Analysis

Scripts in this directory summarize validated experiment outputs, produce
final reproduction figures/tables, and support provenance audits.

## Main analysis scripts

- `make_final_figures.py` — generate finalized reproduction figures.
- `finalize_fig4b_type1.py` — finalize Figure 4b type-I results.
- `make_c2st_type1_json.py` — construct structured C2ST type-I output.
- `plot_figure2_hdgm.py` — HDGM figure plotting.
- `print_table3_mnist.py` — summarize the reproduced MNIST Table 3 results.
- `show_results.py` — general result inspection.
- `trainability_audit/` — Phase-1/Phase-2 trainability and provenance audit.

## Legacy helpers

`legacy_helpers/` contains small one-off inspection scripts used during
the reproduction process. They are retained for provenance but separated
from the primary analysis workflow.
