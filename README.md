# Physics-informed adaptive Lindblad propagation

This repository contains the reproducible benchmark scripts, processed outputs, plotting routines, and table-generation files for the manuscript:

**Physics-informed adaptive Lindblad propagation: structure-aware control, positivity correction, and benchmark analysis**

## Repository contents

- `sim_lindblad_regimes_physinf.py`  
  Nominal regime-dependent benchmark simulations.

- `sim_lindblad_regimes_physinf_strict.py`  
  Strict physics-informed reruns for Regimes II and III.

- `sim_lindblad_regimes_physinf_stress.py`  
  Stress-test tolerance sweeps used to demonstrate active trial-step rejection.

- `make_fig_regimes_acc_cost.py`  
  Generates the regime-dependent cost--accuracy figure.

- `make_fig_regimes_physicality.py`  
  Generates the regime-dependent physicality diagnostics figure.

- `make_fig_controller_decomposition.py`  
  Generates the controller-channel decomposition figure.

- `make_fig_stress_rejections.py`  
  Generates the stress-test rejection figure.

- `make_table_regimes_summary.py`  
  Generates the regime summary table.

- `make_table_stress_summary.py`  
  Generates the stress-test summary table.

- `outputs/`  
  Processed `.npz` benchmark outputs.

- `figures/`  
  Manuscript figures generated from the processed outputs.

- `tables/`  
  LaTeX tables generated from the processed outputs.

## Reproducing the results

Install the required Python packages:

```bash
pip install -r requirements.txt
