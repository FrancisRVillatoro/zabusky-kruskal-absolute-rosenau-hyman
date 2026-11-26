# The Zabusky–Kruskal experiment for the absolute Rosenau–Hyman equation: code and data

This repository contains the numerical codes, the exported data and the analysis scripts of the article

> F. R. Villatoro, *The Zabusky–Kruskal experiment for the absolute Rosenau–Hyman equation*, manuscript submitted for publication (2026).

The article follows a periodic cosine under the absolute Rosenau–Hyman equation, u_t + (∂_x + ∂_x^3)(|u|^(p-1) u) = 0, and studies its returns, the zeros of the solution and the sensitivity of the computations for the exponent p = 9/4, with a fourth-order Padé scheme and Fourier–Galerkin truncations.

The repository is an archive of the numerical study and a self-contained route to reproducing the twelve figures of the article, the numbers quoted in it and the checks of the exported diagnostics. The figures that use only archived data need no integration. Those that depend on the analysis package `analysis/` need short computations (about 80 minutes on one core, no job submission). Full production HDF5 trajectories and certain full step-by-step histories remain on the Picasso supercomputer (SCBI, Universidad de Málaga); the archives include the available derived data, selected fields, code and checks. A computed diagnostic is not a certified continuum limit.

## Contents

| Path | Content |
|---|---|
| `archives/` | Ten ZIP files with the solvers, the campaigns, the exported data and the audits of the numerical study, preserved byte for byte. |
| `analysis/` | Scripts that compute the remaining quantities of the article from the archives, the short additional computations, the figure script and the script that prints the numbers quoted in the text. See `analysis/README.md`. |
| `check_key_results.py` | Check of the headline numbers against the source exports. |
| `audit_exports.sh` | Runs the three independent audits stored in the archives. |
| `validation/` | Logs of the audits and of the 43 unit tests, as executed during the preparation of the manuscript, and the environment used. |
| `SHA256SUMS` | Checksums of every file of the repository. |

## Quick start

Use Python 3 with NumPy, SciPy, pandas, matplotlib and h5py (`requirements.txt`). The Picasso production environment was Python 3.14.3, NumPy 2.3.5, SciPy 1.17.0 and h5py 3.15.1; the original source packages record their environments, and `analysis/README.md` records that of the analysis package. Use the following from the repository root, with `python` pointing to a suitable installation:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum -c SHA256SUMS --quiet
python check_key_results.py
bash audit_exports.sh
bash analysis/run_all.sh
```

The scripts operate on the immutable ZIPs in `archives/`. The audits are numerical recomputations from the exported fields and tables; they do not repeat production runs or certify the HDF5 contents not included in these archives. `analysis/run_all.sh` computes the quantities of the article that are not exported by the campaigns, writes them to `analysis/results/`, generates the 40 panels of the twelve figures in `figures_reproduced/` and writes the numbers quoted in the article to `analysis/results/paper_numbers.txt`. Figure generation is deterministic plotting of numerical arrays. No generative image synthesis is used. The outputs of these commands are not distributed, since the scripts regenerate them.

## Archives and provenance

| Archive | Role |
|---|---|
| `base_revision.zip` | Original Python solver, campaigns and exports for the earlier phases, AVF, unprojected and p=3 controls, positive-data extension, older figures and their source scripts. See the internal README, FILE_PROVENANCE.tsv and SOURCE_ARCHIVES.tsv. |
| `CNSNS_Fase9A_20261124.zip` | Initial-velocity scaling, lattice coefficients, 396 evaluations, local code and tests, profile extraction protocol. |
| `Fase9A_HDF5_3831810.zip` | Original Picasso exports: 42 minima and 42 matched-time profiles, layer metrics, and 12 replay comparisons. |
| `CNSNS_Fase9A_cierre_20261125.zip` | Independent profile audit, source, tables and figures. |
| `CNSNS_Fase9B_20261125.zip` | 39 local Galerkin experiments, length sweep, detuning, multiplier interventions, full-trace summaries and reduced plot traces; Padé integration and postprocessing source. |
| `Fase9B_CNSNS_resultados_20261125.zip` | Actual Picasso production: 13/14 new complete trajectories, one partial trajectory, 12 archived histories, reduced modal traces and full-step aggregate balance tables, logs and audits. |
| `CNSNS_Fase9B_cierre_20261125.zip` | Independent audit of modal identity, detuning, chronological return catalogues and power budgets; exact source tables and reduced-trace checks are distinguished. |
| `CNSNS_Fase9C_20261125.zip` | Exact implicit-stage tangent equations, AVF differentiation, nonlinear controls, pilots, manifests and tests. |
| `Fase9C_CNSNS_resultados_20261125.zip` | Actual Picasso production: 41 complete tasks and one partial tangent task, 122 segment audits and exported histories and diagnostics. |
| `CNSNS_Fase9C_cierre_20261125.zip` | Independent audit of finite-time gain, comparison envelopes, localization and perturbation tests. |

These archives are preserved byte for byte as a historical record (`SHA256SUMS` covers them). Their names and their internal reports, some of them in Spanish, follow the phases of the study (9A, zeros of the solution; 9B, spectrum and recurrence; 9C, sensitivity). Older README files contain status descriptions at the time of each campaign; the associated production and closure archives supersede those pending statuses, and the article supersedes the interpretations given in the internal reports. Historical Slurm files are preserved for provenance, not as launch instructions.

## Analysis package

`analysis/` is distributed as plain files because its scripts are run from the repository root. It contains the postprocessing of the archived data (momentum, catalogue of returns and slice coefficients, structure of the zeros), the short additional computations (initial layer of the Padé semidiscrete system, Galerkin runs at other lengths, variational equations and relative periodic orbits of the Galerkin truncations), the figure script and a script that prints the numbers quoted in the article. Its README lists inputs, outputs and cost of each script.

## Figure sources

Each figure of the article consists of two or four panels, stored as separate files `<name>_<k>.pdf`. For the figures with four panels the list gives the indices k from top left to bottom right.

* Figure 1 (`recurrence`, panels 1 and 2): space-time export of the base archive (series A, N=1800) and the 9B production trace `s7.2_N1800_d4`.
* Figure 2 (`returns`, panels 1, 2, 3, 4): `return_catalog.csv` and `return_time_scaling.csv` (built from the 9B traces and event tables, the AVF controls and experiments of the base archive and the 9C reports), `length_returns.csv` and `length_curves.npz` (Galerkin runs of `analysis/lengths.py`).
* Figure 3 (`modal_exchange`, panels 1, 4, 2, 3): 9B production traces (N=900, 1800), series A trace (N=450) and local Galerkin trace `central_K51`; the momentum and its contributions by harmonic are evaluated from the stored harmonics.
* Figure 4 (`null_law`, panels 1 and 2): 9B local `near_null.csv`; time averages of the seventh harmonic from the Galerkin and production traces.
* Figure 5 (`return_map`, panels 4, 1, 2, 3): `return_catalog.csv`.
* Figure 6 (`rpo`, panels 1, 4, 2, 3): `rpo_orbits.json` and `rpo_linear_check.csv`.
* Figure 7 (`initial_velocity`, panels 1, 2, 4, 3): 9A `initial_scaling.csv` and `scaling_summary.csv`; the lattice sum is evaluated by the figure script and checked against the archived values.
* Figure 8 (`initial_layer`, panels 1 and 2): `initial_layer.csv` and `initial_layer_profiles.csv`.
* Figure 9 (`zero_structure`, panels 1 and 2): `zero_local_profiles.csv`, extracted from the 9A production `minima/profiles.csv`.
* Figure 10 (`return_localization`, panels 1, 2, 4, 3): 9A production `minima/profiles.csv` and closure `finest_returns.csv`; the fourth fine-grid profile is the one recomputed with the refined step.
* Figure 11 (`transverse`, panels 1, 2, 4, 3): 9C closure `fixed_time_gains.csv` and `return_gains.csv`; 9C production `report/gain_traces.csv`; `gain_ladder_traces.npz`.
* Figure 12 (`regularization_positive`, panels 1, 2, 4, 3): 9A closure `finest_returns.csv`; 9C closure `regularized_final_gains.csv`; positive-data `experiments/results/positive{1000,2000}/history.csv` of the base archive.

The files without an archive label are written by the scripts of `analysis/` into `analysis/results/`. All plotted values are obtained from source files by `analysis/make_figures.py`. No curves are manually digitized.

## Code checks from a fresh directory

To reproduce the 43 unit tests without submitting jobs, create a new directory, extract `base_revision.zip` inside it, copy the contents of its `code/` into the test root, then extract the 9A, 9B and 9C **code** packages at that root (not the result archives). With that root as the working directory and `PYTHONPATH=.`, run:

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s fase9a/tests -v
python -m unittest discover -s fase9b/tests -v
python -m unittest discover -s fase9c/tests -v
```

The local Galerkin histories and initial-velocity calculations can be regenerated with the respective package README and case manifests. Production re-execution requires the original Picasso environment and HDF5 directory layout (or regeneration of the prerequisite trajectories); archive audits and figure regeneration do not. Do not rerun old production campaigns into their original output paths. Some raw historical README and manifest files refer to archived directories and older Slurm policies; these are not prerequisites for the quick start.

## Development

The codes, the numerical experiments and their audit were developed with the assistance of AI tools (OpenAI ChatGPT and Anthropic Claude), as declared in the article. The author reviewed the outputs and takes full responsibility for the content of this repository.

## Citation

If you use this code or these data, please cite the article and the version of this repository archived in Zenodo (`CITATION.cff`).

## Funding and acknowledgements

This work was supported by the Universidad de Málaga, Spain, through project PRO-B4-2026-011 of its Plan Propio de Investigación, Transferencia y Divulgación Científica. The author thanks the SCBI (Supercomputing and Bioinformatics Center) of the Universidad de Málaga for technical assistance and for access to the Picasso supercomputer.

## License

MIT License. See `LICENSE`.
