# MatterGen + Fixed CHGNet Ranking Baseline

This is an adapted internal baseline for budget-constrained crystal discovery. It uses MatterGen to create a candidate pool, CHGNet single-point forces to rank unique valid structures, and CHGNet relaxation on a fixed Top-K selected before relaxation starts. It is not a reproduction of MADE (ICML 2026).

## Ranking and scientific scope

For candidate structure `X`, the ranking score is `s(X) = max_i ||F_i(X)||_2` from an unrelaxed CHGNet single-point calculation. Candidates are sorted by ascending score, with the stable candidate ID as the deterministic tie-breaker. The first K entries are frozen in `selected_candidates.json` before any relaxation begins. Relaxation outcomes never alter that ranking.

This initial-force heuristic is not an energy-above-hull estimate or a thermodynamic stability measure. CHGNet energies from different compositions are not compared to infer stability. Small force does not imply stability, novelty, or uniqueness. CIF parsing and StructureMatcher duplicate detection only provide structural validity/deduplication under the configured matcher settings. Relaxation success is recorded separately from whether the final atomic force meets `fmax`; neither establishes phase stability.

## Formal 64-to-8 configuration and result

The formal run used `mattergen_base`, seed 42, 64 generated candidates, CHGNet 0.4.2 single-point scoring, initial maximum atomic force ranking, selection budget 8, and CHGNet relaxation with cell relaxation enabled, `fmax=0.05 eV/Å`, and a 500-step maximum. The ranking score is minimized. Software/checkpoint revisions and the complete run parameters are in [`configs/fixed_v0.json`](../configs/fixed_v0.json) and the local `results/fixed_v0/run_config.json`.

All 64 generated CIFs parsed and passed the configured minimum-distance check; StructureMatcher found zero duplicates. All 64 single-point calls succeeded. Eight candidates were selected, all eight relaxations succeeded, and all eight met the independently checked final-force threshold. There was no selection shortfall. Total measured run time was about 160 seconds.

The wrapper accepted the requested `--generation-batch-size 4`, but a bug in that version of the wrapper invoked MatterGen with batch size 64 and one batch. That run completed successfully (peak observed GPU memory was about 4.3 GB). The wrapper now passes the requested batch size and computes the matching number of batches; a rerun with batch size 4 would use 16 batches. The historical run configuration records both requested and effective settings. See [`docs/experiment-log.md`](experiment-log.md) for the selected structures and force values.

The current policy is an implementation of the replaceable `RankingPolicy` protocol in `fixed_screening.py`. Future policies should document which precomputed information and compute budget they can access. For fair Random/Fixed/Agent comparisons, use the same candidate pool and explicitly align information access and physical-computation budgets.

## Environment and commands

Use the already configured Python 3.10 environment; this workflow does not install or update packages:

```bash
cd ~/gxq/5040/AIST5040-Materials-Agent
source ../.venvs/mattergen/bin/activate
```

Run against an existing CIF pool, with no MatterGen generation:

```bash
CUDA_VISIBLE_DEVICES=0 materials-fixed-screening \
  --candidate-dir results/smoke/cifs \
  --output-dir results/fixed_v0 \
  --top-k 8 --seed 42 --checkpoint mattergen_base \
  --device cuda:0 --fmax 0.05 --max-steps 500
```

To generate a 64-structure pool with MatterGen and then screen/select/relax Top 8 in one command:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 materials-fixed-screening \
  --candidate-dir results/fixed_v0/candidate_pool \
  --output-dir results/fixed_v0 \
  --generate-count 64 --top-k 8 --seed 42 \
  --checkpoint mattergen_base --device cuda:0 \
  --fmax 0.05 --max-steps 500
```

Equivalent module invocation, useful before refreshing the editable install, is:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python -m materials_agent.baselines.fixed_screening \
  --candidate-dir results/fixed_v0/candidate_pool \
  --output-dir results/fixed_v0 --generate-count 64 --top-k 8 \
  --seed 42 --device cuda:0 --fmax 0.05 --max-steps 500
```

Generation and screening are separable: run once with `--generate-count 64`, then rerun against that candidate directory without `--generate-count` (or point another policy at the same pool). MatterGen's current API writes `generated_crystals_cif.zip` and `generated_crystals.extxyz`; the wrapper reads the CIF archive into the candidate directory. The screening run copies each input into `cifs/` and does not modify the source pool.

## Input validation and compute accounting

Every CIF receives an ID based on its sorted relative path stem and SHA-256 content digest. The ID is stable for the same path/content, and duplicate names/content receive deterministic suffixes to keep IDs unique. Invalid CIFs, structures with non-finite periodic distance matrices, and any pair closer than the configurable default 0.5 Å are retained in the output records. Valid structures are compared by pymatgen `StructureMatcher(primitive_cell=False, scale=True, attempt_supercell=False)`; duplicate rows point to the first canonical candidate and are not scored again.

One CHGNet instance is loaded for all single-point scores and reused in the selected relaxations. Single-point records include eV/atom energy, forces in eV/Å, maximum atomic-force norm, status, elapsed time and failure reason. Ranking includes only successfully scored canonical structures. Exactly `min(K, successful unique scores)` relaxation calls are attempted; failed calls still consume a relaxation attempt and have their elapsed time recorded. The final force convergence flag is computed from a fresh CHGNet force prediction on the final structure, not from the optimizer return code.

## Two-candidate GPU smoke test

The implemented baseline was run against the two existing Smoke Test CIFs with `--top-k 2` on `cuda:0`. Both structures passed validation and deduplication; both were scored and selected. The baseline ranked InPF6 first (initial max force 0.2354 eV/Å) and Ba3CaRe2WO12 second (1.9071 eV/Å). Both relaxations completed and met the independently checked 0.05 eV/Å atomic-force threshold; final maxima were 0.0238 and 0.0458 eV/Å. The run took about 4.5 seconds after model loading, with about 0.33 s recorded for single-point calls and 3.14 s for relaxation calls. These results verify execution only, not stability.

Outputs are under the ignored local directory `results/fixed_v0_smoke/`; the two relaxed CIFs, CSV, selected list and metadata are present. The run metadata records CHGNet 0.4.2, pymatgen 2024.10.29, PyTorch 2.7.1+cu128, MatterGen source revision and the cached Hugging Face checkpoint revision.

## Output files

`results/fixed_v0/` contains:

- `generated/`: MatterGen archive and generated extxyz when generation is requested.
- `cifs/`: a copy of every input CIF, including invalid and duplicate inputs.
- `relaxed/`: CIF files for successful selected relaxations.
- `screening.csv`: one row per candidate, including validity, duplicate link, single-point force/energy/cost, rank, fixed selection, relaxation status/cost and final properties.
- `selected_candidates.json`: immutable ranking decision written before relaxation.
- `summary.json`: candidate, validity, deduplication, single-point and relaxation counts/costs, plus shortfall when fewer than K can be selected.
- `run_config.json`: seeds, budgets, model/device settings, package versions and available Git revisions.

Generated structures, model weights, and run outputs are not tracked by Git.
