# AIST5040 Materials Agent

Research software for budget-constrained, physics-guided crystal discovery. The project compares candidate search policies using a fixed MatterGen generation pool and explicit CHGNet scoring and relaxation budgets.

The current internal baseline is an **adapted MatterGen + Fixed CHGNet Ranking** workflow. It ranks valid, unique candidates by the maximum atomic-force norm from an unrelaxed CHGNet single-point calculation, then relaxes a fixed Top-K. This is not a full reproduction of MADE and the force score is not a thermodynamic stability metric.

## Current status

- MatterGen `mattergen_base` candidate generation with a recorded seed.
- CIF validation, minimum periodic-distance screening, stable candidate IDs and `StructureMatcher` deduplication.
- CHGNet single-point energy/force calculation and deterministic fixed ranking.
- CHGNet structure and cell relaxation with independent final-force convergence checks.
- CSV/JSON experiment records, a GPU smoke test, a 64-to-8 run, and Mock-based unit tests.
- Agent integration, random-search comparisons, reference phase-diagram stability metrics and DFT validation remain future work.

## Repository layout

Keep this repository and Microsoft's MatterGen repository independent:

```text
~/gxq/5040/
├── AIST5040-Materials-Agent/
├── mattergen/
└── .venvs/mattergen/
```

The source packages are under `src/materials_agent/`. Experiment inputs live in `configs/`, documentation in `docs/`, and local generated data/results in the Git-ignored `results/` directory.

## Environment

The tested environment is Python 3.10.20, PyTorch 2.7.1+cu128, PyG 2.7.0, CHGNet 0.4.2 and an NVIDIA RTX 5090. The environment is isolated from Conda base at `~/gxq/5040/.venvs/mattergen`.

Activate the existing environment and enter this repository:

```bash
cd ~/gxq/5040/AIST5040-Materials-Agent
source ../.venvs/mattergen/bin/activate
```

`configs/blackwell-overrides.txt` holds the PyTorch/PyG versions required by the tested RTX 5090 setup; `configs/mattergen-environment.freeze.txt` records installed packages. See [docs/smoke-test.md](docs/smoke-test.md) for the GPU checks. To provision a new environment, use `bash scripts/setup_mattergen.sh`; it installs to the project-local environment, not Conda base.

## Run fixed screening

Run the recorded formal configuration, generating 64 candidates and relaxing the ranked Top-8:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 materials-fixed-screening \
  --candidate-dir results/fixed_v0/candidate_pool \
  --output-dir results/fixed_v0 \
  --generate-count 64 --generation-batch-size 4 --top-k 8 --seed 42 \
  --checkpoint mattergen_base --device cuda:0 --fmax 0.05 --max-steps 500
```

The exact configuration and completed run are recorded in [configs/fixed_v0.json](configs/fixed_v0.json) and [docs/experiment-log.md](docs/experiment-log.md). To screen an existing candidate pool, omit `--generate-count` and provide its path with `--candidate-dir`. Generation is optional so different ranking policies can share one pool.

The executable can also be run without relying on the installed console script:

```bash
PYTHONPATH=src python -m materials_agent.baselines.fixed_screening --help
```

## Tests and scientific scope

Run CPU-side unit tests with:

```bash
PYTHONPATH=src python -m pytest -q
```

The initial-force heuristic is only a ranking proxy. Low force does not establish stability, novelty or uniqueness; raw CHGNet energies from different compositions are not compared as stability scores. Relaxation success and force convergence are reported separately. See [docs/fixed-screening.md](docs/fixed-screening.md) for the algorithm, output schema, experiment settings and limitations.
