# Phase 1 mock Agent

Run from the repository root with the existing Python environment.

```bash
source ../.venvs/mattergen/bin/activate
run_root=$(mktemp -d "${TMPDIR:-/tmp}/aist5040-agent.XXXXXX")
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m materials_agent.agent.mock_experiment \
  --output-dir "$run_root/synthetic" --seed 42 --budget 8
```

The synthetic pool has 64 candidates. Their CIF paths are placeholders. The mock evaluator generates synthetic observations from the seed and candidate ID. The mock planner reruns its selection after every attempt using the latest status and final force. It deliberately ignores initial force scores, so enabling score visibility exercises the information and cost plumbing alone. At a fixed seed the decisions, synthetic observations, and exports repeat. The summary also reports measured wall time, which varies by run.

To exercise the existing MatterGen pool with synthetic physics, use the Fixed Baseline CSV for IDs, formulas, and the baseline's pre-physics validity and duplicate flags. The loader matches CIFs by filename and the ID's content hash prefix, checks coverage against the input pool, and records full SHA256 hashes. It uses the eligibility flags independently of whether CHGNet screening passed.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m materials_agent.agent.mock_experiment \
  --output-dir "$run_root/fixed-pool" --seed 42 --budget 8 \
  --candidate-dir results/fixed_v0/candidate_pool \
  --screening-csv results/fixed_v0/screening.csv
```

For score visibility, add `--allow-initial-max-force` and use a new output directory. This exposes only the initial maximum force and records the original single-point call count and time under reused screening. New screening costs stay at zero. By default the score is absent and the historical screening computations are uncharged because the planner uses only pre-physics pool metadata. This choice and its associated costs must be part of the comparison protocol. The CSV's ranks, selections, relaxation results, and other columns have no path into planner state.

To expose structure-only descriptors parsed from each input CIF, add `--include-structure-features`. This is CPU-side pymatgen parsing and geometry/composition arithmetic; it does not call CHGNet. The versioned `Candidate.features` mapping contains atom count, element count, cell volume, density, lattice lengths and angles, and elemental atomic fractions. The run config records the feature version, source and candidate-pool preparation time. These descriptors are disabled by default. Structure-similarity descriptors are not included: they require a declared metric and candidate-pair comparisons, so their costs and meaning need to be agreed as a separate experiment setting.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q tests
```

## Runtime and exports

The runtime uses the shared `Candidate`, `State`, `Action`, `Observation`, `Planner`, `PhysicsEvaluator`, and `validate_action`. It gives the planner a fresh state with an allowlisted candidate projection and only observations from its own completed attempts. `visible_candidate_pool` is the shared projection for every policy; with structure features enabled, Agent and Classical Adaptive Planner implementations receive the same versioned `Candidate.features` values. Arbitrary candidate features are rejected. Evaluator errors are reduced to a generic failure message in the state. The run log records full error details. Planners may expose a short rationale as `last_rationale` without changing the shared `Action` contract.

The current repository has no Classical Adaptive Baseline implementation (`heuristic_search.py` is empty). The shared state path and feature extractor are ready for that policy, but its integration cannot be verified until the baseline implementation uses the same candidate pool, `RunConfig`, and `run_agent` input projection. This is an outstanding cross-policy integration check.

The runtime validates every proposed action against its own state and spends one attempt before calling the evaluator. Failed attempts, including exceptions and observations for the wrong ID, consume budget and cannot be repeated. A `stop` action consumes no attempt. Three consecutive invalid decisions or planner errors terminate by default. Each decision is flushed to the journal as it completes. Outputs live in a new directory and an interrupted run has no automatic resume.

| Artifact | Contents |
| --- | --- |
| `run_config.json` | Seed, limits, planner and evaluator names, score visibility, and screening provenance and cost |
| `candidate_manifest.json` | Planner-visible candidate metadata in pool order |
| `decisions.jsonl` | Actions, rationales, budget before and after, observations, errors, and CIF paths |
| `selected_candidates.json` | Baseline-shaped selection rows. `rank` is the attempt order and includes failures |
| `relaxation_results.csv` | Baseline field names where available, plus shared observation status and synthetic flag |
| `summary.json` | Termination reason, attempt counts, convergence, and separate new and reused screening costs |
| `relaxed/` | Optional evaluator output at `relaxed/<candidate_id>.cif`, recorded relative to the run directory |

In the CSV, `source_file` is relative to the candidate pool and `relaxation_status=success` denotes a returned calculation, with force convergence in its own column. The shared observation has no final total energy or CIF field, so the runtime omits the former and discovers the latter at the existing Fixed Baseline path convention. Use the synthetic flag in the CSV or run config to identify test data. The export can be verified independently while the MatterSim benchmark interface is pending.

Phase 2 can plug a structured LLM planner into the same contract and test schema validation, retries, and usage tracking with mocked API responses. Phase 3 can adapt the existing CHGNet wrapper with the fixed protocol's force threshold, step limit, and cell relaxation setting. Confirm the exported CIF and result format with the shared Benchmark before the authorized minimal real-physics run. Numerical force convergence is distinct from thermodynamic stability, and raw energy per atom across unrelated compositions does not establish a stability ranking.
