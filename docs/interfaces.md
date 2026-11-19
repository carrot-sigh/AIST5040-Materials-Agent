# Shared Agent–Physics Contracts

This document defines the version 1 boundary between the Agent/Planner work and the Physics/Benchmark work. It is intentionally a small Python interface in `materials_agent.contracts`; it adds no runtime dependency and does not implement an LLM or an experiment runner.

## Architecture and decision cycle

The initial experiment uses a fixed pool of 64 CIF candidates and permits at most 8 CHGNet relaxation attempts. The Agent sees a `State`, returns one constrained `Action`, and the runtime validates it before dispatching one `PhysicsEvaluator.relax` call. The resulting `Observation` is appended to history and the remaining relaxation budget is decremented before the next decision.

```text
fixed candidate manifest -> State -> Planner.select_action -> validate_action
                                                    | relax(candidate)
                                                    v
                                           PhysicsEvaluator
                                                    |
                                      append Observation, spend 1
                                                    +----> next State
```

The current contract contains only `relax` and `stop`; it does not let a Planner issue shell commands, paths, arbitrary tool names or online generation requests.

## Data contracts

All four types are frozen dataclasses with explicit `to_dict()`, `from_dict()`, `to_json()` and `from_json()` methods. JSON is deterministic (sorted keys, compact separators) and rejects NaN/Infinity. Field names are stable and units appear in names.

### Candidate

`Candidate` describes data the Agent is permitted to see, not the hidden benchmark record.

```python
Candidate(
    candidate_id="gen_12-569f4de39f2f",
    formula="BiPdPt2",
    cif_path="gen_12.cif",             # relative to candidate-pool root
    initial_max_force_ev_ang=0.004385,  # omit/None when score access is disabled
    cif_sha256="<64 lowercase hex characters>",
    features={},
)
```

Candidate IDs are opaque stable strings. For a shared pool, preserve CIF filenames and bytes; the existing baseline derives IDs from the sanitized filename stem plus a SHA256 content prefix. `cif_path` must be a relative `.cif` path and cannot escape the pool root. `cif_sha256` is optional in memory, but should be populated in a shared manifest. `features` is reserved for explicitly approved public/pre-screen fields. Contract validation rejects nested feature keys indicating relaxation, observations, final values, benchmark, MatterSim, stability labels or energy above hull; do not encode hidden labels in generic features.

The optional `geometry-v1` feature extractor in `materials_agent.structure_features` parses the candidate CIF with pymatgen and derives atom count, element count, cell volume, density, six lattice parameters and elemental atomic fractions. It performs no model inference. Runtime visibility is opt-in with `RunConfig.allow_structure_features`; the exact allowlisted fields are copied to every planner's `State`, so a Classical Adaptive policy using the same runtime receives the same values as the LLM Agent. CIF parsing time is recorded separately from physics costs. `heuristic_search.py` is currently empty, so actual Classical Baseline integration remains to be checked when that policy is implemented.

Pairwise structure similarity is not part of this initial feature set. It requires an agreed representation and matching threshold, and incurs pairwise CPU work; it should be separately configured and costed if added.

### Observation

One `Observation` corresponds to one actual relaxation attempt. Status means:

- `converged`: the evaluator's declared force criterion was met.
- `not_converged`: execution returned a structure, but the force criterion was not met.
- `failed`: the physical calculation failed; `error` must preserve the reason and results may be `None`.

`converged` only describes numerical force convergence. It does **not** mean thermodynamically stable, novel, unique, or DFT validated.

```python
Observation(
    candidate_id="gen_12-569f4de39f2f",
    status="converged",
    final_energy_ev_atom=-5.31,
    final_max_force_ev_ang=0.021,
    elapsed_seconds=4.82,
    error=None,
)
```

### State

`State(candidates, history, remaining_relaxations)` is the complete Agent-visible state for one decision. History contains only completed observations tied to known candidates, with at most one entry per candidate. A failed attempt is still history and counts as an evaluation. `evaluated_candidate_ids` supports membership checks. All budget values are non-negative integers.

The State deliberately has no MatterSim label, hidden final benchmark table, or results for candidates that have not been relaxed. Candidate's optional initial force is populated only when the experiment policy explicitly grants access.

### Action

`Action("relax", candidate_id)` requests one allowed relaxation. `Action("stop")` ends the search. No free-form command or tool arguments exist in this schema. The runtime must call `validate_action(state, action)` immediately before dispatching a relaxation.

## Interfaces

```python
from typing import Protocol
from materials_agent.contracts import Action, Candidate, Observation, State

class Planner(Protocol):
    def select_action(self, state: State) -> Action: ...

class PhysicsEvaluator(Protocol):
    def relax(self, candidate: Candidate) -> Observation: ...
```

Minimal runtime control flow belongs to the integrating application:

```python
action = planner.select_action(state)
validate_action(state, action)
if action.action_type == "stop":
    return state
candidate = next(c for c in state.candidates if c.candidate_id == action.candidate_id)
observation = evaluator.relax(candidate)  # evaluator returns failed Observation on failure
state = State(
    candidates=state.candidates,
    history=[*state.history, observation],
    remaining_relaxations=state.remaining_relaxations - 1,
)
```

Do not decrement the budget only on success. The attempt was spent as soon as the evaluator was invoked. If the evaluator itself raises unexpectedly, the runtime should convert that exception into a `failed` Observation and decrement the budget.

## JSON exchange example

The following is a complete state with one completed relaxation and one remaining attempt:

```json
{
  "candidates": [
    {
      "candidate_id": "gen_12-569f4de39f2f",
      "cif_path": "gen_12.cif",
      "cif_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "features": {},
      "formula": "BiPdPt2",
      "initial_max_force_ev_ang": null
    },
    {
      "candidate_id": "gen_23-b777dac96b3b",
      "cif_path": "gen_23.cif",
      "cif_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "features": {},
      "formula": "SrCdHgPd",
      "initial_max_force_ev_ang": null
    }
  ],
  "history": [
    {
      "candidate_id": "gen_12-569f4de39f2f",
      "elapsed_seconds": 4.82,
      "error": null,
      "final_energy_ev_atom": -5.31,
      "final_max_force_ev_ang": 0.021,
      "status": "converged"
    }
  ],
  "remaining_relaxations": 1
}
```

Possible next-action JSON:

```json
{"action_type":"relax","candidate_id":"gen_23-b777dac96b3b"}
```

Failure observation JSON preserves the consumed attempt:

```json
{"candidate_id":"gen_23-b777dac96b3b","elapsed_seconds":2.1,"error":"optimizer timed out","final_energy_ev_atom":null,"final_max_force_ev_ang":null,"status":"failed"}
```

Read and write these values with `State.from_json(...)`, `Action.from_json(...)`, and each object's `to_json()` method rather than maintaining parallel ad hoc schemas.

## Budget rules

- Candidate-generation and relaxation budgets are separate. This stage uses a pre-generated fixed pool of 64 and does not generate online.
- The experiment config/runtime owns a maximum of 8 CHGNet relaxation attempts; the LLM does not control this limit.
- Every evaluator invocation consumes one relaxation attempt, even if it raises or returns `status="failed"`.
- Repeated relaxation of any candidate already present in history is rejected.
- `validate_action` rejects unknown candidate IDs, repeats and requests with no remaining budget.
- `stop` is always valid and consumes no relaxation budget, including when the budget is zero.
- Future physical tools (for example, another force field or DFT) need separate explicitly configured budgets; they must not silently share or bypass the CHGNet counter.

The current State carries remaining CHGNet attempts only. The application initializes it from the experiment's configured budget of 8 and is responsible for decrementing on every attempt. A future multi-tool protocol can add separate budget fields without changing the meaning of this one.

## Information access and leakage prevention

The completed Fixed Ranking baseline computed CHGNet initial scores for all 64 candidates. Those scores are **precomputed screening information**, not free Agent observations. Whether `initial_max_force_ev_ang` is populated must be an explicit experiment-config choice. Record that choice and the score-generation cost for every Random/Fixed/Agent comparison.

The Agent may only receive candidates, approved pre-screen features, its own completed relaxation observations, and remaining budget. It must never receive:

- relaxation results for candidates before it requests and spends an attempt;
- hidden observations for unselected candidates;
- MatterSim final benchmark labels, stability labels, or held-out evaluation summaries.

The benchmark evaluator keeps its final labels in a separate store and only joins them after the policy run is complete. Use the same candidate pool, feature permissions and physical-computation accounting when comparing Random, Fixed and Agent policies.

## Developer ownership

- **Agent Lead:** planner/LLM prompts, memory, policy logic and runtime orchestration. Return `Action`; do not import private benchmark stores or invoke shell commands from an action.
- **Benchmark/Physics Lead:** candidate manifests, generation, CHGNet/MatterSim adapters, evaluator, per-tool budgets and held-out labels. Implement `PhysicsEvaluator`; keep benchmark-only labels outside `State` and `Candidate`.
- **Integration owner:** validates each action, dispatches at most one allowed operation, records every returned/converted observation, decrements budgets, and serializes the state transition.

Both developers should import these contracts from the same repository revision and keep integration tests against the public types.

## Mock evaluator tests

No GPU, MatterGen or CHGNet import is needed to test an Agent policy. A deterministic fake evaluator can be used:

```python
class MockEvaluator:
    def __init__(self):
        self.calls = []

    def relax(self, candidate):
        self.calls.append(candidate.candidate_id)
        return Observation(
            candidate_id=candidate.candidate_id,
            status="failed",
            final_energy_ev_atom=None,
            final_max_force_ev_ang=None,
            elapsed_seconds=0.01,
            error="mock failure",
        )
```

Validate an Agent's action before calling the mock. Append its result and decrement remaining attempts even for the failure. Contract tests run with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_contracts.py
```

## Connecting the existing CHGNet wrapper

`materials_agent.tools.chgnet.relax_structure(model, structure, output_path, device, fmax, steps, relax_cell)` returns existing fields named `final_energy_per_atom_eV`, `final_max_force_eV_per_A`, `converged`, and `elapsed_seconds`. A physics adapter should load the CHGNet model once, resolve `candidate.cif_path` under a configured pool root, load its pymatgen `Structure`, call this helper, and map fields into the shared schema:

```python
Observation(
    candidate_id=candidate.candidate_id,
    status="converged" if result["converged"] else "not_converged",
    final_energy_ev_atom=result["final_energy_per_atom_eV"],
    final_max_force_ev_ang=result["final_max_force_eV_per_A"],
    elapsed_seconds=result["elapsed_seconds"],
)
```

Wrap exceptions as `Observation(status="failed", ..., error="Type: message")`; keep measured elapsed time and any partial finite results when available. This preserves CHGNet's existing public output fields and avoids changes to the model wrapper.

## Sharing a candidate pool across servers

Share the candidate CIFs and a relative-path SHA256 manifest, not absolute paths. From the candidate-pool root:

```bash
find . -type f -name '*.cif' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
```

Transfer the directory while preserving filenames and bytes, then run `sha256sum -c SHA256SUMS` on the receiving server before constructing Candidates. Keep the same manifest and candidate IDs in all policy runs. `Candidate.cif_sha256` can carry each full digest. Commit small manifests/configs, but do not commit the large CIF pool, model weights, generated output or virtual environment; the existing `results/` and model-weight ignore rules apply.

## Future closed-loop generation

The first version has a fixed candidate list and no generation Action. A future closed-loop version should add a separately typed generation request/result and a separate generation budget, then append newly generated Candidates to a new State. It must keep generator execution behind a constrained runtime API, record seeds and pool hashes, and continue to expose only completed relaxation Observations. Do not add arbitrary tools or shell payloads to `Action`.
