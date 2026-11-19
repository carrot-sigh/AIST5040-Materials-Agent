"""Sequential runtime for the shared planner and physics interfaces."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Iterable

from materials_agent.contracts import (
    Action, ActionValidationError, Candidate, ContractError, Observation,
    PhysicsEvaluator, Planner, State, validate_action,
)

from .artifacts import RunArtifacts
from .budget import MAX_RELAXATIONS, RelaxationBudget
from .memory import Decision, Memory
from materials_agent.structure_features import STRUCTURE_FEATURE_KEYS


@dataclass(frozen=True)
class ScreeningCosts:
    """Actual single-point work attributed to a run, or reused from another run."""

    source: str | None = None
    reused_calls: int = 0
    run_calls: int = 0
    failures: int = 0
    reused_elapsed_seconds: float = 0.0
    run_elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 0 for n in (self.reused_calls, self.run_calls, self.failures)):
            raise ValueError('screening call counts must be non-negative integers')
        calls = self.reused_calls + self.run_calls
        if self.failures > calls:
            raise ValueError('invalid screening cost')
        if calls and not self.source:
            raise ValueError('screening costs require provenance')
        for seconds, count in ((self.reused_elapsed_seconds, self.reused_calls),
                               (self.run_elapsed_seconds, self.run_calls)):
            if not isfinite(seconds) or seconds < 0 or (seconds and not count):
                raise ValueError('invalid screening elapsed time')


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    seed: int = 42
    relaxation_budget: int = MAX_RELAXATIONS
    max_consecutive_invalid: int = 3
    allow_initial_max_force: bool = False
    allow_structure_features: bool = False
    structure_feature_extraction_seconds: float = 0.0
    screening: ScreeningCosts = field(default_factory=ScreeningCosts)
    pool_source: str = 'provided candidates'
    pool_root: str | None = None
    synthetic: bool = True
    planner_settings: dict = field(default_factory=dict)  # Public settings only.
    evaluator_settings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')
        if type(self.max_consecutive_invalid) is not int or self.max_consecutive_invalid < 1:
            raise ValueError('max_consecutive_invalid must be a positive integer')
        if type(self.allow_initial_max_force) is not bool or type(self.synthetic) is not bool:
            raise ValueError('visibility and synthetic flags must be booleans')
        if type(self.allow_structure_features) is not bool:
            raise ValueError('allow_structure_features must be a boolean')
        if not isfinite(self.structure_feature_extraction_seconds) or self.structure_feature_extraction_seconds < 0:
            raise ValueError('structure feature extraction time must be finite and non-negative')


def visible_candidate_pool(candidates: Iterable[Candidate], config: RunConfig) -> list[Candidate]:
    """Build the shared allowlisted pool used by any Planner, including baselines."""
    pool = []
    for candidate in candidates:
        features = dict(candidate.features) if config.allow_structure_features else {}
        if config.allow_structure_features and not set(features).issubset(STRUCTURE_FEATURE_KEYS):
            raise ValueError(f'candidate {candidate.candidate_id} has unapproved public features')
        pool.append(replace(
            candidate,
            features=features,
            initial_max_force_ev_ang=(candidate.initial_max_force_ev_ang
                                      if config.allow_initial_max_force else None),
        ))
    return pool


def _evaluate(physics: PhysicsEvaluator, candidate: Candidate) -> Observation:
    started = perf_counter()
    try:
        observation = physics.relax(candidate)
        if not isinstance(observation, Observation) or observation.candidate_id != candidate.candidate_id:
            raise ContractError('evaluator must return an Observation for the requested candidate')
        if observation.error is not None and not isinstance(observation.error, str):
            raise ContractError('evaluator error must be a string or None')
        return observation
    except Exception as exc:
        return Observation(candidate.candidate_id, 'failed', None, None, perf_counter() - started,
                           f'{type(exc).__name__}: {exc}')


def run_agent(
    candidates: Iterable[Candidate], planner: Planner, physics: PhysicsEvaluator, config: RunConfig,
) -> dict:
    """Run a fresh fixed-pool experiment and write outputs to a new directory."""
    started = perf_counter()
    budget = RelaxationBudget(config.relaxation_budget)
    pool = visible_candidate_pool(candidates, config)
    initial_state = State(pool, [], budget.remaining)
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    if any(c.initial_max_force_ev_ang is not None for c in pool):
        screening_successes = config.screening.run_calls + config.screening.reused_calls - config.screening.failures
        if sum(c.initial_max_force_ev_ang is not None for c in pool) > screening_successes:
            raise ValueError('visible screening scores require a cost record of their single-point calls')

    memory = Memory()
    artifacts = RunArtifacts(config.output_dir)
    metadata = asdict(config)
    metadata.update(output_dir=str(config.output_dir), planner=type(planner).__name__,
                    evaluator=type(physics).__name__, schema_version=1)
    metadata['visible_candidate_fields'] = ['candidate_id', 'formula', 'cif_path', 'cif_sha256']
    if config.allow_initial_max_force:
        metadata['visible_candidate_fields'].append('initial_max_force_ev_ang')
    if config.allow_structure_features:
        metadata['visible_candidate_fields'].append('features')
        metadata['structure_features'] = {
            'version': 'geometry-v1',
            'keys': sorted(STRUCTURE_FEATURE_KEYS),
            'extraction_seconds': config.structure_feature_extraction_seconds,
            'source': 'input CIF; pymatgen parsing and geometry/composition only',
        }
    reason = 'budget_exhausted'
    invalid_streak = 0
    with artifacts.start(metadata, initial_state.candidates) as journal:
        while budget.remaining:
            history = memory.feedback()
            if len(history) == len(pool):
                reason = 'pool_exhausted'
                break

            # Validate on the runtime's snapshot. The planner receives a separate copy.
            state = State(pool, history, budget.remaining)
            action = None
            rationale = None
            observation = None
            relaxed_cif = None
            error = None
            try:
                proposed = planner.select_action(State.from_dict(state.to_dict()))
                rationale_value = getattr(planner, 'last_rationale', None)
                if isinstance(rationale_value, str):
                    rationale = rationale_value[:1000]
            except Exception as exc:
                outcome = 'planner_error'
                error = f'{type(exc).__name__}: {exc}'
            else:
                try:
                    if isinstance(proposed, Action):
                        action = Action(proposed.action_type, proposed.candidate_id)
                    if action is None:
                        raise ActionValidationError('action must be an Action instance')
                    validate_action(state, action)
                except ContractError as exc:
                    outcome = 'invalid_action'
                    error = str(exc)
                else:
                    if action.action_type == 'stop':
                        outcome = 'stop'
                    else:
                        candidate = by_id[action.candidate_id]
                        budget.spend_attempt()
                        observation = _evaluate(physics, candidate)
                        relaxed_cif = artifacts.relaxed_cif(candidate.candidate_id)
                        outcome = 'evaluated'

            entry = Decision(len(memory.decisions) + 1, action, rationale, state.remaining_relaxations,
                             budget.remaining, outcome, observation, relaxed_cif, error)
            memory.decisions.append(entry)
            artifacts.record(journal, entry)
            if outcome == 'stop':
                reason = 'planner_stop'
                break
            if outcome == 'evaluated':
                invalid_streak = 0
            else:
                invalid_streak += 1
                if invalid_streak >= config.max_consecutive_invalid:
                    reason = 'invalid_decision_limit'
                    break

    return artifacts.finish(memory, by_id, config.relaxation_budget, budget.remaining,
                            reason, perf_counter() - started, config.synthetic, asdict(config.screening))
