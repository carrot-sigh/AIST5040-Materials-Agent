"""Version 1 data contracts shared by agent and physics implementations."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from numbers import Real
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping, Protocol

ActionType = Literal['relax', 'stop']
ObservationStatus = Literal['converged', 'not_converged', 'failed']


class ContractError(ValueError):
    """Invalid contract data or an invalid cross-object relationship."""


class ActionValidationError(ContractError):
    """An action cannot be executed in the supplied state."""


def _finite_optional(value: float | None, field_name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
    ):
        raise ContractError(f'{field_name} must be finite or None')


def _json_round_trip(value: Any) -> Any:
    """Copy JSON-compatible values and reject NaN/Infinity and custom objects."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ContractError(f'value must be JSON-compatible: {exc}') from exc


def _validate_public_features(value: Any, parent: str = 'features') -> None:
    forbidden_tokens = ('relax', 'observation', 'benchmark', 'mattersim', 'stability', 'energy_above_hull')
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized.startswith('final_') or any(token in normalized for token in forbidden_tokens):
                raise ContractError(f'{parent} cannot expose hidden evaluation field: {key}')
            _validate_public_features(item, parent)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_public_features(item, parent)


@dataclass(frozen=True)
class Candidate:
    """Agent-visible candidate metadata; CIF paths are relative to the pool root."""

    candidate_id: str
    formula: str
    cif_path: str
    initial_max_force_ev_ang: float | None = None
    cif_sha256: str | None = None
    features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id or self.candidate_id.strip() != self.candidate_id:
            raise ContractError('candidate_id must be a non-empty stable identifier')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*', self.candidate_id):
            raise ContractError('candidate_id contains unsupported characters')
        if not isinstance(self.formula, str) or not self.formula.strip():
            raise ContractError('formula must be non-empty')
        if not isinstance(self.cif_path, str):
            raise ContractError('cif_path must be a relative path string')
        path = PurePosixPath(self.cif_path)
        if (
            not self.cif_path
            or path.is_absolute()
            or '..' in path.parts
            or '\\' in self.cif_path
            or re.match(r'^[A-Za-z]:', self.cif_path)
        ):
            raise ContractError('cif_path must be a relative path within the candidate pool')
        if path.suffix.lower() != '.cif':
            raise ContractError('cif_path must point to a .cif file')
        _finite_optional(self.initial_max_force_ev_ang, 'initial_max_force_ev_ang')
        if self.cif_sha256 is not None and not re.fullmatch(r'[a-fA-F0-9]{64}', self.cif_sha256):
            raise ContractError('cif_sha256 must be a 64-character hexadecimal SHA256 digest')
        public_features = _json_round_trip(dict(self.features))
        _validate_public_features(public_features)
        object.__setattr__(self, 'features', public_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            'candidate_id': self.candidate_id,
            'formula': self.formula,
            'cif_path': self.cif_path,
            'initial_max_force_ev_ang': self.initial_max_force_ev_ang,
            'cif_sha256': self.cif_sha256,
            'features': _json_round_trip(self.features),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Candidate:
        return cls(**dict(data))

    @classmethod
    def from_json(cls, value: str) -> Candidate:
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class Observation:
    """Result of one attempted relaxation; convergence is not thermodynamic stability."""

    candidate_id: str
    status: ObservationStatus
    final_energy_ev_atom: float | None
    final_max_force_ev_ang: float | None
    elapsed_seconds: float
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*', self.candidate_id):
            raise ContractError('candidate_id must be a non-empty stable identifier')
        if self.status not in ('converged', 'not_converged', 'failed'):
            raise ContractError(f'unsupported observation status: {self.status!r}')
        _finite_optional(self.final_energy_ev_atom, 'final_energy_ev_atom')
        _finite_optional(self.final_max_force_ev_ang, 'final_max_force_ev_ang')
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, Real)
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ContractError('elapsed_seconds must be a finite non-negative number')
        if self.status == 'failed' and not self.error:
            raise ContractError('failed observations must preserve an error message')

    def to_dict(self) -> dict[str, Any]:
        return {
            'candidate_id': self.candidate_id,
            'status': self.status,
            'final_energy_ev_atom': self.final_energy_ev_atom,
            'final_max_force_ev_ang': self.final_max_force_ev_ang,
            'elapsed_seconds': self.elapsed_seconds,
            'error': self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Observation:
        return cls(**dict(data))

    @classmethod
    def from_json(cls, value: str) -> Observation:
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class State:
    """Agent-visible fixed pool, completed relaxation history and remaining attempt budget."""

    candidates: list[Candidate]
    history: list[Observation]
    remaining_relaxations: int

    def __post_init__(self) -> None:
        candidates = list(self.candidates)
        history = list(self.history)
        if isinstance(self.remaining_relaxations, bool) or not isinstance(self.remaining_relaxations, int):
            raise ContractError('remaining_relaxations must be an integer')
        if self.remaining_relaxations < 0:
            raise ContractError('remaining_relaxations must be non-negative')
        if any(not isinstance(candidate, Candidate) for candidate in candidates):
            raise ContractError('candidates must contain Candidate objects')
        if any(not isinstance(observation, Observation) for observation in history):
            raise ContractError('history must contain completed Observation objects')
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ContractError('candidate IDs must be unique within a state')
        candidate_id_set = set(candidate_ids)
        history_ids = [observation.candidate_id for observation in history]
        if len(history_ids) != len(set(history_ids)):
            raise ContractError('history cannot contain repeated evaluations')
        unknown = set(history_ids) - candidate_id_set
        if unknown:
            raise ContractError(f'history contains unknown candidate IDs: {sorted(unknown)}')
        object.__setattr__(self, 'candidates', candidates)
        object.__setattr__(self, 'history', history)

    @property
    def evaluated_candidate_ids(self) -> frozenset[str]:
        return frozenset(observation.candidate_id for observation in self.history)

    def to_dict(self) -> dict[str, Any]:
        return {
            'candidates': [candidate.to_dict() for candidate in self.candidates],
            'history': [observation.to_dict() for observation in self.history],
            'remaining_relaxations': self.remaining_relaxations,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> State:
        values = dict(data)
        values['candidates'] = [Candidate.from_dict(item) for item in values['candidates']]
        values['history'] = [Observation.from_dict(item) for item in values['history']]
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> State:
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class Action:
    """A constrained decision; this contract deliberately has no shell/tool payload."""

    action_type: ActionType
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        if self.action_type not in ('relax', 'stop'):
            raise ContractError(f'unsupported action_type: {self.action_type!r}')
        if self.action_type == 'relax':
            if not isinstance(self.candidate_id, str) or not self.candidate_id:
                raise ContractError('relax actions require candidate_id')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*', self.candidate_id):
                raise ContractError('candidate_id contains unsupported characters')
        elif self.candidate_id is not None:
            raise ContractError('stop actions must not specify candidate_id')

    def to_dict(self) -> dict[str, Any]:
        return {'action_type': self.action_type, 'candidate_id': self.candidate_id}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Action:
        return cls(**dict(data))

    @classmethod
    def from_json(cls, value: str) -> Action:
        return cls.from_dict(json.loads(value))


class Planner(Protocol):
    """Agent-side policy that chooses one constrained next action."""

    def select_action(self, state: State) -> Action:
        """Return a relaxation request or stop decision from the visible state."""
        ...


class PhysicsEvaluator(Protocol):
    """Physics-side implementation that executes one relaxation attempt."""

    def relax(self, candidate: Candidate) -> Observation:
        """Attempt relaxation and return an Observation, including on failure."""
        ...


def validate_action(state: State, action: Action) -> None:
    """Raise ActionValidationError unless ``action`` is legal in ``state``.

    Runtime code should call this before dispatching any evaluator operation.
    A failed observation still appears in history and therefore consumes budget.
    """
    if not isinstance(action, Action):
        raise ActionValidationError('action must be an Action instance')
    if action.action_type == 'stop':
        return
    if action.action_type != 'relax':
        raise ActionValidationError(f'unsupported action type: {action.action_type!r}')
    candidate_ids = {candidate.candidate_id for candidate in state.candidates}
    if action.candidate_id not in candidate_ids:
        raise ActionValidationError(f'unknown candidate_id: {action.candidate_id!r}')
    if action.candidate_id in state.evaluated_candidate_ids:
        raise ActionValidationError(f'candidate already evaluated: {action.candidate_id}')
    if state.remaining_relaxations < 1:
        raise ActionValidationError('no relaxation budget remains')
