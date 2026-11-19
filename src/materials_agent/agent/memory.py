"""Decision records and the feedback projection supplied to the planner."""
from dataclasses import dataclass, field, replace
from typing import Literal

from materials_agent.contracts import Action, Observation


@dataclass(frozen=True)
class Decision:
    iteration: int
    action: Action | None
    rationale: str | None
    budget_before: int
    budget_after: int
    outcome: Literal['evaluated', 'invalid_action', 'planner_error', 'stop']
    observation: Observation | None = None
    relaxed_cif: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            **vars(self),
            'action': self.action.to_dict() if self.action else None,
            'observation': self.observation.to_dict() if self.observation else None,
        }


@dataclass
class Memory:
    decisions: list[Decision] = field(default_factory=list)

    def feedback(self) -> list[Observation]:
        # Error details are for the audit log. Evaluator diagnostics can contain local paths.
        return [
            replace(entry.observation, error='evaluation failed' if entry.observation.status == 'failed' else None)
            for entry in self.decisions
            if entry.observation is not None
        ]
