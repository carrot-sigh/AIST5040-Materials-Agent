"""Optional rationale support layered on the shared planner contract."""
from typing import Protocol

from materials_agent.contracts import Planner


class ReasonedPlanner(Planner, Protocol):
    @property
    def last_rationale(self) -> str:
        """Rationale for the most recent successful select_action call."""
        ...
