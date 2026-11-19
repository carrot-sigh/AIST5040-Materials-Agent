"""Logical attempt budget for the first experiment."""
from dataclasses import dataclass, field

MAX_RELAXATIONS = 8


@dataclass
class RelaxationBudget:
    limit: int
    _used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if type(self.limit) is not int or not 0 <= self.limit <= MAX_RELAXATIONS:
            raise ValueError(f'relaxation budget must be an integer between 0 and {MAX_RELAXATIONS}')

    @property
    def remaining(self) -> int:
        return self.limit - self._used

    def spend_attempt(self) -> None:
        if self.remaining <= 0:
            raise RuntimeError('relaxation budget exhausted')
        self._used += 1
