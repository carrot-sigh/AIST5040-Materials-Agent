"""Synthetic test data with no CIF reads or physics model calls."""
import hashlib

from materials_agent.contracts import Candidate, Observation


class MockPhysicsEvaluator:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def relax(self, candidate: Candidate) -> Observation:
        payload = f'{self.seed}|{candidate.candidate_id}'.encode()
        outcome, energy, force = hashlib.sha256(payload).digest()[:3]
        if outcome % 7 == 0:
            return Observation(candidate.candidate_id, 'failed', None, None, 0.0,
                               'synthetic test failure')
        converged = outcome % 3 != 0
        return Observation(
            candidate.candidate_id,
            'converged' if converged else 'not_converged',
            round(-1.0 - energy / 100, 2),
            round((0.001 if converged else 0.051) + force / 10000, 4),
            0.0,
        )
