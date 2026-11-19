"""Seeded policy that exercises sequential feedback for the mock experiment."""
import hashlib

from materials_agent.contracts import Action, Candidate, State


class MockPlanner:
    def __init__(self, seed: int = 42, stop_after: int | None = None):
        self.seed = seed
        self.stop_after = stop_after
        self.last_rationale = ''

    def select_action(self, state: State) -> Action:
        available = [candidate for candidate in state.candidates
                     if candidate.candidate_id not in state.evaluated_candidate_ids]
        if (not available or state.remaining_relaxations == 0
                or (self.stop_after is not None and len(state.history) >= self.stop_after)):
            self.last_rationale = 'Mock stop condition reached.'
            return Action('stop')

        # Use only the most recent result. The seeded key makes decisions reproducible
        # without depending on call count, clock time, or the input list order.
        last = state.history[-1] if state.history else None
        feedback = last.status if last else 'initial'
        if last:
            feedback += f':{last.final_max_force_ev_ang}'
        self.last_rationale = f'Mock choice using seeded ordering and latest status/force: {feedback}.'

        def key(candidate: Candidate) -> tuple[bytes, str]:
            payload = f'{self.seed}|{feedback}|{candidate.candidate_id}'.encode()
            return hashlib.sha256(payload).digest(), candidate.candidate_id

        return Action('relax', min(available, key=key).candidate_id)
