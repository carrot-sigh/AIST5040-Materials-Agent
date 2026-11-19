"""Export the Agent audit trail and baseline-compatible relaxation rows."""
import csv
import json
from pathlib import Path
from typing import TextIO

from materials_agent.contracts import Candidate

from .memory import Decision, Memory

RESULT_FIELDS = (
    'iteration', 'rank', 'candidate_id', 'source_file', 'formula', 'max_force_eV_per_A',
    'observation_status', 'relaxation_status', 'relaxation_seconds', 'relaxed_cif',
    'final_energy_per_atom_eV', 'final_max_force_eV_per_A', 'converged', 'relaxation_error',
    'synthetic',
)


class RunArtifacts:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def _write_json(self, name: str, value: object) -> None:
        with (self.output_dir / name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    def start(self, metadata: dict, pool: list[Candidate]) -> TextIO:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / 'relaxed').mkdir()
        self._write_json('run_config.json', metadata)
        self._write_json('candidate_manifest.json', [candidate.to_dict() for candidate in pool])
        return (self.output_dir / 'decisions.jsonl').open('x', encoding='utf-8')

    @staticmethod
    def record(journal: TextIO, entry: Decision) -> None:
        journal.write(json.dumps(entry.to_dict(), sort_keys=True, allow_nan=False) + '\n')
        journal.flush()

    def relaxed_cif(self, candidate_id: str) -> str | None:
        path = self.output_dir / 'relaxed' / f'{candidate_id}.cif'
        if path.is_file() and path.resolve().is_relative_to(self.output_dir.resolve()):
            return path.relative_to(self.output_dir).as_posix()
        return None

    def finish(
        self, memory: Memory, pool: dict[str, Candidate], limit: int, remaining: int,
        reason: str, elapsed: float, synthetic: bool, screening: dict,
    ) -> dict:
        selected = []
        rows = []
        for entry in memory.decisions:
            observation = entry.observation
            if observation is None:
                continue
            candidate = pool[observation.candidate_id]
            rank = len(selected) + 1
            selected.append({
                'rank': rank,
                'candidate_id': candidate.candidate_id,
                'score_max_force_eV_per_A': candidate.initial_max_force_ev_ang,
            })
            rows.append({
                'iteration': entry.iteration,
                'rank': rank,
                'candidate_id': candidate.candidate_id,
                'source_file': candidate.cif_path,
                'formula': candidate.formula,
                'max_force_eV_per_A': candidate.initial_max_force_ev_ang,
                'observation_status': observation.status,
                'relaxation_status': 'failed' if observation.status == 'failed' else 'success',
                'relaxation_seconds': observation.elapsed_seconds,
                'relaxed_cif': entry.relaxed_cif,
                'final_energy_per_atom_eV': observation.final_energy_ev_atom,
                'final_max_force_eV_per_A': observation.final_max_force_ev_ang,
                'converged': None if observation.status == 'failed' else observation.status == 'converged',
                'relaxation_error': observation.error,
                'synthetic': synthetic,
            })

        self._write_json('selected_candidates.json', selected)
        with (self.output_dir / 'relaxation_results.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        failures = sum(row['relaxation_status'] == 'failed' for row in rows)
        summary = {
            'synthetic': synthetic,
            'candidate_count': len(pool),
            'selection_budget': limit,
            'remaining_relaxations': remaining,
            'termination_reason': reason,
            'decision_count': len(memory.decisions),
            'invalid_decisions': sum(d.outcome in ('invalid_action', 'planner_error') for d in memory.decisions),
            'selected_count': len(selected),
            'selection_budget_shortfall': limit - len(selected),
            'relaxation_attempts': len(rows),
            'relaxation_successes': len(rows) - failures,
            'relaxation_failures': failures,
            'force_converged_count': sum(row['converged'] is True for row in rows),
            'single_point_calls': screening['run_calls'],
            'reused_single_point_calls': screening['reused_calls'],
            'total_single_point_seconds': screening['run_elapsed_seconds'],
            'reused_single_point_seconds': screening['reused_elapsed_seconds'],
            'single_point_failures': screening['failures'],
            'total_relaxation_seconds': sum(row['relaxation_seconds'] for row in rows),
            'total_runtime_seconds': elapsed,
        }
        self._write_json('summary.json', summary)
        return summary
