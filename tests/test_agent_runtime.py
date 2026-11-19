import csv
import hashlib
import json

import pytest

from materials_agent.agent.mock_evaluator import MockPhysicsEvaluator
from materials_agent.agent.mock_experiment import load_fixed_pool, synthetic_pool
from materials_agent.agent.mock_planner import MockPlanner
from materials_agent.agent.runtime import RunConfig, ScreeningCosts, run_agent
from materials_agent.contracts import Action, Candidate, Observation, State
from materials_agent.structure_features import extract_structure_features
from pymatgen.core import Lattice, Structure


def json_file(path):
    return json.loads(path.read_text())


def decisions(directory):
    return [json.loads(line) for line in (directory / 'decisions.jsonl').read_text().splitlines()]


class ScriptedPlanner:
    def __init__(self, *actions):
        self.actions = iter(actions)
        self.states = []

    def select_action(self, state):
        self.states.append(state)
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        return action


class RecordingPhysics(MockPhysicsEvaluator):
    def __init__(self):
        super().__init__()
        self.calls = []

    def relax(self, candidate):
        self.calls.append(candidate.candidate_id)
        return super().relax(candidate)


def test_mock_64_to_8_and_reproducible_exports(tmp_path):
    pool = synthetic_pool()
    outputs = [tmp_path / 'one', tmp_path / 'two']
    for output in outputs:
        summary = run_agent(pool, MockPlanner(42), MockPhysicsEvaluator(42), RunConfig(output))
        assert summary['candidate_count'] == 64
        assert summary['relaxation_attempts'] == 8
        assert summary['remaining_relaxations'] == 0
        assert summary['termination_reason'] == 'budget_exhausted'
        assert summary['synthetic'] and summary['single_point_calls'] == 0

    selected = json_file(outputs[0] / 'selected_candidates.json')
    ids = {row['candidate_id'] for row in selected}
    assert len(ids) == 8 and ids <= {candidate.candidate_id for candidate in pool}
    assert [row['rank'] for row in selected] == list(range(1, 9))
    records = decisions(outputs[0])
    assert all(record['rationale'] for record in records)
    assert [record['budget_after'] for record in records] == list(range(7, -1, -1))
    assert [record['iteration'] for record in records] == list(range(1, 9))
    assert all(record['relaxed_cif'] is None for record in records)
    for name in ('candidate_manifest.json', 'selected_candidates.json', 'decisions.jsonl', 'relaxation_results.csv'):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    summaries = [json_file(output / 'summary.json') for output in outputs]
    for summary in summaries:
        del summary['total_runtime_seconds']
    assert summaries[0] == summaries[1]
    config = json_file(outputs[0] / 'run_config.json')
    assert config['seed'] == 42 and config['planner'] == 'MockPlanner'
    with pytest.raises(FileExistsError):
        run_agent(pool, MockPlanner(), MockPhysicsEvaluator(), RunConfig(outputs[0]))
    assert json_file(outputs[0] / 'selected_candidates.json') == selected


def test_changed_feedback_can_change_the_next_mock_choice():
    pool = synthetic_pool()
    planner = MockPlanner(42)
    first = planner.select_action(State(pool, [], 8)).candidate_id
    converged = State(pool, [Observation(first, 'converged', -1.0, 0.02, 0.0)], 7)
    failed = State(pool, [Observation(first, 'failed', None, None, 0.0, 'fixture error')], 7)
    assert planner.select_action(converged) != planner.select_action(failed)


def test_invalid_and_duplicate_actions_cannot_spend_attempts(tmp_path):
    pool = synthetic_pool()
    first, second = (candidate.candidate_id for candidate in pool[:2])
    planner = ScriptedPlanner(
        {'action_type': 'shell'}, Action('relax', 'missing'), Action('relax', first),
        Action('relax', first), Action('relax', second), Action('stop'),
    )
    physics = RecordingPhysics()
    output = tmp_path / 'run'
    summary = run_agent(pool, planner, physics, RunConfig(output))
    assert physics.calls == [first, second]
    assert summary['remaining_relaxations'] == 6
    assert summary['termination_reason'] == 'planner_stop'
    assert summary['invalid_decisions'] == 3
    assert [state.remaining_relaxations for state in planner.states] == [8, 8, 8, 7, 7, 6]
    assert [state.evaluated_candidate_ids for state in planner.states] == [
        set(), set(), set(), {first}, {first}, {first, second},
    ]
    records = decisions(output)
    assert 'already evaluated' in records[3]['error']
    assert records[-1]['budget_before'] == records[-1]['budget_after'] == 6


def test_invalid_streak_ends_run_without_physics(tmp_path):
    planner = ScriptedPlanner(ValueError('fixture planner error'), None, Action('relax', 'missing'))
    physics = RecordingPhysics()
    output = tmp_path / 'run'
    summary = run_agent(synthetic_pool(), planner, physics, RunConfig(output))
    assert not physics.calls
    assert summary['termination_reason'] == 'invalid_decision_limit'
    assert summary['remaining_relaxations'] == 8
    assert [entry['outcome'] for entry in decisions(output)] == ['planner_error', 'invalid_action', 'invalid_action']


def test_evaluator_failures_are_charged_and_audited(tmp_path):
    candidates = synthetic_pool()[:4]
    ids = [candidate.candidate_id for candidate in candidates]
    output = tmp_path / 'run'

    class Physics:
        def relax(self, candidate):
            candidate_id = candidate.candidate_id
            if candidate_id == ids[0]:
                raise RuntimeError('fixture exception with /private/path')
            if candidate_id == ids[1]:
                return Observation(candidate_id, 'failed', None, None, 0.2, 'failed fixture')
            if candidate_id == ids[2]:
                return Observation(ids[3], 'converged', -1.0, 0.01, 0.5)
            (output / 'relaxed' / f'{candidate_id}.cif').write_text('data_test_fixture\n')
            return Observation(candidate_id, 'not_converged', -2.0, 0.06, 0.5)

    planner = ScriptedPlanner(*(Action('relax', candidate_id) for candidate_id in ids))
    summary = run_agent(candidates, planner, Physics(), RunConfig(output, relaxation_budget=4))
    assert summary['remaining_relaxations'] == 0
    assert summary['relaxation_attempts'] == 4 and summary['relaxation_failures'] == 3
    assert summary['force_converged_count'] == 0
    assert [entry.candidate_id for entry in planner.states[-1].history] == ids[:3]
    assert all(entry.error == 'evaluation failed' for entry in planner.states[-1].history)
    records = decisions(output)
    assert '/private/path' in records[0]['observation']['error']
    assert records[1]['observation']['elapsed_seconds'] == 0.2
    assert records[2]['observation']['final_energy_ev_atom'] is None
    assert records[-1]['relaxed_cif'] == f'relaxed/{ids[-1]}.cif'
    with (output / 'relaxation_results.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [row['candidate_id'] for row in rows] == ids
    assert rows[-1]['relaxed_cif'] == records[-1]['relaxed_cif']
    assert rows[-1]['converged'] == 'False'
    assert rows[-1]['synthetic'] == 'True'


def test_state_is_a_projection_and_planner_mutation_cannot_change_validity(tmp_path):
    candidate = Candidate('real', 'NaCl', 'real.cif', 0.2, features={'generic_label': -100})
    physics = RecordingPhysics()

    class MutatingPlanner:
        def select_action(self, state):
            assert state.candidates[0].initial_max_force_ev_ang is None
            assert state.candidates[0].features == {} and state.history == []
            state.candidates.append(Candidate('injected', 'LiF', 'injected.cif'))
            return Action('relax', 'injected')

    output = tmp_path / 'run'
    run_agent([candidate], MutatingPlanner(), physics, RunConfig(output, max_consecutive_invalid=1))
    assert not physics.calls
    assert decisions(output)[0]['outcome'] == 'invalid_action'
    assert candidate.initial_max_force_ev_ang == 0.2
    assert json_file(output / 'candidate_manifest.json')[0]['features'] == {}


def test_low_cost_structure_features_are_shared_across_planners(tmp_path):
    structure = Structure(
        Lattice.from_parameters(3, 4, 5, 80, 90, 100),
        ['Na', 'Cl', 'Cl'],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    features = extract_structure_features(structure)
    assert features['structure_feature_version'] == 'geometry-v1'
    assert features['num_atoms'] == 3
    assert features['num_elements'] == 2
    assert features['volume_angstrom3'] == pytest.approx(structure.volume)
    assert features['density_g_cm3'] == pytest.approx(structure.density)
    assert features['lattice_parameters']['a_angstrom'] == pytest.approx(3)
    assert features['lattice_parameters']['beta_degree'] == pytest.approx(90)
    assert features['element_fractions'] == {'Cl': pytest.approx(2 / 3), 'Na': pytest.approx(1 / 3)}

    candidate = Candidate('crystal', 'NaCl', 'crystal.cif', features=features)
    received = []

    class AdaptivePolicy:
        def select_action(self, state):
            received.append(state.candidates[0].features)
            return Action('stop')

    # Agent and Classical Adaptive implementations use the same run_agent -> State path.
    run_agent([candidate], AdaptivePolicy(), RecordingPhysics(), RunConfig(
        tmp_path / 'agent', allow_structure_features=True,
    ))
    run_agent([candidate], AdaptivePolicy(), RecordingPhysics(), RunConfig(
        tmp_path / 'classical', allow_structure_features=True,
    ))
    assert received == [features, features]
    assert json_file(tmp_path / 'agent' / 'candidate_manifest.json')[0]['features'] == features
    assert json_file(tmp_path / 'classical' / 'candidate_manifest.json')[0]['features'] == features

    with pytest.raises(ValueError, match='unapproved public features'):
        run_agent([Candidate('hidden', 'NaCl', 'hidden.cif', features={'custom_label': 'leak'})],
                  AdaptivePolicy(), RecordingPhysics(),
                  RunConfig(tmp_path / 'reject', allow_structure_features=True))


def test_fixed_pool_import_filters_fields_and_accounts_for_opt_in(tmp_path):
    cif = tmp_path / 'real.cif'
    cif.write_bytes(b'data_pool_fixture\n')
    digest = hashlib.sha256(cif.read_bytes()).hexdigest()
    candidate_id = f'real-{digest[:12]}'
    source = tmp_path / 'screening.csv'
    row = {
        'candidate_id': candidate_id, 'formula': 'LiF', 'source_file': 'old/pool/real.cif',
        'validity': 'valid', 'duplicate': 'False', 'single_point_status': 'success',
        'max_force_eV_per_A': '0.4', 'single_point_seconds': '0.25',
        'final_energy_per_atom_eV': '-1000', 'relaxed_cif': 'hidden.cif',
        'rank': '1', 'mattersim': 'hidden',
    }
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    for allow in (False, True):
        pool, costs, _ = load_fixed_pool(tmp_path, source, allow)
        planner = ScriptedPlanner(Action('stop'))
        output = tmp_path / str(allow)
        summary = run_agent(pool, planner, RecordingPhysics(), RunConfig(
            output, allow_initial_max_force=allow, screening=costs,
        ))
        assert summary['single_point_calls'] == 0
        assert summary['reused_single_point_calls'] == int(allow)
        assert summary['total_single_point_seconds'] == 0.0
        assert summary['reused_single_point_seconds'] == (0.25 if allow else 0.0)
        state = planner.states[0]
        assert state.history == []
        assert state.candidates == [Candidate(candidate_id, 'LiF', 'real.cif',
                                              0.4 if allow else None, digest)]
        assert 'hidden' not in state.to_json() and '-1000' not in state.to_json()
    with pytest.raises(ValueError, match='cost record'):
        run_agent(pool, planner, RecordingPhysics(), RunConfig(
            tmp_path / 'no-ledger', allow_initial_max_force=True, screening=ScreeningCosts(),
        ))
    cif.write_bytes(b'changed CIF bytes')
    with pytest.raises(ValueError, match='matching CIF'):
        load_fixed_pool(tmp_path, source, False)


def test_fixed_pool_can_extract_features_from_cif_without_physics(tmp_path):
    cif = tmp_path / 'input.cif'
    Structure(Lattice.cubic(4), ['Na', 'Cl'], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(
        filename=str(cif), fmt='cif')
    digest = hashlib.sha256(cif.read_bytes()).hexdigest()
    source = tmp_path / 'screening.csv'
    row = {
        'candidate_id': f'input-{digest[:12]}', 'formula': 'NaCl',
        'source_file': 'candidate_pool/input.cif', 'validity': 'valid', 'duplicate': 'False',
        'single_point_status': 'success', 'single_point_seconds': '0.1',
        'max_force_eV_per_A': '0.2',
    }
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    pool, _, extraction_seconds = load_fixed_pool(
        tmp_path, source, False, include_structure_features=True,
    )
    features = pool[0].features
    assert features['num_atoms'] == 2
    assert features['volume_angstrom3'] == pytest.approx(64)
    assert features['element_fractions'] == {'Cl': pytest.approx(0.5), 'Na': pytest.approx(0.5)}
    assert pool[0].initial_max_force_ev_ang is None
    assert extraction_seconds >= 0


@pytest.mark.parametrize('budget', [-1, 9, True, 1.5])
def test_phase_one_budget_bounds(tmp_path, budget):
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='relaxation budget'):
        run_agent(synthetic_pool(), ScriptedPlanner(), RecordingPhysics(),
                  RunConfig(output, relaxation_budget=budget))
    assert not output.exists()


@pytest.mark.parametrize(('pool_size', 'budget', 'reason'), [
    (0, 8, 'pool_exhausted'), (1, 8, 'pool_exhausted'), (1, 0, 'budget_exhausted'),
])
def test_empty_pool_or_budget_needs_no_additional_decision(tmp_path, pool_size, budget, reason):
    pool = synthetic_pool()[:pool_size]
    planner = ScriptedPlanner(*(Action('relax', candidate.candidate_id) for candidate in pool))
    summary = run_agent(pool, planner, RecordingPhysics(), RunConfig(tmp_path / 'run', relaxation_budget=budget))
    assert summary['termination_reason'] == reason
    assert len(planner.states) == min(pool_size, budget)
