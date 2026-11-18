import json

import pytest

from materials_agent.contracts import (
    Action,
    ActionValidationError,
    Candidate,
    ContractError,
    Observation,
    State,
    validate_action,
)


def candidate(candidate_id='cand-001', **kwargs):
    cif_path = kwargs.pop('cif_path', f'{candidate_id}.cif')
    return Candidate(
        candidate_id=candidate_id,
        formula='LiFeO2',
        cif_path=cif_path,
        **kwargs,
    )


def test_candidate_validation_and_relative_paths():
    item = candidate(cif_sha256='a' * 64, initial_max_force_ev_ang=0.12)
    assert Candidate.from_json(item.to_json()) == item
    with pytest.raises(ContractError, match='relative path'):
        candidate(cif_path='/home/alice/pool.cif')
    with pytest.raises(ContractError, match='relative path'):
        candidate(cif_path='../private/pool.cif')
    with pytest.raises(ContractError, match='relative path'):
        candidate(cif_path='C:\\data\\pool.cif')
    with pytest.raises(ContractError, match='candidate_id'):
        candidate(candidate_id='bad/id')
    with pytest.raises(ContractError, match='SHA256'):
        candidate(cif_sha256='not-a-digest')
    with pytest.raises(ContractError, match='finite'):
        candidate(initial_max_force_ev_ang=float('nan'))


def test_action_relax_and_stop_json_round_trip():
    relax = Action('relax', 'cand-001')
    stop = Action('stop')
    assert Action.from_json(relax.to_json()) == relax
    assert Action.from_dict(stop.to_dict()) == stop
    with pytest.raises(ContractError, match='require candidate_id'):
        Action('relax')
    with pytest.raises(ContractError, match='must not specify'):
        Action('stop', 'cand-001')
    with pytest.raises(ContractError, match='unsupported action_type'):
        Action('shell', None)  # type: ignore[arg-type]


def test_action_validator_rejects_unknown_repeat_and_empty_budget():
    candidates = [candidate(), candidate('cand-002')]
    fresh = State(candidates, [], remaining_relaxations=1)
    validate_action(fresh, Action('relax', 'cand-001'))
    validate_action(fresh, Action('stop'))
    with pytest.raises(ActionValidationError, match='unknown candidate'):
        validate_action(fresh, Action('relax', 'cand-missing'))

    failed = Observation('cand-001', 'failed', None, None, 1.5, error='optimizer error')
    after_failure = State(candidates, [failed], remaining_relaxations=0)
    assert 'cand-001' in after_failure.evaluated_candidate_ids
    with pytest.raises(ActionValidationError, match='already evaluated'):
        validate_action(after_failure, Action('relax', 'cand-001'))
    with pytest.raises(ActionValidationError, match='no relaxation budget'):
        validate_action(after_failure, Action('relax', 'cand-002'))
    validate_action(after_failure, Action('stop'))


def test_failed_observation_round_trip_preserves_error_and_optional_results():
    observation = Observation(
        candidate_id='cand-001', status='failed', final_energy_ev_atom=None,
        final_max_force_ev_ang=None, elapsed_seconds=2.25, error='CUDA out of memory',
    )
    assert Observation.from_json(observation.to_json()) == observation
    state = State([candidate()], [observation], remaining_relaxations=7)
    assert State.from_json(state.to_json()) == state
    assert state.evaluated_candidate_ids == frozenset({'cand-001'})


def test_state_rejects_duplicate_or_unknown_history():
    obs = Observation('cand-001', 'not_converged', -3.2, 0.2, 3.0)
    with pytest.raises(ContractError, match='repeated evaluations'):
        State([candidate()], [obs, obs], remaining_relaxations=6)
    with pytest.raises(ContractError, match='unknown candidate'):
        State([candidate()], [Observation('other', 'failed', None, None, 1, 'failed')], 7)
    with pytest.raises(ContractError, match='unique'):
        State([candidate(), candidate()], [], 8)


@pytest.mark.parametrize('budget', [-1, 1.5, True])
def test_state_rejects_invalid_budget(budget):
    with pytest.raises(ContractError, match='remaining_relaxations'):
        State([], [], budget)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'kwargs',
    [
        {'status': 'failed', 'error': None},
        {'status': 'failed', 'error': ''},
        {'status': 'not_converged', 'error': None, 'final_energy_ev_atom': float('inf')},
        {'status': 'converged', 'error': None, 'final_max_force_ev_ang': float('nan')},
        {'status': 'not_converged', 'error': None, 'elapsed_seconds': -0.1},
    ],
)
def test_observation_rejects_missing_failure_or_nonfinite_physics(kwargs):
    values = {
        'candidate_id': 'cand-001', 'status': 'not_converged',
        'final_energy_ev_atom': -3.2, 'final_max_force_ev_ang': 0.2,
        'elapsed_seconds': 1.0, 'error': None,
    }
    values.update(kwargs)
    with pytest.raises(ContractError):
        Observation(**values)  # type: ignore[arg-type]


def test_contract_json_is_stable_and_rejects_non_json_features():
    item = candidate(features={'site_count': 4, 'labels': ['Li', 'Fe', 'O']})
    parsed = json.loads(item.to_json())
    assert list(parsed) == sorted(parsed)
    assert Candidate.from_dict(parsed) == item
    with pytest.raises(ContractError, match='JSON-compatible'):
        candidate(features={'not_json': object()})
    with pytest.raises(ContractError, match='hidden evaluation field'):
        candidate(features={'final_energy_ev_atom': -5.0})
    with pytest.raises(ContractError, match='hidden evaluation field'):
        candidate(features={'public': {'mattersim_stability_label': 'stable'}})
