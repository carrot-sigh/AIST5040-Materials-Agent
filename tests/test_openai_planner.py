import json
import sys
import types

import pytest

from materials_agent.agent.mock_evaluator import MockPhysicsEvaluator
from materials_agent.agent.mock_experiment import synthetic_pool
from materials_agent.agent.openai_experiment import run_openai_experiment
from materials_agent.agent.openai_planner import (
    ACTION_SCHEMA, OpenAIPlanner, OpenAIPlannerError, PlannerConfig,
)
from materials_agent.agent.prompts import SYSTEM_PROMPT, serialize_state
from materials_agent.agent.runtime import RunConfig
from materials_agent.contracts import Action, Observation, State


class Response:
    def __init__(self, data, *, input_tokens=100, output_tokens=20, response_id='resp-test'):
        self.output_text = json.dumps(data)
        self.id = response_id
        self._request_id = 'req-test'
        self.usage = types.SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )


class FakeResponses:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, *results):
        self.responses = FakeResponses(*results)


def state(pool=None, history=(), remaining=8):
    return State(pool or synthetic_pool(), list(history), remaining)


def choice(candidate_id, action_type='relax', justification='A plausible next test.'):
    return {'action_type': action_type, 'candidate_id': candidate_id, 'justification': justification}


def test_structured_response_schema_and_serialization_exclude_paths():
    from pathlib import Path
    from materials_agent.agent.runtime import RunConfig, visible_candidate_pool

    pool = synthetic_pool()[:2]
    pool[0] = type(pool[0])(
        pool[0].candidate_id, pool[0].formula, 'private/path/one.cif',
        initial_max_force_ev_ang=0.3,
        features={},
    )
    pool = visible_candidate_pool(pool, RunConfig(output_dir=Path('unused')))
    obs = Observation(pool[1].candidate_id, 'failed', None, None, 1.25, 'failed at /private/output/secret.cif')
    current = State(pool, [obs], 7)
    request = serialize_state(current)
    assert pool[0].candidate_id in request
    assert [row['candidate_id'] for row in json.loads(request)['available_candidates']] == [pool[0].candidate_id]
    assert 'private/path' not in request
    assert '/private/output' not in request
    assert 'num_atoms' not in request
    assert 'initial_max_force' not in request
    assert 'final_energy_ev_atom' in request
    client = FakeClient(Response(choice(pool[0].candidate_id)))
    planner = OpenAIPlanner(PlannerConfig('gpt-test'), client=client)
    assert planner.select_action(current) == Action('relax', pool[0].candidate_id)
    request_args = client.responses.calls[0]
    schema = request_args['text']['format']['schema']
    assert schema['properties']['candidate_id']['enum'] == [pool[0].candidate_id, None]
    assert schema['required'] == ACTION_SCHEMA['required']
    assert request_args['text']['format']['strict'] is True
    assert request_args['instructions'] == SYSTEM_PROMPT
    assert request_args['model'] == 'gpt-test'
    assert planner.last_rationale == 'A plausible next test.'


def test_explicit_feature_and_force_visibility_only():
    candidate = type(synthetic_pool()[0])(
        'cand-visibility', 'LiFeO2', 'unsafe/path.cif', 0.125,
        features={'num_atoms': 7, 'element_fractions': {'Fe': 0.5, 'Li': 0.25, 'O': 0.25}},
    )
    from pathlib import Path
    from materials_agent.agent.runtime import visible_candidate_pool
    from materials_agent.agent.runtime import RunConfig as RuntimeConfig
    candidate = visible_candidate_pool([candidate], RuntimeConfig(output_dir=Path('unused')))[0]
    client = FakeClient(Response(choice(candidate.candidate_id)))
    planner = OpenAIPlanner(PlannerConfig('model-x'), client=client)
    planner.select_action(State([candidate], [], 8))
    text = client.responses.calls[0]['input']
    assert 'unsafe/path' not in text
    assert 'num_atoms' not in text and 'initial_max_force' not in text

    from materials_agent.agent.runtime import visible_candidate_pool
    visible = State(visible_candidate_pool([type(synthetic_pool()[0])(
        'cand-visible', 'LiFeO2', 'unsafe/path.cif', 0.125,
        features={'num_atoms': 7, 'element_fractions': {'Fe': 0.5, 'Li': 0.25, 'O': 0.25}},
    )], RuntimeConfig(
        output_dir=Path('unused'),
        allow_structure_features=True, allow_initial_max_force=True,
    )), [], 8)
    visible_client = FakeClient(Response(choice(visible.candidates[0].candidate_id)))
    OpenAIPlanner(PlannerConfig('model-x'), client=visible_client).select_action(visible)
    text = visible_client.responses.calls[0]['input']
    assert 'num_atoms' in text and 'initial_max_force_ev_ang' in text
    assert 'unsafe/path' not in text


def test_invalid_candidate_duplicate_and_stop_actions_retry():
    pool = synthetic_pool()[:3]
    history = [Observation(pool[0].candidate_id, 'failed', None, None, 0.1, 'failed')]
    current = State(pool, history, 7)
    responses = FakeClient(
        Response(choice('not-in-pool')),
        Response(choice(pool[0].candidate_id)),
        Response(choice(None, 'stop', 'No useful unseen option remains.')),
    )
    planner = OpenAIPlanner(PlannerConfig('m', max_attempts=3), client=responses)
    assert planner.select_action(current) == Action('stop')
    assert len(responses.responses.calls) == 3


def test_retry_bound_api_failures_and_usage_for_failed_structured_actions(tmp_path):
    client = FakeClient(RuntimeError('temporary'), Response({'bad': 'shape'}), RuntimeError('still down'))
    log = tmp_path / 'api.jsonl'
    planner = OpenAIPlanner(PlannerConfig('m', max_attempts=2), client=client, request_log_path=log)
    with pytest.raises(OpenAIPlannerError, match='2 attempt'):
        planner.select_action(state())
    assert len(client.responses.calls) == 2
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [record['attempt'] for record in records] == [1, 2]
    assert all(record['outcome'] == 'rejected_or_api_error' for record in records)
    assert all('api_key' not in json.dumps(record).lower() for record in records)


def test_usage_cost_and_request_metadata_are_logged(tmp_path):
    pool = synthetic_pool()[:1]
    client = FakeClient(Response(choice(pool[0].candidate_id), input_tokens=150, output_tokens=50))
    planner = OpenAIPlanner(
        PlannerConfig('gpt-test', proxy_url='http://127.0.0.1:17897',
                      input_price_per_million=2.0, output_price_per_million=8.0),
        client=client, request_log_path=tmp_path / 'requests.jsonl',
    )
    planner.select_action(State(pool, [], 8))
    record = json.loads((tmp_path / 'requests.jsonl').read_text())
    assert record['input_tokens'] == 150 and record['output_tokens'] == 50
    assert record['total_tokens'] == 200
    assert record['input_cost_usd'] == pytest.approx(0.0003)
    assert record['output_cost_usd'] == pytest.approx(0.0004)
    assert record['total_cost_usd'] == pytest.approx(0.0007)
    assert record['model'] == 'gpt-test' and record['request_id'] == 'req-test'
    assert record['request_sha256'] and record['candidate_count'] == 1


def test_missing_credentials_and_sdk_client_uses_http1_proxy(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        OpenAIPlanner(PlannerConfig('m'))

    capture = {}

    class HttpClient:
        def __init__(self, **kwargs):
            capture['http'] = kwargs

        def close(self):
            capture['closed'] = True

    class SDKClient:
        def __init__(self, **kwargs):
            capture['sdk'] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setenv('OPENAI_API_KEY', 'test-secret')
    monkeypatch.setitem(sys.modules, 'httpx', types.SimpleNamespace(Client=HttpClient))
    monkeypatch.setitem(sys.modules, 'openai', types.SimpleNamespace(OpenAI=SDKClient))
    planner = OpenAIPlanner(PlannerConfig('m', proxy_url='http://proxy:1234'))
    assert capture['http'] == {'http2': False, 'proxy': 'http://proxy:1234'}
    assert capture['sdk']['api_key'] == 'test-secret'
    assert capture['sdk']['max_retries'] == 0
    planner.close()
    assert capture['closed']


def test_environment_configuration_and_optional_pricing(monkeypatch):
    monkeypatch.setenv('OPENAI_MODEL', 'account-model')
    monkeypatch.setenv('OPENAI_PROXY_URL', 'http://127.0.0.1:20000')
    monkeypatch.setenv('OPENAI_INPUT_PRICE_PER_1M', '1.25')
    monkeypatch.setenv('OPENAI_OUTPUT_PRICE_PER_1M', '5')
    config = PlannerConfig.from_env()
    assert config.model == 'account-model'
    assert config.proxy_url.endswith('20000')
    assert config.input_price_per_million == 1.25
    assert config.public_dict()['output_price_per_million'] == 5


def test_model_must_be_configured_and_is_never_defaulted(monkeypatch):
    monkeypatch.delenv('OPENAI_MODEL', raising=False)
    with pytest.raises(ValueError, match='OPENAI_MODEL is required'):
        PlannerConfig.from_env()


def test_mocked_eight_step_run_preserves_budget_and_writes_usage(tmp_path):
    pool = synthetic_pool()
    results = [Response(choice(candidate.candidate_id), input_tokens=10, output_tokens=4,
                        response_id=f'resp-{i}')
               for i, candidate in enumerate(pool[:8])]
    client = FakeClient(*results)
    output = tmp_path / 'run'
    summary = run_openai_experiment(
        pool, MockPhysicsEvaluator(42), RunConfig(output, seed=42),
        planner_config=PlannerConfig('mock-gpt'), client=client,
    )
    assert summary['relaxation_attempts'] == 8
    assert summary['remaining_relaxations'] == 0
    assert summary['selection_budget'] == 8
    assert len(client.responses.calls) == 8
    assert len((output / 'api_requests.jsonl').read_text().splitlines()) == 8
    usage = json.loads((output / 'api_usage.json').read_text())
    assert usage['request_count'] == 8
    assert usage['input_tokens'] == 80 and usage['output_tokens'] == 32
    assert usage['pricing_configured'] is False
    decisions = [json.loads(line) for line in (output / 'decisions.jsonl').read_text().splitlines()]
    assert [record['budget_after'] for record in decisions] == list(range(7, -1, -1))


def test_budget_is_not_changed_by_stop_or_planner_failure(tmp_path):
    pool = synthetic_pool()[:2]
    stop_client = FakeClient(Response(choice(None, 'stop')))
    output = tmp_path / 'stop'
    summary = run_openai_experiment(
        pool, MockPhysicsEvaluator(), RunConfig(output),
        planner_config=PlannerConfig('m'), client=stop_client,
    )
    assert summary['remaining_relaxations'] == 8
    assert summary['relaxation_attempts'] == 0

    bad_client = FakeClient(RuntimeError('offline'))
    failure = run_openai_experiment(
        pool, MockPhysicsEvaluator(), RunConfig(tmp_path / 'failure'),
        planner_config=PlannerConfig('m', max_attempts=1), client=bad_client,
    )
    assert failure['remaining_relaxations'] == 8
    assert failure['relaxation_attempts'] == 0


def test_run_config_never_records_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY', 'secret-value')
    client = FakeClient(Response(choice(synthetic_pool()[0].candidate_id)))
    output = tmp_path / 'run'
    run_openai_experiment(
        synthetic_pool()[:1], MockPhysicsEvaluator(), RunConfig(output),
        planner_config=PlannerConfig('safe-model'), client=client,
    )
    assert 'secret-value' not in (output / 'run_config.json').read_text()
    assert 'safe-model' in (output / 'run_config.json').read_text()
