# Phase 2 OpenAI Planner

Install the optional client dependencies into the existing environment when network/package access is available:

```bash
source ../.venvs/mattergen/bin/activate
pip install -e '.[llm]'
```

The `OpenAIPlanner` implements the existing `Planner.select_action(State) -> Action` contract using the official OpenAI Python SDK Responses API. It sends a strict JSON Schema containing `action_type`, nullable `candidate_id`, and a concise `justification`. It validates the result against the current State before returning; the Agent Runtime still performs its own action validation and owns all budget/evaluator calls.

At the CLI entry point, `python-dotenv` loads `.env` if present. Configuration is read from the process environment after loading:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | API credential; never written to run artifacts |
| `OPENAI_MODEL` | required | Model name enabled for the API account; no model is hardcoded |
| `OPENAI_PROXY_URL` | `http://127.0.0.1:17897` | Explicit HTTP proxy, including the required SSH tunnel |
| `OPENAI_MAX_ATTEMPTS` | `3` | Bounded request/validation attempts, maximum 5 |
| `OPENAI_MAX_OUTPUT_TOKENS` | `500` | Response output bound |
| `OPENAI_TEMPERATURE` | unset | Optional generation parameter; omitted by default |
| `OPENAI_REASONING_EFFORT` | unset | Optional Responses reasoning effort |
| `OPENAI_INPUT_PRICE_PER_1M` | unset | Optional configured input-token USD price per million |
| `OPENAI_OUTPUT_PRICE_PER_1M` | unset | Optional configured output-token USD price per million |

The SDK client is created with `httpx.Client(http2=False, proxy=OPENAI_PROXY_URL)` and SDK retries disabled, so the planner's configured attempt bound is the only retry loop. Each attempt is appended to `api_requests.jsonl` with the model, request parameters, deterministic prompt/state SHA256, input byte and candidate/history counts, response/request IDs, token usage, estimated cost, result and error. The key and full prompt input are not logged. `api_usage.json` aggregates usage and cost. Costs are estimated only when both per-million prices are configured; pricing is user supplied and is not inferred from a model name.

The prompt receives only remaining budget, unevaluated candidate IDs and formulas, the optional feature and initial-force fields present in the runtime-projected State, and this run's observations. It omits CIF paths, CIF hashes, filesystem locations, and failed-calculation diagnostics beyond a generic failure marker. It warns that relaxation convergence does not establish thermodynamic stability and raw per-atom energies are not universal cross-composition stability scores.

## Real API with mock physics

This is the explicit command for a real API call; it uses synthetic candidate metadata and `MockPhysicsEvaluator`, so it makes no GPU calls. It may incur API charges. Set credentials and model in the environment or a local `.env`, and ensure the SSH tunnel listens on the configured proxy port first.

```bash
cd ~/gxq/5040/AIST5040-Materials-Agent
source ../.venvs/mattergen/bin/activate
export OPENAI_API_KEY='...'
export OPENAI_MODEL='model-enabled-for-your-account'
export OPENAI_PROXY_URL='http://127.0.0.1:17897'
run_root=$(mktemp -d "${TMPDIR:-/tmp}/aist5040-openai.XXXXXX")
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m materials_agent.agent.openai_experiment \
  --output-dir "$run_root/gpt-mock-physics" --seed 42 --budget 8
```

To use the existing MatterGen pool, pass `--candidate-dir results/fixed_v0/candidate_pool --screening-csv results/fixed_v0/screening.csv`. Optional visibility flags are `--include-structure-features` and `--allow-initial-max-force`; initial screening cost accounting follows the mock experiment loader. The CLI defaults to mock physics. A separately configured `--evaluator-factory module:callable` can provide an implementation of the existing `PhysicsEvaluator` interface; no CHGNet adapter or GPU experiment is part of this phase.

Run the offline suite without API credentials:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

Phase 3 still needs a thin `PhysicsEvaluator` adapter that resolves a candidate-relative CIF under a configured pool root, uses the existing `load_chgnet`/`relax_structure` helpers with the agreed Fixed Baseline settings, writes relaxed CIFs under the run directory, maps returned fields to `Observation`, and accounts for evaluator failures. Validate the adapter with an explicitly authorized minimal real calculation before any full run.
