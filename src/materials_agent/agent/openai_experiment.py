"""Optional OpenAI Planner experiment entry point; default evaluator is synthetic."""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

from materials_agent.contracts import Candidate, PhysicsEvaluator

from .mock_evaluator import MockPhysicsEvaluator
from .mock_experiment import load_fixed_pool, synthetic_pool
from .openai_planner import OpenAIPlanner, PlannerConfig
from .runtime import RunConfig, ScreeningCosts, run_agent


def run_openai_experiment(
    candidates: Iterable[Candidate],
    physics: PhysicsEvaluator,
    config: RunConfig,
    *,
    planner_config: PlannerConfig | None = None,
    client: Any | None = None,
) -> dict:
    """Run GPT decisions against any implementation of the shared evaluator interface."""
    planner_config = planner_config or PlannerConfig.from_env()
    with OpenAIPlanner(
        planner_config, client=client,
        request_log_path=config.output_dir / 'api_requests.jsonl',
    ) as planner:
        run_config = replace(config, planner_settings=planner_config.public_dict())
        summary = run_agent(candidates, planner, physics, run_config)
        usage = planner.request_records

    rounds = {}
    for record in usage:
        rounds.setdefault(record['round'], []).append(record)
    (config.output_dir / 'api_usage.json').write_text(json.dumps({
        'request_count': len(usage),
        'response_models': sorted({r['response_model'] for r in usage if r.get('response_model')}),
        'cached_input_tokens': sum(r.get('cached_input_tokens') or 0 for r in usage),
        'billing_actual_usd': None,
        'cost_basis': 'response token counts times configured rates; no invoice reconciliation',
        'rounds': [
            {'round': index, 'requests': len(records),
             'api_elapsed_seconds': sum(r['elapsed_seconds'] for r in records)}
            for index, records in rounds.items()
        ],
        'accepted_count': sum(record.get('outcome') == 'accepted' for record in usage),
        'input_tokens': sum(record.get('input_tokens') or 0 for record in usage),
        'output_tokens': sum(record.get('output_tokens') or 0 for record in usage),
        'total_tokens': sum(record.get('total_tokens') or 0 for record in usage),
        'input_cost_usd': sum(record.get('input_cost_usd') or 0 for record in usage),
        'output_cost_usd': sum(record.get('output_cost_usd') or 0 for record in usage),
        'total_cost_usd': (sum(r['total_cost_usd'] for r in usage if r.get('total_cost_usd') is not None)
                           if any(r.get('total_cost_usd') is not None for r in usage) else None),
        'pricing_configured': bool(usage) and all(record.get('pricing_configured') for record in usage),
    }, indent=2, sort_keys=True) + '\n')
    return summary


def _load_evaluator(spec: str) -> PhysicsEvaluator:
    """Load a user-configured adapter factory returning the shared evaluator interface."""
    module_name, separator, attribute = spec.partition(':')
    if not separator:
        raise ValueError('evaluator factory must use module:callable syntax')
    factory = getattr(importlib.import_module(module_name), attribute)
    evaluator = factory()
    if not callable(getattr(evaluator, 'relax', None)):
        raise TypeError('configured evaluator must implement relax(candidate)')
    return evaluator


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit("Install optional LLM dependencies: pip install -e '.[llm]'") from exc
    load_dotenv(Path(__file__).resolve().parents[3] / '.env')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--candidate-dir', type=Path)
    parser.add_argument('--screening-csv', type=Path)
    parser.add_argument('--allow-initial-max-force', action='store_true')
    parser.add_argument('--include-structure-features', action='store_true')
    evaluators = parser.add_mutually_exclusive_group()
    evaluators.add_argument('--evaluator-factory', help='module:callable returning a PhysicsEvaluator')
    evaluators.add_argument('--chgnet', action='store_true', help='Use real CHGNet relaxation')
    parser.add_argument('--device', default='cuda:0', help='CHGNet device when --chgnet is set')
    args = parser.parse_args(argv)
    if (args.candidate_dir is None) != (args.screening_csv is None):
        parser.error('--candidate-dir and --screening-csv must be supplied together')
    if args.allow_initial_max_force and args.screening_csv is None:
        parser.error('--allow-initial-max-force requires the candidate pool and screening CSV')

    if args.chgnet and args.candidate_dir is None:
        parser.error('--chgnet requires the original candidate pool and screening CSV')

    if args.screening_csv:
        candidates, screening, feature_seconds = load_fixed_pool(
            args.candidate_dir, args.screening_csv,
            args.allow_initial_max_force, args.include_structure_features,
        )
        pool_source = str(args.screening_csv)
    else:
        candidates, screening, feature_seconds = synthetic_pool(), ScreeningCosts(), 0.0
        pool_source = 'synthetic fixture v1'
    evaluator_settings = {}
    if args.chgnet:
        from materials_agent.evaluation.chgnet_evaluator import CHGNetEvaluator, CHGNetSettings
        settings = CHGNetSettings(device=args.device)
        physics = CHGNetEvaluator(args.candidate_dir, args.output_dir, settings)
        evaluator_settings = {**asdict(settings), 'chgnet_version': version('chgnet')}
    elif args.evaluator_factory:
        physics = _load_evaluator(args.evaluator_factory)
    else:
        physics = MockPhysicsEvaluator(args.seed)
    run_config = RunConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        relaxation_budget=args.budget,
        allow_initial_max_force=args.allow_initial_max_force,
        allow_structure_features=args.include_structure_features,
        structure_feature_extraction_seconds=feature_seconds,
        screening=screening,
        pool_source=pool_source,
        pool_root=str(args.candidate_dir) if args.candidate_dir else None,
        synthetic=not (args.chgnet or args.evaluator_factory),
        evaluator_settings=evaluator_settings,
    )
    summary = run_openai_experiment(candidates, physics, run_config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
