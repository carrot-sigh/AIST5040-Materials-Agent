"""Run a CPU-only mock experiment on a synthetic or existing candidate pool."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from materials_agent.contracts import Candidate
from materials_agent.structure_features import extract_structure_features

from .mock_evaluator import MockPhysicsEvaluator
from .mock_planner import MockPlanner
from .runtime import RunConfig, ScreeningCosts, run_agent


def synthetic_pool() -> list[Candidate]:
    formulas = ('LiFeO2', 'NaCl', 'ZnO', 'CaTiO3')
    return [Candidate(f'mock-{i:03}', formulas[i % len(formulas)], f'mock-{i:03}.cif')
            for i in range(64)]


def load_fixed_pool(
    candidate_dir: Path, screening_csv: Path, allow_initial_max_force: bool,
    include_structure_features: bool = False,
) -> tuple[list[Candidate], ScreeningCosts, float]:
    """Read only the identity, eligibility, and explicitly enabled single-point fields."""
    files_by_name: dict[str, list[tuple[Path, str]]] = {}
    pool_root = candidate_dir.resolve()
    pool_files = set()
    for path in candidate_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() == '.cif':
            if not path.resolve().is_relative_to(pool_root):
                raise ValueError(f'CIF outside candidate root: {path}')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files_by_name.setdefault(path.name, []).append((path, digest))
            pool_files.add(path)

    candidates = []
    seen_files = set()
    calls = failures = 0
    seconds = 0.0
    feature_seconds = 0.0
    with screening_csv.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            score = None
            if allow_initial_max_force:
                status = row['single_point_status']
                if status not in ('success', 'failed', 'not_run'):
                    raise ValueError(f'unrecognized single-point status: {status!r}')
                if status in ('success', 'failed'):
                    calls += 1
                    failures += status == 'failed'
                    seconds += float(row['single_point_seconds'])
                if status == 'success':
                    score = float(row['max_force_eV_per_A'])
            candidate_id = row['candidate_id']
            name = Path(row['source_file']).name
            matches = []
            for path, digest in files_by_name.get(name, []):
                if re.search(rf'-{digest[:12]}(?:-\d+)?$', candidate_id):
                    matches.append((path, digest))
            if len(matches) != 1:
                raise ValueError(f'expected one matching CIF for {candidate_id}, found {len(matches)}')
            path, digest = matches[0]
            if path in seen_files:
                raise ValueError(f'multiple screening rows for {path}')
            seen_files.add(path)
            if row['validity'] != 'valid' or row['duplicate'] != 'False':
                continue
            features = {}
            if include_structure_features:
                from time import perf_counter
                from pymatgen.core import Structure
                feature_started = perf_counter()
                features = extract_structure_features(Structure.from_file(path))
                feature_seconds += perf_counter() - feature_started
            candidates.append(Candidate(candidate_id, row['formula'], path.relative_to(candidate_dir).as_posix(),
                                        initial_max_force_ev_ang=score, cif_sha256=digest,
                                        features=features))

    if seen_files != pool_files:
        raise ValueError('screening CSV must cover the exact input CIF pool')
    if not candidates:
        raise ValueError('no eligible CIF candidates found')
    costs = ScreeningCosts(source=str(screening_csv), reused_calls=calls,
                           failures=failures, reused_elapsed_seconds=seconds) if allow_initial_max_force else ScreeningCosts()
    return sorted(candidates, key=lambda item: item.cif_path), costs, feature_seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New run directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--stop-after', type=int, help='Exercise an early mock stop')
    parser.add_argument('--candidate-dir', type=Path, help='Original CIF pool from a fixed screening run')
    parser.add_argument('--screening-csv', type=Path, help='Fixed screening CSV for pool identity and eligibility')
    parser.add_argument('--allow-initial-max-force', action='store_true',
                        help='Opt in to cached initial forces and attribute their original screening cost')
    parser.add_argument('--include-structure-features', action='store_true',
                        help='Expose geometry and composition parsed from the input CIFs')
    args = parser.parse_args(argv)
    if (args.candidate_dir is None) != (args.screening_csv is None):
        parser.error('--candidate-dir and --screening-csv must be supplied together')
    if args.allow_initial_max_force and args.screening_csv is None:
        parser.error('--allow-initial-max-force requires the candidate pool and screening CSV')
    if args.stop_after is not None and args.stop_after < 0:
        parser.error('--stop-after must be non-negative')

    if args.screening_csv is None:
        pool, costs = synthetic_pool(), ScreeningCosts()
        source = 'synthetic fixture v1'
        feature_seconds = 0.0
    else:
        pool, costs, feature_seconds = load_fixed_pool(
            args.candidate_dir, args.screening_csv,
            args.allow_initial_max_force, args.include_structure_features,
        )
        source = str(args.screening_csv)
    summary = run_agent(pool, MockPlanner(args.seed, args.stop_after), MockPhysicsEvaluator(args.seed), RunConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        relaxation_budget=args.budget,
        allow_initial_max_force=args.allow_initial_max_force,
        allow_structure_features=args.include_structure_features,
        structure_feature_extraction_seconds=feature_seconds,
        screening=costs,
        pool_source=source,
        pool_root=str(args.candidate_dir) if args.candidate_dir is not None else None,
        planner_settings={'stop_after': args.stop_after},
    ))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
