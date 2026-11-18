"""Reproducible MatterGen candidate screening with a fixed CHGNet ranking."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from materials_agent.tools.chgnet import load_chgnet, relax_structure, single_point

LOGGER = logging.getLogger(__name__)
MIN_ATOMIC_DISTANCE_A = 0.5


class RankingPolicy(Protocol):
    """Replaceable policy that orders screened candidate records."""

    def key(self, record: dict[str, Any]) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class InitialMaxForceRanking:
    """Rank by cheap initial maximum force, then stable candidate ID."""

    def key(self, record: dict[str, Any]) -> tuple[float, str]:
        return float(record['max_force_eV_per_A']), str(record['candidate_id'])


@dataclass(frozen=True)
class FixedScreeningConfig:
    candidate_dir: Path
    output_dir: Path
    top_k: int = 8
    seed: int = 42
    checkpoint: str = 'mattergen_base'
    device: str = 'cuda:0'
    fmax: float = 0.05
    max_steps: int = 500
    min_atomic_distance: float = MIN_ATOMIC_DISTANCE_A
    generate_count: int = 0
    generation_batch_size: int = 4
    relax_cell: bool = True


def _candidate_files(candidate_dir: Path) -> list[Path]:
    if not candidate_dir.exists():
        raise FileNotFoundError(f'Candidate directory does not exist: {candidate_dir}')
    return sorted(
        (p for p in candidate_dir.rglob('*') if p.is_file() and p.suffix.lower() == '.cif'),
        key=lambda p: p.relative_to(candidate_dir).as_posix(),
    )


def _stable_ids(files: list[Path]) -> dict[Path, str]:
    ids: dict[Path, str] = {}
    used: set[str] = set()
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        label = re.sub(r'[^A-Za-z0-9_.-]+', '-', path.stem).strip('-.') or 'candidate'
        candidate_id = f'{label}-{digest}'
        suffix = 2
        while candidate_id in used:
            candidate_id = f'{label}-{digest}-{suffix}'
            suffix += 1
        ids[path] = candidate_id
        used.add(candidate_id)
    return ids


def _validate_structure(structure: Any, min_distance: float) -> str | None:
    if len(structure) < 1:
        return 'empty structure'
    distances = np.asarray(structure.distance_matrix, dtype=float)
    if distances.shape != (len(structure), len(structure)) or not np.isfinite(distances).all():
        return 'invalid periodic distance matrix'
    if len(structure) > 1:
        upper = distances[np.triu_indices(len(structure), k=1)]
        if upper.size and float(upper.min()) < min_distance:
            return f'minimum periodic atomic distance below {min_distance:g} A'
    return None


def _prepare_generated(config: FixedScreeningConfig) -> None:
    if config.generate_count <= 0:
        return
    from materials_agent.tools.mattergen import generate

    generated_dir = config.output_dir / 'generated'
    generated_dir.mkdir(parents=True, exist_ok=True)
    generate(
        generated_dir, num_candidates=config.generate_count, seed=config.seed,
        batch_size=config.generation_batch_size,
    )
    archive_path = generated_dir / 'generated_crystals_cif.zip'
    if not archive_path.exists():
        raise FileNotFoundError(f'MatterGen did not create expected CIF archive: {archive_path}')
    config.candidate_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.filename.lower().endswith('.cif'):
                target_name = Path(member.filename).name
                target = config.candidate_dir / target_name
                if target.exists():
                    target = config.candidate_dir / f'{archive_path.stem}-{target_name}'
                target.write_bytes(archive.read(member))


def _load_run_metadata(config: FixedScreeningConfig) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    mattergen_repo = project_root.parent / 'mattergen'
    try:
        commit = subprocess.run(
            ['git', '-C', str(project_root), 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions: dict[str, str] = {}
    checkpoint_revision = None
    try:
        from importlib.metadata import version
        import torch

        versions = {
            'chgnet': version('chgnet'),
            'pymatgen': version('pymatgen'),
            'torch': torch.__version__,
        }
        if torch.cuda.is_available():
            device_index = int(config.device.split(':')[1]) if ':' in config.device else 0
            versions['gpu'] = torch.cuda.get_device_name(device_index)
        try:
            from huggingface_hub import scan_cache_dir

            repo_id = f'microsoft/{config.checkpoint}' if config.checkpoint != 'mattergen_base' else 'microsoft/mattergen'
            for repo in scan_cache_dir().repos:
                if repo.repo_id == repo_id:
                    checkpoint_revision = next(
                        (revision.commit_hash for revision in repo.revisions if 'main' in revision.refs),
                        None,
                    )
                    break
        except (ImportError, OSError, ValueError):
            checkpoint_revision = None
    except (ImportError, OSError, Exception) as exc:
        versions['metadata_error'] = str(exc)
    try:
        mattergen_commit = subprocess.run(
            ['git', '-C', str(mattergen_repo), 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        mattergen_commit = None
    return {
        'seed': config.seed,
        'candidate_budget': config.generate_count or len(_candidate_files(config.candidate_dir)),
        'selection_budget': config.top_k,
        'checkpoint': config.checkpoint,
        'checkpoint_revision': checkpoint_revision,
        'scorer': 'CHGNet single-point initial maximum atomic force',
        'device': config.device,
        'relaxation': {
            'model': 'CHGNet', 'fmax_eV_per_A': config.fmax,
            'max_steps': config.max_steps, 'relax_cell': config.relax_cell,
        },
        'minimum_atomic_distance_A': config.min_atomic_distance,
        'git_commit': commit,
        'mattergen_git_commit': mattergen_commit,
        'versions': versions,
    }


def run_fixed_screening(
    config: FixedScreeningConfig,
    *,
    model: Any | None = None,
    structure_loader: Callable[[Path], Any] | None = None,
    scorer: Callable[[Any, Any], dict[str, Any]] = single_point,
    relaxer: Callable[..., dict[str, Any]] = relax_structure,
    ranking: RankingPolicy | None = None,
) -> dict[str, Any]:
    """Run validation, single-point scoring, fixed top-K selection and relaxation."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    start = time.perf_counter()
    if config.top_k < 0 or config.max_steps < 1 or config.fmax <= 0:
        raise ValueError('top_k must be non-negative; max_steps and fmax must be positive')
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = config.output_dir / 'cifs'
    relaxed_dir = config.output_dir / 'relaxed'
    cif_dir.mkdir(exist_ok=True)
    relaxed_dir.mkdir(exist_ok=True)
    _prepare_generated(config)
    files = _candidate_files(config.candidate_dir)
    if not files:
        raise ValueError(f'No CIF candidates found in {config.candidate_dir}')
    id_by_path = _stable_ids(files)
    load = structure_loader or Structure.from_file
    matcher = StructureMatcher(primitive_cell=False, scale=True, attempt_supercell=False)
    structures: dict[str, Any] = {}
    canonical: list[tuple[str, Any]] = []
    records: list[dict[str, Any]] = []

    for path in files:
        candidate_id = id_by_path[path]
        record: dict[str, Any] = {
            'candidate_id': candidate_id,
            'source_file': str(path),
            'formula': '',
            'validity': 'invalid',
            'validity_reason': '',
            'duplicate': False,
            'duplicate_of': '',
            'single_point_status': 'not_run',
            'single_point_seconds': '',
            'energy_per_atom_eV': '',
            'forces_eV_per_A': '',
            'max_force_eV_per_A': '',
            'rank': '',
            'selected': False,
            'relaxation_status': 'not_selected',
            'relaxation_seconds': '',
            'relaxed_cif': '',
            'final_energy_per_atom_eV': '',
            'final_total_energy_eV': '',
            'final_max_force_eV_per_A': '',
            'converged': '',
            'relaxation_error': '',
        }
        try:
            structure = load(path)
            record['formula'] = structure.composition.reduced_formula
            invalid_reason = _validate_structure(structure, config.min_atomic_distance)
            if invalid_reason:
                record['validity_reason'] = invalid_reason
            else:
                record['validity'] = 'valid'
                structures[candidate_id] = structure
                duplicate_of = next(
                    (known_id for known_id, known in canonical if matcher.fit(known, structure)), None
                )
                if duplicate_of:
                    record['duplicate'] = True
                    record['duplicate_of'] = duplicate_of
                else:
                    canonical.append((candidate_id, structure))
            shutil.copy2(path, cif_dir / f'{candidate_id}.cif')
        except Exception as exc:  # preserve parse failures as candidate rows
            record['validity_reason'] = f'{type(exc).__name__}: {exc}'
            LOGGER.warning('Could not validate %s: %s', candidate_id, exc)
        records.append(record)

    scorable = [r for r in records if r['validity'] == 'valid' and not r['duplicate']]
    if scorable:
        model = model if model is not None else load_chgnet(config.device)
    single_point_calls = 0
    for record in scorable:
        single_point_calls += 1
        call_started = time.perf_counter()
        try:
            result = scorer(model, structures[record['candidate_id']])
            record['single_point_status'] = 'success'
            record['single_point_seconds'] = result.get('elapsed_seconds', '')
            record['energy_per_atom_eV'] = result['energy_per_atom_eV']
            record['forces_eV_per_A'] = json.dumps(result.get('forces_eV_per_A', []))
            record['max_force_eV_per_A'] = result['max_force_eV_per_A']
        except Exception as exc:
            record['single_point_status'] = 'failed'
            record['single_point_seconds'] = time.perf_counter() - call_started
            record['validity_reason'] = f'single-point failed: {type(exc).__name__}: {exc}'

    policy = ranking or InitialMaxForceRanking()
    ranked = sorted(
        (r for r in scorable if r['single_point_status'] == 'success'), key=policy.key
    )
    for rank, record in enumerate(ranked, start=1):
        record['rank'] = rank
    selected = ranked[:config.top_k]
    for record in selected:
        record['selected'] = True
        record['relaxation_status'] = 'pending'
    selected_json = [
        {'rank': r['rank'], 'candidate_id': r['candidate_id'], 'score_max_force_eV_per_A': r['max_force_eV_per_A']}
        for r in selected
    ]
    (config.output_dir / 'selected_candidates.json').write_text(
        json.dumps(selected_json, indent=2) + '\n'
    )

    relaxation_attempts = 0
    relaxation_successes = 0
    for record in selected:
        relaxation_attempts += 1
        record['relaxation_status'] = 'failed'
        output_path = relaxed_dir / f"{record['candidate_id']}.cif"
        call_started = time.perf_counter()
        try:
            result = relaxer(
                model, structures[record['candidate_id']], output_path,
                device=config.device, fmax=config.fmax, steps=config.max_steps,
                relax_cell=config.relax_cell,
            )
            record['relaxation_status'] = 'success'
            record['relaxation_seconds'] = result.get('elapsed_seconds', '')
            record['relaxed_cif'] = result.get('relaxed_cif', str(output_path))
            record['final_energy_per_atom_eV'] = result['final_energy_per_atom_eV']
            record['final_total_energy_eV'] = result['final_total_energy_eV']
            record['final_max_force_eV_per_A'] = result['final_max_force_eV_per_A']
            record['converged'] = bool(result['converged'])
            relaxation_successes += 1
        except Exception as exc:
            record['relaxation_seconds'] = time.perf_counter() - call_started
            record['relaxation_error'] = f'{type(exc).__name__}: {exc}'
            LOGGER.exception('Relaxation failed for %s', record['candidate_id'])

    fields = list(records[0])
    with (config.output_dir / 'screening.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    elapsed = time.perf_counter() - start
    summary = {
        'candidate_count': len(records),
        'valid_candidate_count': sum(r['validity'] == 'valid' for r in records),
        'unique_valid_candidate_count': sum(r['validity'] == 'valid' and not r['duplicate'] for r in records),
        'duplicate_count': sum(bool(r['duplicate']) for r in records),
        'single_point_calls': single_point_calls,
        'single_point_failures': sum(r['single_point_status'] == 'failed' for r in records),
        'ranked_candidate_count': len(ranked),
        'selection_budget': config.top_k,
        'selected_count': len(selected),
        'selection_budget_shortfall': max(0, config.top_k - len(selected)),
        'relaxation_attempts': relaxation_attempts,
        'relaxation_successes': relaxation_successes,
        'relaxation_failures': relaxation_attempts - relaxation_successes,
        'force_converged_count': sum(r['converged'] is True for r in records),
        'total_runtime_seconds': elapsed,
        'total_single_point_seconds': sum(float(r['single_point_seconds'] or 0) for r in records),
        'total_relaxation_seconds': sum(float(r['relaxation_seconds'] or 0) for r in records),
    }
    (config.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    run_config = _load_run_metadata(config)
    run_config.update(asdict(config))
    run_config['candidate_dir'] = str(config.candidate_dir)
    run_config['output_dir'] = str(config.output_dir)
    run_config['ranking_policy'] = type(policy).__name__
    run_config['generation'] = {
        'candidate_budget': config.generate_count,
        'requested_batch_size': config.generation_batch_size,
        'effective_batch_size': min(config.generation_batch_size, config.generate_count) if config.generate_count else None,
        'num_batches': (config.generate_count + config.generation_batch_size - 1) // config.generation_batch_size if config.generate_count else None,
    }
    (config.output_dir / 'run_config.json').write_text(json.dumps(run_config, indent=2, default=str) + '\n')
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', type=Path, required=True, help='Directory containing input CIF files')
    parser.add_argument('--output-dir', type=Path, default=Path('results/fixed_v0'))
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', default='mattergen_base')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fmax', type=float, default=0.05)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--min-atomic-distance', type=float, default=MIN_ATOMIC_DISTANCE_A)
    parser.add_argument('--generate-count', type=int, default=0, help='Optionally generate this many MatterGen samples before screening')
    parser.add_argument('--generation-batch-size', type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    config = FixedScreeningConfig(
        candidate_dir=args.candidate_dir, output_dir=args.output_dir, top_k=args.top_k,
        seed=args.seed, checkpoint=args.checkpoint, device=args.device, fmax=args.fmax,
        max_steps=args.max_steps, min_atomic_distance=args.min_atomic_distance,
        generate_count=args.generate_count, generation_batch_size=args.generation_batch_size,
    )
    print(json.dumps(run_fixed_screening(config), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
