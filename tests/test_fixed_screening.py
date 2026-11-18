import csv
import json
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from materials_agent.tools import mattergen
from materials_agent.baselines.fixed_screening import (
    FixedScreeningConfig,
    InitialMaxForceRanking,
    run_fixed_screening,
)


def write_cif(path: Path, lattice_constant: float = 3.0) -> None:
    structure = Structure(
        Lattice.cubic(lattice_constant), ['Li', 'F'] if lattice_constant == 4.0 else ['Na', 'Cl'],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    structure.to(filename=str(path), fmt='cif')


class FakeModel:
    pass


def fake_score(_model, structure):
    force = float(structure.lattice.a)
    return {
        'energy_per_atom_eV': -1.0,
        'max_force_eV_per_A': force,
        'forces_eV_per_A': [[force, 0, 0]] * len(structure),
        'elapsed_seconds': 0.01,
    }


def fake_relax(_model, structure, output_path, **kwargs):
    if structure.lattice.a == 4:
        raise RuntimeError('intentional relaxation error')
    return {
        'relaxed_cif': str(output_path),
        'final_energy_per_atom_eV': -2.0,
        'final_total_energy_eV': -4.0,
        'final_max_force_eV_per_A': 0.04,
        'converged': True,
        'elapsed_seconds': 0.02,
    }


def test_fixed_ranking_deduplication_budgets_and_failure_accounting(tmp_path):
    pool = tmp_path / 'pool'
    pool.mkdir()
    write_cif(pool / 'a.cif', 3.0)
    write_cif(pool / 'duplicate.cif', 3.0)
    write_cif(pool / 'b.cif', 4.0)
    (pool / 'invalid.cif').write_text('not a CIF')

    summary = run_fixed_screening(
        FixedScreeningConfig(pool, tmp_path / 'out', top_k=2, device='cpu'),
        model=FakeModel(), scorer=fake_score, relaxer=fake_relax,
    )

    assert summary['candidate_count'] == 4
    assert summary['valid_candidate_count'] == 3
    assert summary['duplicate_count'] == 1
    assert summary['single_point_calls'] == 2
    assert summary['selection_budget_shortfall'] == 0
    assert summary['relaxation_attempts'] == 2
    assert summary['relaxation_successes'] == 1
    assert summary['relaxation_failures'] == 1

    rows = list(csv.DictReader((tmp_path / 'out/screening.csv').open()))
    assert len(rows) == 4
    assert sum(row['selected'] == 'True' for row in rows) == 2
    assert next(row for row in rows if row['duplicate'] == 'True')['duplicate_of']
    assert next(row for row in rows if row['validity'] == 'invalid')['validity_reason']
    selected = json.loads((tmp_path / 'out/selected_candidates.json').read_text())
    assert [row['rank'] for row in selected] == [1, 2]
    assert InitialMaxForceRanking().key({'max_force_eV_per_A': 1, 'candidate_id': 'b'}) > InitialMaxForceRanking().key({'max_force_eV_per_A': 1, 'candidate_id': 'a'})


def test_selection_shortfall_and_tie_break(tmp_path):
    pool = tmp_path / 'pool'
    pool.mkdir()
    write_cif(pool / 'z.cif', 3.0)
    write_cif(pool / 'a.cif', 4.0)

    summary = run_fixed_screening(
        FixedScreeningConfig(pool, tmp_path / 'out', top_k=5, device='cpu'),
        model=FakeModel(), scorer=lambda *_: {
            'energy_per_atom_eV': -1.0, 'max_force_eV_per_A': 0.2,
            'forces_eV_per_A': [], 'elapsed_seconds': 0.0,
        }, relaxer=lambda *_args, **_kwargs: {
            'relaxed_cif': '', 'final_energy_per_atom_eV': -1.0,
            'final_total_energy_eV': -2.0, 'final_max_force_eV_per_A': 0.01,
            'converged': True, 'elapsed_seconds': 0.0,
        },
    )
    assert summary['selected_count'] == 2
    assert summary['selection_budget_shortfall'] == 3
    selected = json.loads((tmp_path / 'out/selected_candidates.json').read_text())
    assert selected[0]['candidate_id'] < selected[1]['candidate_id']


def test_periodic_close_contacts_are_rejected(tmp_path):
    pool = tmp_path / 'pool'
    pool.mkdir()
    Structure(
        Lattice.cubic(2.0), ['Na', 'Cl'], [[0, 0, 0], [0.1, 0, 0]]
    ).to(filename=str(pool / 'close.cif'), fmt='cif')
    summary = run_fixed_screening(
        FixedScreeningConfig(pool, tmp_path / 'out', top_k=1, device='cpu'),
        model=FakeModel(), scorer=fake_score, relaxer=fake_relax,
    )
    assert summary['valid_candidate_count'] == 0
    assert summary['single_point_calls'] == 0
    assert summary['relaxation_attempts'] == 0
    rows = list(csv.DictReader((tmp_path / 'out/screening.csv').open()))
    assert 'distance' in rows[0]['validity_reason']


def test_bad_budget_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_fixed_screening(FixedScreeningConfig(tmp_path, tmp_path / 'out', top_k=-1))


def test_mattergen_generation_batches_and_seed(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, env, check):
        captured['command'] = command
        captured['env'] = env
        captured['check'] = check

    monkeypatch.setattr(mattergen.subprocess, 'run', fake_run)
    mattergen.generate(tmp_path, num_candidates=64, seed=42, batch_size=4)
    command = captured['command']
    assert '--batch_size=4' in command
    assert '--num_batches=16' in command
    assert '--seed=42' in command
    assert captured['check'] is True
