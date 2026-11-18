"""CHGNet single-point prediction and structure relaxation helpers."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


def load_chgnet(device: str = 'cuda') -> Any:
    """Load one pretrained CHGNet model onto ``device``."""
    from chgnet.model.model import CHGNet

    return CHGNet.load(use_device=device)


def single_point(model: Any, structure: Any) -> dict[str, Any]:
    """Predict energy per atom and forces without changing or relaxing a structure."""
    start = perf_counter()
    prediction = model.predict_structure(structure)
    energy = float(prediction['e'])
    forces = np.asarray(prediction['f'], dtype=float)
    if forces.shape != (len(structure), 3):
        raise ValueError(f'Expected forces with shape {(len(structure), 3)}, got {forces.shape}')
    force_norms = np.linalg.norm(forces, axis=1)
    max_force = float(force_norms.max())
    if not np.isfinite(energy) or not np.isfinite(forces).all() or not np.isfinite(max_force):
        raise ValueError('CHGNet returned non-finite energy or forces')
    return {
        'energy_per_atom_eV': energy,
        'forces_eV_per_A': forces.tolist(),
        'max_force_eV_per_A': max_force,
        'elapsed_seconds': perf_counter() - start,
    }


def relax_structure(
    model: Any,
    structure: Any,
    output_path: str | Path,
    *,
    device: str = 'cuda',
    fmax: float = 0.05,
    steps: int = 500,
    relax_cell: bool = True,
) -> dict[str, Any]:
    """Relax a structure and report final forces independently of optimizer success."""
    from chgnet.model.dynamics import StructOptimizer

    start = perf_counter()
    optimizer = StructOptimizer(model=model, use_device=device)
    result = optimizer.relax(
        structure, fmax=fmax, steps=steps, relax_cell=relax_cell, verbose=False
    )
    final = result['final_structure']
    final_prediction = model.predict_structure(final)
    final_energy = float(final_prediction['e'])
    final_forces = np.asarray(final_prediction['f'], dtype=float)
    max_force = float(np.linalg.norm(final_forces, axis=1).max())
    total_energy = float(result['trajectory'].energies[-1])
    finite = np.isfinite([final_energy, max_force, total_energy]).all()
    if not finite:
        raise ValueError('CHGNet returned non-finite relaxation results')
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final.to(filename=str(path), fmt='cif')
    return {
        'relaxed_cif': str(path),
        'final_energy_per_atom_eV': final_energy,
        'final_total_energy_eV': total_energy,
        'final_max_force_eV_per_A': max_force,
        'converged': max_force <= fmax,
        'trajectory_frames': len(result['trajectory'].energies),
        'elapsed_seconds': perf_counter() - start,
    }


def predict_and_relax(
    cif_path: str | Path,
    output_path: str | Path,
    model: Any | None = None,
    steps: int = 500,
    fmax: float = 0.05,
    device: str = 'cuda',
) -> dict[str, Any]:
    """Compatibility wrapper used by the original smoke-test pipeline."""
    from pymatgen.core import Structure

    model = model if model is not None else load_chgnet(device)
    structure = Structure.from_file(cif_path)
    initial = single_point(model, structure)
    relaxed = relax_structure(
        model, structure, output_path, device=device, fmax=fmax, steps=steps
    )
    return {
        'input': str(cif_path),
        'formula': structure.composition.reduced_formula,
        'num_atoms': len(structure),
        'initial_energy_eV_per_atom': initial['energy_per_atom_eV'],
        'relaxed_energy_eV_per_atom': relaxed['final_energy_per_atom_eV'],
        'relaxed_total_energy_eV': relaxed['final_total_energy_eV'],
        'max_force_eV_per_A': relaxed['final_max_force_eV_per_A'],
        'force_threshold_met': relaxed['converged'],
        'trajectory_frames': relaxed['trajectory_frames'],
        'fmax': fmax,
        'max_steps': steps,
        'single_point_seconds': initial['elapsed_seconds'],
        'relaxation_seconds': relaxed['elapsed_seconds'],
    }
