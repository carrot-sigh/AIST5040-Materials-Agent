"""Predict and relax CIF structures with CHGNet on CUDA."""
from pathlib import Path
import numpy as np
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure


def predict_and_relax(cif_path: str | Path, output_path: str | Path, model: CHGNet, steps: int = 500, fmax: float = 0.05) -> dict:
    structure = Structure.from_file(cif_path)
    prediction = model.predict_structure(structure)
    initial_energy = float(prediction['e'])
    if not np.isfinite(initial_energy):
        raise ValueError('Non-finite CHGNet energy')
    optimizer = StructOptimizer(model=model, use_device='cuda')
    result = optimizer.relax(structure, fmax=fmax, steps=steps, verbose=False)
    final = result['final_structure']
    final_prediction = model.predict_structure(final)
    max_force = float(np.linalg.norm(final_prediction['f'], axis=1).max())
    final_energy = float(final_prediction['e'])
    total_energy = float(result['trajectory'].energies[-1])
    if not np.isfinite([final_energy, total_energy, max_force]).all():
        raise ValueError('Non-finite CHGNet relaxation result')
    final.to(filename=str(output_path), fmt='cif')
    return {
        'input': str(cif_path), 'relaxed_cif': str(output_path),
        'formula': structure.composition.reduced_formula, 'num_atoms': len(structure),
        'initial_energy_eV_per_atom': initial_energy,
        'relaxed_energy_eV_per_atom': final_energy,
        'relaxed_total_energy_eV': total_energy,
        'max_force_eV_per_A': max_force,
        'force_threshold_met': max_force <= fmax,
        'trajectory_frames': len(result['trajectory'].energies),
        'fmax': fmax, 'max_steps': steps,
    }
