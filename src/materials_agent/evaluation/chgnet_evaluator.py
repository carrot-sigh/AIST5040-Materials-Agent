"""Adapt the existing CHGNet wrapper to the shared relaxation interface."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from materials_agent.contracts import Candidate, Observation


@dataclass(frozen=True)
class CHGNetSettings:
    device: str = 'cuda:0'
    fmax: float = 0.05
    max_steps: int = 500
    relax_cell: bool = True


class CHGNetEvaluator:
    def __init__(self, pool_root: Path, output_dir: Path, settings: CHGNetSettings):
        self.pool_root = pool_root.resolve()
        self.output_dir = output_dir
        self.settings = settings
        self._model = None

    def relax(self, candidate: Candidate) -> Observation:
        from pymatgen.core import Structure

        from materials_agent.tools.chgnet import load_chgnet, relax_structure

        source = (self.pool_root / candidate.cif_path).resolve()
        if not source.is_relative_to(self.pool_root):
            raise ValueError('candidate CIF resolves outside the pool')
        if candidate.cif_sha256 and sha256(source.read_bytes()).hexdigest() != candidate.cif_sha256:
            raise ValueError('candidate CIF does not match the manifest hash')
        structure = Structure.from_file(source)
        if self._model is None:
            self._model = load_chgnet(self.settings.device)
        result = relax_structure(
            self._model, structure, self.output_dir / 'relaxed' / f'{candidate.candidate_id}.cif',
            device=self.settings.device, fmax=self.settings.fmax, steps=self.settings.max_steps,
            relax_cell=self.settings.relax_cell,
        )
        return Observation(
            candidate.candidate_id, 'converged' if result['converged'] else 'not_converged',
            result['final_energy_per_atom_eV'], result['final_max_force_eV_per_A'],
            result['elapsed_seconds'],
        )
