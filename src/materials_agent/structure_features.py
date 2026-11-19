"""Low-cost, deterministic descriptors computed from an input crystal structure."""
from __future__ import annotations

from typing import Any

STRUCTURE_FEATURES_VERSION = "geometry-v1"
STRUCTURE_FEATURE_KEYS = frozenset({
    "structure_feature_version",
    "num_atoms",
    "num_elements",
    "volume_angstrom3",
    "density_g_cm3",
    "lattice_parameters",
    "element_fractions",
})


def extract_structure_features(structure: Any) -> dict[str, Any]:
    """Extract geometry and composition descriptors without model inference.

    Requires a pymatgen-compatible Structure. CIF parsing and these operations are
    CPU-only; the returned mapping is JSON-compatible for Candidate.features.
    """
    composition = structure.composition.element_composition
    amounts = composition.get_el_amt_dict()
    total = float(sum(amounts.values()))
    lattice = structure.lattice
    return {
        "structure_feature_version": STRUCTURE_FEATURES_VERSION,
        "num_atoms": int(len(structure)),
        "num_elements": int(len(amounts)),
        "volume_angstrom3": float(lattice.volume),
        "density_g_cm3": float(structure.density),
        "lattice_parameters": {
            "a_angstrom": float(lattice.a),
            "b_angstrom": float(lattice.b),
            "c_angstrom": float(lattice.c),
            "alpha_degree": float(lattice.alpha),
            "beta_degree": float(lattice.beta),
            "gamma_degree": float(lattice.gamma),
        },
        "element_fractions": {
            symbol: float(amount / total) for symbol, amount in sorted(amounts.items())
        },
    }
