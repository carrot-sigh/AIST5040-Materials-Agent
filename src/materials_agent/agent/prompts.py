"""Deterministic prompt construction for budget-aware crystal selection."""
from __future__ import annotations

import hashlib
import json

from materials_agent.contracts import State
from materials_agent.structure_features import STRUCTURE_FEATURE_KEYS

SYSTEM_PROMPT = """You are a materials-science decision planner for a fixed inorganic crystal pool.
Choose exactly one next action: relax one available candidate or stop. The runtime, not you,
executes and validates actions. You cannot call tools, run code, or access files.

Reason from chemical composition, supplied geometry/composition descriptors, explicitly
provided initial-force values, prior CHGNet relaxation feedback, and remaining attempts.
Use feedback adaptively: consider what previous outcomes imply for the remaining pool and
state uncertainty. Distinguish structural plausibility from numerical relaxation convergence,
energetic comparisons, and thermodynamic stability. Convergence does not prove stability.
Raw energy per atom must not be compared across unrelated compositions as a universal stability
score. No MatterSim or ground-truth labels are available. Give a concise scientific justification
and avoid claims the supplied evidence cannot support. When a supplied descriptor informs
your choice, name it and report its value and units when applicable. Stop only when further attempts are not
useful under the remaining budget."""

MAX_CANDIDATES = 64
MAX_HISTORY = 8
MAX_STATE_BYTES = 64_000


def serialize_state(state: State) -> str:
    """Serialize only authorized, compact decision information; omit all internal paths."""
    if len(state.candidates) > MAX_CANDIDATES or len(state.history) > MAX_HISTORY:
        raise ValueError("planner context exceeds the 64-candidate / 8-observation experiment limit")
    evaluated = state.evaluated_candidate_ids
    candidates = []
    for candidate in state.candidates:
        if candidate.candidate_id in evaluated:
            continue
        record = {
            "candidate_id": candidate.candidate_id,
            "formula": candidate.formula,
        }
        features = {
            key: candidate.features[key]
            for key in sorted(STRUCTURE_FEATURE_KEYS)
            if key in candidate.features
        }
        if features:
            record["features"] = features
        if candidate.initial_max_force_ev_ang is not None:
            record["initial_max_force_ev_ang"] = candidate.initial_max_force_ev_ang
        candidates.append(record)
    history = [{
        "candidate_id": observation.candidate_id,
        "status": observation.status,
        "final_energy_ev_atom": observation.final_energy_ev_atom,
        "final_max_force_ev_ang": observation.final_max_force_ev_ang,
        "elapsed_seconds": observation.elapsed_seconds,
        "error": "evaluation failed" if observation.status == "failed" else None,
    } for observation in state.history]
    payload = {
        "remaining_relaxations": state.remaining_relaxations,
        "available_candidates": candidates,
        "completed_observations": history,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(serialized.encode("utf-8")) > MAX_STATE_BYTES:
        raise ValueError(f"planner state exceeds {MAX_STATE_BYTES} byte context bound")
    return serialized


def request_fingerprint(instructions: str, state_json: str) -> str:
    digest = hashlib.sha256()
    digest.update(instructions.encode("utf-8"))
    digest.update(b"\0")
    digest.update(state_json.encode("utf-8"))
    return digest.hexdigest()
