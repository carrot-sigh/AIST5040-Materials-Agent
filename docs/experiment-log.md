# Experiment Log

## Fixed Screening v0: MatterGen 64 to CHGNet Top-8

- Date: 2026-11-18
- Candidate generator: Microsoft MatterGen `mattergen_base`
- Seed: 42
- Generation request: 64 candidates, batch size 4
- Actual generation invocation: batch size 64, 1 batch (wrapper argument propagation bug; fixed afterward)
- Ranking: ascending initial CHGNet maximum atomic-force norm, candidate ID tie-break
- Selection budget: 8
- Relaxation: CHGNet 0.4.2, `cuda:0`, cell relaxation enabled, `fmax=0.05 eV/Å`, maximum 500 steps
- Hardware/software: NVIDIA RTX 5090, Python 3.10.20, PyTorch 2.7.1+cu128, pymatgen 2024.10.29
- MatterGen source revision: `92423660a8bd70e83679086e88f88596d484dc16`
- Hugging Face checkpoint revision: `5244495dd9a979ff71abc7548a0b14b9deb0069a`

### Budget and outcome

| Measure | Result |
| --- | ---: |
| Generated candidates | 64 |
| Valid candidates | 64 |
| Unique valid candidates | 64 |
| Duplicate candidates | 0 |
| CHGNet single-point calls | 64 |
| Single-point failures | 0 |
| Selected candidates | 8 |
| Relaxation attempts | 8 |
| Relaxation successes | 8 |
| Relaxation failures | 0 |
| Final force-converged candidates | 8 |
| Total wall time | 159.96 s |
| Sum of recorded single-point call times | 1.27 s |
| Sum of recorded relaxation call times | 1.97 s |

### Selected structures

| Rank | Candidate ID | Formula | Initial max force (eV/Å) | Final max force (eV/Å) |
| ---: | --- | --- | ---: | ---: |
| 1 | `gen_56-8428c1213ba7` | ZrNbOsRu | 0.003483 | 0.002678 |
| 2 | `gen_12-569f4de39f2f` | BiPdPt2 | 0.004385 | 0.002927 |
| 3 | `gen_55-306289fa612e` | CaPrCu2 | 0.006886 | 0.003843 |
| 4 | `gen_23-b777dac96b3b` | SrCdHgPd | 0.008427 | 0.004189 |
| 5 | `gen_13-ecc73f7c086f` | TbAs | 0.009958 | 0.003281 |
| 6 | `gen_29-a5b3033d048e` | DyCdHgPd | 0.010848 | 0.004050 |
| 7 | `gen_39-1ae3991ace71` | Na2LiSbTe4 | 0.024745 | 0.038041 |
| 8 | `gen_37-3721544830e6` | Ho3AgN | 0.024987 | 0.024987 |

All eight final atomic-force maxima are at or below 0.05 eV/Å. This does not establish thermodynamic stability, novelty, or DFT accuracy. The ranking score is the initial force heuristic and must not be interpreted as energy above hull.

### Artifacts

Local, Git-ignored artifacts are in `results/fixed_v0/`: generated CIF archive and extxyz, the candidate CIF pool, `screening.csv`, `selected_candidates.json`, `summary.json`, `run_config.json`, and eight relaxed CIF files. The run config records the requested batch size and the actual invocation separately. Large CIF outputs are intentionally not committed.
