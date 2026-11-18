# AIST5040 Materials Agent

MatterGen candidate generation and CHGNet prediction/relaxation for AIST5040.

## Layout

Keep the two repositories independent:

```text
5040/
├── AIST5040-Materials-Agent/
├── mattergen/                    # https://github.com/microsoft/mattergen
└── .venvs/mattergen/             # isolated Python 3.10 environment
```

`src/materials_agent` contains tools, agent, baselines and evaluation packages.
`configs/default.json` contains initial experiment settings; agent and baseline modules are placeholders.

## RTX 5090 setup

From this repository, with `uv` available:

```bash
bash scripts/setup_mattergen.sh
source ../.venvs/mattergen/bin/activate
python scripts/gpu_smoke_test.py
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python scripts/materials_smoke_test.py
```

The setup installs into `../.venvs/mattergen`, without modifying Conda base.
`configs/blackwell-overrides.txt` deliberately replaces MatterGen's Linux torch 2.2.1 pins with torch 2.7.1/cu128 and matching PyG wheels. `uv pip check` reports the three original torch/torchvision/torchaudio pin conflicts; these overrides are required for this GPU. Do not reinstall upstream dependencies without the overrides.

- PyTorch Blackwell support: https://pytorch.org/blog/pytorch-2-7/
- PyG installation matrix: https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html
- CHGNet: https://chgnet.lbl.gov/

`configs/mattergen-environment.freeze.txt` records the installed versions. Editable paths are relative to this repository; use the setup script to apply the required package indexes and overrides.

## Smoke pipeline

The pipeline generates two crystals with `mattergen_base`, seed 42, and the default full diffusion schedule. Microsoft weights are downloaded automatically from Hugging Face. The MatterGen subprocess sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to load the official checkpoint's configuration objects with PyTorch >=2.6; use this wrapper only with trusted official checkpoints.

Generated CIFs, relaxed CIFs and `chgnet_report.json` are written under `results/smoke/` (ignored by Git). CHGNet runs on CUDA and relaxes positions and cell, with a maximum of 500 steps and force threshold 0.05 eV/Å. The report records whether the final atomic forces meet that threshold; this alone does not establish cell/stress convergence or thermodynamic stability. Prediction energies are eV/atom; relaxation trajectory energies are total eV.

To rerun only CHGNet on the existing generated archive:

```bash
python scripts/materials_smoke_test.py --skip-generation
```

This validates generation and ML force-field execution. SUN metrics, reference-data comparisons, DFT validation, and the agent/baselines are future work. Model weights, environment files, data and results are excluded from commits.
