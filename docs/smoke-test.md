# GPU and materials smoke test

Validated on 2026-11-18 with Python 3.10.20 in `../.venvs/mattergen`.

- MatterGen source commit: `92423660a8bd70e83679086e88f88596d484dc16` (upstream repository left unmodified).
- Microsoft Hugging Face checkpoint revision: `5244495dd9a979ff71abc7548a0b14b9deb0069a`.
- PyTorch 2.7.1+cu128, PyG 2.7.0, CHGNet 0.4.2, NumPy 1.26.4.
- Both RTX 5090 GPUs (sm_120) passed matrix forward/backward, torch_cluster radius_graph, torch_scatter scatter_add and torch_sparse matmul.
- Generation used GPU 0, seed 42, batch size 2 and the full 1000-step schedule. The first run included loading the official checkpoint and took about 42 seconds for sampling.
- CHGNet used GPU 0 with cell relaxation enabled and a 500-step budget.

| Formula | Atoms | Initial eV/atom | Relaxed eV/atom | Final max force eV/Å |
| --- | ---: | ---: | ---: | ---: |
| InPF6 | 8 | -5.199989 | -5.201487 | 0.023664 |
| Ba3CaRe2WO12 | 19 | -7.961741 | -8.094784 | 0.043631 |

Both final atomic-force maxima meet 0.05 eV/Å. This is an execution smoke test and does not establish thermodynamic stability, SUN scores, or DFT accuracy.

Local artifacts: `results/smoke/gpu_report.json`, `pipeline.log`, `generated_crystals_cif.zip`, `cifs/`, `crystal_*_relaxed.cif`, and `chgnet_report.json`. Artifacts and weights are excluded from Git.

The three `uv pip check` conflicts are the intentional upstream torch/torchvision/torchaudio version overrides. All other installed requirements passed that check.
