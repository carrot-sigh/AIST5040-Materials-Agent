"""Run MatterGen in its installed environment."""
import os
from pathlib import Path
import subprocess
import sys


def generate(output_dir: str | Path, num_candidates: int = 2, seed: int = 42) -> None:
    if num_candidates < 1:
        raise ValueError('num_candidates must be positive')
    env = os.environ.copy()
    # Official Microsoft checkpoints contain configuration objects. PyTorch >=2.6
    # changed torch.load's default; scope legacy loading to this trusted subprocess.
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    subprocess.run([
        sys.executable, '-m', 'mattergen.scripts.generate', str(Path(output_dir).resolve()),
        '--pretrained_name=mattergen_base', f'--batch_size={num_candidates}',
        '--num_batches=1', '--record_trajectories=False', f'--seed={seed}',
    ], env=env, check=True)
