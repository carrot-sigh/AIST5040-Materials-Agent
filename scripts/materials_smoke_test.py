"""Generate two crystals and validate CHGNet prediction and relaxation."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
from chgnet.model.model import CHGNet
from materials_agent.tools.mattergen import generate
from materials_agent.tools.chgnet import predict_and_relax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/smoke'))
    parser.add_argument('--skip-generation', action='store_true')
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_generation:
        generate(args.output)
    cif_dir = args.output / 'cifs'
    cif_dir.mkdir(exist_ok=True)
    with ZipFile(args.output / 'generated_crystals_cif.zip') as archive:
        names = [name for name in archive.namelist() if name.endswith('.cif')]
        assert len(names) == 2, f'Expected 2 CIFs, got {len(names)}'
        for index, name in enumerate(names):
            (cif_dir / f'crystal_{index}.cif').write_bytes(archive.read(name))
    model = CHGNet.load(use_device='cuda')
    reports = [predict_and_relax(path, args.output / f'{path.stem}_relaxed.cif', model, steps=args.steps) for path in sorted(cif_dir.glob('*.cif'))]
    (args.output / 'chgnet_report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
