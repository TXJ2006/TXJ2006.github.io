"""Compile the checked-in TikZ sources, then export vector SVGs and PNG previews."""
from pathlib import Path
import argparse
import subprocess
import fitz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Build directory outside the source tree')
    parser.add_argument('--assets', type=Path, help='Optional destination for the website SVG files')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.assets:
        args.assets.mkdir(parents=True, exist_ok=True)
    for source in sorted(Path(__file__).resolve().parent.glob('tikz_*.tex')):
        result = subprocess.run(['tectonic', '--outdir', str(args.output.resolve()), str(source)], capture_output=True, text=True)
        (args.output / (source.stem + '.build.log')).write_text(result.stdout + result.stderr)
        result.check_returncode()
        with fitz.open(args.output / (source.stem + '.pdf')) as doc:
            page = doc[0]
            svg = page.get_svg_image(text_as_path=True)
            (args.output / (source.stem + '.svg')).write_text(svg)
            page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False).save(args.output / (source.stem + '.png'))
            if args.assets:
                (args.assets / (source.stem + '.svg')).write_text(svg)
        print(source.stem)


if __name__ == '__main__':
    main()
