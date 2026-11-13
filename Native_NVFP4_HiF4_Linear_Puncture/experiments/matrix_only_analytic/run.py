"""CPU-only, real layer-0 q_proj diagnostic for H-only analytic optimization."""
import argparse
from pathlib import Path

from .diagnostic import run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new, independent output directory")
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
