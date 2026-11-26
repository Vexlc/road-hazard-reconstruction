"""Reconstruct the independent PointPSSN surveys selected for the field-rims release."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lane-damage/points'))
from build import build

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('archive', 'materials', 'out'):
        parser.add_argument(argument, type=Path)
    build(**vars(parser.parse_args()), survey=HERE/'scans.json')
