"""Replay calibrated road shots and validate triangulated depth against independent ground truth."""

import argparse
from importlib import import_module
from pathlib import Path

cameras = import_module("check-mei-pits")
ROOT = cameras.ROOT
EXAMPLES = ROOT / "gen2hole/examples/stereo-drains"
PAIRS = ("P279-A280", "P281-M280")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("pack", "replay", "stereo-inputs", "stereo-check")
    )
    for key in ("models", "views", "packets", "stereo"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    if args.stage == "pack":
        cameras.pack(args.packets, args.views, PAIRS)
    elif args.stage == "replay":
        cameras.replay(
            args.packets, args.models, args.views / "replay", EXAMPLES, PAIRS
        )
    elif args.stage == "stereo-inputs":
        from drain_stereo_inputs import prepare

        prepare(args.views, args.stereo, PAIRS)
    else:
        from drain_stereo_checks import check

        check(args.stereo, args.views / "stereo", args.models, EXAMPLES, PAIRS)


if __name__ == "__main__":
    main()
