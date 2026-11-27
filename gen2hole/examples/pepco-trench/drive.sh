#!/usr/bin/env bash
set -eu
raw=$1
masks=$2
models=$3
out=$4
repo=$(cd "$(dirname "$0")/../../.." && pwd)
kit="$repo/releases/field-chambers"
for state in raised offset open; do
  focus=9
  if [ "$state" = open ]; then focus=5; fi
  for side in front back; do
    python "$kit/scripts/render-road-distress.py" "$raw" "$masks" "$models/$state" "$out/$state/$side" \
      --shot 15 --pixel 1190 925 --yaw-deg 15 --camera "${side}_camera" --semantic-occlusion \
      --focus-part "$focus" --relief-daylight "$repo/gen2hole/examples/gervis-rupture/daylight.json" A591
  done
  python "$repo/gen2hole/examples/lane-damage/check_drive.py" "$models/$state" "$raw" "$out/$state" "$out/$state/check.json" --ids A591
done
