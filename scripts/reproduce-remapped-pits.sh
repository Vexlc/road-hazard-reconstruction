#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
src=gen2hole/examples/remapped-pits
models=${PIT_MODELS:-runs/remapped-pits}
out=${PIT_VIEWS:-outputs/remapped-pits}
py=${PIT_PYTHON:-work/envs/streetcrafter/bin/python}
base=runs/streetgs-joint/kitti360-0000/30000/remapped-pits
pair=P250-P251
case "${1:-help}" in
assets)
  for key in P250 P251; do
    "$py" gen2hole/multiview_pits.py --survey "$src/$key.json" --source "${PIT_CAPTURES:-work/remapped-captures}" --out "$models/$key"
  done
  ;;
drive)
  shift
  if [[ $# -eq 0 ]]; then set -- kitti audi; fi
  for site in "$@"; do
    case "$site" in
      kitti) args=(--scene kitti360-0000 --offset 1.4) ;;
      audi) args=(--scene a2d2-frontleft --offset -.5 --ego-mask gen2hole/examples/traffic-cones/audi-body.json) ;;
      *) echo "Unknown site: $site" >&2; exit 2 ;;
    esac
    args+=(--run "$models" --layout "$src/layout.json" --group 26 --paired-preview --export-packet "$pair")
    "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/reference/$site"
    "$py" scripts/render-mixed-road.py "${args[@]}" --appearance-views "$out/reference/$site" --video --out "$out/$site"
    if [[ $site == audi ]]; then
      "$py" scripts/review-asphalt-breakouts.py --layout "$src/layout.json" --views "$out"
      "$py" scripts/check-remapped-pit-drive.py "$out" "$models"
    fi
  done
  ;;
joint-build)
  "$py" streetcrafter/joint/pavement_pair.py "$src/layout.json" "$models" "$src/kitti.json" "$base" --apron-exposure
  ;;
joint-render)
  gut="${OFFICIAL_ROOT:-$(cd ../.. && pwd)/01_official}/3dgrut"
  gpu=${GUT_PYTHON:-$gut/.venv/bin/python}
  PYTHONPATH="$gut${PYTHONPATH:+:$PYTHONPATH}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PATH="$(dirname "$gpu"):$PATH" "$gpu" streetcrafter/joint/roadside_pairs.py render --save-pavement-inputs --assembly "$pair" --site kitti --out "$base" --renders "$out/joint"
  "$py" scripts/review-roadside-props.py --views "$out" --pairs "$pair" --joint-only
  ;;
joint-local)
  "$py" streetcrafter/joint/local_pavement_replay.py "$src/layout.json" "$models" "$src/kitti.json" "$out/joint" "$out/local/joint"
  "$py" scripts/review-roadside-props.py --views "$out/local" --pairs "$pair" --joint-only
  ;;
optics)
  "${PIT_CPU_PYTHON:-$py}" scripts/check-camera-remap.py "${PIT_ORACLE:-work/remap-oracle}" "${COVER_MODELS:-work/remap-covers}" "$out/optics"
  ;;
*) echo 'Usage: reproduce-remapped-pits.sh {assets|drive [kitti|audi ...]|joint-build|joint-render|joint-local|optics}' >&2; exit 2 ;;
esac
