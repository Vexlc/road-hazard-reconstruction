#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
src=gen2hole/examples/mei-pits
models=${MEI_MODELS:-runs/mei-pits}
out=${MEI_VIEWS:-outputs/mei-pits}
py=${MEI_PYTHON:-work/envs/streetcrafter/bin/python}
cpu=${MEI_CPU_PYTHON:-$py}
base=runs/streetgs-joint/kitti360-0000/30000/mei-pits
pairs=(P272-P273 P274-P275)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
stage=${1:-help}
case "$stage" in
assets)
  for id in P272 P273 P274 P275; do
    "$py" gen2hole/multiview_pits.py --survey "$src/$id.json" --source "${MEI_CAPTURES:-work/mei-captures}" --out "$models/$id"
  done ;;
drive)
  sites=(kitti avenue audi)
  if [[ -n ${2:-} ]]; then sites=("$2"); fi
  if [[ -n ${3:-} ]]; then pairs=("$3"); fi
  for site in "${sites[@]}"; do
    case "$site" in
      kitti) host=(--scene kitti360-0000 --group 26 --offset 1.4 --native-mei) ;;
      avenue) host=(--scene kitti360-0003-right --group 24 --offset .35 --native-mei) ;;
      audi) host=(--scene a2d2-frontleft --group 26 --offset -.5 --ego-mask gen2hole/examples/traffic-cones/audi-body.json) ;;
      *) exit 2 ;;
    esac
    for pair in "${pairs[@]}"; do
      args=("${host[@]}" --run "$models" --layout "$src/$pair.json" --unclipped-preview --export-packet "$pair")
      "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/reference/$pair/$site"
      "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/$pair/$site" --appearance-views "$out/reference/$pair/$site" --video
    done
  done ;;
review)
  "$py" scripts/review-round-covers.py --views "$out" --layouts "$src/P272-P273.json" "$src/P274-P275.json" --rear-hold --avenue
  "$cpu" scripts/review-mixed-inlets.py "$out" "$models" "$src" --pairs "${pairs[@]}" ;;
joint-build|joint-render|joint-local)
  for pair in "${pairs[@]}"; do
    anchor="$src/anchors/$pair-kitti.json"
    case "$stage" in
      joint-build) "$py" streetcrafter/joint/pavement_pair.py "$src/$pair.json" "$models" "$anchor" "$base" --apron-exposure ;;
      joint-render)
        gut=${OFFICIAL_ROOT:-$(cd ../.. && pwd)/01_official}/3dgrut
        gpu=${GUT_PYTHON:-$gut/.venv/bin/python}
        PYTHONPATH="$gut${PYTHONPATH:+:$PYTHONPATH}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PATH="$(dirname "$gpu"):$PATH" "$gpu" streetcrafter/joint/roadside_pairs.py render --save-pavement-inputs --assembly "$pair" --site kitti --out "$base" --renders "$out/joint" ;;
      joint-local) "$py" streetcrafter/joint/local_pavement_replay.py "$src/$pair.json" "$models" "$anchor" "$out/joint" "$out/local/joint" ;;
    esac
  done
  if [[ $stage == joint-render ]]; then "$py" scripts/review-roadside-props.py --views "$out" --pairs "${pairs[@]}" --joint-only; fi
  if [[ $stage == joint-local ]]; then "$py" scripts/review-roadside-props.py --views "$out/local" --pairs "${pairs[@]}" --joint-only; fi ;;
pack)
  "$py" scripts/check-mei-pits.py pack --views "$out" --packets "${MEI_EXPOSURES:-work/mei-exposures}" ;;
control)
  "${MEI_CONTROL_PYTHON:-work/envs/mei-control/bin/python}" scripts/render-mei-control.py --assets "$models" --packets "${MEI_EXPOSURES:-work/mei-exposures}" --out "${MEI_CONTROLS:-work/mei-controls}" --calibrations "${MEI_CALIBRATIONS:-work/data/kitti360-0000-0200/calibration}" ;;
replay|optics)
  packets=${MEI_EXPOSURES:-work/mei-exposures}
  "$cpu" scripts/check-mei-pits.py "$stage" --packets "$packets" --controls "${MEI_CONTROLS:-work/mei-controls}" --assets "$models" --out "$out/$stage" ;;
*) echo 'Use assets, drive [kitti|avenue|audi [PAIR]], review, joint-build, joint-render, joint-local, pack, control, replay or optics' >&2; exit 2 ;;
esac
