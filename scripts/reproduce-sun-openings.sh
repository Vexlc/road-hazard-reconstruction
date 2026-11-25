#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
src=gen2hole/examples/sun-openings
models=${OPENING_MODELS:-runs/sun-openings}
out=${OPENING_VIEWS:-outputs/sun-openings}
py=${OPENING_PYTHON:-work/envs/streetcrafter/bin/python}
cpu=${OPENING_CPU_PYTHON:-$py}
base=runs/streetgs-joint/kitti360-0000/30000/sun-openings
pairs=(P276-A277 P278-M277)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
stage=${1:-help}
case "$stage" in
assets)
  "$py" gen2hole/survey.py --survey "$src/survey.json" --out "$models/iron"
  for id in P276 P278; do
    "$py" gen2hole/multiview_pits.py --survey "$src/$id.json" --source "${OPENING_CAPTURES:-work/opening-captures}" --out "$models/$id"
  done ;;
drive)
  sites=(kitti audi)
  if [[ -n ${2:-} ]]; then sites=("$2"); fi
  if [[ -n ${3:-} ]]; then pairs=("$3"); fi
  for site in "${sites[@]}"; do
    case "$site" in
      kitti) host=(--scene kitti360-0000 --offset 1.4 --native-mei); light=(--pavement-sun "$src/kitti-sun.json");;
      audi) host=(--scene a2d2-frontleft --offset -.5 --ego-mask gen2hole/examples/traffic-cones/audi-body.json); light=();;
      *) exit 2 ;;
    esac
    for pair in "${pairs[@]}"; do
      args=("${host[@]}" --group 26 --run "$models" --layout "$src/$pair.json" --unclipped-preview --export-packet "$pair")
      "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/reference/$pair/$site"
      "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/$pair/$site" --appearance-views "$out/reference/$pair/$site" "${light[@]}" --video
    done
  done ;;
review)
  "$py" scripts/review-round-covers.py --views "$out" --layouts "$src/P276-A277.json" "$src/P278-M277.json" --rear-hold ;;
pack|replay|oracle|control)
  runner=$cpu
  if [[ $stage == pack ]]; then runner=$py; fi
  if [[ $stage == oracle ]]; then runner=${OPENING_MESH_PYTHON:-work/envs/assets/bin/python}; fi
  "$runner" scripts/check-opening-sun.py "$stage" --models "$models" --views "$out" --packets "${OPENING_EXPOSURES:-work/opening-exposures}" --controls "${OPENING_CONTROLS:-work/opening-controls}" ;;
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
*) echo 'Use assets, drive [kitti|audi [PAIR]], review, joint-build, joint-render, joint-local, pack, replay, oracle or control' >&2; exit 2 ;;
esac
