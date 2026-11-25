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
default_sites=(kitti audi)
source scripts/paired-pavement.sh
sun_survey=$src/kitti-sun.json
case "$stage" in
assets)
  "$py" gen2hole/survey.py --survey "$src/survey.json" --out "$models/iron"
  for id in P276 P278; do
    "$py" gen2hole/multiview_pits.py --survey "$src/$id.json" --source "${OPENING_CAPTURES:-work/opening-captures}" --out "$models/$id"
  done ;;
drive) drive_pavement_pairs "$@" ;;
review) review_pavement_pairs ;;
pack|replay|oracle|control)
  runner=$cpu
  if [[ $stage == pack ]]; then runner=$py; fi
  if [[ $stage == oracle ]]; then runner=${OPENING_MESH_PYTHON:-work/envs/assets/bin/python}; fi
  "$runner" scripts/check-opening-sun.py "$stage" --models "$models" --views "$out" --packets "${OPENING_EXPOSURES:-work/opening-exposures}" --controls "${OPENING_CONTROLS:-work/opening-controls}" ;;
joint-build|joint-render|joint-local) native_pavement_pairs ;;
*) echo 'Use assets, drive [kitti|audi [PAIR]], review, joint-build, joint-render, joint-local, pack, replay, oracle or control' >&2; exit 2 ;;
esac
