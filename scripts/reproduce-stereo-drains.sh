#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
src=gen2hole/examples/stereo-drains
models=${DRAIN_MODELS:-runs/stereo-drains}
out=${DRAIN_VIEWS:-outputs/stereo-drains}
py=${DRAIN_PYTHON:-work/envs/streetcrafter/bin/python}
cpu=${DRAIN_CPU_PYTHON:-$py}
base=runs/streetgs-joint/kitti360-0000/30000/stereo-drains
pairs=(P279-A280 P281-M280)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
stage=${1:-help}
default_sites=(kitti avenue audi)
source scripts/paired-pavement.sh
sun_survey=gen2hole/examples/sun-openings/kitti-sun.json
case "$stage" in
assets)
  "$py" gen2hole/survey.py --survey "$src/survey.json" --out "$models/grate"
  for id in P279 P281; do
    "$py" gen2hole/multiview_pits.py --survey "$src/$id.json" --source "${DRAIN_CAPTURES:-work/drain-captures}" --out "$models/$id"
  done ;;
drive) drive_pavement_pairs "$@" ;;
review) review_pavement_pairs --avenue ;;
cycles)
  "${DRAIN_MESH_PYTHON:-work/envs/assets/bin/python}" scripts/render-drain-stereo.py "$models" "$out/cycles" "$src/${pairs[0]}.json" "$src/${pairs[1]}.json" ;;
pack|replay|stereo-inputs|stereo-check)
  runner=$cpu
  if [[ $stage == pack || $stage == stereo-inputs ]]; then runner=$py; fi
  "$runner" scripts/check-drain-stereo.py "$stage" --models "$models" --views "$out" --packets "${DRAIN_EXPOSURES:-work/drain-exposures}" --stereo "${DRAIN_STEREO:-work/drain-stereo}" ;;
joint-build|joint-render|joint-local) native_pavement_pairs ;;
*) echo 'Use assets, drive [kitti|avenue|audi [PAIR]], review, joint-build, joint-render, joint-local, pack, replay, cycles, stereo-inputs or stereo-check' >&2; exit 2 ;;
esac
