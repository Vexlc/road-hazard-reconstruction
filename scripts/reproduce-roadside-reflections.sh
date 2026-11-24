#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
src=gen2hole/examples/roadside-reflections
models=${REFLECTION_MODELS:-runs/roadside-reflections}
out=${REFLECTION_VIEWS:-outputs/roadside-reflections}
py=${REFLECTION_PYTHON:-work/envs/streetcrafter/bin/python}
cpu=${REFLECTION_CPU_PYTHON:-$py}
mesh=${REFLECTION_MESH_PYTHON:-work/envs/assets/bin/python}
pair=P256-P258
base=runs/streetgs-joint/kitti360-0000/30000/roadside-reflections
stage=${1:-help}
shift || true
case "$stage" in
beds)
  for id in P256 P257; do
    "$py" gen2hole/multiview_pits.py --survey "$src/$id.json" --source "${REFLECTION_CAPTURES:-work/reflector-captures}" --out "$models/beds/$id"
  done
  ;;
survey|seat|drive)
  if [[ $# -eq 0 ]]; then set -- urban avenue audi; fi
  for site in "$@"; do
    stereo=(--stereo-reflections)
    case "$site" in
      urban) host=(--scene kitti360-0000 --group 26 --offset 1.4) ;;
      avenue) host=(--scene kitti360-0003-right --group 24 --offset .35) ;;
      audi) host=(--scene a2d2-frontleft --group 26 --offset -.5 --ego-mask gen2hole/examples/traffic-cones/audi-body.json); stereo=() ;;
      *) echo "Expected urban, avenue or audi, got $site" >&2; exit 2 ;;
    esac
    case "$stage" in
      survey)
        "$py" scripts/render-mixed-road.py "${host[@]}" --survey-only --run "$models/beds" --layout "$src/dry.json" --out "$out/survey/$site" ;;
      seat)
        anchor="$src/anchors/$site.json"
        if [[ -n ${REFLECTION_SURVEYS:-} ]]; then anchor="$REFLECTION_SURVEYS/$site/anchor.json"; fi
        "$py" gen2hole/seat_water.py "$src/dry.json" "$anchor" "$src/pair.json" "$models/beds" "$models/$site" ;;
      drive)
        args=("${host[@]}" --run "$models/$site" --layout "$src/pair.json" --paired-preview --export-packet "$pair")
        "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/reference/$site"
        "$py" scripts/render-mixed-road.py "${args[@]}" --metric-reflections "${stereo[@]}" --appearance-views "$out/reference/$site" --out "$out/$site" --video ;;
    esac
  done
  ;;
review) "$py" scripts/review-roadside-reflections.py "$out" "$models" ;;
joint-build) "$py" streetcrafter/joint/pavement_pair.py "$src/pair.json" "$models/urban" "$models/urban/anchor.json" "$base" --apron-exposure ;;
joint-render)
  gut="${OFFICIAL_ROOT:-$(cd ../.. && pwd)/01_official}/3dgrut"
  gpu=${GUT_PYTHON:-$gut/.venv/bin/python}
  PYTHONPATH="$gut${PYTHONPATH:+:$PYTHONPATH}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PATH="$(dirname "$gpu"):$PATH" "$gpu" streetcrafter/joint/roadside_pairs.py render --save-pavement-inputs --assembly "$pair" --site kitti --out "$base" --renders "$out/joint"
  "$py" scripts/review-roadside-props.py --views "$out" --pairs "$pair" --joint-only ;;
joint-local)
  "$py" streetcrafter/joint/local_pavement_replay.py "$src/pair.json" "$models/urban" "$models/urban/anchor.json" "$out/joint" "$out/local/joint"
  "$py" scripts/review-roadside-props.py --views "$out/local" --pairs "$pair" --joint-only ;;
cycles)
  "$mesh" scripts/render-roadside-reflection-control.py "$out/control"
  "$cpu" scripts/check-roadside-reflections.py "$out/cameras" "$out/optics" --from-cycles "$out/control" ;;
optics) "$cpu" scripts/check-roadside-reflections.py "${REFLECTION_CAMERAS:-work/reflector-control}" "$out/optics" ;;
replay)
  packet=${REFLECTION_EXPOSURE:-work/reflector-exposure}
  "$cpu" scripts/transfer-mesh-view.py render "$src/pair.json" "$packet/appearance.json" "$packet/urban-front.npz" "$out/replay" --asset-root "$models/urban"
  sums="$(realpath "$packet/products.sha256")"
  (cd "$out/replay" && sha256sum -c "$sums")
  "$cpu" scripts/check-reflected-bank.py "$models/urban" "$packet" "$out/replay" ;;
*) echo 'Stages: beds, survey, seat, drive [urban|avenue|audi ...], review, joint-build, joint-render, joint-local, cycles, optics, replay' >&2; exit 2 ;;
esac
