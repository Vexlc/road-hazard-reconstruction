#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
src=gen2hole/examples/cube-pits
models=${CUBE_MODELS:-runs/cube-pits}
out=${CUBE_VIEWS:-outputs/cube-pits}
py=${CUBE_PYTHON:-work/envs/streetcrafter/bin/python}
cpu=${CUBE_CPU_PYTHON:-$py}
base=runs/streetgs-joint/kitti360-0000/30000/cube-pits
pairs=(P266-P267 P268-P269)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
stage=${1:-help}
case "$stage" in
assets)
  for id in P266 P267 P268 P269; do
    "$py" gen2hole/multiview_pits.py --survey "$src/$id.json" --source "${CUBE_CAPTURES:-work/cube-captures}" --out "$models/$id"
  done ;;
drive)
  sites=(kitti audi)
  if [[ -n ${2:-} ]]; then sites=("$2"); fi
  if [[ -n ${3:-} ]]; then pairs=("$3"); fi
  for site in "${sites[@]}"; do
    case "$site" in
      kitti) host=(--scene kitti360-0000 --offset 1.4) ;;
      audi) host=(--scene a2d2-frontleft --offset -.5 --ego-mask gen2hole/examples/traffic-cones/audi-body.json) ;;
      *) exit 2 ;;
    esac
    for pair in "${pairs[@]}"; do
      args=("${host[@]}" --group 26 --run "$models" --layout "$src/$pair.json" --unclipped-preview --export-packet "$pair")
      "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/reference/$pair/$site"
      "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/$pair/$site" --appearance-views "$out/reference/$pair/$site" --video
    done
  done ;;
review)
  "$py" scripts/review-round-covers.py --views "$out" --layouts "$src/P266-P267.json" "$src/P268-P269.json" --rear-hold
  "$cpu" scripts/review-mixed-inlets.py "$out" "$models" "$src" --pairs "${pairs[@]}" ;;
gl|optics)
  layouts=("$src/${pairs[0]}.json" "$src/${pairs[1]}.json")
  cameras=${CUBE_CAMERAS:-work/cube-cameras}
  if [[ $stage == gl ]]; then
    "${CUBE_GL_PYTHON:-work/envs/cubegl/bin/python}" scripts/render-cube-pits.py "$models" "$cameras" "${layouts[@]}"
  else
    "$cpu" scripts/check-cube-pits.py "$models" "$cameras" "$out/optics" "${layouts[@]}"
  fi ;;
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
replay)
  packets=${CUBE_EXPOSURES:-work/cube-exposures}
  for pair in "${pairs[@]}"; do
    for sensor in image_00 image_03 front_left rear_center; do
      site=kitti
      if [[ $sensor == front_left || $sensor == rear_center ]]; then site=audi; fi
      folder="$packets/$pair/$site"
      target="$out/replay/$pair/$sensor"
      "$cpu" scripts/transfer-mesh-view.py render "$src/$pair.json" "$folder/appearance.json" "$folder/$sensor/view.npz" "$target" --asset-root "$models"
      sums=$(realpath "$folder/$sensor/products.sha256")
      (cd "$target" && sha256sum -c "$sums")
    done
  done ;;
*) echo 'Use assets, drive [kitti|audi [PAIR]], review, joint-build, joint-render, joint-local replay, gl or optics' >&2; exit 2 ;;
esac
