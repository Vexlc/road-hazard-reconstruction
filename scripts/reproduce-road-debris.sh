#!/usr/bin/env bash
# Reproduce road-debris-v1.0.0 from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
src=gen2hole/examples/road-debris
models=${DEBRIS_MODELS:-runs/road-debris}
out=${DEBRIS_VIEWS:-outputs/road-debris}
cams=${DEBRIS_CAMERAS:-work/debris-cameras}
base=runs/streetgs-joint/kitti360-0000/30000
py=${DEBRIS_CPU_PYTHON:-work/envs/roadside-cpu/bin/python}
native=${DEBRIS_NATIVE_PYTHON:-work/envs/streetcrafter/bin/python}
pairs=(R242-R243 R244-R245)

case "${1:-help}" in
assets)
  "$py" gen2hole/road_debris.py --out "$models"
  for pair in "${pairs[@]}"; do
    "$py" gen2hole/roadside_layout.py "$models" "$src/$pair.json" "$models/$pair"
  done
  ;;
replay)
  read -ra frame < <(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(*j["seed"], *j["heading"], *j["up"])' "$cams/frame.json")
  for pair in "${pairs[@]}"; do
    cam="$cams/$pair"
    dest="$out/replay/$pair"
    mkdir -p "$dest"
    "$py" scripts/transfer-mesh-view.py road-points "$dest/road.npy" "$cam/front.npz" "$cam/edge.npz" --stride 2
    "$py" scripts/transfer-mesh-view.py fit "$dest/road.npy" "$dest/anchor.json" --seed "${frame[@]:0:3}" --heading "${frame[@]:3:3}" --up "${frame[@]:6:3}" --radius 1.5 --asset "$models/$pair" --lighting "$cam/light.json"
    cmp "$cam/anchor.json" "$dest/anchor.json"
    for view in front edge rear; do
      "$py" scripts/transfer-mesh-view.py render "$models/$pair" "$dest/anchor.json" "$cam/$view.npz" "$dest/$view"
      sums=$(realpath "$cam/$view.sha256")
      (cd "$dest/$view" && sha256sum -c "$sums")
    done
  done
  ;;
joint-build|joint-render)
  for pair in "${pairs[@]}"; do
    site=urban
    if [[ $pair == R244-R245 ]]; then site=sun; fi
    args=(--assets "$models" --views "$src/anchors/$pair" --assembly "$pair" --site "$site" --out "$base/road-debris" --renders "$out/joint")
    if [[ $1 == joint-build ]]; then
      "$native" streetcrafter/joint/roadside_pairs.py build "${args[@]}"
    else
      gut="${OFFICIAL_ROOT:-$(cd ../.. && pwd)/01_official}/3dgrut"
      CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PYTHONPATH="$gut" PATH="$gut/.venv/bin:$PATH" "$gut/.venv/bin/python" streetcrafter/joint/roadside_pairs.py render "${args[@]}"
    fi
  done
  if [[ $1 == joint-render ]]; then
    "$native" scripts/review-roadside-props.py --views "$out" --pairs "${pairs[@]}" --joint-only
  fi
  ;;
physical)
  mkdir -p "$out"
  "$native" scripts/fit-roadside-light.py --out "$out/light.json"
  for pair in "${pairs[@]}"; do
    common=(--surveyed-depth --run "$models" --ids "$pair" --export-packet "$pair" --video)
    "$native" scripts/render-mixed-road.py "${common[@]}" --scene kitti360-0000 --group 26 --offset 1.86 --out "$out/$pair/urban"
    "$native" scripts/render-mixed-road.py "${common[@]}" --scene kitti360-0000 --group 12 --offset 4 --seed-lift .12 --support-radius 1.15 --lighting "$out/light.json" --out "$out/$pair/sun"
    "$native" scripts/render-mixed-road.py "${common[@]}" --scene a2d2-frontleft --group 26 --offset 1.96 --ego-mask gen2hole/examples/traffic-cones/audi-body.json --out "$out/$pair/audi"
  done
  "$native" scripts/check-debris-visibility.py --assets "$models" --views "$out" --out "$out/visibility.json"
  "$native" scripts/review-road-debris.py --views "$out" --plate "$out/physical.jpg"
  ;;
cycles)
  "${DEBRIS_BLENDER_PYTHON:-work/envs/assets/bin/python}" scripts/render-debris-cycles.py --assets "$models" --out "$out/cycles"
  "$py" scripts/check-debris-cycles.py --assets "$models" --views "$out/cycles" --out "$out/cycles-check"
  ;;
*)
  echo 'Usage: bash scripts/reproduce-road-debris.sh {assets|replay|joint-build|joint-render|physical|cycles}' >&2
  exit 2
  ;;
esac
