#!/usr/bin/env bash
# Run from a checkout of lost-cargo-v1.0.0.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
src=gen2hole/examples/lost-cargo
models=${CARGO_MODELS:-runs/lost-cargo/assets}
out=${CARGO_VIEWS:-outputs/lost-cargo}
py=${CARGO_CPU_PYTHON:-work/envs/roadside-cpu/bin/python}
native=${CARGO_NATIVE_PYTHON:-work/envs/streetcrafter/bin/python}
base=runs/streetgs-joint/kitti360-0000/30000
pairs=(R262-R263 R264-R265)
case "${1:-help}" in
assets)
  "$py" gen2hole/road_debris.py --recipe "$src/sources.json" --out "$models"
  for pair in "${pairs[@]}"; do
    "$py" gen2hole/roadside_layout.py "$models" "$src/$pair.json" "$models/$pair"
  done
  ;;
physical)
  mkdir -p "$out"
  "$native" scripts/fit-roadside-light.py --out "$out/light.json"
  for pair in "${pairs[@]}"; do
    for site in urban sun audi; do
      opts=(--scene kitti360-0000 --group 26 --offset 1.86)
      if [[ $site == sun ]]; then opts=(--scene kitti360-0000 --group 12 --offset 4 --seed-lift .12 --support-radius 1.15 --lighting "$out/light.json"); fi
      if [[ $site == audi ]]; then opts=(--scene a2d2-frontleft --group 26 --offset 1.96 --ego-mask gen2hole/examples/traffic-cones/audi-body.json); fi
      "$native" scripts/render-mixed-road.py --cargo-pair --unclipped-preview --surveyed-depth --run "$models" --ids "$pair" --export-packet "$pair" --video --out "$out/$pair/$site" "${opts[@]}"
    done
  done
  "$native" scripts/review-road-debris.py --views "$out" --plate "$out/physical.jpg" --layouts "$src/R262-R263.json" "$src/R264-R265.json"
  ;;
joint-build|joint-render)
  for pair in "${pairs[@]}"; do
    site=urban
    if [[ $pair == R264-R265 ]]; then site=sun; fi
    args=(--assets "$models" --views "${CARGO_ANCHORS:-$src/anchors}/$pair" --site "$site" --assembly "$pair" --out "$base/lost-cargo" --renders "$out/joint")
    if [[ $1 == joint-build ]]; then
      "$native" streetcrafter/joint/roadside_pairs.py build "${args[@]}"
    else
      gut="${OFFICIAL_ROOT:-$(cd ../.. && pwd)/01_official}/3dgrut"
      CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PATH="$gut/.venv/bin:$PATH" PYTHONPATH="$gut" "$gut/.venv/bin/python" streetcrafter/joint/roadside_pairs.py render "${args[@]}"
    fi
  done
  if [[ $1 == joint-render ]]; then
    "$native" scripts/review-roadside-props.py --views "$out" --pairs "${pairs[@]}" --joint-only
  fi
  ;;
cycles)
  "${CARGO_BLENDER_PYTHON:-work/envs/assets/bin/python}" scripts/check-cargo-source.py "$models" "$src/sources.json" "$out/source-check.json"
  "${CARGO_BLENDER_PYTHON:-work/envs/assets/bin/python}" scripts/render-debris-cycles.py --size 896 --assets "$models" --detail --pairs "${pairs[@]}" --out "$out/cycles"
  "$py" scripts/check-debris-cycles.py --assets "$models" --detail --pairs "${pairs[@]}" --views "$out/cycles" --out "$out/cycles-check"
  ;;
replay)
  "$py" scripts/check-cargo-replay.py "$models" "${CARGO_PACKETS:-work/cargo-packets}" "$out/replay"
  ;;
*) echo 'Use assets, physical, cycles, replay, joint-build or joint-render' >&2; exit 2 ;;
esac
