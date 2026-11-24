#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
src=gen2hole/examples/rolling-covers
models=${COVER_MODELS:-runs/rolling-covers}
out=${COVER_VIEWS:-outputs/rolling-covers}
py=${COVER_PYTHON:-work/envs/streetcrafter/bin/python}
cpu=${COVER_CPU_PYTHON:-$py}
base=runs/streetgs-joint/kitti360-0000/30000/rolling-covers
pairs=(A248-M249 M248-A249)
layouts=("$src/A248-M249.json" "$src/M248-A249.json")
case "${1:-help}" in
assets)
  for scan in 003 006; do
    "$py" gen2hole/survey.py --survey "$src/$scan/survey.json" --out "$models/$scan"
  done
  ;;
drive)
  for pair in "${pairs[@]}"; do
    for site in kitti audi; do
      args=(--scene kitti360-0000 --offset 1.4)
      if [[ $site == audi ]]; then args=(--scene a2d2-frontleft --offset -.5 --ego-mask gen2hole/examples/traffic-cones/audi-body.json); fi
      args+=(--run "$models" --layout "$src/$pair.json" --group 26 --paired-preview --export-packet "$pair")
      "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/reference/$pair/$site"
      "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/$pair/$site" --appearance-views "$out/reference/$pair/$site" --video
    done
  done
  "$py" scripts/review-round-covers.py --views "$out" --rear-hold --layouts "${layouts[@]}"
  ;;
joint-build)
  for pair in "${pairs[@]}"; do
    "$py" streetcrafter/joint/pavement_pair.py "$src/$pair.json" "$models" "$src/anchors/$pair-kitti.json" "$base" --apron-exposure
  done
  ;;
joint-render)
  gut="${OFFICIAL_ROOT:-$(cd ../.. && pwd)/01_official}/3dgrut"
  for pair in "${pairs[@]}"; do
    PYTHONPATH="$gut" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PATH="$gut/.venv/bin:$PATH" "$gut/.venv/bin/python" streetcrafter/joint/roadside_pairs.py render --save-pavement-inputs --assembly "$pair" --site kitti --out "$base" --renders "$out/joint"
  done
  "$py" scripts/review-roadside-props.py --views "$out" --pairs "${pairs[@]}" --joint-only
  ;;
joint-local)
  for pair in "${pairs[@]}"; do
    "$py" streetcrafter/joint/local_pavement_replay.py "$src/$pair.json" "$models" "$src/anchors/$pair-kitti.json" "$out/joint" "$out/local/joint"
  done
  "$py" scripts/review-roadside-props.py --views "$out/local" --pairs "${pairs[@]}" --joint-only
  ;;
cycles)
  flags=(--assets "$models" --layout "${layouts[0]}" --layout "${layouts[1]}")
  "${COVER_BLENDER_PYTHON:-work/envs/assets/bin/python}" scripts/render-round-cover-cycles.py "${flags[@]}" --readout --out "$out/cycles"
  "$cpu" scripts/check-round-cover-cycles.py "${flags[@]}" --views "$out/cycles" --out "$out/oracle"
  "$cpu" scripts/check-rolling-covers.py --assets "$models" --views "$out"
  ;;
replay)
  cam=${COVER_CAMERAS:-work/rolling-cameras}
  read -ra frame < <(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(*j["seed"],*j["heading"],*j["up"])' "$cam/frame.json")
  cli=("$cpu" scripts/transfer-mesh-view.py)
  for pair in "${pairs[@]}"; do
    dest="$out/replay/$pair"
    ref="$cam/$pair"
    "${cli[@]}" road-points "$dest/road.npy" "$ref/ortho.npz" --stride 3
    "${cli[@]}" fit "$dest/road.npy" "$dest/anchor.json" --seed "${frame[@]:0:3}" --heading "${frame[@]:3:3}" --up "${frame[@]:6:3}" --radius 1.05 --asset "$src/$pair.json" --asset-root "$models"
    cmp "$ref/anchor.json" "$dest/anchor.json"
    "${cli[@]}" calibrate-pavement "$src/$pair.json" "$dest/anchor.json" "$dest/appearance.json" "$ref/ortho.npz" --asset-root "$models"
    cmp "$ref/appearance.json" "$dest/appearance.json"
    for shot in top-down bottom-up; do
      "${cli[@]}" register-readout "$ref/$shot-timed.npz" "$ref/trajectory.npz" "$dest/$shot.npz"
      "${cli[@]}" render "$src/$pair.json" "$dest/appearance.json" "$dest/$shot.npz" "$dest/$shot" --asset-root "$models"
      sums=$(realpath "$ref/$shot.sha256")
      (cd "$dest/$shot" && sha256sum -c "$sums")
    done
  done
  ;;
*) echo 'Usage: reproduce-rolling-covers.sh {assets|drive|joint-build|joint-render|joint-local|cycles|replay}' >&2; exit 2 ;;
esac
