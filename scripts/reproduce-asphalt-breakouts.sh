#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
src=gen2hole/examples/asphalt-breakouts
models=${PIT_MODELS:-runs/asphalt-breakouts}
out=${PIT_VIEWS:-outputs/asphalt-breakouts}
pair=P246-P247
py=${PIT_PYTHON:-work/envs/streetcrafter/bin/python}
cpu=${PIT_CPU_PYTHON:-$py}
base=runs/streetgs-joint/kitti360-0000/30000/asphalt-breakouts
case "${1:-help}" in
assets)
  for key in P246 P247; do
    "$py" gen2hole/multiview_pits.py --survey "$src/$key.json" --source "${PIT_CAPTURES:-work/asphalt-captures}" --out "$models/$key"
  done
  ;;
drive)
  for site in kitti audi; do
    args=(--scene kitti360-0000 --offset 1.4)
    if [[ $site == audi ]]; then args=(--scene a2d2-frontleft --offset -.5 --ego-mask gen2hole/examples/traffic-cones/audi-body.json); fi
    args+=(--run "$models" --layout "$src/layout.json" --group 26 --paired-preview --export-packet "$pair")
    "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/reference/$site"
    "$py" scripts/render-mixed-road.py "${args[@]}" --video --appearance-views "$out/reference/$site" --out "$out/$site"
  done
  "$py" scripts/review-asphalt-breakouts.py --views "$out"
  ;;
joint-build)
  "$py" streetcrafter/joint/pavement_pair.py "$src/layout.json" "$models" "$src/kitti.json" "$base" --apron-exposure
  ;;
joint-render)
  gut="${OFFICIAL_ROOT:-$(cd ../.. && pwd)/01_official}/3dgrut"
  PYTHONPATH="$gut" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PATH="$gut/.venv/bin:$PATH" "$gut/.venv/bin/python" streetcrafter/joint/roadside_pairs.py render --assembly "$pair" --site kitti --out "$base" --renders "$out/joint"
  "$py" scripts/review-roadside-props.py --views "$out" --pairs "$pair" --joint-only
  ;;
replay)
  camera_root=${PIT_CAMERAS:-work/asphalt-cameras}
  for suffix in "" /diffuse-host; do
    cam="$camera_root$suffix"
    dest="$out/replay$suffix"
    read -ra pose < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(*d["seed"],*d["heading"],*d["up"])' "$cam/frame.json")
    cli=("$cpu" scripts/transfer-mesh-view.py)
    "${cli[@]}" road-points "$dest/road.npy" "$cam/center.npz" "$cam/edge.npz" --stride 2
    "${cli[@]}" fit "$dest/road.npy" "$dest/anchor.json" --seed "${pose[@]:0:3}" --heading "${pose[@]:3:3}" --up "${pose[@]:6:3}" --radius .85 --asset "$src/crown.json" --asset-root "$models"
    cmp "$cam/anchor.json" "$dest/anchor.json"
    "${cli[@]}" calibrate-pavement "$src/crown.json" "$dest/anchor.json" "$dest/appearance.json" "$cam/center.npz" "$cam/edge.npz" --asset-root "$models"
    cmp "$cam/appearance.json" "$dest/appearance.json"
    for view in center edge rear; do
      "${cli[@]}" render "$src/crown.json" "$dest/appearance.json" "$cam/$view.npz" "$dest/$view" --asset-root "$models"
      sums=$(realpath "$cam/$view.sha256")
      (cd "$dest/$view" && sha256sum -c "$sums")
    done
  done
  ;;
cycles)
  opts=(--assets "$models" --layout "$src/crown.json")
  for suffix in "" /diffuse-host; do
    flags=(--matched-apron)
    if [[ -n $suffix ]]; then flags=(--albedo-pass); fi
    dest="$out$suffix"
    "${PIT_BLENDER_PYTHON:-work/envs/assets/bin/python}" scripts/render-pavement-cycles.py "${opts[@]}" --paint-control "${flags[@]}" --out "$dest/cycles"
    "$cpu" scripts/check-pavement-cycles.py "${opts[@]}" --views "$dest/cycles" --out "$dest/controls"
    "$cpu" scripts/check-asphalt-breakouts.py --views "$dest" --assets "$models" --cycles-only
  done
  ;;
*)
  echo 'Usage: bash scripts/reproduce-asphalt-breakouts.sh {assets|drive|joint-build|joint-render|replay|cycles}' >&2
  exit 2
  ;;
esac
