#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
src=gen2hole/examples/wet-pavement
models=${WET_MODELS:-runs/wet-pavement}
out=${WET_VIEWS:-outputs/wet-pavement}
py=${WET_PYTHON:-work/envs/streetcrafter/bin/python}
cpu=${WET_CPU_PYTHON:-$py}
mesh=${WET_MESH_PYTHON:-work/envs/assets/bin/python}
base=runs/streetgs-joint/kitti360-0000/30000/wet-pavement
pairs=(P252-P255 P254-P253)
stage=${1:-help}
shift || true
case "$stage" in
beds)
  for key in P252 P253; do
    "$py" gen2hole/multiview_pits.py --survey "$src/$key.json" --source "${WET_CAPTURES:-work/wet-captures}" --out "$models/beds/$key"
  done
  ;;
survey|seat|drive)
  if [[ $# -eq 0 ]]; then set -- kitti audi; fi
  for site in "$@"; do
    case "$site" in
      kitti) host=(--scene kitti360-0000 --offset 1.4) ;;
      audi) host=(--scene a2d2-frontleft --offset -.5 --ego-mask gen2hole/examples/traffic-cones/audi-body.json) ;;
      *) echo "Expected kitti or audi, got $site" >&2; exit 2 ;;
    esac
    if [[ $stage == survey ]]; then
      "$py" scripts/render-mixed-road.py "${host[@]}" --group 26 --run "$models/beds" --layout "$src/P252-P253.json" --survey-only --out "$out/survey/$site"
      continue
    fi
    for pair in "${pairs[@]}"; do
      layout="$src/$pair.json"
      placed="$models/$site/$pair"
      if [[ $stage == seat ]]; then
        anchor="$src/anchors/$site.json"
        if [[ -n ${WET_SURVEYS:-} ]]; then anchor="$WET_SURVEYS/$site/anchor.json"; fi
        "$py" gen2hole/seat_water.py "$src/P252-P253.json" "$anchor" "$layout" "$models/beds" "$placed"
      else
        args=("${host[@]}" --group 26 --run "$placed" --layout "$layout" --paired-preview --export-packet "$pair")
        "$py" scripts/render-mixed-road.py "${args[@]}" --out "$out/reference/$pair/$site"
        "$py" scripts/render-mixed-road.py "${args[@]}" --appearance-views "$out/reference/$pair/$site" --video --out "$out/$pair/$site"
      fi
    done
  done
  ;;
review) "$py" scripts/review-wet-pavement.py "$out" "$models" ;;
joint-build|joint-render|joint-local)
  for pair in "${pairs[@]}"; do
    placed="$models/kitti/$pair"
    case "$stage" in
      joint-build)
        "$py" streetcrafter/joint/pavement_pair.py "$src/$pair.json" "$placed" "$placed/anchor.json" "$base" --apron-exposure ;;
      joint-render)
        gut="${OFFICIAL_ROOT:-$(cd ../.. && pwd)/01_official}/3dgrut"
        gpu=${GUT_PYTHON:-$gut/.venv/bin/python}
        PYTHONPATH="$gut${PYTHONPATH:+:$PYTHONPATH}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" PATH="$(dirname "$gpu"):$PATH" "$gpu" streetcrafter/joint/roadside_pairs.py render --save-pavement-inputs --assembly "$pair" --site kitti --out "$base" --renders "$out/joint" ;;
      joint-local)
        "$py" streetcrafter/joint/local_pavement_replay.py "$src/$pair.json" "$placed" "$placed/anchor.json" "$out/joint" "$out/local/joint" ;;
    esac
  done
  if [[ $stage != joint-build ]]; then
    views=$out
    if [[ $stage == joint-local ]]; then views="$out/local"; fi
    "$py" scripts/review-roadside-props.py --views "$views" --pairs "${pairs[@]}" --joint-only
  fi
  ;;
crown-seat) "$py" gen2hole/seat_water.py "$src/crown-dry.json" "$src/anchors/crown.json" "$src/crown-wet.json" "$models/beds" "$models/crown" ;;
cycles)
  "$mesh" gen2hole/examples/standing-water/check_solver.py
  "$mesh" scripts/render-water-interface.py "$out/interface"
  "$mesh" scripts/render-pavement-cycles.py --assets "$models/crown" --layout "$src/crown-wet.json" --matched-apron --out "$out/crown/wet"
  "$py" scripts/check-wet-pavement.py --views "$out" --models "$models"
  ;;
replay)
  packet=${WET_PACKETS:-work/wet-cameras}
  "$cpu" scripts/transfer-mesh-view.py calibrate-pavement "$src/crown-wet.json" "$models/crown/anchor.json" "$out/replay-appearance.json" "$packet/center.npz" "$packet/edge.npz" --asset-root "$models/crown"
  for view in center edge rear; do
    "$cpu" scripts/transfer-mesh-view.py render "$src/crown-wet.json" "$out/replay-appearance.json" "$packet/$view.npz" "$out/replay/$view" --asset-root "$models/crown"
  done
  sums="$(realpath "$packet/products.sha256")"
  (cd "$out/replay" && sha256sum -c "$sums")
  ;;
*) echo 'Stages: beds, survey [kitti|audi], seat [kitti|audi], drive [kitti|audi], review, joint-build, joint-render, joint-local, crown-seat, cycles, replay' >&2; exit 2 ;;
esac
