#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
scene=${1:?expected driving scene}
op=${2:-all}
steps=${3:-3000}
case "$scene" in 055|056|a2d2-frontleft|kitti360-0000|kitti360-0003-right) ;; *) exit 2;; esac
case "$op" in train|render|video|all|prepare|pit-pairs|pit-pairs-rear) ;; *) exit 2;; esac
[[ $steps =~ ^[1-9][0-9]*$ ]] || exit 2
official=${OFFICIAL_ROOT:-$(cd "$root/../.." && pwd)/01_official}
python=${GUT_PYTHON:-$official/3dgrut/.venv/bin/python}
scpython=${SCGS_PYTHON:-$root/work/envs/fisheye-sc-gs/bin/python}
export PYTHONPATH="$official/3dgrut${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$python"):$PATH" OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
out="$root/runs/streetgs-joint/$scene/$steps"
mkdir -p "$out"
exec 9>"$root/runs/streetgs-joint/$scene/.lock"
flock 9
if [[ $op == pit-pairs || $op == pit-pairs-rear ]]; then
    if [[ $scene != kitti360-0000 || $steps != 30000 || ! -s "$out/joint.pt" ]]; then
        echo "pit-pairs requires the trained kitti360-0000 30000 joint.pt" >&2
        exit 2
    fi
    cd "$official/3dgrut"
    if [[ $op == pit-pairs-rear ]]; then
        "$python" "$here/pit_pairs.py" render-rear
        exec "$scpython" "$here/review_pair_rear.py"
    fi
    "$python" "$here/pit_pairs.py" build
    "$python" "$here/pit_pairs.py" render
    "$python" "$here/pit_pairs.py" render-hd
    exec "$scpython" "$here/review_pit_pairs.py"
fi
if [[ ! -s "$root/work/streetcrafter-rig/$scene/all/split.json" ]]; then
    STREETCRAFTER_RIG=1 bash "$root/streetcrafter/backends/run.sh" "$scene" prepare
fi
if [[ $scene == kitti360-* || $scene == a2d2-* ]]; then
    "$scpython" "$root/streetcrafter/backends/rig_replay_source.py" "$root/work/streetcrafter-rig/$scene" "$root/work/data"
fi
if [[ $op == prepare || $op == train || $op == all ]]; then
    "$scpython" "$here/prepare.py" "$root" "$scene"
fi
cd "$official/3dgrut"
if [[ $op == train || $op == all ]]; then
    "$python" -u "$here/run.py" "$scene" train "$steps"
fi
if [[ $op == render || $op == all ]]; then
    "$python" -u "$here/run.py" "$scene" render "$steps"
fi
if [[ $op == video || $op == all ]]; then
    "$scpython" "$here/video.py" "$root" "$scene" "$steps"
fi
