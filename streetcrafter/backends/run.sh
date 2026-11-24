#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
scene=${1:?expected a configured driving clip}
op=${2:?expected prepare, 3dgut, sc-gs, video or all}
steps=${3:-3000}
case "$scene" in 055|056|kitti360-0000|kitti360-0003-right|a2d2-frontleft) ;; *) exit 2;; esac
case "$op" in prepare|3dgut|sc-gs|video|all) ;; *) exit 2;; esac
[[ $steps =~ ^[1-9][0-9]*$ && $steps -gt 150 ]] || { echo "Use more than 150 steps for dynamic training" >&2; exit 2; }
work="$root/work"
repo="$work/street_crafter"
official=${OFFICIAL_ROOT:-$(cd "$root/../.." && pwd)/01_official}
bundle="$work/streetcrafter-backends/$scene"
run="$root/runs/streetcrafter-backends/$scene"
scpython=${SCGS_PYTHON:-$work/envs/fisheye-sc-gs/bin/python}
rig=${STREETCRAFTER_RIG:-1}
if [[ $rig == 1 ]]; then
    bundle="$work/streetcrafter-rig/$scene"
    run="$root/runs/streetcrafter-rig/$scene"
fi
mkdir -p "$run"
exec 9>"$run/.backend.lock"
flock 9
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MAX_JOBS=${MAX_JOBS:-2} OPENBLAS_NUM_THREADS=2
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} MPLBACKEND=Agg

if [[ ! -f $bundle/all/split.json ]]; then
    if [[ $rig == 1 ]]; then
        if [[ $scene == kitti360-* || $scene == a2d2-* ]]; then
            # A2D2 uses the metric trajectory estimated from its front-left training exposures.
            if [[ $scene == a2d2-* && ! -s "$work/streetcrafter-backends/$scene/all/split.json" ]]; then
                STREETCRAFTER_RIG=0 bash "$here/run.sh" "$scene" prepare
            fi
            "$scpython" "$here/download_rig.py" "$scene" "$work/data"
        elif [[ ! -s "$work/data/processed/$scene/extrinsics/079_5.txt" ]]; then
            bash "$root/scripts/scene.sh" prepare "$scene"
        fi
        "$scpython" "$here/prepare_rig.py" "$scene" "$work" "$bundle"
    elif [[ $scene == kitti360-* ]]; then
        raw="$work/data/kitti360-0000-0200"
        clip_args=()
        if [[ $scene == kitti360-0003-right ]]; then
            raw="$work/data/$scene"
            clip_args=(--drive 0003 --camera image_03 --start 480 --count 64)
        fi
        "$scpython" "$here/download_kitti.py" "$raw" "${clip_args[@]}"
        "$scpython" "$here/prepare_kitti.py" "$raw" "$bundle" "${clip_args[@]}"
    elif [[ $scene == a2d2-frontleft ]]; then
        "$scpython" "$here/download_a2d2.py" "$work/data/$scene"
        "$scpython" "$here/prepare_a2d2.py" "$work/data/$scene" "$bundle"
    else
        cd "$repo"
        SCENE_BUNDLE="$bundle" PYTHONPATH="$repo:$work/EasyVolcap" \
            "${STREETCRAFTER_PYTHON:-$work/envs/streetcrafter/bin/python}" "$here/prepare_pandaset.py" \
            --config "configs/pandaset_${scene}_warmup.yaml" mode trajectory source_path "$work/data/processed/$scene"
    fi
fi
if [[ $rig == 1 && ( $scene == kitti360-* || $scene == a2d2-* ) ]]; then
    "$scpython" "$here/rig_replay_source.py" "$bundle" "$work/data"
fi
if [[ $op == 3dgut || $op == all ]]; then
    cd "$official/3dgrut"
    gutpython=${GUT_PYTHON:-$PWD/.venv/bin/python}
    export PATH="$(dirname "$gutpython"):$PATH"
    output="$run/3dgut-$steps"
    mkdir -p "$output"
    train="$bundle/train"
    [[ ! -d $bundle/fish_train ]] || train="$bundle/fish_train"
    [[ $rig != 1 ]] || train="$bundle/gut_train"
    shopt -s nullglob
    checkpoints=("$output"/fit/*/ckpt_last.pt)
    if (( ${#checkpoints[@]} == 0 )); then
        "$gutpython" train.py --config-name apps/colmap_3dgut "path=$train" "out_dir=$output" experiment_name=fit \
            "n_iterations=$steps" num_workers=2 dataset.load_exif=false dataset.test_split_interval=0 test_last=false \
            compute_extra_metrics=false log_frequency=100 export_ply.enabled=true "export_ply.path=$output/scene.ply" \
            2>&1 | tee "$output/train.log"
        checkpoints=("$output"/fit/*/ckpt_last.pt)
    fi
    (( ${#checkpoints[@]} == 1 )) || { echo "Expected one completed 3DGUT fit in $output" >&2; exit 1; }
    replay="$bundle/all"
    [[ $rig != 1 ]] || replay="$bundle/eval"
    PYTHONPATH="$PWD" "$gutpython" "$here/render_gut.py" "${checkpoints[0]}" "$replay" "$output/frames"
    if [[ -d $bundle/sensor_eval ]]; then
        PYTHONPATH="$PWD" "$gutpython" "$here/render_gut.py" "${checkpoints[0]}" "$bundle/sensor_eval" "$output/sensor_frames"
    fi
    if [[ -d $bundle/fish_all ]]; then
        PYTHONPATH="$PWD" "$gutpython" "$here/render_gut.py" "${checkpoints[0]}" "$bundle/fish_all" "$output/fisheye_frames"
    fi
fi
if [[ $op == sc-gs || $op == all ]]; then
    cd "$official/SC-GS"
    action=train
    output="$run/sc-gs-$steps"
    if [[ -d $bundle/fish_train ]]; then
        action=train-fisheye
        output="$run/sc-gs-fisheye-$steps"
    fi
    mkdir -p "$output"
    model="$output/fit_node"
    if [[ ! -s $model/deform/iteration_$steps/deform.pth || ! -s $model/point_cloud/iteration_$steps/point_cloud.ply ]]; then
        memory_args=()
        [[ $rig != 1 ]] || memory_args=(--load2gpu_on_the_fly)
        "$scpython" "$here/scgs.py" "$action" "${memory_args[@]}" --source_path "$bundle/all" --model_path "$model" --eval --deform_type node \
            --node_num 256 --init_isotropic_gs_with_all_colmap_pcl --resolution 1 --iterations_node_sampling 100 \
            --iterations_node_rendering 150 --node_warm_up 50 --warm_up 150 --no_arap_loss --deform_lr_scale 0.005 \
            --iterations "$steps" --test_iterations "$steps" --save_iterations "$steps" 2>&1 | tee "$output/train.log"
    fi
    "$scpython" "$here/scgs.py" render --model_path "$model" --deform_type node --iteration "$steps" --frames "$output/frames"
fi
if [[ $op == video || $op == all ]]; then
    compare=compare.py
    [[ $rig != 1 ]] || compare=rig_video.py
    "$scpython" "$here/$compare" "$root" "$scene" "$steps"
fi
