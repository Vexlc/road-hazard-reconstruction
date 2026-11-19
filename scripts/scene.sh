#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || ! "$2" =~ ^05[56]$ ]]; then
    echo "Usage: $0 {prepare|train|export} {055|056} [StreetCrafter config overrides]" >&2
    exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$root/work"
repo="$work/street_crafter"
python="${STREETCRAFTER_PYTHON:-$work/envs/streetcrafter/bin/python}"
scene="$2"
export PYTHONPATH="$work/EasyVolcap${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
case "$1" in
    prepare)
        cd "$root"
        prep=outputs/reproducibility/preparation_scripts
        raw="$work/data/raw/$scene"
        if [[ ! -s "$raw/camera/front_camera/poses.json" || ! -s "$raw/lidar/79.pkl.gz" || ! -s "$raw/annotations/cuboids/79.pkl.gz" ]]; then
            "$python" "$prep/download_scene.py" --scene "$scene"
        fi
        sdk="$(find "$work/pandaset_sdk" -type f -path '*/pandaset/sequence.py' -print -quit)"
        [[ -n "$sdk" ]] || { echo "PandaSet SDK not found" >&2; exit 1; }
        export PYTHONPATH="$(dirname "$(dirname "$sdk")"):$PYTHONPATH"
        cd "$repo/data_processor"
        converter=pandaset_processor/pandaset_convertor.py
        params=(--data_root "$work/data/raw" --target_dir "$work/data/processed" --scene_ids "$scene")
        "$python" "$converter" "${params[@]}" --process_keys timestamps images calib pose dynamic_masks objects
        "$python" "$converter" "${params[@]}" --process_keys lidar_forward
        cd "$root"
        "$python" "$prep/make_sky.py" "$scene"
        ;;
    train|export)
        cd "$repo"
        config="configs/pandaset_${scene}_warmup.yaml"
        command="$1"
        shift 2
        if [[ "$command" == train ]]; then
            "$python" train.py --config "$config" mode train "$@"
        else
            "$python" export_reconstruction.py --config "$config" mode trajectory "$@"
        fi
        ;;
    *)
        echo "Unknown operation: $1" >&2
        exit 2
        ;;
esac
