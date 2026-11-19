#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || ! "$2" =~ ^(034|109|110)$ ]]; then
    echo "Usage: $0 {prepare|train|insert|reload} {034|109|110} [batch IDs or reload ID]" >&2
    exit 2
fi
command="$1"; site="$2"
work="$root/tools/multiscene"
repo="$root/work/street_crafter"
python="${STREETCRAFTER_PYTHON:-$root/work/envs/streetcrafter/bin/python}"
sdk_import="$root/work/pandaset_sdk"
if [[ -d "$sdk_import" ]]; then
    sdk_module="$(find "$sdk_import" -type f -path '*/pandaset/sequence.py' -print -quit)"
    [[ -z "$sdk_module" ]] || sdk_import="$(dirname "$(dirname "$sdk_module")")"
fi
export PYTHONPATH="$sdk_import:$root/work/EasyVolcap:$root/work/asset_pipeline:$repo${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$root/work/hf_cache}"
export ROAD_BATCH_WORK="$work"
"$python" "$root/scripts/multiscene-configs.py"
case "$command" in
  prepare)
    "$python" "$work/download_scene.py" --scene "$site"
    cd "$repo/data_processor"
    "$python" pandaset_processor/pandaset_convertor.py --data_root "$work/data/raw" --target_dir "$work/data/processed" --scene_ids "$site" --workers 1 --process_keys timestamps images calib pose dynamic_masks objects
    "$python" pandaset_processor/pandaset_convertor.py --data_root "$work/data/raw" --target_dir "$work/data/processed" --scene_ids "$site" --workers 1 --process_keys lidar_forward
    "$python" "$work/sky_masks.py" "$site"
    ;;
  train)
    cd "$repo"
    "$python" train.py --config "$work/pandaset_${site}.yaml" mode train
    ;;
  insert)
    export ROAD_ONLY_BATCHES="${3:-}"
    cd "$repo"
    "$python" "$work/render_scenes.py" --config "$work/pandaset_${site}.yaml" mode trajectory
    ;;
  reload)
    batch="${3:-${site}_01}"
    [[ "$batch" =~ ^${site}_[0-9][0-9]$ ]] || { echo "Invalid batch ID" >&2; exit 2; }
    cd "$repo"
    "$python" "$work/reload_check.py" --config "$root/outputs/道路病害_多场景迁入/迁入场景/$batch/render_config.yaml" mode trajectory
    ;;
  *) echo "Unknown command: $command" >&2; exit 2 ;;
esac
