#!/usr/bin/env bash
# Run the portable, sequential GPU pipeline for the five final 1080p videos.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:-help}"
dry=0
[[ "${2:-}" != --dry-run ]] || dry=1
case "$stage" in all|native|views|compose-native|scaled|compose-scaled|export|verify) ;; *)
    echo "Usage: $0 {all|native|views|compose-native|scaled|compose-scaled|export|verify} [--dry-run]" >&2
    exit 2 ;;
esac
python="${STREETCRAFTER_PYTHON:-$root/work/envs/streetcrafter/bin/python}"
work="$root/tools/multiscene"
repo="$root/work/street_crafter"
export PYTHONPATH="$root/work/EasyVolcap:$root/work/asset_pipeline:$repo${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export ROAD_BATCH_WORK="$work" PYTHONUTF8=1 PYTHONIOENCODING=utf-8
unset ROAD_ONLY_BATCHES SCALED_ONLY_BATCHES
run() {
    printf '+ '; printf '%q ' "$@"; printf '\n'
    if [[ "$dry" == 0 ]]; then "$@"; fi
}
if [[ "$stage" != export && "$stage" != verify ]]; then
    run "$python" "$root/scripts/multiscene-configs.py"
fi
if [[ "$dry" == 0 ]]; then
    if [[ "$stage" != export && "$stage" != verify ]]; then
        for site in 034 109 110; do
            [[ -s "$root/outputs/道路病害_多场景迁入/重建场景/$site/trained_model/iteration_7000.pth" ]] || {
                echo "Restore scenes-$site and sample-multiscene-50 first (docs/01-downloads.md)." >&2; exit 2;
            }
        done
        cd "$repo"
    fi
fi
native() {
    for site in 034 109 110; do run bash "$root/scripts/multiscene.sh" insert "$site"; done
}
views() {
    for site in 034 109 110; do
        run "$python" "$work/render_presentation_views.py" --config "$work/pandaset_$site.yaml" mode trajectory
        run "$python" "$work/render_stereo_detail.py" --config "$work/pandaset_$site.yaml" mode trajectory
    done
    run "$python" "$work/render_presentation_orbits.py"
    run "$python" "$work/render_stereo_orbits.py"
}
scaled() {
    for site in 034 109 110; do
        run "$python" "$work/render_enlarged_stereo.py" --config "$work/pandaset_$site.yaml" mode trajectory
    done
}
if [[ "$stage" == all || "$stage" == native ]]; then native; fi
if [[ "$stage" == all || "$stage" == views ]]; then views; fi
if [[ "$stage" == all || "$stage" == compose-native ]]; then run "$python" "$work/compose_dynamic_presentation.py"; fi
if [[ "$stage" == all || "$stage" == scaled ]]; then scaled; fi
if [[ "$stage" == all || "$stage" == compose-scaled ]]; then
    run "$python" "$work/compose_enlarged_stereo.py"
    run "$python" "$work/verify_enlarged_stereo.py"
fi
if [[ "$stage" == all || "$stage" == export ]]; then run "$python" "$work/export_motion_confirmation.py"; fi
if [[ "$stage" == all || "$stage" == verify ]]; then run "$python" "$work/verify_motion_confirmation.py"; fi
