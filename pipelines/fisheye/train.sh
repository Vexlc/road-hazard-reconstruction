#!/usr/bin/env bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
project=$(cd "$here/../.." && pwd)
workspace=$(cd "$project/../.." && pwd)
official=${OFFICIAL_ROOT:-$workspace/01_official}
method=${1:?choose 3dgut, sc-gs or self-cali-gs}
case "$method" in 3dgut|sc-gs|self-cali-gs) ;; *) echo "Unknown method: $method" >&2; exit 2;; esac
scene=$(realpath -e "${2:?prepared scene directory}")
output=$(realpath -m "${3:?new output directory}")
steps=${4:-1500}
[[ $steps =~ ^[1-9][0-9]*$ && -f $scene/split.json ]] || exit 2
if [[ -e $output ]]; then echo "Refusing to overwrite: $output" >&2; exit 2; fi
mkdir -p "$output"
export MAX_JOBS=${MAX_JOBS:-2} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=2
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} MPLBACKEND=Agg WANDB_MODE=disabled
case "$method" in
  3dgut)
    cd "$official/3dgrut"
    export PATH="$PWD/.venv/bin:$PATH"
    exec .venv/bin/python train.py --config-name apps/colmap_3dgut "path=$scene/fish" \
      "out_dir=$output" experiment_name=fit "n_iterations=$steps" num_workers=2 \
      dataset.load_exif=false dataset.test_split_interval=8 log_frequency=100 \
      compute_extra_metrics=true export_ply.enabled=true "export_ply.path=$output/scene.ply";;
  sc-gs)
    cd "$official/SC-GS"
    exec "$project/work/envs/fisheye-sc-gs/bin/python" "$here/train-scgs.py" \
      --source_path "$scene" --model_path "$output/fit_node" --eval --is_scene_static --deform_type node \
      --node_num 256 --init_isotropic_gs_with_all_colmap_pcl --resolution 1 \
      --iterations_node_sampling 100 --iterations_node_rendering 150 --node_warm_up 50 \
      --iterations "$steps" --test_iterations "$steps" --save_iterations "$steps";;
  self-cali-gs)
    cd "$official/Self-Cali-GS"
    exec "$project/work/envs/fisheye-self-cali-gs/bin/python" train.py -s "$scene" -m "$output" \
      --eval --resolution 1 --opt_distortion --outside_rasterizer --flow_scale 2 2 --port 0 \
      --r_t_noise 0 0 1 --control_point_sample_scale 16 --iresnet_lr 1e-7 \
      --iterations "$steps" --test_iterations "$steps" --save_iterations "$steps" --checkpoint_iterations "$steps";;
esac
