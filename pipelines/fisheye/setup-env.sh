#!/usr/bin/env bash
set -eu
project=$(cd "$(dirname "$0")/../.." && pwd)
workspace=$(cd "$project/../.." && pwd)
official=${OFFICIAL_ROOT:-$workspace/01_official}
method=${1:?choose sc-gs or self-cali-gs}
case "$method" in
  sc-gs) base=${BASE_ENV:-$workspace/02_projects/dynamic_benchmark/.venv}; source_dir=$official/SC-GS/submodules;;
  self-cali-gs) base=${BASE_ENV:-$official/3dgrut/.venv}; source_dir=$official/Self-Cali-GS;;
  *) exit 2;;
esac
envdir=$project/work/envs/fisheye-$method
"$base/bin/python" -m venv "$envdir"
sitepath() { "$1/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))'; }
sitepath "$base" > "$(sitepath "$envdir")/base-environment.pth"
export PATH="$envdir/bin:$base/bin:$PATH" CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} MAX_JOBS=${MAX_JOBS:-2}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}
export NVCC_PREPEND_FLAGS="--pre-include cstdint --pre-include cfloat ${NVCC_PREPEND_FLAGS:-}"
python -m pip install wheel 'setuptools<81' pycolmap==3.13.0
if [ "$method" = sc-gs ]; then
  python -m pip install piq dearpygui pytorch-msssim lpips
  raster=$source_dir/diff-gaussian-rasterization
else
  python -m pip install FrEIA==0.2 visdom selenium pywavefront ipdb trimesh easydict termcolor
  raster=$source_dir/3dgs-pose
fi
python -m pip install --no-deps --no-build-isolation "$source_dir/simple-knn" "$raster"
