#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$root/work"
repo="$work/street_crafter"
sources="$root/outputs/reproducibility/sources.json"
python="$work/envs/streetcrafter/bin/python"

for name in gsplat nvdiffrast easyvolcap pandaset_sdk; do
    mapfile -t revision < <("$python" -c 'import json,sys; data=json.load(open(sys.argv[1]))[sys.argv[2]]; print(data["url"]); print(data["commit"])' "$sources" "$name")
    directory="$work/$name"
    [[ "$name" != easyvolcap ]] || directory="$work/EasyVolcap"
    if [[ ! -e "$directory" ]]; then
        git clone --no-checkout "${revision[0]}" "$directory"
        git -C "$directory" checkout --detach "${revision[1]}"
    fi
    if [[ "$(git -C "$directory" rev-parse --show-toplevel)" != "$directory" || "$(git -C "$directory" rev-parse HEAD)" != "${revision[1]}" ]]; then
        echo "Expected ${revision[1]} at $directory" >&2
        exit 1
    fi
done

sdk_patch="$root/streetcrafter/pandaset-sdk.patch"
if ! git -C "$work/pandaset_sdk" apply --reverse --check "$sdk_patch" >/dev/null 2>&1; then
    git -C "$work/pandaset_sdk" apply "$sdk_patch"
fi

bash "$root/scripts/checkout-streetcrafter.sh"
git -C "$repo" submodule update --init submodules/simple-knn
git -C "$work/gsplat" submodule update --init --depth 1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-2}"

"$python" -m pip install setuptools==79.0.1 wheel==0.45.1
"$python" -m pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
"$python" -m pip install -r "$root/streetcrafter/requirements.txt"
"$python" -m pip install --no-build-isolation --no-deps "$work/gsplat" "$work/nvdiffrast" "$repo/submodules/simple-knn"
"$python" -m pip check
