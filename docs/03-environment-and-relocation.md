# 环境和换机

推荐 Linux x86-64。查看 GLB 与 MP4 不需要训练环境；生成 GLB/高度场需要资产 Python 环境；生成或渲染 3DGS、训练道路场景需要 NVIDIA GPU、合适的驱动、CUDA Toolkit 和 C++ 编译器。本机验证使用 RTX 4070 Ti SUPER 16 GiB；其他 GPU 的显存需求和运行时间未在本项目逐一测量。

## 克隆

```bash
git clone https://github.com/Vexlc/road-hazard-reconstruction.git
cd road-hazard-reconstruction
```

代码仓库已公开，克隆代码无需 GitHub 登录。下载数据文件仍需要相应 Google Drive 访问权限。

## 资产环境

```bash
conda create -y -p "$PWD/work/envs/assets" python=3.11 pip
work/envs/assets/bin/python -m pip install -r scripts/requirements-assets.txt
```

版本文件固定 NumPy、SciPy、Trimesh 和 Pillow；重建后的纹理编码和浮点计算可能受版本影响，请保留固定版本。

## CUDA / StreetCrafter 环境

```bash
conda create -y -p "$PWD/work/envs/streetcrafter" python=3.9.23 pip
# 以下值按你的 CUDA 安装路径和 GPU 计算能力填写。
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  bash scripts/install-streetcrafter.sh
```

安装脚本获取固定提交的 StreetCrafter、gsplat、nvdiffrast、EasyVolcap 与 PandaSet SDK，应用项目补丁并编译 CUDA 扩展。依赖来源在 `outputs/reproducibility/sources.json`，Python 依赖在 `streetcrafter/requirements.txt`。原始环境是 PyTorch 2.4.0+cu121、CUDA Toolkit 12.8、gsplat 1.4.0。若 CUDA 编译报告架构或工具链不匹配，应按本机安装调整变量，不能直接拷贝另一台电脑的编译扩展或虚拟环境。

`TORCH_CUDA_ARCH_LIST=8.9` 对应本机 GPU，不是所有 NVIDIA GPU 的通用值。准备环境需要能访问 GitHub、Python 包索引和 PyTorch 下载源；从原数据准备道路时还需访问 Hugging Face 的固定天空分割模型。

## 恢复选择的数据

```bash
python scripts/packages.py check --group scenes-055-056 --group sample-scene-001
python scripts/packages.py restore --group scenes-055-056 --group sample-scene-001
bash scripts/scene.sh export 055
bash scripts/scene.sh export 056
```

新道路 034/109/110 的包恢复后，运行以下命令重写本机路径：

```bash
work/envs/streetcrafter/bin/python scripts/multiscene-configs.py
```

此脚本按当前克隆目录生成配置，包括 parent_cfg、数据路径、模型路径、训练记录路径和已有迁入场景的渲染配置。复制其他机器生成的 YAML 而不重定位路径，无法保证正常加载。

## 目录与空间

代码、原始重建输入、成品资产与场景运行包互相分离。生成文件写入 `work/` 或 `outputs/`，均保持在 Git 忽略规则中。按完整清单下载前先查看 `scripts/packages.py list` 的大小；解压空间与临时网格缓存需要另留余量。训练重跑会产生多个检查点，所需空间高于最终发布包。

不上传本机 Python 环境、浏览器数据、OAuth 配置、缓存和临时下载块。重装依赖和恢复数据包才是换机流程。
