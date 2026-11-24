# StreetGS、3DGUT 与 SC-GS 联合重建

项目入口为 `bash scripts/scene.sh joint <scene> all 3000`。支持 `a2d2-frontleft`、`kitti360-0003-right`、`kitti360-0000`、`055` 和 `056`。前两个名称沿用旧片段 ID，当前训练覆盖该片段的所有物理相机。`backend <scene> all` 也进入联合训练。独立模型对照另用 `backend-baselines`。

## 一张图像怎样参与训练

1. 读取该次曝光的 RGB、内参、畸变模型、时间和相机位姿。训练/留出按整个采集组划分。在训练组的 LiDAR 上取色，从头初始化各层高斯。
2. 原项目的 `StreetGaussianModel` 持有背景、天空立方体和物体局部高斯。它用 `ActorPose` 将当前曝光可见的物体变换到世界坐标。PandaSet 依据训练组的标注轨迹、框内 LiDAR 和 RGB 建立物体层，轨迹按各相机曝光时间插值。轨迹固定，物体的高斯参数可学习。
3. 在没有接入实例轨迹的三段鱼眼片段中，从相同镜头的多个训练时刻重投影 LiDAR，使用粗深度缓冲去除遮挡。外观变化较大的点组成一个单独的残余层。光照、反光和配准误差也可能进入该层，因此这部分没有车辆身份或速度标签。
4. 官方 SC-GS 的 `ControlNodeWarp` 预测物体层在刚体变换之后的残余平移、旋转和尺度，使用 128 个控制节点、每点 4 个邻居。静态背景使用自己的高斯参数。所有可见高斯一起送入官方 3DGUT 的可微光栅化器，在当前物理相机的投影模型下完成深度排序与透明度合成。天空用同一相机的像素光线采样。
5. 同一个优化器从合成 RGB 反向传播到背景、天空、物体高斯和 SC-GS 的形变网络及节点权重、半径。损失包含 L1、结构相似性损失和一个弱位移惩罚。每轮随机遍历全部训练曝光。每个场景生成一个 `joint.pt`，包含场景图参数、轨迹、天空、控制节点、形变网络和渲染配置。

## 回放与验证

`all` 依次准备数据、训练、在新进程重新加载检查点、回放所有相机并编码视频。这些步骤可以分别用 `prepare`、`train`、`render`、`video` 调用。相机标定及留出清单位于 `work/streetcrafter-rig/<scene>/all/split.json`。A2D2 的五路鱼眼使用全画幅缩放图，正前长焦在校正后的视场回放。KITTI-360 两路侧摄从 MEI 重采样为水平 125° 等距鱼眼。PandaSet 使用六路针孔。

每个场景的 `runs/streetgs-joint/<scene>/<steps>/training.json` 保存一次真实光度损失对背景位置、物体位置及形变网络权重的梯度范数，以及各训练曝光的实际取样次数。`warp_effect.json` 记录了同一检查点在可见形变层的留出机位上启用和关闭形变的差异。该检查衡量已训练模型对形变的依赖。视频与各相机的留出 PSNR、SSIM 位于 `outputs/streetgs-joint/<scene>/<steps>/`。

五段均完成后可以生成视频合辑和报告。

```bash
work/envs/fisheye-sc-gs/bin/python streetcrafter/joint/video.py . all 3000
```

## 在 KITTI-360 0000 回放双坑

20 件车行道坑洞分为 10 个独立场景。每组从同一个 30,000 步检查点开始，在车辆转弯前和弯道里分别放一件。两处地面的倾斜由 KITTI 的 Velodyne 扫描拟合。坑底高斯来自各自的原始米制模型，按空间降采样，路面接缝和明暗参考该检查点在同一路段的渲染。每组保存完整的 `joint.pt`，可重新载入场景渲染器。

从仓库根目录执行以下命令。输入包括原始联合检查点、该路段的相机标定和扫描，以及本地 20 件坑洞合集。[从发布包恢复模型和宿主](../../docs/16-reproduction-cookbook.md#kitti-360-双坑与后视)；运行后分组索引写入 `outputs/streetgs-joint/kitti360-0000/30000/pit-pairs/list.md`。

```bash
bash scripts/scene.sh joint kitti360-0000 pit-pairs 30000
```

每组的 `placement.json` 记录两个模型的原始 PLY 校验值、实际放置的世界坐标、尺度和路面拟合误差。单组纯前视视频从该组检查点渲染 48 个原始相机时刻。对比版另外停留在两坑都可见的画面，下方两组近景各使用完全相同的相机位姿对照迁入前后。

已有双坑检查点和前视回放时，可用 `bash scripts/scene.sh joint kitti360-0000 pit-pairs-rear 30000` 生成后视。该片段有前视双目及左右侧视鱼眼。虚拟相机固定在车尾，沿原车轨迹回望，水平视场 108°。回放在车尾越过第一处完整坑沿后切到后视，两处坑洞均位于车尾后方并在画面内时标注编号。后视是重建场景的新视角，路面及街景部分区域可见拉伸。[十组后视视频](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/reproduction-2026.11.25/paired-rear.mp4) · [前视驶入、后视驶离](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/reproduction-2026.11.25/paired-front-to-rear.mp4)

## 环境

使用已有的 StreetCrafter 和 EasyVolcap 源码，以及 `01_official/3dgrut` 和 `01_official/SC-GS`。联合训练使用 3DGUT 的 Python 3.11 / Torch 2.7 CUDA 12.8 环境，并加载相同 ABI 的 SC-GS 和 PyTorch3D 扩展。数据准备和视频使用现有 `work/envs/fisheye-sc-gs`。本机还在 3DGUT 虚拟环境中由 EasyVolcap 补齐了 pip、pdbr、ujson 和 ruamel.yaml。bidict 与 roma 单独安装在工作目录。

```bash
../../01_official/3dgrut/.venv/bin/python -m pip install --no-deps --target work/joint-deps bidict==0.23.1 roma==1.8.0
```

`GUT_PYTHON`、`SCGS_PYTHON`、`SCGS_SITE` 与 `OFFICIAL_ROOT` 可用于指定其他相容安装。此次重建使用 StreetCrafter 的 StreetGS 场景图；视频扩散蒸馏为单独训练流程。

## 配置联合渲染环境

先按[环境与换机](../../docs/03-environment-and-relocation.md)准备仓库内的 `work/street_crafter`、`work/EasyVolcap`、`work/nvdiffrast` 及资产计算环境。下面是 Linux / NVIDIA 的独立 Python 3.11 渲染环境；在仓库根目录执行。源码和场景清单均可放在新目录，使用 `OFFICIAL_ROOT` 指向外部项目目录。请为本机安装 uv、CUDA Toolkit、C++ 编译工具和 OpenGL 开发头文件。

~~~bash
export OFFICIAL_ROOT="$PWD/work/joint-upstream"
mkdir -p "$OFFICIAL_ROOT"
git clone https://github.com/nv-tlabs/3dgrut.git "$OFFICIAL_ROOT/3dgrut"
git -C "$OFFICIAL_ROOT/3dgrut" checkout --detach e6f4552b408af020a96de09ab5757f9b57430aaa
git -C "$OFFICIAL_ROOT/3dgrut" submodule update --init --recursive
git clone https://github.com/CVMI-Lab/SC-GS.git "$OFFICIAL_ROOT/SC-GS"
git -C "$OFFICIAL_ROOT/SC-GS" checkout --detach 3a9d2ad4e4fc058b0763d446ae9e6b1be120b872
git -C "$OFFICIAL_ROOT/SC-GS" submodule update --init --recursive
git clone https://github.com/facebookresearch/pytorch3d.git "$OFFICIAL_ROOT/pytorch3d"
git -C "$OFFICIAL_ROOT/pytorch3d" checkout --detach 204fe5831b3a5bbe079f09d72afb4515178f8892
(cd "$OFFICIAL_ROOT/3dgrut" && ./install_env_uv.sh)
export GUT_PYTHON="$OFFICIAL_ROOT/3dgrut/.venv/bin/python"
"$GUT_PYTHON" -m ensurepip --upgrade
"$GUT_PYTHON" -m pip install pdbr ujson ruamel.yaml opencv-python==4.11.0.86 plyfile==1.1.3
BASE_ENV="$OFFICIAL_ROOT/3dgrut/.venv" bash pipelines/fisheye/setup-env.sh sc-gs
export SCGS_PYTHON="$PWD/work/envs/fisheye-sc-gs/bin/python"
"$SCGS_PYTHON" -m pip install fvcore iopath imageio-ffmpeg==0.6.0 pandas pyyaml easydict
"$SCGS_PYTHON" -m pip install --no-deps --no-build-isolation "$OFFICIAL_ROOT/pytorch3d" "$PWD/work/nvdiffrast"
"$GUT_PYTHON" -m pip install --no-deps --target work/joint-deps bidict==0.23.1 roma==1.8.0
"$SCGS_PYTHON" -c 'import torch, pytorch3d.ops, nvdiffrast.torch, simple_knn._C, cv2, imageio_ffmpeg; print(torch.__version__, torch.cuda.is_available())'
~~~

实跑环境为 Torch 2.7.0+cu128、Torchvision 0.22.0+cu128、PyTorch3D 0.7.9、nvdiffrast 0.4.0。GPU 架构、CUDA 和 Torch 必须与扩展匹配。加载发布检查点即可走通所有双坑步骤。自行重新训练时，`run.py` 还会导入 `fused_ssim`；若 3DGUT 安装没有提供它，用 `GUT_PYTHON` 从 `rahul-goel/fused-ssim` 的提交 `1272e21a282342e89537159e4bad508b19b34157` 编译安装。回放字幕使用 `fonts-noto-cjk` 的 `NotoSansCJK-Regular.ttc`。[模型、输入和输出目录](../../docs/16-reproduction-cookbook.md#kitti-360-双坑与后视)
