# 鱼眼实拍场景重建

**视频回放：** [街边挖掘机](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/sun-openings-v1.0.0/dozer-host.mp4) · [红杉林](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/sun-openings-v1.0.0/redwoods2-host.mp4)。每段 32 秒：0–12 秒为 3DGUT（7,000 步）与静态 SC-GS（1,500 步）同轨迹漫游，使用共同针孔视场；12–22 秒为完整鱼眼实拍对照；22–32 秒为三种方法各 1,500 步的共同视场对照。对照部分每个机位停留 1.25 秒，分数对应当帧，而非序列均值。

从有标定、多视角的鱼眼照片训练三维高斯场景。这里不使用道路病害资产生成器，也不把普通照片加畸变来充当鱼眼数据。

| 方法与本地源码（相对 `0_test/01_official`） | 实际输入和输出 |
| --- | --- |
| [3DGUT](https://github.com/nv-tlabs/3dgrut) · `3dgrut` @ `e6f4552` | 直接用 `OPENCV_FISHEYE` 和畸变图；导出 `scene.ply`、训练 checkpoint 和鱼眼渲染。 |
| [SC-GS：Sparse-Controlled Gaussian Splatting](https://github.com/CVMI-Lab/SC-GS) · `SC-GS` @ `3a9d2ad` | 走官方静态场景、256 个控制节点分支；针孔相机需先去畸变。输出 `fit_node/point_cloud/iteration_1500/point_cloud.ply` 和 `deform/`。没有检验动态目标编辑。 |
| [Self-Cali-GS](https://github.com/denghilbert/Self-Cali-GS) · `Self-Cali-GS` @ `a269f90` | 透视面光栅化，利用 COLMAP 畸变系数初始化镜头网络，再用原鱼眼图监督；保存点云、`lens_net1500.pth` 与渲染。单独的 PLY 不包含镜头模型。 |

## 数据与划分

输入来自 [Nerfstudio 团队的数据仓](https://huggingface.co/datasets/nerfstudioteam/datasets/tree/461701c17e83c3f4d2481db32315aa7df703d2f8)，两个真实采集场景：街边挖掘机 `dozer` 和红杉林 `redwoods2`。下载 `images_4/` 和 COLMAP 二进制模型，验证 HF 发布的文件哈希；不用全分辨率照片或特征数据库。原始数据在 `../../03_datasets/NerfstudioFisheye/raw/`（本文命令均从 RoadGS 仓库根目录执行）。

- 各从有 COLMAP 位姿的缩小版图片中按文件名等间隔选 64 张，固定每第 8 张留出，56 训 / 8 测。默认宽 480，Dozer 高 319，Redwoods2 高 270；`prepared/<scene>/split.json` 记录原始文件名和缩放后的标定。
- `prepare.py` 保留同一世界坐标和位姿，分别输出 `fish/`（原鱼眼）及场景根目录（OpenCV 重映射后的针孔视图）。保留至少被两张训练图观察到、在两种视场内可用的点，颜色仅从训练像素取样；这也限制了边缘的初始化覆盖。Dozer 初始 31,557 点，Redwoods2 初始 46,156 点。
- **这是已知位姿的新视图合成**：几何坐标、相机和位姿来自发布方的全序列 SfM，不是仅用训练图从零解算。因此不能把留出分数当成无位姿重建或米制深度精度。
- 官方 SC-GS 的 COLMAP 路径在 `--eval` 下仍把测试相机放入训练集；本入口显式按 `split.json` 重新划分，并在官方评估时保存测试渲染，不修改上游训练损失。

## 本机复现

需要 NVIDIA GPU、CUDA 编译器和本机已有的两个 Torch 环境：`../../01_official/3dgrut/.venv`（可训练 3DGUT）及 `../dynamic_benchmark/.venv`（含匹配的 PyTorch3D）。本次为 Python 3.11、Torch 2.7.0+cu128、RTX 4070 Ti SUPER 16 GB。两个 SC 方法的 CUDA 扩展同名但不兼容，安装到不同的 `work/envs/fisheye-*`，不向上述基础环境装包。首次运行需联网获取依赖和感知指标权重；遵循各上游的使用条款。

```bash
# 若本地没有对应上游，先在 ../../01_official 获取上述仓库及指定版本，再运行：
git -C ../../01_official/SC-GS submodule update --init --recursive
git -C ../../01_official/Self-Cali-GS submodule update --init --recursive
bash pipelines/fisheye/setup-env.sh sc-gs
bash pipelines/fisheye/setup-env.sh self-cali-gs

data=../../03_datasets/NerfstudioFisheye
py=work/envs/fisheye-sc-gs/bin/python
for scene in dozer redwoods2; do
  python3 pipelines/fisheye/download.py "$data/raw" "$scene"
  # 已经准备好的场景可以跳过本行；不会覆盖已有目录。
  "$py" pipelines/fisheye/prepare.py "$data/raw/$scene" "$data/prepared/$scene"
  out="outputs/fisheye-repeat/$scene"
  for method in 3dgut sc-gs self-cali-gs; do
    bash pipelines/fisheye/train.sh "$method" "$data/prepared/$scene" "$out/$method" 1500
  done
  bash pipelines/fisheye/train.sh 3dgut "$data/prepared/$scene" "$out/3dgut-7000" 7000
  "$py" pipelines/fisheye/evaluate.py "$data/prepared/$scene" "$out"
done
```

## 如何看结果

本次成品在 `outputs/fisheye/<scene>/`。打开 `full-fisheye.jpg` 看最宽视场和边缘；打开 `shared-perspective.jpg` 比较三种方法在同一针孔机位上的画面。每组预览固定展示第一张、中间一张、最后一张留出图，不按分数挑图。`evaluation.json` 保存所有留出帧的实测值及高斯数；`3dgut-7000/scene.ply` 是额外长跑的模型。

统一指标为渲染 PNG 和参考图的逐帧 RGB PSNR，再在 8 帧上取平均，不做色彩匹配。完整鱼眼视场不遮罩（包括 Self-Cali-GS 的黑边）；共同视场把鱼眼渲染重投影到 SC-GS 的针孔视图，只比较标定覆盖的像素。两种视场不要混排。1,500 指场景训练步数，不含 Self-Cali-GS 的 5,000 步镜头初始化或 SC-GS 的约 150 步节点阶段；7,000 步的 3DGUT 另列，不代表同等算力。

| 场景 / 评估视场 | 3DGUT · 1,500 | SC-GS · 1,500 | Self-Cali-GS · 1,500 | 3DGUT · 7,000 |
| --- | ---: | ---: | ---: | ---: |
| Dozer / 完整鱼眼 | 19.81 | — | 14.80 | 21.92 |
| Dozer / 共同针孔 | 21.11 | 20.96 | 15.52 | 23.46 |
| Redwoods2 / 完整鱼眼 | 15.20 | — | 21.01 | 19.91 |
| Redwoods2 / 共同针孔 | 15.49 | 23.21 | 22.07 | 20.53 |

单位 dB，越高越好。Dozer 的 3DGUT 长跑保留了机械臂、锥桶和建筑轮廓，但第一张留出图的近地面仍模糊；红杉林在同样的 1,500 步下，共同针孔视场的 SC-GS 表现最好，不能据此推断它能重建最外围的鱼眼像素。共同针孔视场内有效像素占比为 Dozer 100%、Redwoods2 98.55%。

本机对照图：[Dozer 完整鱼眼](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/sun-openings-v1.0.0/dozer-full-fisheye.jpg) · [Redwoods2 完整鱼眼](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/sun-openings-v1.0.0/redwoods2-full-fisheye.jpg) · [Dozer 共同针孔](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/sun-openings-v1.0.0/dozer-shared-perspective.jpg) · [Redwoods2 共同针孔](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/sun-openings-v1.0.0/redwoods2-shared-perspective.jpg)。模型、日志和原始图像不放入 Git。

## 重做视频

下面只读取训练结果，不重训。从按采集编号排序的 64 个位姿线性插值相机中心、用 SLERP 插值旋转，两种渲染器共享 288 个位姿。首帧与各自保存的留出渲染核对。第二、三段直接读取八个留出机位的原图、渲染和评估记录。

```bash
official=$(realpath ../../01_official)
export CUDA_HOME=/usr/local/cuda MAX_JOBS=2 OMP_NUM_THREADS=4
export PATH="$official/3dgrut/.venv/bin:$PATH"
py=work/envs/fisheye-sc-gs/bin/python
for scene in dozer redwoods2; do
  data="../../03_datasets/NerfstudioFisheye/prepared/$scene"
  runs="outputs/fisheye/$scene"
  cache="work/fisheye-video/$scene"
  PYTHONPATH="$official/3dgrut" "$official/3dgrut/.venv/bin/python" pipelines/fisheye/render-video-path.py 3dgut "$data" "$runs" "$cache"
  PYTHONPATH="$official/SC-GS" "$py" pipelines/fisheye/render-video-path.py sc-gs "$data" "$runs" "$cache"
  "$py" pipelines/fisheye/compose-video.py "$data" "$runs" "$cache" "outputs/fisheye/videos/$scene.mp4"
done
```
