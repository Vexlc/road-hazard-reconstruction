# StreetCrafter 独立后端对照

三种方法融合训练使用 `bash scripts/scene.sh joint <scene> all 3000`，[联合模型说明](../joint/README.md)。下文为分别训练两个模型的对照实验。

从项目根目录运行。每个场景的 `all` 会准备数据，训练两个模型，重新加载检查点，沿采集轨迹渲染并编码视频。

```bash
bash scripts/scene.sh backend-baselines 056 all 3000
bash scripts/scene.sh backend-baselines 055 all 3000
bash scripts/scene.sh backend-baselines kitti360-0000 all 3000
bash scripts/scene.sh backend-baselines kitti360-0003-right all 3000
bash scripts/scene.sh backend-baselines a2d2-frontleft all 3000
```

默认按每段数据集的全部物理相机联合训练。每个后端、每个场景各产生一个检查点。可将 `all` 换成 `prepare`、`3dgut`、`sc-gs` 或 `video`。同一采集组的所有物理相机、切平面和 LiDAR 同时进训练集或留出集。每第 8 个时间间隔留出一组，从组索引 8 开始。动态模型使用各相机自己的曝光时间。

| 场景 | 物理相机 | 采集组 | 训练／留出组 | 训练／留出原图 |
|---|---|---:|---:|---:|
| PandaSet 055、056（各段） | 前、左前、右前、左、右、后 | 80 | 71／9 | 426／54 |
| KITTI-360 0000 | 前方双目、左右侧鱼眼 | 48 | 43／5 | 172／20 |
| KITTI-360 0003 | 前方双目、左右侧鱼眼 | 64 | 57／7 | 228／28 |
| Audi A2D2 | 左前、右前、正前、左侧、右侧、后方 | 64 | 57／7 | 342／42 |

PandaSet 使用 1920 × 1080 原图、逐曝光的官方相机位姿和环视／前向 LiDAR，六路各自缩放到 480 × 270。KITTI-360 使用 1408 × 376 的校正双目图、1400 × 1400 的 MEI 鱼眼图、Velodyne 和车体轨迹。双目输入为 640 × 171，侧摄用 MEI 标定重采样成 400 × 400、水平视场 125° 的等距鱼眼。车体位姿按曝光时间插值。A2D2 使用五路原始鱼眼和一路前向长焦，原图 1920 × 1208，3DGUT 输入为 480 × 300。以左前训练数据估计的米制轨迹配合六组刚体外参，并按微秒时间戳匹配各路曝光。各路 LiDAR 子云仅在训练采集组取色。

3DGUT 使用各物理镜头独立的投影参数。SC-GS 用每个鱼眼的中心、左偏和右偏切平面训练，三者共享曝光时间和光心。SC-GS 回放时按目标鱼眼射线从同一个动态模型的多方向渲染采样。视频每列是一台物理相机，每行依次为实拍、3DGUT、SC-GS。留出 PSNR 和 SSIM 在原尺寸评测投影上逐图计算，涵盖所有相机。A2D2 正前长焦使用校正后的中心透视图回放。视频统一按每秒 10 个采集组播放。

训练数据写入 `work/streetcrafter-rig/<scene>/`。其中 `all/split.json` 保存传感器名称、源帧、逐相机时间、原始标定、各后端内参与训练划分。`eval/sparse/0/images.txt` 保存每次曝光的世界到相机位姿，采用 COLMAP 四元数 wxyz 和米制平移。模型、PLY 和回放帧在 `runs/streetcrafter-rig/<scene>/`，视频与分相机指标在 `outputs/streetcrafter-rig/<scene>/<steps>/`。

五段训练和视频完成后，运行以下命令生成合辑与逐相机内外参和指标报告。

```bash
work/envs/fisheye-sc-gs/bin/python streetcrafter/backends/rig_gallery.py . 3000
```

## 历史单相机实验

用 `bash scripts/scene.sh backend-single <scene> all 3000` 可复现下列早期实验。其结果保存在 `outputs/streetcrafter-backends/`。

### 单相机数据与视场

- **PandaSet 055、056**。调用 StreetCrafter 的 `readPandasetInfo` 和对应 warmup 配置，使用前视针孔相机、960 × 540 图像、逐帧外参和时间戳。每段 80 帧，训练 71 帧，留出第 8、16、…、72 帧。两个新后端共用训练帧的有色 LiDAR 点和从训练帧天空掩码采样的远点。如需从原始数据开始，可先运行 `bash scripts/scene.sh prepare 056` 或 `055`。
- **KITTI-360**。使用 drive 0000 左侧 `image_02` 第 200–247 帧和 drive 0003 右侧 `image_03` 第 480–543 帧，分别留出 5 和 7 帧。入口按需下载图像、Velodyne、相机时间戳、标定和车体位姿。训练帧扫描在原镜头取色，按 MEI 标定生成 125° 水平视场的等距鱼眼图。[官方数据说明](https://www.cvlibs.net/datasets/kitti-360/documentation.php)
- **Audi A2D2**。使用 2018-08-10 行程序列的左前鱼眼第 1500–1563 帧，保留完整 1920 × 1208 画幅，缩放到 768 × 483 后用原始四参数鱼眼畸变训练 3DGUT。相邻训练图像用光流跟踪特征，配准 LiDAR 提供米制深度，PnP 估计相对位姿。7 个留出机位按时间插值，点云和颜色取自其余 57 帧。位姿估计的内点数和重投影误差可在场景目录的 `all/split.json` 查看。[数据论文](https://arxiv.org/abs/2004.06320)

鱼眼场景中的 SC-GS 以 92° 中心透视图和 8 个覆盖边缘的切平面视图训练。每组共用同一相机中心和时间戳。评测时从 SC-GS 检查点渲染多个透视方向，再沿鱼眼标定的像素光线采样。两种模型的鱼眼 PSNR、SSIM 在相同留出帧和像素上计算。带 `_fisheye` 后缀的指标对应鱼眼域，其余两项对应 92° 针孔域。

生成鱼眼实拍、两个模型的原机位回放和 3DGUT 右移 35 厘米的新视角合辑：

```bash
work/envs/fisheye-sc-gs/bin/python streetcrafter/backends/fisheye_video.py a2d2-frontleft kitti360-0003-right kitti360-0000
```

## 模型与环境

3DGUT 用无迹变换渲染一组静态高斯。SC-GS 用 256 个控制节点学习随时间变化的形变，训练和回放均应用标定主点。两个训练器各自产生检查点。PandaSet 视频里原有的 StreetCrafter 列读取 7,000 步动态 warmup 导出的 JPEG，沿用它原来的初始化方式；该列只作参考。以上对照均未运行 StreetCrafter 的扩散蒸馏。

默认读取同一工作区 `01_official/3dgrut`、`01_official/SC-GS` 及项目现有的 `work/envs/streetcrafter`、`work/envs/fisheye-sc-gs`。可用 `OFFICIAL_ROOT`、`GUT_PYTHON`、`SCGS_PYTHON`、`STREETCRAFTER_PYTHON` 指向其他安装。3DGUT 使用自己的 CUDA 环境和 Slang，SC-GS 的专用扩展可用 `bash pipelines/fisheye/setup-env.sh sc-gs` 构建。本地使用的上游版本分别为 `e6f4552` 和 `3a9d2ad`。[3DGUT](https://github.com/nv-tlabs/3dgrut) · [SC-GS](https://github.com/CVMI-Lab/SC-GS)
