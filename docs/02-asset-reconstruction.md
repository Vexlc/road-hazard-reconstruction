# 五类资产从输入到 GLB 与 3DGS

先按 [环境文档](03-environment-and-relocation.md) 安装 Python 3.11 资产环境和 Python 3.9 CUDA 环境。以下命令都在仓库根目录执行。`reconstruct-assets.py` 只重建版本化资产清单中已经发布的 ID，避免上游数据新增、排序或筛选变化导致资产数量漂移。

## 1. 双目实拍坑洞：65 条

输入来自 Fan Stereo Pothole Datasets 固定提交 `a6cc8e7354da4503b0ee03d15058df131a645195`。每条有对齐的 RGB PNG、标签 PNG 和 MATLAB `xyzPoints`；XYZ 原单位为毫米，乘 0.001 转为米。本次发布的是扩充库的 65 条记录，早期单独发布的 dataset1/02 与 dataset2/25 不混入这 65 条。

处理流程：取最大病害连通域并填孔；在病害周围的非病害实测点上迭代拟合道路平面；以平面法线构造局部坐标；对局部 XYZ 用近邻加权构建高度场；将原 RGB 投影回网格作为纹理；剔除没有测量支持的边界，生成网格和路面透明过渡字段。边界宽度按 0.065、0.040、0.026 米依次尝试，并记录实际采用值。

通过条件包括病害和周边点支持率、正向凹陷、平面拟合误差与尺度范围。源数据作者排除的评估帧仍会标记 `author_excluded_from_evaluation`，本资产库的生成通过不代表它适用于源论文的基准评测。

```bash
python scripts/packages.py restore --group inputs-stereo
work/envs/assets/bin/python scripts/reconstruct-assets.py --category stereo --sample
# 重新生成全部 65 条，去掉 --sample
work/envs/assets/bin/python scripts/reconstruct-assets.py --category stereo
```

源码：`outputs/road_hazard_expansion/reproduction/batch_stereo.py` 与 `reconstruct_pothole.py`。重建结果写入 `work/rebuilt-assets/stereo/`，不会覆盖下载的原始成品。

## 2. RGB-D 实拍坑洞：822 条

输入是 PothRGBD/PData 固定数据快照：`images.zip`、`-depths.zip`、`labels.zip`；来源仓库提交 `7596dc1c745e68b90403d7e82f89e7492fa95edc`。按时间戳配对 RGB、uint16 深度 NPY 和归一化多边形标签，重复时间戳存在多份 RGB/标签变体；发布入口使用 catalog/capture-inputs.json 固定每件资产的精确输入成员名和 SHA-256。深度每单位 0.001 米；默认 `fx=fy=462` 像素，主点为图像中心。这是标称内参，重建尺寸与深度是近似值，不能作为经过独立标定的高精度测量。

处理流程：保留最大标签连通域；筛除截断、过小或深度支持不足的图像；过滤无效深度并以 3×3 中值减轻量化噪声；按针孔模型反投影 XYZ；复用局部道路平面、高度场、源 RGB 投影网格和高斯导出流程。边界依次尝试 0.085、0.050 米；网格间距目标 0.0023 米，单边不超过 340 个采样点。只发布版本化清单中通过筛选的 822 条记录。

```bash
python scripts/packages.py restore --group inputs-rgbd
work/envs/assets/bin/python scripts/reconstruct-assets.py --category rgbd --sample
work/envs/assets/bin/python scripts/reconstruct-assets.py --category rgbd
```

源码：`batch_rgbd.py` 与共享的 `reconstruct_pothole.py`。每条成品的 `capture_metadata.json` 记录输入成员名、SHA-256、路面平面残差和深度分位数。

## 3. 程序化坑洞：120 条

10 种形态 × 4 种路面外观 × 3 个确定性变体。形态为剥落、碗状坑、深度破损、月牙坑、车辙长坑、连通坑、群坑、修补破损、板角破损和积水坑。`variant_catalog.py` 确定每个 ID 的尺寸、深度和随机种子；`make_variants.py` 生成不规则坑口、破碎骨料、裂缝与路面高度场。积水变体另生成水面几何。

```bash
python scripts/packages.py restore --group inputs-procedural
work/envs/assets/bin/python scripts/reconstruct-assets.py --category pothole --sample
work/envs/assets/bin/python scripts/reconstruct-assets.py --category pothole
```

## 4. 缺盖窨井：64 条

4 种井口形状 × 4 种井沿状态 × 4 种路面。形状为圆、椭圆、方形、长方形；状态为齐平、砖砌、破损井沿、沉降井框。模型有实际井沿、向内井壁、井底与适用时的踏步。结构是参数化建模，表面外观来自 ambientCG 材质，不是 64 个独立实物的扫描。

```bash
work/envs/assets/bin/python scripts/reconstruct-assets.py --category missing --sample
work/envs/assets/bin/python scripts/reconstruct-assets.py --category missing
```

## 5. 掀盖、倾斜与移位窨井：128 条

4 种形状 × 8 种盖板姿态 × 4 种路面。姿态为微开 14°、倾斜 29°、支起 43°、掀开 62°、近直立 82°、平移、斜移、移到井旁。井筒与井框共用缺盖流程；盖板独立生成正面、背面、厚度和加强筋，以铰接支点、转角和位移构成姿态，参数记录在 `cover_pose`。

```bash
work/envs/assets/bin/python scripts/reconstruct-assets.py --category raised --sample
work/envs/assets/bin/python scripts/reconstruct-assets.py --category raised
```

后三类共用 `inputs-procedural`，只需恢复一次。也可用 `--asset 完整ID` 重建任意单个资产，或多次传入 `--asset`。

## GLB 到高斯的共同步骤

程序化网格按表面积分层随机采样，固定种子来自资产 ID 对应元数据；沿表面切线设置各向异性尺度，法线方向保留薄层；几何遮挡与直接光估计用于外观烘焙。实拍资产保留由实际高度场产生的网格采样与采集 RGB。二者均用固定球面采样拟合一阶球谐系数，再保存 Gaussian PLY，并用 gsplat 渲染预览确认有限值与覆盖率。

`reconstruct-assets.py` 自动调用 CUDA 环境中的 `export_gaussians.py`。只生成 GLB/高度场可加 `--mesh-only`；指定环境位置可用 `--gaussian-python /absolute/path/to/python`。

## 复现核对

```bash
# 下载成品要求 SHA-256 完全一致
python scripts/verify-assets.py --category stereo

# 重新计算的结果使用显式数值阈值比较
work/envs/streetcrafter/bin/python scripts/compare-reconstruction.py \
  --reference outputs/road_hazard_expansion --rebuilt work/rebuilt-assets \
  --report work/reconstruction-comparison.json
```

浮点库、GPU、边界采样和并行归约顺序可能使重新生成的二进制哈希变化。比较脚本报告点数、几何位置、高度场、尺度、旋转、透明度与球谐系数。当前审阅稿中仍有 13 个双目资产的精度与外观验收待完成，不能声称已通过全部数值阈值；详见[验证状态](06-validation.md)。需要原始字节时直接使用已校验的成品包。
