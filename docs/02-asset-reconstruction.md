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

### 自带 RGB-D 文件

在仓库根目录运行下列入口可重建清单以外的干坑。提供同一相机坐标系中已配准的 RGB、光轴深度和二值病害掩膜。深度支持单通道图像或数值 NPY，掩膜中非零像素表示病害。用采集标定替换示例内参，`--depth-scale` 表示一个深度单位对应多少米。主点默认取图像中心，可显式传入 `--cx`、`--cy`。

~~~bash
work/envs/assets/bin/python scripts/reconstruct-rgbd.py --id survey_001 \
  --rgb capture/color.jpg --depth capture/depth.npy --mask capture/mask.png \
  --fx 462 --fy 462 --depth-scale 0.001 --out runs/survey-001
~~~

程序沿用实拍 RGB-D 库的有效深度、道路平面与凹陷支持度检查，选用最大连通病害。结果写入新目录，包含资产清单、纹理 GLB、Gaussian PLY、路面场、预览和三份输入的指纹。仅需网格时加 `--mesh-only`。本次用独立文件重建 20250227_135438，GLB 与 PLY 均与历史交付逐字节相同。

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

## 两张实拍重建可动圆井盖

从井圈照片和盖面照片分别标四个平面点，顺序为左、上、右、下。在 `gen2hole/examples/photo-chamber/survey.json` 中可改照片、控制点、设计尺寸、转角和水平位移。示例采用 Zhang 等人论文图 5 的 D、B 两个裁切，两张照片来自不同设施。井圈和盖面纹理取自照片，遮挡面取自补充材质，井深及盖板厚度采用设计值。[论文与原图](https://www.mdpi.com/1424-8220/23/5/2676#fig_body_display_sensors-23-02676-f005)

~~~bash
python gen2hole/survey.py --survey gen2hole/examples/photo-chamber/survey.json --out runs/my-chamber --preview
python gen2hole/downstream/place.py --bundle runs/my-chamber/series --site 034 --group 93 --assets A93 --anchor-category pothole --front
work/envs/assets/bin/python scripts/check-chamber-family.py --run runs/my-chamber
~~~

`--anchor-category pothole` 让单件资产使用 034 的第一个锚点。同一井体切换状态时使用不同组号、相同锚点。结果中的井圈、井壁和井底共享相同顶点、UV 和材质，每次变换只移动井盖。落座态用作对照。GLB 以 Y 为上，3DGS 与插入场以 Z 为上，单位均为米。

### 使用标定深度测井

对齐到井圈原图的深度可用单通道 PNG 或 NPY。深度值表示相机光轴 Z。将下面的对象加入 survey JSON，路径相对于该 JSON。内参按原图分辨率填写。

~~~json
"rgbd": {
  "source_kind": "sensor_capture",
  "depth": "frame_z_mm.png",
  "metres_per_unit": 0.001,
  "intrinsics_px": [540.0, 532.0, 319.5, 239.5],
  "bottom_mask": "visible_floor.png"
}
~~~

程序用井圈外的有效深度拟合路面，沿相机射线将控制点投到路面上测直径。提供 `frame.aperture_landmarks_xy` 可测净孔直径。提供 `bottom_mask` 时，白色像素应只标可见井底，程序用这些返回点到路面的有符号距离中位数估计井深。没有相应观测的尺寸从设计文件读取。每次构建会保留相机内参、深度输入摘要、路面内点残差和有效井底覆盖率。

仓库内的受控标定样例包含斜视、量化、噪声、深度缺测与离群值。它只验证几何计算。实拍深度的设备误差与分割误差仍需在对应采集条件下测量。

~~~bash
work/envs/assets/bin/python scripts/benchmark-chamber-rgbd.py runs/my-calibration-input
python gen2hole/survey.py --survey runs/my-calibration-input/survey.json --out runs/my-calibration --preview
work/envs/assets/bin/python scripts/benchmark-chamber-rgbd.py runs/my-calibration-input --score-run runs/my-calibration
~~~

### 只有一张敞井照片

如果所有状态都标为 `"absent": true`，只需提供井圈图。可以省略 `lid`、`lid_diameter_m` 和 `lid_thickness_m`。若同一张照片同时拍到井圈与盖面，可以让两个部件指向同一文件，各自标四个控制点。

## 从实拍裂缝与像素掩膜重建

为近俯视路面准备正方形 RGB 裁切和同尺寸的二值掩膜，每个非零像素表示缝迹。在 `gen2hole/examples/photo-cracks` 中，`references/P102.png` 对应 `calibration/P102_outline.png`。配方 `masked_groove` 保留单像素分支与分离的缝迹，以局部像素宽度、设计跨度、侧壁坡度和深度上限构造 V 形截面。参数 `span`、`depth` 单位为米，`wall_slope` 表示垂直变化量与水平距离之比。

~~~bash
python gen2hole/gen2hole.py build --inputs gen2hole/examples/photo-cracks --out runs/my-cracks --preview P101 P102 P103
python gen2hole/downstream/place.py --bundle runs/my-cracks/series --site 034 --group 102 --assets P102 --front
work/envs/assets/bin/python scripts/check-crack-assets.py --run runs/my-cracks
~~~

图谱将它们显示为裂缝。现有道路回放继续使用三个传感器大类，裂缝共用路面凹陷标签，资产元数据中的 `surface_class=crack` 保留具体形态。实拍来源为 Rossini 等人的沥青路面数据，三份裁切的原始文件名、坐标及摘要保存在示例的 `sources.json` 中。[原始数据与授权](https://doi.org/10.5281/zenodo.20040434)

## 从观测恢复井盖姿态

如果还有同一设施开盖时的标定 RGB-D，可以在 survey JSON 添加 `pose_capture`。它使用自己的相机内参和路面井圈控制点，可以从另一机位拍摄。标记可见盖板**上表面**的二值掩膜和四个边缘点。井圈与盖面的控制点都按各自部件上的左、上、右、下顺序标注，方向要与用来提取纹理的对应部件一致。程序分别拟合局部路面和盖面，恢复盖面中心、完整三维旋转、盖板直径及倾角。

~~~json
{
  "image": "deployed_rgb.png",
  "depth": "deployed_z_mm.png",
  "lid_mask": "deployed_top.png",
  "metres_per_unit": 0.001,
  "intrinsics_px": [540.0, 532.0, 319.5, 239.5],
  "source_kind": "sensor_capture",
  "frame_landmarks_xy": [[166,232],[316,137],[457,238],[314,340]],
  "lid_landmarks_xy": [[245,246],[295,177],[367,207],[306,281]]
}
~~~

把这个对象放在 `pose_capture` 下，将对应状态设为 `{"id":"A98","label":"观测开盖","pose_from_capture":true}`。上述像素坐标仅演示字段格式。可运行前述标定样例生成器获得一套相互配准、可以直接复跑的 RGB、深度、掩膜和控制点。厚度与不可见底面继续使用设计值。

[受控回投图](figures/registered-lid-pose.jpg)使用带噪声和缺测的合成姿态观测。独立设定盖面倾角为 56°，重建得到 56.004°。用导出的 GLB 回投输入相机，盖面轮廓 IoU 为 0.9994，共同有效像素的深度绝对差中位数为 1.45 mm。它验证了同视角配准和几何实现。真实传感器采集的倾角精度仍需现场基准验证。

~~~bash
work/envs/streetcrafter/bin/python scripts/reproject-chamber-pose.py --run runs/my-calibration --asset A98 --figure runs/my-calibration/reprojection.jpg --report runs/my-calibration/reprojection.json
~~~
