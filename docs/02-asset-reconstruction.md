# 道路病害资产从输入到 GLB 与 3DGS

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

## 从自驾相机和原始激光点重建落座井盖

PandaSet 034 的 A104 使用前视第 13、11 帧的外圈和盖缝椭圆。原始激光点已在世界坐标系中，邻近路面确定局部基准面，盖面内的点确定落座高度。两张照片按投影面积加权融合纹理。A105 与 M104 共用该井体，分别演示掀开 55° 与缺盖。井口净径、盖厚、井壁、井深与盖背采用设计值。

从仓库根目录运行以下命令，两个输出目录应为空缺路径。

~~~bash
work/envs/streetcrafter/bin/python gen2hole/fit_road_cover.py \
  --raw tools/multiscene/data/raw --capture gen2hole/examples/pandaset-034/capture.json \
  --out runs/my-034-input
python gen2hole/survey.py --survey runs/my-034-input/survey.json --out runs/my-034 --preview
work/envs/streetcrafter/bin/python scripts/check-road-cover.py \
  --capture gen2hole/examples/pandaset-034/capture.json --raw tools/multiscene/data/raw \
  --run runs/my-034 --out runs/my-034/reprojection.json
~~~

采集清单将几何与纹理拟合帧、附加配准帧、留出帧分别列出。相机的局部像素平移按深度倒数拟合，仅用于这处静态目标，原始相机位姿保留。验收程序直接读取落座 GLB，对前后视留出照片做逐像素射线求交，再用独立原始扫描的盖面点核验网格高度。图谱可分别查看实拍、原始位姿及局部配准三个画面。[采集标注](../gen2hole/examples/pandaset-034/capture.json) · [拟合记录](../validation/pandaset-034-fit.json) · [导出网格核验](../validation/pandaset-034-reprojection.json)

PandaSet 109 的 A107、A108、M107 使用同一入口。金属组件外径约 0.347 米，浅色修补圈约 0.799 × 0.838 米。可在各拟合帧中增加 `apron_edge` 像素多边形，程序结合原始激光点估计修补圈的米制边界和五个径向高度。盖缝无法分辨时可用 `lid_radius_clearance_design_m` 指定径向间隙，图册单独标出设计盖面直径。错位盖板按盖背加强筋与路面、修补圈或底座的首次接触落位。用同一套命令替换采集清单和输出路径即可复建。[109 采集标注](../gen2hole/examples/pandaset-109/capture.json) · [留出帧、修补圈和盖背接触核验](../validation/pandaset-109-reprojection.json)

## 多帧实拍重建同一组坑洞

近景可以各自拍到病害的一部分。选择一个能与各帧通过坑外路面配准的机位作为 `anchor`，在每帧标出所有可见坑体。P112 的两张近景各记录一个坑，两帧都用全景的外围路面定位。导出的 GLB 含有两处独立坑缘，比较全景标注时，单帧重建的轮廓交并比（IoU）分别为 0.398、0.536，融合后为 0.898。[原图与回投](figures/double-pothole.jpg)

~~~bash
work/envs/streetcrafter/bin/python gen2hole/multiview_pits.py \
  --survey gen2hole/examples/multiview-potholes/survey.json --out runs/my-pit-survey --preview
~~~

`fit` 列出参与表面与纹理融合的帧，`holdout` 列出用于检验的帧。拍摄时应让坑外有可匹配的纹理与有效深度。程序在提取相机配准特征前遮去坑内照片，再用坑外三维点估计相对位姿。各拟合帧提供米制凹深、坑缘和颜色。正射纹理优先使用分辨率高、与融合表面深度相符的观测。未拍全的外围路面用邻近沥青纹理补齐，图谱的「实测覆盖次数」可查看各格点有几帧几何观测。

自带数据使用 `source: "files"`。复制[文件清单示例](../gen2hole/examples/multiview-potholes/files-example.json)，将路径改为相对该 JSON 的原图、深度和掩膜。每帧三者必须像素对齐，深度取相机光轴方向的 Z 值，可用单通道 PNG 或二维 NPY。毫米深度使用 `depth_scale: 0.001`，米使用 `1.0`。二值掩膜非零处标病害，`fx_fy_cx_cy` 使用该图实际针孔内参（像素），示例数值只演示格式。将上述命令的 `--survey` 替换为新清单即可重建。输出目录须为新路径。

每个 `series/assets/pothole/<ID>/` 含 GLB、Gaussian PLY 和米制表面。留出核验使用导出的 GLB、实际 UV 和内嵌底色，与每帧的轮廓、光轴深度、照片进行比较，另用每张拟合帧单独重建作为对照。结果与回投图保存在 `evaluation/`。PothRGBD 使用标称内参，双坑中心距的毫米级吻合描述同一套标定下的观测。FanStereo 使用原始 XYZ 点阵恢复投影，其中 P114 的拟合帧 03、04 机位很接近。两个小坑的可直接回投纹理覆盖约 79% 和 87%，三组较大坑均超过 99%。[五组图版](figures/multiview-potholes.jpg) · [六个留出帧的测量记录](../validation/multiview-potholes.json) · [留出像素隔离与异坑反例](../validation/multiview-pit-inputs.json)

从项目根目录运行以下命令，可在同一处 PandaSet 109 道路放置五组病害，并把 P113 放到第二段路。模拟时沿实拍激光束计算更换路面后的返回距离。核验同时检查模型摘要、原生检查点回读、资产尺度和可见相机帧。这六次放置共有 480 个案例帧，10 组前后对照中前视 6 组、后视 4 组。图谱另附每处放置的完整行车画面，黄色框和放大区来自同一帧。[迁入核验](../validation/multiview-pit-transfer.json)

~~~bash
for n in 110 111 112 113 114; do
  python gen2hole/downstream/place.py --bundle runs/my-pit-survey/series --site 109 --group "$n" --assets "P$n" --front
done
python gen2hole/downstream/place.py --bundle runs/my-pit-survey/series --site 034 --group 113 --assets P113 --front
python gen2hole/downstream/simulate.py --run runs/my-pit-survey --out runs/my-pit-survey/sensors
python scripts/check-pit-transfer.py --run runs/my-pit-survey --out runs/my-pit-survey/transfer.json
work/envs/streetcrafter/bin/python scripts/check-pit-inputs.py --run runs/my-pit-survey --out runs/my-pit-survey/input-checks.json
work/envs/assets/bin/python scripts/render-pit-survey.py --run runs/my-pit-survey --out runs/my-pit-survey/figures
~~~

## 从实拍盖板重建贯穿孔与断条

矩形雨水篦来自一张斜视照片。标注盖面四角、可见下角和 60 个孔口，在方形像素、给定主点和边缘正交的假设下恢复相机、盖板长宽比和厚宽比。参考宽度取 0.45 米，换算后盖长 0.601 米、厚 66.0 毫米。将完整 GLB 投回原机位，同一张照片中留出的 6 个下缘标记与模型底边距离中位数为 1.77 像素。绝对尺度随参考宽度改变。[原始照片与署名](https://www.dvidshub.net/image/9735953/374-ces-airmen-maintain-installation-readiness-through-storm-mitigating-flood) · [重建与道路图版](figures/perforated-grate.jpg)

铺石圆盖的 16 个孔、石材与盖面凹凸取自同一份摄影测量彩色图和相对位移图。来源公布的图幅宽 1.25 米，对应可见盖缝直径约 0.655 米。一处孔被树枝遮住，按两侧可见边界补全。垂直起伏幅度 7 毫米、盖厚 26 毫米与井深 0.75 米采用设计值。两组资产的隐藏井壁采用同一份混凝土材质。[原始摄影测量图](https://ambientcg.com/a/ManholeCover004) · [井壁材质](https://ambientcg.com/a/Concrete049) · [圆盖图版](figures/perforated-cobblestone.jpg)

自己的输入可以沿用两份示例清单。将原图和同分辨率的黑白 PNG 孔口掩膜放在一起，白色标出每个孔口内圈，黑色保留金属。斜视照片的四角从盖面左上起顺时针标注。正射圆盖标外接矩形四角。在 `perforations` 中给出井深、座圈重叠宽度、取纹理用的图像区域和井壁纹理。已知盖板尺寸时填写 `size_m`、`thickness_m`，矩形单照也可以填写 `edge_calibration`。对齐的单通道整数位移图和米制起伏跨度为可选输入。[矩形输入清单](../gen2hole/examples/perforated-covers/yokota/survey.json) · [圆盖输入清单](../gen2hole/examples/perforated-covers/cobblestone/survey.json)

```bash
python gen2hole/survey.py --survey gen2hole/examples/perforated-covers/yokota/survey.json --out work/grate-rebuild --preview
python gen2hole/survey.py --survey gen2hole/examples/perforated-covers/cobblestone/survey.json --out work/round-cover-rebuild --preview
work/envs/streetcrafter/bin/python scripts/check-perforated-covers.py --run work/grate-rebuild --out work/grate-check.json
```

孔缘约束三角化生成盖面、底面和每个孔的侧壁。角度与位移对整块盖板施加刚体变换，移位盖板按盖底与座圈或路面的首次接触落位。缺盖状态只导出井体。可用归一化盖面坐标中的 `break_polygon_uv` 指定缺损格条。对 A115 和 A117 的实际 GLB 作同位置射线比较，移除约 5.0 平方厘米金属，两个独立孔连成一个。六个有盖状态的网格均通过封闭性、连通性和孔数检查。完整矩形、圆盖的孔口回投与输入标注的 IoU 分别为 0.997、0.960。

在 PandaSet 109，两组资产的全部状态分别放到相同的道路位置。A117 和 A120 还迁到 034。共 10 次原生放置、20 组前后视相机对照和 800 个雷达案例帧，复用两段路的 160 帧原始扫描。断条、掀盖、移位、缺盖都属于由同一来源派生的状态。核验文件逐状态记录 GLB、Gaussian PLY 摘要、同源井体和回波命中部件。[雨水篦核验](../validation/yokota-grate.json) · [铺石圆盖核验](../validation/cobble-cover.json)

图谱的道路菜单可以直接比较落座和异常状态。两幅图共享相机、帧、曝光来源和像素坐标，黄色标出可见部件改变或光轴深度差超过 1 厘米的像素。对 109 的断条篦子，前视第 29 帧有 10 个这样的像素，后视第 40 帧有 3 个。沿该路段 80 帧原始激光束逐行比较，两束的模拟距离差超过 1 厘米，分别约为 24.6 和 74.3 毫米。断条回波俯视图把两个状态的命中点投在同一张盖面纹理上。

## 两张彩照重建坑底起伏

从同一处病害的两个不同位置拍照，两张都要能看到坑底和有纹理的周边路面。在主照片中标记同一路面上的两个点，提供它们的实际间距、相机内参和两张图的病害掩膜。主照片决定纹理和可恢复的范围，第二张照片提供视差。相同位置仅转动相机无法恢复凹深。四个发布样例的尺度各来自坑外两个小区的参考距离。其中三组使用 PothRGBD 标称内参，另一组使用 Fan 数据的相机投影标定。构建入口仅读取 RGB、标注和尺度清单。[输入清单](../gen2hole/examples/photo-parallax/survey.json)

算法匹配坑外路面特征以估计平面单应，用坑内和坑沿特征区分候选相机姿态。随后在各候选米制高度上比较两幅图的局部灰度纹理，交换机位重复匹配，剔除单向匹配与歧义像素。图谱用绿色标出有双向支持的网格，橙色标出插值区域。可在“米制剖面对照”中看到对应照片上的截线，以及同一位置的模型高度和传感器高度。

| ID | 实拍形态 | 模型 P95 凹深 | 坑内双向匹配覆盖 |
| --- | --- | ---: | ---: |
| P121 | 四边破损的方形浅坑 | 36.1 毫米 | 64% |
| P122 | 粗骨料裸露的深坑 | 104.2 毫米 | 68% |
| P123 | 弧形陡边坑 | 73.0 毫米 | 88% |
| P124 | 斑块状剥落坑 | 43.8 毫米 | 58% |

用自己的彩照时，复制本地文件示例，替换图片、掩膜、内参、尺度点和间距。二值掩膜支持 0/1 和 0/255，白色为病害。两部相机可分别在各自的帧记录中填写 `fx_fy_cx_cy`。输入应为已去畸变的针孔图像。可用 `search_depth_m` 修改候选凹深上限，单位为米。对示例的四组资产，用仅包含彩照和掩膜的独立目录复跑，GLB 和路面场与发布结果逐字节一致。[自采文件示例](../gen2hole/examples/photo-parallax/files-example.json) · [独立复跑记录](../validation/photo-parallax-inputs.json)

```bash
work/envs/streetcrafter/bin/python gen2hole/multiview_pits.py --survey gen2hole/examples/photo-parallax/survey.json --out work/rgb-pair-demo --preview
work/envs/streetcrafter/bin/python scripts/check-photo-parallax.py --run work/rgb-pair-demo --check gen2hole/examples/photo-parallax/check.json --compare-run runs/multiview-pit-survey
python gen2hole/downstream/place.py --bundle work/rgb-pair-demo/series --site 109 --group 121 --assets P121 --front
```

核验时从导出的 GLB 向原相机逐像素求交，与原始传感器深度比较。光轴深度误差包含尺度与位姿偏差。相对路面高度误差则分别以模型周边路面和传感器拟合的周边路面为零。P124 还在完成建模后仅用坑外路面纹理定位三个留出机位，坑缘 IoU 为 0.857–0.940，坑内光轴深度绝对误差中位数为 1.7–4.4 毫米。其余三件在输入照片的机位用留出的深度检查，P122 两个机位的光轴误差中位数分别为 20.6 和 18.1 毫米。P110 与 P124 记录同一处剥落，可直接比较含深度和仅用彩照的两种重建。[逐机位核验](../validation/photo-parallax.json) · [留出机位图版](figures/photo-parallax-heldout.jpg)

四件资产按重建尺寸放到 PandaSet 109，方形浅坑和骨料深坑还放到了 034。六个场景重载后逐像素复现迁入画面，在两段各 80 帧的原始扫描上共回放 480 个案例帧。小型剥落坑 P124 只有前视有效，其两部激光雷达分别得到 0 和 5 个模拟回波。其余五次放置均有前后视对照和两部雷达的回波。[道路迁移图版](figures/photo-parallax.jpg) · [分场景记录](../validation/photo-parallax-transfer.json) · [PData 照片来源](https://github.com/ymyurdakul/PData) · [Fan 双目照片来源](https://github.com/ruirangerfan/stereo_pothole_datasets)

从 P110 或 P124 打开第 10、13 帧的「双照片与多帧双目 · 同像素剖面」，可比较同一位置的两个模型。两个机位对两种重建都留出，截线上的高度统一以该帧实测路面为零。第 10 帧有 18,865 个双方均可见的坑内像素，P124 与 P110 的光轴深度绝对误差中位数分别为 4.35 和 1.22 毫米。第 13 帧共有 16,655 个这样的像素，对应误差为 4.23 和 3.29 毫米。P124 在建模时使用预先量好的坑外参考长度，P110 在建模时使用第 09、11、12 帧的双目深度。

## 井盖断片与井内剖视

同一块铸铁圆盖可以保留在座圈上，也可以只留下一部分。A125 使用完整盖板，A126 缺失断片，A127 的断片平放在井底，A128 的断片放在井旁路面。原始摄影测量资料记录了完整盖面。14 个孔由配准的彩色和相对位移图辨认，生成盖面的实心部分、通孔和孔壁。来源未提供绝对尺度，示例将盖径定为 0.62 米、盖厚定为 28 毫米、井深定为 0.54 米。断线、断片方位和不可见的井内结构由设计给定。[原始盖面](https://ambientcg.com/a/ManholeCover009) · [四态对照](figures/fractured-iron-cover.jpg)

在输入的正射照片坐标中，`fracture_path_uv` 为从上向下的一条折线，原点在左上角，纵坐标严格递增，首尾延伸到盖面范围之外。折线右侧是可移走的部分。状态中的 `fragment.location` 可填 `missing`、`floor` 或 `road`。放下碎片时，`center_xy_m` 指定其投影面积中心的本地米制坐标，`yaw_deg` 给出绕竖轴的转角。构建时检查碎片能否放入净井口、井旁碎片是否避开固定座圈。断面两侧保留各自的实体侧壁，导出的 GLB 和 Gaussian PLY 均包含实际留在场景中的部件。[完整输入配置](../gen2hole/examples/perforated-covers/split-iron/survey.json)

完整盖面的 18,809 个三角形中，13,863 个留在座圈上，另 4,946 个组成碎片。把碎片按刚体逆变换放回后，两部分在原盖面上的三角形刚好互补。三角形纹理坐标完全一致。两个碎块的封闭性、接触支承和体积分别从已导出的 GLB 核验。按均匀密度计算的重心投影均在接触点的凸包内。原盖的 14 个贯穿孔中，10 个完整保留在余盖上、2 个完整保留在碎片上，另 2 个被断线穿过，成为断面边缘缺口。图谱和「交互三维」都可切换井体剖视。剖视 GLB 只裁切近侧井壁、座圈和路面。逐件验证了盖板、断片、井底的三角形及纹理坐标均与原资产一致，全部部件的底色贴图逐像素一致。米制截面则直接由原网格三角形与竖直平面求交得到。[实体与支承核验](../validation/split-iron-cover.json)

在 PandaSet 109 和 034 上完成六次原尺寸放置。两段路各 80 帧原始扫描共回放 480 个案例帧。相机在相同世界位置、标定参数、曝光帧和裁切上比较状态。在 109 的前视第 38 帧和后视第 51 帧，断片从缺失改为躺在井底，部件或光轴深度差超过 1 厘米的像素均为零。断片从缺失改放在井旁，两帧分别改变 900 和 209 个像素，整个路段有 315 束模拟回波距离差超过 1 厘米。两段道路的前后相机中，井底碎片的单独投影覆盖过像素，但每帧都被完整井体或余盖遮挡。图谱的道路选项可查看逐帧可见性，以及井旁碎片覆盖最多的原始相机照片。[自驾对照图](figures/fractured-cover-roads.jpg) · [逐帧计数](../validation/split-iron-visibility.json)

原始 4K 颜色、位移和路面贴图存于本机素材库。可用以下第一条命令按 SHA-256 核对原始贴图，重制示例的四张输入图片。其余命令从发布的输入重建资产、剖视模型和几何检查。

```bash
work/envs/streetcrafter/bin/python gen2hole/examples/perforated-covers/split-iron/prepare.py --inputs work/reconstruction-inputs/pbr --out work/split-source
python gen2hole/survey.py --survey gen2hole/examples/perforated-covers/split-iron/survey.json --out work/split-cover-demo --preview
work/envs/assets/bin/python scripts/render-cover-cutaways.py --run work/split-cover-demo
work/envs/streetcrafter/bin/python scripts/check-perforated-covers.py --run work/split-cover-demo --out work/split-cover-geometry.json
```

完整的道路示例保存在 `runs/split-iron-cover`。用 `gen2hole/downstream/place.py --front` 将不同状态放入同一路段，再用 `gen2hole/downstream/simulate.py` 回放。断片家族会额外输出同帧前后视相机画面到 `sensors/matched_camera`。给几何检查加上 `--with-sensors` 可复核状态间的相机与回波差异。用 `scripts/check-fragment-visibility.py --run runs/split-iron-cover --out work/split-visibility.json` 可重新统计整个行车序列中碎片的可见像素。

## 实拍坑底上的积水

P129–P132 复用 P122 的双照片粗骨料深坑。P133 复用 P113 的多帧 RGB-D 双坑，在其中一个坑里加入 75 毫升水。每个状态都有独立水面三角网格、带实拍纹理的完整坑底、3DGS 和米制水深场。在图谱里选择「交互三维」后可以排空、恢复积水或切换同一坑的其他状态。展开高程剖面可读取首个交点和下方坑底，选中道路的「干湿对照」可比较同一相机像素。[坑底、岸线与剖面](figures/standing-water.jpg) · [道路相机对照](figures/standing-water-roads.jpg)

| 资产 | 坑底 | PandaSet 路段 / 局部坡度 | 加入 | 留存 | 溢出 | 最大铅直水深 |
|---|---|---|---:|---:|---:|---:|
| P129 | P122 | 109 / 5.1% | 1 L | 1 L | 0 L | 35.1 mm |
| P130 | P122 | 109 / 5.1% | 5 L | 5 L | 0 L | 66.6 mm |
| P131 | P122 | 034 / 14.2% | 5 L | 5 L | 0 L | 84.6 mm |
| P132 | P122 | 034 / 14.2% | 9 L | 6.444 L | 2.556 L | 93.9 mm |
| P133 | P113 | 109 / 5.1% | 0.075 L | 0.075 L | 0 L | 7.5 mm |

建模时读取目标锚点的世界竖直方向，沿原始网格边从注水点向低处寻找洼地。算法沿三角形边检查积水区域是否与路面片的外边界连通，求出首次外排的高程。对水位以下连通的三角形逐个沿岸线裁切，积分坑底与水面间的体积，再由注水量反求水位。超出容量的水计入溢出量。P133 的两个坑分别覆盖 25,847 和 22,558 个重建网格顶点，较大坑保持干燥；较小坑有 3,879 个病害顶点被水覆盖，岸线另延伸至周边 109 个路面顶点。[五态几何与观测覆盖核验](../validation/standing-water.json)

干坑来自照片和深度观测。P122 用两幅 RGB 和一段已知路面长度恢复尺度，输入图像所对应的深度留作复核，两帧内的光轴深度绝对误差中位数分别为 20.6 和 18.1 毫米。P113 在未参与拟合的全景帧上，该值为 2.7 毫米。湿态的容量与水深是基于重建曲面的计算值。坑底材质与三角形在所有干湿 GLB 间完全相同；水面外观使用深度衰减后的实拍颜色和低粗糙度材质。原始资料拍摄的是干坑，湿态外观尚无对应实拍核验。

五个状态均按原尺寸迁入目标路段。两段原始双雷达扫描各 80 帧，共回放 400 个案例帧。P130 在 109 前视第 29 帧占 1,612 个水面像素，相对同机位干坑，入水区域的光轴距离缩短中位数为 98.5 毫米。相同原始激光束中有 239 束从坑底改为先交水面，沿射线距离缩短中位数为 97.8 毫米。这里统计的是理想几何交点。真实水体的镜面反射、透射和激光丢点仍缺少湿态观测。[相机与雷达干湿对照](../validation/standing-water-observations.json) · [原生场景重载与扫描核验](../validation/standing-water-transfer.json)

从项目根目录执行下面的命令。样例用已重建的干坑和目标道路的放置记录，支持为其他已导出的实拍高度场指定注水量、坑内注水点和场景竖轴。更换道路坡度时重新求解水面。输入多洼地在鞍点发生跃变、目标水量无法形成单个稳定连通水面时，构建会给出诊断。当前模型输出静止水面和累计溢出量。

```bash
work/envs/assets/bin/python gen2hole/examples/standing-water/check_solver.py
python gen2hole/fill_pits.py --survey gen2hole/examples/standing-water/survey.json --out work/standing-water-demo --preview
work/envs/streetcrafter/bin/python scripts/check-standing-water.py --run work/standing-water-demo --out work/standing-water-demo/evaluation/measurements.json
```

配方中的 `seed_uv` 使用正射图归一化坐标，原点在左上。`anchors` 可以指定相对于 `--runs` 的场景记录，也可以给出本地坐标中的单位竖轴，例如水平地面用 `[0,0,1]`。完整的几何检查从湿态 GLB 全部水面三角形的中心向干坑 GLB 发射射线，重新积分水量，同时检查岸线的每个边界顶点是否贴合原坑底。另起新目录完整重建五态，逐件比对了 GLB、Gaussian PLY、水深场和逐点材料标签，四类文件的 SHA-256 均可复现。[输入配方](../gen2hole/examples/standing-water/survey.json) · [独立检查代码](../scripts/check-standing-water.py)

## 从道路扫描重建裂缝

P134–P137 来自两份 CC0 道路扫描。混凝土整片和加密局部保留同一批碎板的空间关系。沥青纵缝的两个版本共享原图、网格平面坐标和裂缝掩膜，各自采用一种竖向尺度。项目从作者的 Blender 材质读取连接到输出节点的位移比例与中点，并按来源给出的铺面宽度把像素换算为米。[沥青原图与材质](https://polyhaven.com/a/asphalt_02) · [混凝土原图与材质](https://polyhaven.com/a/cracked_concrete) · [输入文件指纹及裁切](../gen2hole/examples/scanned-cracks/survey.json)

| 资产 | 取景边长 | 网格间距 | 示例截线凹深 | 留出缝内与作者位移的中位绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| P134 沥青纵缝，作者材质尺度 | 1.125 m | 2.93 mm | 22.23 mm | 0.496 mm |
| P137 同一纵缝，法线估计尺度 | 1.125 m | 2.93 mm | 4.84 mm | 4.147 mm |
| P135 混凝土网裂整片 | 1.750 m | 3.91 mm | 2.19 mm | 0.107 mm |
| P136 网裂局部加密 | 0.750 m | 1.95 mm | 2.17 mm | 0.040 mm |

程序只用颜色找暗缝，从稀疏位移样本网格恢复凹陷、断边和周边路面，外圈渐变接入宿主路面。原图的纹理坐标与导出 GLB 逐点核对，解码后四个模型的 RGB 均与原裁切逐像素相同。PLY 从各自带纹理的网格表面采样。图谱里的「无纹理网格」和可旋转三维模型里的「查看无纹理网格」都能单独检查几何。[资产图版](figures/scanned-cracks.jpg)

核验向导出 GLB 发射竖直射线，与未进入建模格点的源位移像素比较，排除与宿主接合的渐变外圈。在 P136 局部的同一组 62,055 个缝内像素上，P135 和 P136 的误差中位数分别为 0.109 和 0.040 mm。P134 的材质尺度能较好复现作者位移，P137 取其 0.218 倍，更符合该材质的法线方向。尺度用交错的图块估计，另外 4,152 个缝内像素的法线夹角中位数从 34.44° 降到 9.04°。这些误差描述对源扫描的复现。沥青裂缝的真实深度仍需独立现场测量。[逐项几何与法线核验](../validation/scanned-cracks.json)

四件资产在 PandaSet 034 和 109 上共做七次原尺寸放置，每段道路使用固定位置。载入场景检查点后的图像逐像素一致，两部激光雷达共回放 560 个放置帧。对 14 个前后视机位，将源图板块身份与相机可见网格逐像素对应。P135 源图中有 12 个被裂缝完全包围且面积至少 25 cm² 的板块，路段 034 前、后视各保留 1 个闭合轮廓，109 前视保留 3 个、后视 1 个。计数要求区域至少 3 个像素、其中至少 80% 属于同一源板块，并覆盖该板块可见投影的至少一半。这衡量给定机位下的几何可分辨程度。[相机原色与几何标记](figures/scanned-crack-roads.jpg) · [逐机位与雷达命中](../validation/scanned-crack-visibility.json) · [放置身份与重载核验](../validation/scanned-crack-transfer.json)

P134 和 P137 在每个路段的位置、相机姿态、帧和实拍底图逐项对齐。在图谱中选「纵缝深度对照」可以切换两套网格的原色画面。对同一批原始射线回放，两种深度让 034 的 170 束、109 的 77 束模拟回波量程相差超过 1 cm。相机光轴深度和雷达斜距还受视角及遮挡影响。[同一截线与两个道路机位的深度差异](figures/asphalt-crack-depth.jpg)

从仓库根目录运行以下命令。每次构建选择新的输入准备目录和运行目录。下列路段与资产组合可复现图版中的七次放置。

~~~bash
src=work/reconstruction-inputs/scanned-pavement
prep=work/prepared-crack-demo
run=work/scanned-crack-demo
python gen2hole/examples/scanned-cracks/download.py --out "$src"
work/envs/streetcrafter/bin/python gen2hole/examples/scanned-cracks/prepare.py --inputs "$src" --out "$prep"
python gen2hole/gen2hole.py build --inputs "$prep" --preview --out "$run" P134 P135 P136 P137
work/envs/streetcrafter/bin/python scripts/check-scanned-cracks.py --run "$run" --inputs "$src" --out "$run/evaluation/measurements.json"
work/envs/assets/bin/python scripts/render-scanned-mesh.py --run "$run" P134 P135 P136 P137
for site in 034 109; do
  for id in P134 P135 P136 P137; do
    if [ "$site/$id" = "034/P136" ]; then continue; fi
    python gen2hole/downstream/place.py --bundle "$run/series" --site "$site" --group "${id#P}" --assets "$id" --front
  done
done
python gen2hole/downstream/simulate.py --run "$run" --out "$run/sensors"
work/envs/streetcrafter/bin/python scripts/check-crack-visibility.py --run "$run" --out "$run/evaluation/visibility.json"
python scripts/check-pit-transfer.py --run "$run" --out "$run/evaluation/transfer.json"
work/envs/streetcrafter/bin/python scripts/render-scanned-cracks.py --run "$run" --out "$run/figures"
~~~

## 原始高斯资产与网格的同机位对照

图谱收录的 1,384 个 PLY 共有 191,499,749 个高斯。每件资产都有一张从自身 PLY 渲染的离线预览。在图谱中点「看高斯渲染」即可查看，点「3DGS 与网格对照」可绕任意方向检查原始高斯，或用分界线在原位切换到 GLB。对照页下方列出同一资产已有的道路观测。[独立资产与实景对照](figures/gaussian-asset-comparison.jpg)

浏览器按 PLY 中的三轴对数尺度、WXYZ 四元数和不透明度恢复各向异性高斯。渲染使用完整点集和原始一阶球谐颜色，按当前透视机位投影协方差、补偿屏幕滤波造成的不透明度变化，再按深度合成。能写入半浮点缓冲区的设备使用 16 位浮点颜色累积。GPU 与浏览器共享米制尺度，原始 PLY 为 Z 向上，GLB 为 Y 向上。[gsplat 光栅化说明](https://docs.gsplat.studio/main/apis/rasterization.html)

直接双击本地页面可以查看预览图。要旋转高斯，可在同一页选择本地 PLY，同时选择对应 GLB 即可滑动对照。页面按 SHA-256 识别与核对文件，识别后加载该资产的来源和道路记录。使用本地预览服务时可直接按图谱 ID 打开模型。

~~~bash
python scripts/serve-asset-atlas.py
~~~

在浏览器打开 [掀开圆盖的高斯与网格](http://127.0.0.1:8765/tools/asset_atlas/splats.html#A119) 或 [矩形敞井](http://127.0.0.1:8765/tools/asset_atlas/splats.html#M11)。两种表示使用同一相机进行旋转、缩放和对照，共用局部坐标的井盖状态可保持机位切换。

### 如何读深度差

「核对高斯与网格的深度」提供固定机位的网格、高斯和像素级绝对差。网格深度取射线首次命中的正面。高斯深度为沿光线的透明度加权期望。图中计入不透明度大于 85%、共同命中且远离网格轮廓的像素。可将统计范围切换到井圈、井壁、盖板、水面或周边路面。数字描述两种资产表示的一致程度。[逐机位与逐部件核验](../validation/atlas-gaussians.json)

浏览器解码器已逐个读入并排序全库 1,384 个 PLY。另选覆盖 11 种来源或重建方法的 13 件资产，在 23 个机位逐像素比较浏览器画面与 CUDA 参考渲染。以 255 为满值，每幅画面不透明区域的 RGB 平均绝对差均小于 0.16。[全库 PLY 解码记录](../validation/atlas-gaussian-ply.json) · [逐机位颜色对照](../validation/atlas-gaussians.json)

矩形敞井 M11 的低角度画面中，井圈、井壁和周边路面的差异相距很大。GLB 的路面网格遮住地下结构，独立高斯资产的路面边缘逐渐透明，部分像素仍收到地下井壁的颜色与深度。迁入画面使用宿主路面和遮挡几何。可对照俯视、斜视和低角度三组图，区分部件本身和道路过渡区的贡献。[敞井深度图版](figures/gaussian-shaft-depth.jpg)

### 重新渲染和核验

运行 `render-atlas-gaussians.py` 可生成全库预览与相机清单，也可用 `--ids M11 A119` 选取一组资产。程序核对源 PLY 和预览指纹，已有且匹配的预览可复用。在仓库根目录执行以下命令。

~~~bash
python scripts/build-asset-atlas.py
work/envs/streetcrafter/bin/python scripts/render-atlas-gaussians.py
python scripts/build-asset-atlas.py
node scripts/check-gaussian-ply.cjs validation/atlas-gaussian-ply.json
~~~

交互核验使用 Chrome、Node.js 和 ws。先启动上面的本地预览服务，再用下列命令记录实际浏览器的两个渲染结果。CUDA 独立读取原始 PLY，Open3D 独立读取原始 GLB 并对每个像素求交。完成核验后重建图谱索引即可展示这些证据。

~~~bash
node scripts/capture-atlas-gaussians.cjs runs/atlas-gaussian-checks/browser A119 M11 P135 P130 P112 P121 P110 A104 P90 rgbd_pothole_20250227_135438 stereo_pothole_fan_1_03 model_raised_rectangular_removed_side_chip_seal A128
work/envs/streetcrafter/bin/python scripts/check-atlas-gaussians.py --captures runs/atlas-gaussian-checks/browser --out validation/atlas-gaussians.json
work/envs/streetcrafter/bin/python scripts/render-gaussian-audit.py
python scripts/build-asset-atlas.py
~~~

### 井类资产的开孔路面

308 件带井体的资产可加铺一层沥青。打开 M11 后拖动旋转，可观察矩形井口内的结构和路面遮挡。也可切换 A119 掀盖、A128 井旁断片或 A117 断条雨水篦。勾选「标出地下 7.5 cm 以下的结构」会把对应高斯染为洋红色，盖板上的孔和篦条间的可见部分随相机移动。页面底部保留同编号资产的实景道路观测。[矩形敞井](http://127.0.0.1:8765/tools/asset_atlas/splats.html?road=1#M11) · [掀盖](http://127.0.0.1:8765/tools/asset_atlas/splats.html?road=1#A119) · [六例对照](figures/gaussian-road-openings.jpg)

每个开孔取自该资产 GLB 铺面三角形的内边界。导出程序焊接纹理接缝并以 0.25 毫米容差简化轮廓，浏览器按 GLB 指纹选择轮廓。观察者位于路面上方时，浏览器逐像素计算视线与水平面的交点，仅允许地下高斯通过开孔显现。路面比资产的标称零高程低 3 毫米，以保留铺面颗粒。示范路面按 3 米幅宽平铺 Rob Tuytel 的 CC0 沥青扫描材质。PandaSet 的车载视图使用各自路段实际的道路、放置位置和相机。[沥青来源与指纹](../gen2hole/examples/scanned-cracks/survey.json)

在 10 件资产的 30 个机位，用原始 GLB 铺面三角形独立求交，并排除两像素边界。地下染色前后，任一 RGB 通道相差超过 8 级计作一个响应像素。全部机位累计的井口外响应从 412,881 个降到 0。井口内纳入核验的 1,174,743 个像素在铺路前后逐通道相同。这项检查衡量开孔遮挡和透视一致性。[逐机位核验记录](../validation/atlas-host-road.json)

在仓库根目录运行以下命令可复现。浏览器捕获默认使用 SwiftShader，具备原生 OpenGL 的 Linux 工作站可设置 `ATLAS_CHROME_ANGLE=gl`。

~~~bash
work/envs/streetcrafter/bin/python scripts/build-gaussian-road.py
node scripts/capture-atlas-gaussians.cjs runs/atlas-host-road --road M11 M10 A119 A128 A125 model_raised_rectangular_removed_side_chip_seal A104 A105 M104 A117
work/envs/streetcrafter/bin/python scripts/check-gaussian-road.py
work/envs/streetcrafter/bin/python scripts/render-gaussian-road.py
python scripts/build-asset-atlas.py
~~~
