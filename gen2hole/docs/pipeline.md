# 生成参考图到米制资产

三维重建的观测量来自生成图，包括破损轮廓、井口内外边界、盖板投影以及可见纹理。整幅图覆盖多少米、坑洞尺度、井深、盖厚和不可见结构来自设计配方。车载相机和 LiDAR 实测的是置入病害以前的道路。三类信息分别保存在校准、资产元数据和场景记录中。[资产清单](../catalog/assets.json) · [PandaSet 034 放置记录](../runs/demo/series/scenes/034_13/scene.json)

~~~mermaid
flowchart TD
    D["病害设计、米制尺寸、提示词"] --> I["GPT Image 垂直俯视主图"]
    I --> P["坑洞轮廓、相对逆深度与路面基准"]
    I --> M["井框、开口、盖板投影及铰点"]
    P --> T["三角网格与分部件材质"]
    M --> T
    S["沥青断面、盖板背面和敞井内壁参考"] --> T
    T --> G["PBR GLB"]
    T --> H["surface.npz 米制切口与过渡权重"]
    T --> A["表面积采样、可见性和一阶球谐 → 3DGS PLY"]
    H --> R["LiDAR 路面拟合、切除宿主、按 1∶1 迁入"]
    A --> R
    R --> C["标定相机合成与类别掩膜"]
    R --> L["按车载射线与已放置网格求交"]
    G --> L
    L --> Q["首个交点、回波增量与高度剖面"]
~~~

## 逐阶段输入输出

| 阶段 | 读入 | 方法与输出 | 主要实现 |
|---|---|---|---|
| 1. 设计与生图 | 类别、形态、材料、姿态、设计尺寸、提示词 | 每件独立生成一张正俯视方图，完整保留病害和周边完好路面。共享断面和背面补图供不可见区域取材 | [首批完整提示词](../inputs/pilot/references/prompts.json) · [扩充批设计与提示词](../inputs/series/calibration/designs.json) |
| 2. 坑洞拟合 | 主图、设计 `span/depth`、可选轮廓与水线标记 | 双尺度单目逆深度、路面基准、窄带 GrabCut，得到 `*_depth.npz`、`*_outline.png` 和适用时的 `*_water.png` | [深度推理](../recipes/common/estimate_depth.py) · [首批标注细化](../recipes/pilot/trace_masks.py) · [批量拟合](../recipes/series/fit_images.py) |
| 3. 井框与盖板拟合 | 主图、米制外框、井深，必要时加入敞井参考 | 在 480 × 480 的归一化图像坐标中拟合内外框、椭圆或四角，保存井圈、盖板姿态、遮挡区取材信息 | [人工修订](../inputs/series/calibration/overrides.json) · [最终拟合](../inputs/series/calibration/layout.json) |
| 4. 建网格与材质 | 校准、原图和共享补图 | 坑沿、井壁、井底、盖板正反面等以实际几何表达。RGB 投影和材质贴图写入 GLB，同时导出局部路面场与网格采样缓存 | [建模入口](../recipes/common/build.py) · [坑洞](../recipes/common/pavement.py) · [扩充批井类](../recipes/series/chambers.py) · [材质与坐标变换](../recipes/series/mesh_io.py) |
| 5. 导出高斯 | 与 GLB 共源的三角面、UV、材质、`surface.npz` | 按面积和部件重要性采样，计算切向尺度、旋转、透明度、遮蔽和一阶 SH 颜色，产出 `asset.ply` 与逐点材质标签 | [表面采样](../recipes/common/export_splats.py) · [球谐拟合和 PLY 格式](../recipes/common/gaussian_sh.py) |
| 6. 道路迁入 | PLY、切口、部件标签、原生道路检查点、标注道路点与相机标定 | 估计局部基底，切除原路面、补面并匹配道路配色，按 1∶1 写入新检查点。用原轨迹回放，合成前视 RGB、掩膜和包围框 | [道路入口](../downstream/place.py) · [迁入适配](../downstream/native_entry.py) · [前视合成](../downstream/sensor_views.py) |
| 7. 传感器与验证 | 已放置的网格和高斯、原始扫描和相机数据 | 输出前后相机裁切、光轴深度、世界法线、部件、实例、激光雷达增量和厘米级剖面。独立求交复核网格与回波 | [仿真入口](../downstream/simulate.py) · [射线命中](../downstream/sensors/build_returns.py) · [独立 BVH 复核](../downstream/sensors/verify_mesh.py) |

## 三类几何如何得到

### 坑洞 P

Depth Anything V2 Small 对原图及其水平翻转分别推理，在 518 和 770 两个分辨率上估计相对逆深度，以 0.45 和 0.55 加权。坑外完好路面约束一个二次趋势面。移除此趋势后，在轮廓内按设计深度缩放，结合到坑沿的距离和纹理细节生成高程。陡壁使用单独展开的截面纹理。可见积水有独立的平面几何与低粗糙度材质。以 P23 为例，整张图设为 1.423 m，拟合出的坑口约 1.129 × 0.845 m，设计深 0.131 m，成品最低点为 −0.137 m。[P23 设计与拟合](../inputs/series/calibration/P23_fit.json) · [P23 成品元数据](../published/道路病害图生三维_新增100/assets/pothole/P23/metadata.json)

### 缺盖井 M

图像给出内外框像素位置，设计的外框宽长把像素换算成米。沿轮廓生成井框、承口、向下的井壁和井底，分别映射井圈、衬壁及沉积物纹理。模型的 `width_m/length_m` 描述净开口，`frame_width_m/frame_length_m` 描述外框，切口文件的 `width/length` 则覆盖带过渡带的整个局部地块。以 M13 为例，外框为 0.918 × 0.918 m，净开口为 0.783 × 0.774 m，设计井深为 0.971 m。[M13 原图](../inputs/series/references/M13.png) · [M13 元数据](../runs/demo/series/assets/missing/M13/metadata.json)

### 异常井盖 A

侧铰圆盖结合椭圆投影、井框尺度和铰点估算倾角。方盖用四角相对于铰边的有向投影恢复姿态。脱座盖用盖心与开口中心的水平偏移定位。盖面、背面、边缘、厚度与适用的铰链均有实体。A13 的提示词要求约 55°，按实际生成图拟合后是 **74.23°**。其盖厚 0.027 m、井深 1.03 m 采用设计值，遮挡内壁使用 M13 参考图。[A13 图像标记](../inputs/series/calibration/A13_fit.json) · [A13 元数据](../runs/demo/series/assets/raised/A13/metadata.json) · [反向几何视图](../runs/demo/series/views/A13_reverse.jpg)

### 外观与 3DGS

可见表面使用源图颜色，正常路面取样延拓边界。GLB 内嵌底色、切线空间法线、金属度与粗糙度贴图。网格以固定 seed 的 Sobol 点按带权三角面积采样，井圈、盖面、断边获得更高密度。高斯的两个轴沿切平面、第三轴沿法线。Open3D 对三角网格求环境遮蔽与直射光可见性，48 个球面方向用于拟合一阶 SH。高斯位置受网格表面约束，颜色保留部分参考图照明。[采样与几何可见性](../recipes/common/visibility.py) · [SH 拟合](../recipes/common/gaussian_sh.py)

## 文件、坐标和导入

| 单件文件 | 如何使用 |
|---|---|
| `asset.glb` | 米制、glTF Y-up。含多层几何与嵌入材质，适合网格引擎、检查盖板背面、求交。 |
| `asset.ply` | 资产局部 Z-up 的三维高斯。含位置、DC 和其余 SH 系数、log 尺度、四元数及 logit 透明度。用支持 Gaussian Splatting 的渲染器打开。 |
| `splat_material.npz` | `part`、`road`、`ambient_visibility` 与 PLY 高斯逐点同序，`part_names` 对应本资产的局部部件表。 |
| `surface.npz` | `width/length/height` 单位米，`alpha` 为融合权重，`defect_mask` 为病害区域。井类有 `opening_mask`，坑洞有 `water_mask`。行从 +Y 到 −Y，列从 −X 到 +X。 |
| `metadata.json` | ID、类别、源图 SHA-256、设计深度与网格实际高低点、部件三角形数、高斯数及坐标约定。`assets/asset_manifest.json` 给出本次构建的相对路径。 |

资产原点在病害或井口中心的局部路面，Z = 0。行向量按 `world = local @ basis_columns.T + center_world` 变换。将资产从 GLB 引擎坐标转换回此局部坐标时使用 `(x, −z, y)`。宿主路面要按切口挖空，否则井下几何会被原路面遮住。高度场保存每个平面位置的单个值，井口上方的盖板及下方井腔通过 GLB 的多层表面查询。[坐标与逐次放置基底](../runs/demo/series/scenes/034_13/scene.json)

## 道路和传感器

原交付的三条 PandaSet 路段为 034、109、110。每个资产各放三次，109 件合计 327 次放置、111 个组合案例。01–35 组各包含同号的 P/M/A 三件，36 组用 P36 与 M36，37 组用 P37 与 A36。不同组合复用同一路段的采集轨迹。历史道路原生检查点位于 `scenes/<案例>/trained_model/iteration_7000.pth`。[场景映射](../published/道路病害图生三维_新增100/场景映射.json)

- **车载前视**为每个案例保存 40 帧、10 Hz、1920 × 1080 的合成视频。034 和 109 使用源帧 020–059，110 使用 000–039。`classes.npz` 中的 `class_id` 取 0 背景、1 坑洞、2 缺盖井、3 异常盖板。`source_frames` 与原帧号对齐。相机标定与实拍背景沿用原帧，同时合成新病害并保留遮挡。[样例视频](../runs/demo/series/sensor_views/034_13/front_after.mp4) · [逐帧标注](../runs/demo/series/sensor_views/034_13/frames.json)
- **相机几何**覆盖前后两种视角。增强包为每次放置分别选择一次前视和一次后视曝光，累计 654 组裁切。`views.json` 记录全图内参、位姿、源帧、时间和裁切框。使用裁切图时，主点减去框的左、上偏移。`depth_z_m` 为相机光轴深度，另有 `world_normal/part/instance/protected`。局部网格提供轮廓和遮挡。场景高斯提供颜色。[本次前后相机配对](../runs/demo/sensors/camera/034_13/views.json)
- **激光雷达回放**分别计算 Pandar64 与 PandarGT 发射中心，按每条回波时间插值，再与已放置的网格求最近交点。低于路面的命中需要穿过切口。更近的原有前景保留。每个案例 80 帧，历史 8,880 个案例帧复用 240 次原始扫描。`base` 保存实测点，`cases/<案例>.npz` 保存逐帧变更的原始行号与命中。合成回波强度为 NaN，原强度可按行号找回。由回放程序导出的点云 PLY 保存回波，与资产目录中的高斯 PLY 用途不同。[本次雷达回放记录](../runs/demo/sensors/cases/034_13.json)
- **局部几何**以 1 cm 间距从上方求交得到 `top_z_m`。去掉盖板、铰链和水面后得到 `substrate_z_m`。`rigid_z_m` 保存可见固体高度。原库 7 个积水资产的部分水下坑底保留为未知。几何一致性可用 Blender BVH 独立复核。现场病害绝对尺度仍属于设计值。[几何网格](../runs/demo/sensors/geometry/inventory.json) · [独立求交核验](../runs/demo/sensors/checks/independent_geometry.json)

## 运行环境

`gen2hole.py` 的调度、目录查询和指纹核对只依赖 Python 标准库。配置使用 [config/local.json](../config/local.json) 中的绝对路径。`python gen2hole.py --config /path/to/local.json build ...` 可使用另一套配置。`place.py` 和 `simulate.py` 接受 `--config`，`downstream/replay.py` 接受 `--runtime`。

| 阶段 | 本机已验证环境与外部输入 |
|---|---|
| 网格与预览 | `mesh_python` 使用 Python 3.11、NumPy 1.26.4、SciPy 1.17.1、Pillow 13.0.0、trimesh 5.2.1。GLB 双视角与独立 BVH 复核使用 bpy 4.2.0。 |
| 图像拟合、深度、高斯 | `compute_python` 使用 Python 3.9、OpenCV 4.10、SciPy 1.13.1、PyTorch 2.4.0+cu121、Open3D 0.18.0、plyfile 1.1.3。深度推理使用 Transformers 4.49.0。本机通过 `depth_pythonpath` 加载，已装入解释器时可将该字段置空。原推理权重在 `models/depth-anything-v2-small/`。高斯外观拟合使用 CUDA。 |
| 道路迁入与传感器 | `road_project` 指向含 StreetCrafter、EasyVolcap 和 `scripts/hazard_factory` 的原工程。`road_config` 指向 v3 运行配置，其中的 `plan/configs/work/annotated_road_cache` 必须能读到道路锚点、父 YAML、掩膜和带路面标注的 LiDAR 缓存。`pandaset_raw`、`pandaset_processed` 分别指向按路段分目录的原始与预处理数据。前后相机裁切使用 gsplat 1.4.0。 |

本项目归档的 Depth Anything V2 Small 权重 SHA-256 为 `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`。迁移到另一台机器时，更新解释器和道路路径，将 `published` 链接指向 RoadGS 发布目录 `data/releases/2026-11-21`。回放时从当前案例目录读取检查点，并按本机道路配置加载该路段的父 YAML。[模型版本与输入来源](../inputs/pilot/calibration/provenance.json)
