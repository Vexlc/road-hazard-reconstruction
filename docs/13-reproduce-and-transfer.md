# 从 GitHub 克隆到车载相机与激光回放

首次克隆或交接项目，请先看[资产重建与场景迁入复现手册](16-reproduction-cookbook.md)。其中列出按编号查询的方法、可下载输入、各宿主的代码入口和验收标准。

本指南有两个入口：[我有原始数据，要重建资产](#按输入选择重建方法)；[我有模型，要放进另一条路](#按交付物选择迁入方法)。第一次运行可从随仓库提供的 [P103 裂缝照片与 CPU 样例](../gen2hole/examples/photo-cracks/README.md)开始；要跑通道路与传感器，再做 [P154 完整示例](#从真实-rgb-d-重建-p154)。命令均在仓库根目录执行，复跑请用新的输出目录。

## 按资产编号找到源码

[逐资产复现索引](../catalog/reconstruction-index.json)按编号给出构建入口、配方、必要的预处理程序和手册章节。[统计口径](14-reconstruction-principles.md#一资产库有多大一个编号代表什么)区分历史发布与后续样例。`locate` 是查询，不会下载或重建；不要把不同专题的编号一律传给通用 `build`。

~~~bash
python3 gen2hole/gen2hole.py locate P154
python3 gen2hole/gen2hole.py locate P202
~~~

`locate` 只需 Python 标准库，可在安装项目依赖前运行。查无编号时会列出共享数字的其他类别，交接时可据此核对源图。例如 P103 是路面裂缝，当前索引中没有 A103。P103 的照片、掩膜与配方已随克隆提供。[从 CPU 网格到 Gaussian PLY 的完整示例](../gen2hole/examples/photo-cracks/README.md)

## 按输入选择重建方法

同样是一个坑，带深度的观测可以估计米制凹深；单张照片通常只能约束可见边缘和外观。先按手头证据选行，再打开对应专题拿命令。

| 输入 → 典型资产 | 怎样重建、哪些量可信 | 配方、操作与核验 |
| --- | --- | --- |
| 标定深度或双目 XYZ + RGB + 病害掩膜 → 坑洞、双排槽 P148、缺砖和翘砖 P149/P154 | 在病害外拟合道路平面，按相对平面高度建网格，再投回彩色纹理。槽口或碎片要保留独立边界；有量程支持的表面才算观测。 | [历史双目/RGB-D 与自带文件](02-asset-reconstruction.md#自带-rgb-d-文件) · [部件示例 P154](#从真实-rgb-d-重建-p154) |
| 多帧 RGB-D/双目，或有平移视差、内参与标尺的两张彩照 → P110、P121、P142 | 前者把有效深度配准到共同路面；后者靠匹配和视差恢复起伏。彩照补深只填受约束的缺口，用留出帧或量程核验，不把补值当实测。 | [多帧融合](02-asset-reconstruction.md#多帧实拍重建同一组坑洞) · [双照片](02-asset-reconstruction.md#两张彩照重建坑底起伏) · [补深](02-asset-reconstruction.md#用彩照补回双目深度缺口) |
| 单张实拍或生成 RGB → 普通坑 P90、设计井盖、大面积坑 P190–P201 中的单图样例 | 用轮廓、相对深度和配方尺寸约束可见表面；大坑还可单独约束断面和局部材质。无标尺时不能把估计面积、深度当测量。P192–P196 有 RGB-D，应走深度路线。 | [图生几何](../gen2hole/docs/pipeline.md#三类几何如何得到) · [斜拍与 RGB-D 大坑分流](02-asset-reconstruction.md#从斜拍照片重建大面积坑洞) · [大坑输入包](15-large-dry-pits.md) |
| 裂缝照片 + 掩膜，或配准的扫描颜色、位移和法线 → P103、P134–P137、碎板 P144–P147 | 照片提供缝迹，截面和深度按配方设计；扫描路线按材质标尺恢复起伏。闭合缝可分出板块用于抬升、下沉或脱落，这些是派生状态。 | [照片裂缝](../gen2hole/examples/photo-cracks/README.md) · [扫描裂缝](02-asset-reconstruction.md#从道路扫描重建裂缝) · [可拆碎板](02-asset-reconstruction.md#从闭合网裂中提取可拆卸的混凝土碎板) |
| 井口与盖面照片，可选标定深度或车载 LiDAR → 圆井、六边形盖、雨水篦、断盖 | 分别建固定井体与活动盖面；真实通孔建成开口，而非绘制黑斑。同一井框派生开合、移位、断裂、安装高差；隐藏井壁、盖厚或人为姿态须与测量值区分。 | [可动井盖与测井](02-asset-reconstruction.md#两张实拍重建可动圆井盖) · [车载点测井](02-asset-reconstruction.md#从自驾相机和原始激光点重建落座井盖) · [通孔与断条](02-asset-reconstruction.md#从实拍盖板重建贯穿孔与断条) |
| 已有坑床 + 水量或水位配方 → 积水 P129–P133 | 保留干态坑床，单独求水面、岸线及水深。换坡道时按目标重力方向重新求水面，不能把水和坑一起刚性倾斜。 | [干湿状态、坡度与核验](02-asset-reconstruction.md#实拍坑底上的积水) |
| 无实测形状，只有参数、种子和扫描材质 → 设计坑、缺盖井、异常盖板 | 用参数生成高度场或井体部件，再赋 PBR 材质。按清单 ID 或固定种子复建；已封存的大坑路线供复现，不是实拍数据入口。 | [历史五类](02-asset-reconstruction.md) · [工厂配方与种子](10-factory.md#生成与追加) · [封存大坑](10-factory.md#大面积坑洞) |
| 单张路面修补带截图 → P202 | 校正路面透视，沿标注追踪断续接缝；把设计高程与相对亮度分开，换沥青时只换基色。标注随项目提供，复建需与哈希匹配的原截图。 | [修补带重建与重新取色](../gen2hole/examples/transverse-repair/README.md) |

## 按交付物选择迁入方法

| 你要交付什么 | 输入与入口 | 怎样判断做成了 |
| --- | --- | --- |
| 在普通网格引擎里放一件资产 | 带材质的 GLB、目标路面、米制锚点及切口。[坐标转换与接入步骤](#接入自己的道路) | 用低机位检查坑底未被原路面填平、井圈外不漏空、凸起部分没有被遮掉。看得到单件 GLB 不等于已改动宿主。 |
| 能重新加载并继续渲染的动态高斯场景 | 新构建的完整 `pilot/` 或 `series/`（非 `--mesh-only`）走 [034/109/110 原生放置](#迁移-p154-到-pandaset-034-并回放)；历史 055/056 走[联合场景最小组](04-scene-insertion.md)，工厂走[工厂计划](10-factory.md#迁入)。P190、P198–P201 另有[放置与行车回放封装](15-large-dry-pits.md#放入实景并回放)。 | 拿到 `scene.json` 和原生 `.pth`，核对资产指纹、锚点和尺度；重新加载同帧比较。单个 PLY、视频或对照图都不是动态检查点。 |
| 标定相机图或模拟激光回波 | 对已放置的场景，用原帧位姿、标定、动态掩膜和原始扫描[回放传感器](#迁移-p154-到-pandaset-034-并回放)。 | 相机看同帧可见轮廓与前景遮挡；雷达核对原束、发射时间、首个命中部件和距离。图像深度不是雷达量程。 |
| 修补带在不同沥青上的同帧外观对照 | P202 的独立目录不是 `series/`；按[专用迁色与投影流程](../gen2hole/examples/transverse-repair/README.md#迁到另一种沥青)取宿主基色、重着色，再投影到实拍。 | 看主体色差、缝芯亮度和轮廓外改动；比较迁色前后的几何指纹。[两路段记录](../validation/transverse-repair-transfer.json)只验证图像迁移和形态保留，不是原生场景或雷达交付。 |

## 安装和配置

~~~bash
git clone https://github.com/Vexlc/road-hazard-reconstruction.git
cd road-hazard-reconstruction
conda create -y -p "$PWD/work/envs/assets" python=3.11 pip
work/envs/assets/bin/python -m pip install -r scripts/requirements-assets.txt
python scripts/configure-gen2hole.py
~~~

配置器将本次克隆的绝对路径写入 `gen2hole/config/local.json`。它也接受 `--mesh-python` 和 `--compute-python` 指定已有解释器。移机后重新运行配置器。对于普通照片井盖，以下命令只需资产环境与 Git 中的样例资料，可生成六个井盖状态的带纹理 GLB。

~~~bash
python gen2hole/survey.py --survey gen2hole/examples/photo-chamber/survey.json \
  --mesh-only --out runs/first-cover
python gen2hole/gen2hole.py verify --run runs/first-cover
~~~

用 Blender 打开 `runs/first-cover/series/assets/raised/A93/asset.glb` 可检查掀盖姿态。`runs/first-cover/series/checks.json` 记录几何和文件指纹。深度拟合、完整 3DGS、场景和传感器流程还要安装计算环境。下面的安装脚本会拉取固定版本的依赖并编译 CUDA 扩展。请先把 `TORCH_CUDA_ARCH_LIST` 设为自己的 NVIDIA GPU 计算能力，项目测试机使用 8.9。Linux 上恢复 `.tar.zst` 数据包还须有 `zstd` 命令。[环境、CUDA 版本及编译说明](03-environment-and-relocation.md)

~~~bash
conda create -y -p "$PWD/work/envs/streetcrafter" python=3.9.23 pip
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 bash scripts/install-streetcrafter.sh
~~~

## 重建 109 件生成 RGB 资产

在私有 GitHub 可以一次下载全部输入和逐文件校验表。新克隆从仓库根目录执行。归档包括两批原图、共享材质、已登记的轮廓、米制尺寸和坑洞相对深度，解压目标为 gen2hole。随后可直接运行下方 build 与 verify 命令。模型权重仅在重新推断图像深度时需要。[GitHub 输入包](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/sun-openings-v1.0.0/generated-rgb-inputs.zip)

~~~bash
repo=Vexlc/road-hazard-reconstruction
tag=sun-openings-v1.0.0
gh release download "$tag" -R "$repo" -D "downloads/$tag" -p generated-rgb-inputs.zip -p SHA256SUMS --skip-existing
(cd "downloads/$tag" && sha256sum --ignore-missing -c SHA256SUMS)
python3 -m zipfile -e "downloads/$tag/generated-rgb-inputs.zip" gen2hole
(cd gen2hole && sha256sum -c SHA256SUMS)
~~~

也可从 Google Drive 的分卷恢复 109 张主图及其轮廓、相对深度等逐图输入。共享纹理、设计 JSON、人工修订和发布指纹由 Git 提供。下载器逐卷检查 SHA-256，按清单顺序拼合后再校验完整 ZIP。网络中断后重跑同一条命令，已经校验的卷可复用。[Google Drive 输入目录](https://drive.google.com/drive/folders/12y0agwLAAcPllCtBWqWOc2rctY19gG42) · [逐包下载清单](../catalog/downloads.json)

~~~bash
python scripts/packages.py fetch --group gen2hole-inputs-109
python scripts/packages.py restore --group gen2hole-inputs-109
python gen2hole/gen2hole.py build P13 M13 A13 --out runs/three-assets
python gen2hole/gen2hole.py verify --run runs/three-assets
# 两批全部 109 件，由同一入口写入 pilot/ 和 series/
python gen2hole/gen2hole.py build --all --out runs/all-generated
~~~

构建完成后，下面这条命令直接读取两个批次下的 218 个 GLB、PLY，把实际字节与历史发布清单中的 SHA-256 对照。产物缺失或不一致时输出具体路径并返回失败状态。参数指向含 pilot 和 series 的整次构建目录。[历史模型指纹](../gen2hole/catalog/assets.json)

~~~bash
python3 gen2hole/gen2hole.py audit --rebuilt runs/all-generated
~~~

如果通过浏览器下载了分卷，把 `part-00` 至 `part-28` 对应的原名文件放入 `downloads/roadgs-generated-inputs-109.zip.parts/`，再执行同一条 `fetch` 命令。完整 ZIP 为 471,110,229 字节，SHA-256 为 `9ed232597b6c5c0913e5516994fff1a9b3a16e7c42e3ce196b0138b8b97035a6`。

默认复用保存的拟合参数与相对深度。要从相同原始 RGB 重算，运行下面的命令。推理脚本将模型固定在提交 `5426e4f0f36572d16453bbda7a8389317b1bef99`。第一次从 Hugging Face 获取权重。离线机器可用 `--model` 指向包含完整权重、配置和预处理器的本地目录。重算后可与各件 `checks.json` 和归档输入指纹比较。[深度推理源码](../gen2hole/recipes/common/estimate_depth.py) · [输入 SHA-256](../gen2hole/catalog/inputs.sha256)

~~~bash
python gen2hole/gen2hole.py build P13 M13 A13 --from-images \
  --model depth-anything/Depth-Anything-V2-Small-hf --preview --out runs/three-from-rgb
~~~

历史发布的 A06、A09、A12、A17、A20、A23 使用较早的平移盖板取样参数。六件的盖面纹理裁切比例为 0.985，路面擦除的归一化半径为 1.075，过渡宽度为 0.055。统一入口按发布编号恢复这些参数，保存拟合和从 RGB 重新拟合两条路径共用同一设置。[逐件参数与发布指纹](../gen2hole/catalog/assets.json)

要同时核验原图、校准和官方模型，下载固定提交的 `model.safetensors` 到 `gen2hole/models/depth-anything-v2-small/`，保留 Git 随附的两份模型配置。运行 `python gen2hole/gen2hole.py audit` 会对 302 个归档输入逐项验证 SHA-256。权重为 99,173,660 字节，SHA-256 为 `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`。[上游固定版本权重](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/5426e4f0f36572d16453bbda7a8389317b1bef99/model.safetensors)

自带照片可从以下入口开始，每次选一个新的输出目录。单张 RGB 可修改粗框、轮廓和设计尺度。RGB-D 需要同一机位配准的 RGB、光轴深度、掩膜、相机内参与深度单位。两张 RGB 需要对应照片、标尺和掩膜。多帧拟合和留出评估使用不同帧。[单图 RGB 样例](../gen2hole/docs/new-assets.md) · [自带 RGB-D 命令](02-asset-reconstruction.md#自带-rgb-d-文件) · [双张 RGB 输入格式](../gen2hole/examples/photo-parallax/files-example.json) · [多帧输入格式](../gen2hole/examples/multiview-potholes/files-example.json)

## 从真实 RGB-D 重建 P154

P154 包含一个泥水坑和一块倾斜的联锁砖。配方指定 PothRGBD 帧 `20250305_110414`、原坑轮廓之外的砖顶及外露侧面。恢复原始包之后，程序按文件成员和 SHA-256 选取 RGB、深度及标签，在拟合前留出同次曝光的深度块。[P154 配方](../gen2hole/examples/displaced-paver/survey.json)

~~~bash
python scripts/packages.py fetch --group inputs-rgbd
python scripts/packages.py restore --group inputs-rgbd
run=runs/p154
work/envs/streetcrafter/bin/python gen2hole/multiview_pits.py \
  --survey gen2hole/examples/displaced-paver/survey.json --preview --out "$run"
work/envs/streetcrafter/bin/python scripts/check-displaced-paver.py \
  --run "$run" --id P154 --out "$run/evaluation/paver.json"
~~~

资产位于 `runs/p154/series/assets/pothole/P154/`。GLB 采用米制 Y 向上坐标，Gaussian PLY 和道路局部坐标采用米制 Z 向上。`surface.npz` 保存局部表面和融合所需信息，`splat_material.npz` 保存逐高斯材料标签。不要只从网格浏览器导出一个 PLY 替代这些文件。独立砖块在 `series/comparison/P154/isolated.glb`，整坑的留出深度误差在 `evaluation/measurements.json`。

## 迁移 P154 到 PandaSet 034 并回放

**准备宿主。** 下载 034 的预处理道路与基础检查点，以及 v3 迁入需要的道路掩膜、标注路面缓存等输入。在渲染车载前后帧和原始激光束时，还要恢复 PandaSet 原始序列。下面的原始场景下载器使用随代码保存的 ZIP 成员索引并检查成员大小与 CRC。已按同样布局准备好原始数据的机器可直接使用它。配置器为当前克隆生成道路配置与固定锚点。[历史道路包与数据来源](01-downloads.md) · [迁入配置实现](../scripts/hazard_factory/configure.py)

~~~bash
python scripts/packages.py fetch --group scenes-034 --group factory-v3-inputs
python scripts/packages.py restore --group scenes-034 --group factory-v3-inputs
work/envs/streetcrafter/bin/python tools/multiscene/download_scene.py --scene 034
work/envs/streetcrafter/bin/python scripts/hazard_factory/configure.py \
  --output outputs/道路病害_稳定生成与迁移_v3
~~~

**放置与回放。** `--bundle` 指向一次完整构建的 `series/` 或 `pilot/`，`--assets` 是此次构建中的 ID。`--site` 选择宿主，`--group` 标识该宿主上的组合。同组每类可放一件。命令在原尺度合并后写出原生动态检查点，再读取相同时间戳和标定进行前后相机与两套激光雷达回放。[放置实现](../gen2hole/downstream/place.py) · [相机与激光流水线](../gen2hole/downstream/simulate.py)

~~~bash
run=runs/p154
python gen2hole/downstream/place.py --bundle "$run/series" \
  --site 034 --group 154 --assets P154
python gen2hole/downstream/simulate.py --run "$run" --out "$run/sensors"
python scripts/check-pit-transfer.py --run "$run" --out "$run/transfer.json"
~~~

需要额外的 40 帧前视视频、类别掩膜和框时，在放置命令加 `--front`。要对照 034 与 109，先恢复 `scenes-109` 并下载 109 的原始序列，再对 109 执行放置，最后对包含两次放置的运行目录执行一次 `simulate.py`。重跑仿真时另选新的传感器输出目录。上面的标准核验命令读取 `$run/sensors`。比较不同井盖状态时用不同组号、相同的 `--anchor-category` 和车辆帧，避免锚点变化混入对照。

**几何与外观如何对齐。** 放置计划给出世界中心 `center_world` 与旋转矩阵 `basis_columns`，局部 Z 轴沿路面法线。局部米制点乘旋转矩阵后平移到道路中心。程序按资产切口处理宿主道路的背景高斯，并变换资产高斯的位置、方向和外观系数。相机画面使用原车标定和动态遮挡，激光按原束的时间与发射器位置对导出网格求最近交点。缺盖井的开口、翘砖的高端、原有前景各有不同的遮挡关系。湿态资产的重力方向还要匹配目标道路坡度。[宿主平面、路面融合与自定义道路](04-scene-insertion.md#迁入方法及自定义宿主) · [积水的目标坡度](02-asset-reconstruction.md#实拍坑底上的积水)

| 要检查的问题 | 输出 |
| --- | --- |
| 是否装入正确的模型、世界中心、旋转基与原尺寸 | `series/scenes/034_154/scene.json` |
| 动态场景能否重新加载 | `series/scenes/034_154/trained_model/iteration_7000.pth` 和 `transfer.json` 中的重载误差 |
| 车上能看到哪些部位 | `sensors/camera/034_154/views.json` 和同目录迁入前后图、首个交点深度、部件图 |
| 哪些原始回波被替换 | `sensors/cases/034_154.npz` 及 `sensors/checks/scan_audit.json` |
| 量程是否经独立几何复核 | `sensors/checks/independent_geometry.json`、`transfer.json` |

当次验收使用当时可匿名读取的源码和数据。当前 GitHub 克隆和 Release 下载需要私有仓库读取权限。验收记录中，109 件生成 RGB 资产按保存的拟合重建，109 个 GLB 和 109 个 Gaussian PLY 的 SHA-256 全部与发布基线一致。原图、校准、纹理与权重等共 302 个输入通过指纹核验。另选坑洞 P13 和平移井盖 A06 从原图重算深度与拟合，两件的 GLB 和 PLY 也与基线一致。复跑复用了测试机已安装的 Python、CUDA 与 StreetCrafter 依赖。[公开下载与复建核验记录](../validation/github-reproduction.json)

实拍 RGB-D 案例 P154 的 GLB 和 PLY 均与基线逐字节一致。迁入 PandaSet 034 后，场景重载前后像素最大差为 0。80 帧回放中，两套激光分别有 433、1,485 个回波因资产而改变。历史库曾有 13 个双目模型因路面拟合的掩膜范围产生差异。恢复原版规则后，全部 65 件双目模型均复现发布哈希。[P154 两段道路的基线](../validation/displaced-paver-transfer.json) · [历史双目复核](../validation/stereo-replay.json)

## 查看结果和其他迁入方式

GitHub 可直接预览下列图版。离线交互图谱从本地模型、相机画面和已有的运行目录构建，完整图谱需要对应的数据树。对单项复建可以直接查看输出的 GLB、预览和传感器记录。[资产与驾驶画面总览](figures/road-asset-portfolio.jpg) · [P154 实拍与道路对照](figures/displaced-paver.jpg) · [六边形井盖四态](figures/hexagonal-cover.jpg)

历史 1,199 件资产走 055/056 的联合场景入口。先恢复 `scenes-055-056` 和仅含 8 件资产的 `sample-scene-001` 即可跑最小组。程序化工厂和 034/109/110 的五类组合有单独的分组计划和冷启动重载步骤。接入新宿主需要动态检查点、标定、相机和道路几何，然后按新宿主重建锚点与缓存。[055/056 最小联合迁入](04-scene-insertion.md) · [工厂迁入](10-factory.md#迁入) · [历史交付与数据版本](12-integration.md)

### 接入自己的道路

1. **先确定宿主表示。** 普通网格引擎可直接导入带材质的 GLB，并在路面网格上做开口。要沿用本项目的动态相机与 Gaussian 渲染，先准备能由 StreetCrafter 重载的宿主检查点、标定、车辆位姿、帧号和前景实例。要回放实车 LiDAR，还需要原始发射器位置、逐点时间和回波记录。
2. **在目标道路上测锚点。** 在一块覆盖资产占地的实测路面上拟合局部平面，记录 `center_world`、`basis_columns`、支持点数、平面拟合残差和参考帧。固定此基底即可在同位置切换掀盖、缺盖或沉降状态。现有计划的 `slots_by_site` 按道路保存锚点，`drive_start_by_site` 记录对应的轨迹起始帧。[计划字段与实值样例](../catalog/factory-v3-plan.json)
3. **用同一次刚体变换放置几何。** 若读入的顶点采用 glTF 原始轴向，先将 `(x, y, z)` 转成资产局部的 `(x, -z, y)`。PLY 已处于资产局部 Z-up 坐标。用列向量计算 `p_world = basis_columns @ p_local + center_world`，距离单位均为米。网格引擎若已在导入时转换坐标轴，此处直接使用转换后的坐标。Gaussian 还需旋转朝向与方向相关的球谐系数。[变换、外观和边缘融合源码](../gen2hole/downstream/native_entry.py)
4. **处理宿主路面。** 按局部切口清除会遮住病害的原路面几何或背景高斯，在坑缘过渡带保留连续的路面外观。盖板、井圈、井筒等多层表面用 GLB 做可见性与最近交点查询。检查斜向低机位能否从开口看见井壁，同时保留更近的人、车和遮挡物。[宿主背景处理](../scripts/hazard_factory/migrate.py) · [射线与部件实现](../gen2hole/downstream/sensors/road_geometry.py)
5. **核验后再导出。** 先检查同一机位迁入前后图、切口边缘和资产尺寸，再重新加载保存的场景，比较回放像素。复用传感器链时，按同一束激光和时间检查首个命中的部件、距离和新旧回波。内置放置命令接受 `034/109/110`。接入新的道路 ID 时需要给它配置场景、锚点和传感器适配，并用自己的数据验证。已有的 055/056 道路使用上面的历史联合场景入口。
