# 道路病害资产重建与场景迁入复现手册

有资产编号，先运行 `locate`。只有参考图或深度，按下表的输入选重建入口。已有资产要放入道路，直接看下文的宿主表。所有命令均从仓库根目录执行，路径为相对路径。下载固定版本需要能读取该私有仓库的 GitHub 账号。[版本源码与下载](https://github.com/Vexlc/road-hazard-reconstruction/releases/tag/reproduction-2026.11.25)

## 拿到代码并找到正确入口

~~~bash
gh auth login
gh repo clone Vexlc/road-hazard-reconstruction
cd road-hazard-reconstruction
git switch --detach reproduction-2026.11.25
python3 scripts/check-reproduction-index.py
python3 gen2hole/gen2hole.py locate P154
python3 gen2hole/gen2hole.py locate P203
python3 gen2hole/gen2hole.py locate P211
~~~

第一项检查仅用 Python 标准库。该版本应输出 **1,463 个唯一编号、36 个重建系列、9 条迁入或展示路线、20 件 KITTI 双坑资产**。`locate` 的 `prepare` 指输入准备，`launcher` 指实际构建程序，`recipe` 指配方，`guide / section` 定位到该案例的操作步骤，`transfer` 给出对应的迁入程序及说明。P154 是带实测深度的翘砖，P203 是单图路面剥落，P211 是生成参考图的大坑，三者使用不同入口。统计以编号为单位，共用同一次采集的多种井盖姿态分别有编号。[机器可读索引](../catalog/reconstruction-index.json)

## 各类资产怎样重建

| 资产与例子 | 必须准备什么 | 重建代码、复现依据 |
|---|---|---|
| 历史双目坑洞 65 条、RGB-D 坑洞 822 条 | 双目 XYZ 或注册深度、RGB、掩膜、相机参数、输入档案 | [历史构建器](../scripts/reconstruct-assets.py) 按发布清单选原始采集，拟合坑外路面，导出相对路面的高度和带纹理网格。自带 RGB-D 走[自定义文件入口](../scripts/reconstruct-rgbd.py)。[五类操作与输入包](02-asset-reconstruction.md) |
| 设计坑、缺盖井、开启或移位盖板；v3 工厂；封存的大坑 P160–P189 | 扫描 PBR 材质、固定配方、种子或已登记的 ID | 历史三类也用上述历史构建器。可追加工厂用[工厂入口](../scripts/hazard_factory/run.py)，按计划生成井体、活动盖面和坑床。P160–P189 用[封存生成器](../scripts/build-large-potholes.py)，保留种子和原始面积核验。[工厂运行与封存说明](10-factory.md) |
| 生成 RGB 的 109 件资产、单图实拍 P90、照片裂缝 P101–P103 | 已发布的照片、轮廓、相对深度和设计尺寸；裂缝另有像素掩膜 | [Gen2Hole 构建器](../gen2hole/gen2hole.py) 分派首批与后续配方。照片提供轮廓及外观，绝对尺度按标尺或配方指定。P103 可直接从克隆生成并核对 GLB。[最小 CPU 案例](../gen2hole/examples/photo-cracks/README.md) · [109 件输入获取及复建](13-reproduce-and-transfer.md#重建-109-件生成-rgb-资产) |
| 可动圆井盖、六边形盖、通孔篦板、断条和断盖、安装高差，及 PandaSet 实测井盖 | 井圈和盖面照片、部件轮廓、孔洞、姿态与尺寸；车载实测路线还需标定和 LiDAR | [井盖构建器](../gen2hole/survey.py) 分开生成井体和盖板，按同一配方派生开启、错位、脱落等状态；车载尺寸由[点云拟合](../gen2hole/fit_road_cover.py)准备。检查真实孔洞、井内遮挡与部件姿态。[专题操作](02-asset-reconstruction.md#两张实拍重建可动圆井盖) |
| 多帧坑、双张彩照坑、双目补深、双排凹槽 P148、缺砖碎片 P149/P150、翘砖 P154 | 对应帧、相机标定、掩膜、深度或视差与米制参照；各案例的输入成员由 survey 指定 | [深度与多视角入口](../gen2hole/multiview_pits.py) 选择配准、融合或[双张彩照匹配](../gen2hole/photo_parallax.py)。分别记录观测区域、补全区域与留出帧误差。槽口与砖块另保留部件几何。[P154 完整命令](13-reproduce-and-transfer.md#从真实-rgb-d-重建-p154) |
| 扫描裂缝 P134–P137、网裂碎板 P144–P147 | 扫描材质的颜色、位移或法线、标尺、缝迹和面片标注 | 分别运行[扫描裂缝预处理](../gen2hole/examples/scanned-cracks/prepare.py)或[碎板预处理](../gen2hole/examples/faulted-slab/prepare.py)，再用 Gen2Hole 构建。扫描提供表面起伏，砖板状态偏移量取自配方。[裂缝操作](02-asset-reconstruction.md#从道路扫描重建裂缝) · [碎板操作](02-asset-reconstruction.md#从闭合网裂中提取可拆卸的混凝土碎板) |
| 真实坑床积水 P129–P133，照片水线 P204 | 前者需要已重建干坑和水量，后者需要原图、水线标注、深度缓存和假设的水下坑床 | [实测坑床加水](../gen2hole/fill_pits.py)；P204 使用[照片水面构建](../gen2hole/photo_puddle.py)。迁往不同坡度时重新求水平水面；验证岸线、深度和可见水面。[P204 的按站点命令](02-asset-reconstruction.md#p204-路缘积水坑) |
| 斜拍大坑 P190/P191/P197、浅剥落 P203、20 件车行道大坑；P192–P196 有实测深度 | 单图需原图、坑沿、相机/深度缓存及尺度；精细干坑还需断壁、局部深度和材质；RGB-D 另走测深路线 | [基础斜拍重建](../gen2hole/oblique_pits.py)；[精细干坑子模块](../gen2hole/large_pits/__main__.py)。P203 使用专属浅剥落轮廓及深度先验。20 件合集包含 P190、P198–P201、P205–P219。[大坑说明](15-large-dry-pits.md) · [实拍剥落带](02-asset-reconstruction.md#p203-路口横向剥落带) |
| 横向沥青修补带 P202 | 原图和标注可从头拟合；复用校正后的高度与相对亮度场可直接导出网格 | [修补带构建与换色](../gen2hole/repair_patch.py) 保留缝迹与高程，按目标路面颜色重着色。下文提供从发布中间量恢复原 GLB 的命令。[完整案例](../gen2hole/examples/transverse-repair/README.md) |

## 先跑一个最小案例

首次验证建议选 P103。Linux 上用 Python 3.11 建立环境，运行后会得到约 81.7 万个三角面的裂缝网格。仅执行这一节时不用下载场景和 GPU 模型。

~~~bash
python3.11 -m venv work/envs/cracks
py=work/envs/cracks/bin/python
"$py" -m pip install -r gen2hole/examples/photo-cracks/requirements-mesh.txt
"$py" scripts/configure-gen2hole.py --mesh-python "$py" --compute-python "$py" --output work/cracks-config.json
"$py" gen2hole/gen2hole.py --config work/cracks-config.json build P103 --mesh-only --out runs/p103-mesh
"$py" gen2hole/gen2hole.py --config work/cracks-config.json verify --run runs/p103-mesh
"$py" scripts/check-crack-assets.py --run runs/p103-mesh
echo '49759ba9e543495993f33a23896c7ead3bc83f2c42f6d0c47d6100ca3ccc5f38  runs/p103-mesh/series/assets/pothole/P103/asset.glb' | sha256sum -c -
~~~

完整资产环境按[环境与移机](03-environment-and-relocation.md)安装。配置器默认使用 `work/envs/assets/bin/python` 做网格、`work/envs/streetcrafter/bin/python` 做 Gaussian 和传感器，写入当前克隆的 `gen2hole/config/local.json`。生成预览的路线还需能调用 Blender。跨机运行需重新生成配置。原始输入的归档通过组名和 SHA-256 恢复；`fetch` 下载、`check` 校验、`restore` 解包。[可选数据与分组清单](01-downloads.md)

## 各类资产怎样迁入道路

| 已有输入与目标 | 运行入口 | 应看到的输出及验收 |
|---|---|---|
| 1,199 件历史库中的成品 → PandaSet 055/056 | [历史联合迁入脚本](../outputs/road_hazard_joint_scenes/reproduction/run.sh)。最小组恢复 `scenes-055-056` 和 `sample-scene-001` 后运行 055 的 001 组。[命令与输出目录](04-scene-insertion.md) | 一个可重载的 `iteration_7000.pth`、`scene.json`、同帧前后与行车视频 |
| 工厂资产 → 034/109/110 的多类组合 | [迁入调度](../scripts/hazard_factory/migrate_all.py) 读取已重定位的 job、组计划、资产库及目标道路缓存。[配置、生成和迁入](10-factory.md) | 每组只替换指定锚点的道路背景。检查场景重载、尺度、地面穿插和原演员保留 |
| 完整 Gen2Hole `pilot/` 或 `series/` → 034/109/110 | [原生放置](../gen2hole/downstream/place.py) → [相机及双 LiDAR 仿真](../gen2hole/downstream/simulate.py) → [迁入验收](../scripts/check-pit-transfer.py)。积水先按站点重新求水面。[可复制的 P154 全流程](13-reproduce-and-transfer.md#迁移-p154-到-pandaset-034-并回放) | 在 `series/scenes/<道路>_<组>` 找到合并检查点；在 `sensors/` 查原始射束、命中部件及前后视机位。检查重载像素、最近交点、前景遮挡 |
| 单个大坑 → PandaSet 行车回放 | [large_pits drive](../gen2hole/large_pits/__main__.py) 调用原生放置后，以[道路相机和表面光照](../scripts/render-placed-pit-drive.py)渲染。[实操](15-large-dry-pits.md#放入实景并回放) | `drive.mp4` 和 `placement.json`；用其中的 PLY 指纹、尺度和检查点重载记录核对选中的模型 |
| 20 个大坑 → KITTI-360 0000，每次两个 | [建场与重载渲染](../streetcrafter/joint/pit_pairs.py) → [前视核验](../streetcrafter/joint/review_pit_pairs.py) → [后视核验](../streetcrafter/joint/review_pair_rear.py) | 10 个单独的 `joint.pt`，20 个不同 ID。前视来自标定双目中的左相机，后视沿原车轨迹用虚拟尾部机位渲染 |
| 修补带 → 另一种沥青的实拍机位 | [修补带同帧投影](../scripts/review-repair-transfer.py)，从目标沥青取色，按标定位姿求 GLB 射线交点。[运行说明](../gen2hole/examples/transverse-repair/README.md#迁到另一种沥青) | 交付物是同机位图像对照和 `transfer.json`。几何指纹、保护区像素与色差在这里核验 |

接入自己的网格引擎时，GLB 使用米制 Y-up，Gaussian PLY 使用米制 Z-up。同一资产的高程场与切口需要随模型一起放置；坑口内要移除会遮挡坑底的宿主路面。原生高斯迁入还会旋转椭球朝向与球谐方向。自定义宿主需要世界坐标、锚点、原始相机、行车时间戳及遮挡信息。[坐标约定与自定义宿主接入](13-reproduce-and-transfer.md#接入自己的道路)

## 从发布输入复建 20 个大坑

在配置好上述网格与 Gaussian 两个环境后运行。压缩包包含每件真正使用的图片、整图和局部深度、边缘标注、材质及来源，共用文件按内容只存一次。可直接使用已缓存深度。脚本先校验整包，再逐文件校验输入，最后比对几何指纹及 GLB、PLY、高度场、元数据和 Gaussian 材质五个文件。[构建脚本](../scripts/rebuild-common-pits.py) · [20 件预期指纹和模型路径](../catalog/common-pit-kit.json)

~~~bash
release=reproduction-2026.11.25
repo=Vexlc/road-hazard-reconstruction
kit=downloads/reproduction-2026.11.25
mkdir -p "$kit"
gh release download "$release" -R "$repo" -p 'common-pit-inputs.zip.part-*' -p PART-SHA256SUMS -p SHA256SUMS --skip-existing -D "$kit"
(cd "$kit" && sha256sum -c PART-SHA256SUMS && cat common-pit-inputs.zip.part-* > common-pit-inputs.zip && sha256sum --ignore-missing -c SHA256SUMS)
python3 scripts/configure-gen2hole.py
python3 scripts/rebuild-common-pits.py "$kit/common-pit-inputs.zip" P211
# 同一条命令省略编号即可补齐全部 20 件。已存在的目标目录会重新校验。
python3 scripts/rebuild-common-pits.py "$kit/common-pit-inputs.zip"
~~~

P211 的成品在 `runs/common-large-pits/P211/series/assets/pothole/P211/`。原有五件沿用原发布结构，以清单里的 `model` 路径为准。新加入的 15 件使用独立生成的参考图，其绝对尺寸来自设计尺度。若已有目标目录与基准不符，请先移走那一次输出，再对该编号运行命令。

补充输入可用以下命令取得。P191 和 P197 提供公有领域原图及匹配的深度缓存；P202 提供校正后的场和标定记录；P203、P204 提供深度缓存，重算照片几何仍需使用者提供与配方 SHA-256 一致的原截图。同一份 P202 场在原始路面基色下可恢复原网格。

~~~bash
kit=downloads/reproduction-2026.11.25
gh release download reproduction-2026.11.25 -R Vexlc/road-hazard-reconstruction -p supplementary-inputs.zip -p SHA256SUMS --skip-existing -D "$kit"
(cd "$kit" && sha256sum --ignore-missing -c SHA256SUMS)
python3 -m zipfile -e "$kit/supplementary-inputs.zip" work/supplementary
work/envs/assets/bin/python gen2hole/repair_patch.py mesh work/supplementary/P202 runs/P202-from-field/asset.glb --rgb 108 109 121
echo '72779a56687e74be653cca268617b98bbeec81b29e47676e3205eed5fa556889  runs/P202-from-field/asset.glb' | sha256sum -c -
~~~

## KITTI-360 双坑与后视

先复建上节的 20 件模型，再准备联合渲染环境。它使用 Python 3.11、Torch 2.7.0+cu128、3DGUT、SC-GS、StreetCrafter 和 PyTorch3D。PandaSet 渲染器使用另一套 Torch；两套环境的 CUDA 扩展分别编译。固定提交、安装顺序及环境预检见[联合渲染环境](../streetcrafter/joint/README.md#配置联合渲染环境)。

~~~bash
kit=downloads/reproduction-2026.11.25
gh release download reproduction-2026.11.25 -R Vexlc/road-hazard-reconstruction -p kitti360-0000-30000.pt -p SHA256SUMS --skip-existing -D "$kit"
(cd "$kit" && sha256sum --ignore-missing -c SHA256SUMS)
host=runs/streetgs-joint/kitti360-0000/30000
mkdir -p "$host"
# 在新克隆中安装宿主。若目标已存在，先核对再继续。
if [ ! -e "$host/joint.pt" ]; then cp "$kit/kitti360-0000-30000.pt" "$host/joint.pt"; fi
cmp "$kit/kitti360-0000-30000.pt" "$host/joint.pt"
bash scripts/scene.sh joint kitti360-0000 prepare 30000
bash scripts/scene.sh joint kitti360-0000 render 30000
bash scripts/scene.sh joint kitti360-0000 pit-pairs 30000
bash scripts/scene.sh joint kitti360-0000 pit-pairs-rear 30000
~~~

`prepare` 通过数据适配器按需取得第 200–247 帧的相机、Velodyne、标定和时间戳。原始 KITTI 数据不在本 Release 中。`render` 生成放置时取路面光照所需的原场景视图。需要自行训练宿主时，从一个空的运行目录用 `bash scripts/scene.sh joint kitti360-0000 all 30000`；精确对照发布动画时使用发布的宿主权重。

每组的原生检查点在 `runs/streetgs-joint/kitti360-0000/30000/pit-pairs/<组>/joint.pt`。`outputs/streetgs-joint/kitti360-0000/30000/pit-pairs/` 下的 `list.md`、`rear-list.md`、`index.json`、`rear-index.json` 分别给出单组播放和逐帧核验。后视画面可能出现背景拉伸。可直接下载发布时的[前后视合集](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/reproduction-2026.11.25/paired-front-to-rear.mp4)和[后视合集](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/reproduction-2026.11.25/paired-rear.mp4)。

## 验收与排错

| 现象 | 如何核对 |
|---|---|
| 编号查得到，通用 `build` 却不接受 | 用 `locate` 查看该编号的 launcher 和 recipe。P154 走 survey，P211 走发布输入包，P202 走修补带入口。 |
| 只生成了网格，迁入命令找不到 PLY | 用同一输入做完整构建。保留 `series/` 或 `pilot/`、部件标签和切口场。历史库的目录及文件名遵照历史清单。 |
| 恢复后仍提示找不到输入、道路配置或帧 | 确认在仓库根目录，先 fetch、check、restore 对应数据组，再生成本机配置。KITTI 另需先生成原检查点的 frames。 |
| 水面倾斜、坑底被盖住或模型浮空 | 核对单位、局部坐标轴、目标道路法线、切口及水位求解。用同帧原路面、近景、射线命中部件复核。 |
| 重载了场景，但指标或二进制略有差异 | 先核对输入和宿主哈希、依赖与 GPU，再比对几何、留出深度、保护区像素和逐束量程。发布基准要求字节一致的案例会明确报告具体文件。 |
| 仅需看效果 | 直接下载 Release 视频、查看仓库内图版。旧文档里的 `runs/`、`outputs/` 和本地图谱链接是完成对应步骤后的本地产物。 |

本轮在独立检出目录复用了已安装的环境、道路数据与发布输入，从空输出目录重建。20 件大坑的 100 个原生文件均与基准 SHA-256 一致。10 组 KITTI 双坑的 480 张前视及 280 张虚拟后视渲染逐像素一致。P154 在 PandaSet 034 的检查点重载像素误差为 0，80 帧双雷达回放分别得到 433 和 1,485 个模拟命中。其余系列按代表样例执行，完整范围见[本次复现结果](../validation/reproduction-2026-11-25.json)。历史双目有 13 件存在已记录的数值差异，逐项容差与旧库全量检查见[既有验收记录](06-validation.md)。
