# 从 GitHub 克隆到车载相机与激光回放

本指南对应 1,403 条资产记录。先用随仓库提供的井框和盖面照片跑通网格，再按资产编号选数据。最后用实拍 RGB-D 的 P154 跑一遍米制重建、道路放置和传感器回放。大型输入按需从 Google Drive 下载。所有命令在仓库根目录执行。

## 按资产编号找到源码

[逐资产复现索引](../catalog/reconstruction-index.json)的 27 个分组覆盖全部 1,403 个编号。每组列出主程序、配方文件、必要的预处理程序，以及手册和对应章节标题。在 GitHub 上打开索引后可以搜索完整编号，也可在克隆后运行下面的查询。

~~~bash
python - P154 <<'PY'
import json, sys
from pathlib import Path
key = sys.argv[1]
groups = json.loads(Path('catalog/reconstruction-index.json').read_text())['families']
for group in groups.values():
    if key in group['assets']:
        for field in ('name', 'prepare', 'launcher', 'recipe', 'guide', 'section'):
            if field in group:
                print(group[field])
        break
else:
    raise SystemExit('未找到编号 ' + key)
PY
~~~

| 资产来源 | 数量 | 重建入口与要准备的输入 |
| --- | ---: | --- |
| 历史实拍和设计库 | 1,199 | 65 条双目 XYZ、822 条 RGB-D、312 个设计坑洞和井盖。按类别下载 `inputs-stereo`、`inputs-rgbd` 或 `inputs-procedural`，用 `--asset` 指定完整 ID。省略时可按 `--category` 复建整类。[五类命令](02-asset-reconstruction.md) · [批量重建源码](../scripts/reconstruct-assets.py) |
| 工厂追加的程序化资产 | 30 | 恢复 `inputs-procedural` 和 `factory-v3-extra-material`，运行工厂配置器。已发布的两组种子为 20261119、20261120，各用索引 0 至 14。以 `--seed <种子> --start 0 --count 15` 分别运行两次，具体输出目录见工厂文档。[配方与每件尺寸](../catalog/factory-v3-assets.json) · [工厂操作](10-factory.md) |
| 生成 RGB | 109 | 9 件首批、100 件后续批次，覆盖坑洞、敞井和异常盖板。提示词、设计尺寸、拟合修订和配方随 Git 发布。原图及逐图拟合输入在 `gen2hole-inputs-109` 数据包中。[生成与重建方法](../gen2hole/docs/pipeline.md) |
| 后续实拍、融合和状态派生 | 65 | 覆盖单图裂缝、双照片坑底、RGB-D 或双目融合、相机与 LiDAR 测井、开孔及断片盖板、积水、混凝土碎板、缺砖和翘砖。每组的源帧、标尺、掩膜或状态参数均有独立配方。[逐类步骤与误差记录](02-asset-reconstruction.md) |

对实拍单目或生成图，平面跨度、坑深、隐藏井壁和盖背厚度可能采用设计值。RGB-D 或双目有量程的位置按标定换算为米。同一井体的多个开合状态、同一病害的多个采集帧分别计数。

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

以下步骤恢复 109 张主图及其轮廓、相对深度等逐图输入。共享纹理、设计 JSON、人工修订和发布指纹由 Git 提供。下载器逐卷检查 SHA-256，按清单顺序拼合后再校验完整 ZIP。网络中断后重跑同一条命令，已经校验的卷可复用。[Google Drive 输入目录](https://drive.google.com/drive/folders/12y0agwLAAcPllCtBWqWOc2rctY19gG42) · [逐包下载清单](../catalog/downloads.json)

~~~bash
python scripts/packages.py fetch --group gen2hole-inputs-109
python scripts/packages.py restore --group gen2hole-inputs-109
python gen2hole/gen2hole.py build P13 M13 A13 --out runs/three-assets
python gen2hole/gen2hole.py verify --run runs/three-assets
# 两批全部 109 件，由同一入口写入 pilot/ 和 series/
python gen2hole/gen2hole.py build --all --out runs/all-generated
~~~

默认复用保存的拟合参数与相对深度。要从相同原始 RGB 重算，运行下面的命令。推理脚本将模型固定在提交 `5426e4f0f36572d16453bbda7a8389317b1bef99`。第一次从 Hugging Face 获取权重。离线机器可用 `--model` 指向包含完整权重、配置和预处理器的本地目录。重算后可与各件 `checks.json` 和归档输入指纹比较。[深度推理源码](../gen2hole/recipes/common/estimate_depth.py) · [输入 SHA-256](../gen2hole/catalog/inputs.sha256)

~~~bash
python gen2hole/gen2hole.py build P13 M13 A13 --from-images \
  --model depth-anything/Depth-Anything-V2-Small-hf --preview --out runs/three-from-rgb
~~~

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

本次在独立源码目录重建的 P154，GLB 和 PLY 的 SHA-256 与此前发布基线一致。034 原生场景重载像素最大差为 0，回放 80 帧，两套激光分别得到 433、1,485 个替换回波。此结果只覆盖这个案例。历史库仍有 13 个双目模型的跨环境复建差异记录。[P154 两段道路的基线](../validation/displaced-paver-transfer.json) · [历史双目差异](../validation/stereo-reconstruction-differences.json)

## 查看结果和其他迁入方式

GitHub 可直接预览下列图版。离线交互图谱从本地模型、相机画面和已有的运行目录构建，完整图谱需要对应的数据树。对单项复建可以直接查看输出的 GLB、预览和传感器记录。[资产与驾驶画面总览](figures/road-asset-portfolio.jpg) · [P154 实拍与道路对照](figures/displaced-paver.jpg) · [六边形井盖四态](figures/hexagonal-cover.jpg)

历史 1,199 件资产走 055/056 的联合场景入口。先恢复 `scenes-055-056` 和仅含 8 件资产的 `sample-scene-001` 即可跑最小组。程序化工厂和 034/109/110 的五类组合有单独的分组计划和冷启动重载步骤。接入新宿主需要动态检查点、标定、相机和道路几何，然后按新宿主重建锚点与缓存。[055/056 最小联合迁入](04-scene-insertion.md) · [工厂迁入](10-factory.md#迁入) · [历史交付与数据版本](12-integration.md)
