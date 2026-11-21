# 在 I2hole2road 开发

日常从项目根目录工作。图生资产与传感器的维护源码在 [pipelines](../pipelines)，11 月 21 日的完整交付在 [data/releases/2026-11-21](../data/releases/2026-11-21)。两批图生资产和传感器包沿用各自原名并排存放，离线页面可以按相对路径打开原图、模型和点云。

| 目录 | 用途 |
| --- | --- |
| [pipelines/image9](../pipelines/image9)、[pipelines/image100](../pipelines/image100) | 首批和新增批次的图像拟合、建模、Gaussian 导出和场景迁入。两批保留各自的配方和编号。 |
| [pipelines/sensors](../pipelines/sensors) | 相机几何、光照、网格求交、雷达回放、浏览器页面与校验。 |
| [scripts/hazard_factory](../scripts/hazard_factory)、[tools/multiscene](../tools/multiscene) | 五类资产的原尺度工厂和三路段运行工具，也是图生资产迁入的基础。 |
| [catalog](../catalog)、[validation](../validation) | 版本清单、文件 SHA-256、基线指标和本次迁移的实际验证结果。 |
| [work](../work)、[outputs](../outputs) | 本机 Python 环境、原始输入、派生缓存，以及沿用原仓库路径的历史资产和道路模型。 |
| [data/history](../data/history)、[runs](../runs) | 历史交付与工作记录，以及每次新实验的独立目录。 |

## 改一个资产

从当前基线新建一个功能分支，在 `pipelines/image100/rebuild/` 修改源码。下面以 P04 为例。初始化会记录当前提交与实际工作区源码的逐文件哈希，并复制参考图、校准参数和放置计划。新模型写入本次实验目录。

~~~bash
git switch -c feat/p04-geometry
python scripts/roadgs.py init p04-geometry --component image100

bundle="$PWD/runs/p04-geometry/道路病害图生三维_新增100"
cache="$PWD/runs/p04-geometry/mesh"
work/envs/assets/bin/python "$bundle/rebuild/build.py" --cache "$cache" P04
work/envs/streetcrafter/bin/python "$bundle/rebuild/export_splats.py" \
  --project "$PWD" --cache "$cache" P04
~~~

生成结果在 `$bundle/assets/pothole/P04/`。`asset.glb` 是 Y 向上的米制网格，`asset.ply` 是 Z 向上的米制 Gaussian，`surface.npz` 给出路面切口和融合场。对照基线时，先检查坑口、深度和单位，再比较几何或渲染变化。可单独选 `image9` 或 `sensors`，也可以省略 `--component` 准备三个流程。每轮有意保留源码快照，修改维护源码后用新的实验名初始化。

## 回放一个已交付的案例

下列命令从项目内的 CUDA 环境和本地检查点回放。在本机迁移到另一目录、或恢复依赖后，先用 v3 配置脚本重建本机路径。场景回放工具读取临时重定位的 YAML。

~~~bash
work/envs/streetcrafter/bin/python scripts/hazard_factory/configure.py \
  --output outputs/道路病害_稳定生成与迁移_v3
work/envs/streetcrafter/bin/python scripts/replay-roadgs-scene.py --case 034_01 110_37
~~~

检查自己实验生成的原生场景时，加 `--bundle "$bundle"`。查看完整雷达扫描只需 Python 和 NumPy。下面导出路段 034 第 46 帧的 A13 局部回波，将 `after` 换成 `before` 可按同一批原始点行号取出对照。

~~~bash
mkdir -p runs/lidar-review
work/envs/streetcrafter/bin/python pipelines/sensors/tools/replay.py \
  --root data/releases/2026-11-21/道路病害_传感器与几何增强 \
  --case 034_13 --frame 46 --asset A13 --mode after \
  --output runs/lidar-review/A13-after.npz
~~~

本机的 `work/envs/assets` 用于网格，`work/envs/streetcrafter` 用于 Gaussian、PandaSet 和 GPU 回放。Depth Anything V2 Small 的本地权重在 `work/depth_model/`。三路段的原始和预处理数据在 `tools/multiscene/data/{raw,processed}/`。换机安装沿用[基础环境说明](03-environment-and-relocation.md)，同时恢复所需版本的数据与道路运行包。首批和新增批次的独立几何缓存分别在 `work/roadgs-cache/image9/` 与 `work/roadgs-cache/image100/`。

## 新旧版本如何管理

1. **日常改动用分支。** `main` 保存已验收的工作版本，每项改动从 `main` 建 `feat/...` 或 `fix/...`。修改代码时保留资产 ID 和源帧号，以便用同一个坑洞、同一帧比较前后效果。
2. **每个实验单独落盘。** `runs/<实验名>/run.json` 记录输入数据版本、Git 提交和实际源码哈希。模型、检查点和视频写入该实验目录，已交付数据保持可复查。
3. **用标签配对代码和数据。** 达到里程碑后，将新交付放入独立的 `data/releases/<版本>/`，生成相对路径 SHA-256 清单，在 `catalog/roadgs-releases.json` 增加版本与计数，验证后更新 `current` 并给对应提交打标签。当前标签为 `roadgs-2026.11.21`。同一天再次发布可采用 `2026-11-21.1` 这类修订号。
4. **旧版从标签和清单找回。** `git show roadgs-2026.11.21:catalog/roadgs-releases.json` 可直接查看当时的索引。需要修改旧代码时，从对应标签新建修复分支，例如 `git switch -c fix/old-replay roadgs-2026.11.21`。`python scripts/roadgs.py --release 2026-11-21 verify` 复核该版本的数据。

11 月 18 日的 150 份旧联合场景检查点在本机与原始交付共享硬链接，直接改写文件内容会影响两处。比较或调整这些旧权重时，先复制到新的 `runs/` 目录。11 月 21 日的发布树采用独立副本。三个 ZIP 提供两套资产交付和传感器包；完整街景检查点与前视序列还需保留发布树中的 `scenes/` 和 `sensor_views/`。换机要完整复现本版时，复制整棵版本目录并运行全量校验。
