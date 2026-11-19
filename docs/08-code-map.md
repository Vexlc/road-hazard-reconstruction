# 代码作用与执行顺序

本文件说明核心重建、场景迁入和全部可视化代码的作用。大文件不存入 Git；固定依赖的第三方源码由安装脚本按提交号获取。

## 目录分工

| 位置 | 作用 |
|---|---|
| `scripts/` | 面向新机器的安装、解包、按类重建、迁入与视频生成入口 |
| `catalog/` | 下载文件和 SHA-256、精确实拍输入、资产与视频的索引 |
| `outputs/road_hazard_expansion/reproduction/` | 五类扩充资产的几何、高斯、预览与原始筛选算法 |
| `outputs/road_hazard_joint_scenes/reproduction/` | 055/056 道路中的全库 150 组混合迁入与验证 |
| `tools/multiscene/` | 新道路 034/109/110、50 个选中资产、原尺度与放大版视频 |
| `tools/visualization/` | 全部 1,199 件资产的五类 4×4 旋转视频 |
| `streetcrafter/` | 固定版本上游的依赖、补丁、配置、导出与插入 overlay |
| `work/asset_pipeline/`、`work/road_hazard_pipeline/` | 网格/高斯通用函数及早期资产管线；这里只跟踪源码和材料索引 |
| `outputs/reproducibility/` | 原始来源、原数据准备、标定核对与早期实拍脚本快照 |
| `validation/` | 已执行的检查结果和明确保留的差异记录 |

## 核心入口

| 脚本 | 作用 |
|---|---|
| `scripts/checkout-streetcrafter.sh` | 获取固定上游提交，应用补丁和项目 overlay |
| `scripts/install-streetcrafter.sh` | 获取固定依赖、安装 Python 包、编译 CUDA 扩展 |
| `scripts/packages.py` | 按分组列出下载，校验 SHA-256，再恢复项目目录 |
| `scripts/reconstruct-assets.py` | 按类别、ID 或样例重建网格/高度场，再导出高斯；RGB-D 使用精确输入成员名，避免重复时间戳选错图片 |
| `scripts/verify-assets.py` | 用成品清单检查 GLB/PLY 是否逐字节一致 |
| `scripts/compare-reconstruction.py` | 对重新计算的几何与参数作数值比较，保留边界差异；不能替代精确哈希验收 |
| `scripts/scene.sh` | 055/056 的原数据准备、训练和重建导出 |
| `scripts/multiscene-configs.py` | 在当前克隆目录生成 034/109/110 的数据、模型和渲染配置 |
| `scripts/multiscene.sh` | 新道路的 prepare、train、insert、reload 入口 |
| `scripts/presentation.sh` | 顺序执行最终五类 1080p 综合视频全流程，避免同时启动多份重型 GPU 作业 |

五类具体算法由 `batch_stereo.py`、`batch_rgbd.py`、`reconstruct_pothole.py`、`variant_catalog.py`、`make_variants.py`、`export_gaussians.py` 组成。双目/RGB-D 使用测量表面和源 RGB，后三类使用参数化网格与固定 PBR 输入。方法与命令见 [五类重建](02-asset-reconstruction.md)。

## 全库 4×4 可视化

| 文件 | 作用 |
|---|---|
| `tools/visualization/render_catalog.py` | 读取各资产 PLY，独立取景并旋转 360°，加尺寸/ID 标签，按 4×4 分页编码五份视频；`--category` 可只渲染一类 |
| `tools/visualization/verify_catalog.py` | 逐帧解码，核对帧数、1,199 个 ID 覆盖和旋转画面变化 |
| `library_previews.py` | 制作扩充库概览、静态缩略图集和编号索引 |

```bash
work/envs/streetcrafter/bin/python tools/visualization/render_catalog.py --category missing
```

## 多道路迁入及综合视频

以下文件均在 `tools/multiscene/`，全部随代码上传。

| 文件 | 作用与依赖 |
|---|---|
| `download_scene.py`、`sky_masks.py` | 下载并校验所选道路数据，生成天空掩膜 |
| `prepare_train.py`、`prepare_cpu.py` | 原数据转换、LiDAR 初始化与训练辅助；使用固定配置和来源 |
| `road_plan.py`、`plan.json`、`selected_assets.json` | LiDAR 拟合道路锚点，固定每类 10 个 ID、随机种子和分组 |
| `render_scenes.py` | 按原尺度把 50 个资产迁入三条道路，产生 30 个动态检查点、前后行车和分类近景 |
| `reload_check.py`、`verify.py` | 重载原生检查点，核对位置、尺度、动态对象和输出视频 |
| `present.py` | 合成原尺度的三道路总览、4×4 对照、分类近景与资产检索 |
| `render_presentation_views.py` | 生成同相机、同时间的全局前后，以及匹配的局部迁入前视频 |
| `render_presentation_orbits.py` | 生成 50 个选中资产的独立环绕视频 |
| `compose_presentation.py` | 共用五分区布局和原版静态全局综合展示；后续动态合成复用其排版函数 |
| `render_stereo_detail.py`、`render_stereo_orbits.py` | 收近双目资产观察镜头，增强原尺度细小坑洞的局部观察；不改变其物理尺寸 |
| `compose_dynamic_presentation.py` | 使用连续 40 帧真实行车序列，逐帧同步全局前后，再与环绕和局部前后合成 |
| `render_enlarged_stereo.py` | 单独生成双目几何 10 倍演示：同步缩放 XYZ、高斯尺度和高度场，其余四类仍保持原尺度 |
| `compose_enlarged_stereo.py` | 合成 10 倍双目版环绕、连续行车与局部前后，并标注原尺寸和演示尺寸 |
| `export_motion_confirmation.py` | 将五份最终综合展示输出为 1080p、H.264 Baseline，生成全局预览与循环 GIF |
| `verify_presentation.py`、`verify_dynamic_presentation.py` | 验证原版与连续行车综合版的帧数、覆盖和同步信息 |
| `verify_enlarged_stereo.py` | 核对 30 处放大的点位/尺度、其他资产尺度、动态对象与天空参数 |
| `verify_motion_confirmation.py` | 直接解码最终五份视频，检查全部 150 段两格全局场景确实运动，而不只检查帧号文字 |
| `build_presentation.py`、`build_enlarged_stereo.py` | 较早的分阶段构建辅助入口，默认顺序执行 GPU 作业 |
| `finish_pipeline.py`、`early_render.py`、`parallel_render.py` | 训练/分阶段渲染的辅助调度；新机器优先使用 `scripts/` 中的统一入口 |

## 从新克隆生成最终五份综合视频

先安装环境，恢复 `scenes-034`、`scenes-109`、`scenes-110` 和 `sample-multiscene-50`。这些文件必须已经下载完成；源代码完整不代表云盘的待上传文件已经可用。

```bash
python scripts/packages.py restore --group scenes-034 --group scenes-109 \
  --group scenes-110 --group sample-multiscene-50

# 先查看将执行的命令，不加载 GPU、不修改配置
bash scripts/presentation.sh all --dry-run

# 实际顺序执行，生成的中间视频和检查点需要额外本地空间
bash scripts/presentation.sh all
```

全流程顺序为：原尺度迁入 → 匹配前后视角与环绕 → 原尺度动态合成 → 双目实际几何放大迁入 → 放大版合成与几何检查 → 五份 1080p 转码 → 最终全局运动检查。

可用 `native`、`views`、`compose-native`、`scaled`、`compose-scaled`、`export`、`verify` 选择阶段。后续阶段依赖前序产物，不会自行替代缺失的原生检查点。

最终文件位于 `outputs/道路病害_多场景迁入/全局动态确认版/`；原尺度结果、10 倍演示和 4×4 全库巡览分别保留。双目 10 倍版是演示尺寸，不能用作实测坑洞尺度。

## 早期代码的范围

基础资产管线和 `capture_to_3d` 快照用于解释早期扫描/照片辅助资产，不等同于本次 1,199 条扩充库的默认入口；其额外源素材可能需要另行准备。复现本次五类资产请使用 `scripts/reconstruct-assets.py`，复现最新综合视频请使用 `scripts/presentation.sh`。

源码完整性、实际执行验证和跨硬件字节一致性是三个不同检查。1,186 件资产已在本机重新生成并逐字节一致，13 件双目资产的数值/边界差异继续在 [验证记录](06-validation.md) 中保留，没有将其写成全部通过。
