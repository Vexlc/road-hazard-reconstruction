# 资产迁入道路场景

将自己重建的 `pilot/` 或 `series/` 资产放入 034、109、110 时，见[准备宿主、原尺度迁入和传感器输出](13-reproduce-and-transfer.md#迁移-p154-到-pandaset-034-并回放)。

此流程在三维高斯场景中替换道路背景，资产和原有动态车辆、轨迹、天空经过同一次 3DGS 渲染。它不是在最终图片上贴图。

## 最小样例：原有道路 055 / 第 001 组

安装 CUDA 环境并下载 `scenes-055-056` 与 `sample-scene-001`，然后在仓库根目录：

```bash
python scripts/packages.py restore --group scenes-055-056 --group sample-scene-001
pipeline="$PWD/outputs/road_hazard_joint_scenes/reproduction"
bash "$pipeline/run.sh" "$PWD" "$pipeline" 055 001
```

结果位于 `outputs/road_hazard_joint_scenes/scene_001/`：原生迁入检查点 `trained_model/iteration_7000.pth`、行车 MP4、近景 MP4/JPEG、放置参数 `scene.json`，以及 showcase 组的整场 PLY 快照。渲染器发现已有 `scene.json` 时会跳过该组，因此重跑对比应使用干净克隆或先将原输出目录改名保存。

## 全库 150 组

下载并恢复五类全部资产包，核验 1,199 个成品：

```bash
python scripts/packages.py restore --group assets-stereo --group assets-rgbd \
  --group assets-pothole --group assets-missing --group assets-raised
python scripts/verify-assets.py
pipeline="$PWD/outputs/road_hazard_joint_scenes/reproduction"
bash "$pipeline/run.sh" "$PWD" "$pipeline" 055
bash "$pipeline/run.sh" "$PWD" "$pipeline" 056
work/envs/streetcrafter/bin/python "$pipeline/verify.py" "$PWD" "$pipeline"
```

`plan.json` 已固定每组资产、宿主道路和米制放置锚点。150 组各最多 8 个资产，合计覆盖 1,199 条记录，每个 ID 一次。所有组输出包含动态检查点，完整重新生成约需数十 GiB；前四个展示组另生成较长视频和静态 PLY。

只下载一个类别时，不要直接执行混合组的完整计划。可从该类别挑选资产，再按下面的锚点格式编辑自己的计划；缺失的类别不会被程序凭空补齐。

## 三条新增道路：034、109、110

下载清单中对应的 `scenes-034`、`scenes-109`、`scenes-110` 包，包含预处理相机/标定/实例信息、LiDAR 初始化和各自的基础检查点。选中 50 个资产的列表是 `tools/multiscene/selected_assets.json`：每类固定随机抽取 10 条，种子 `20261119`。相同 50 个 ID 分别放入三条道路，每条道路 10 组、每组五类各一件。

```bash
python scripts/packages.py restore --group scenes-034 --group scenes-109 --group scenes-110
work/envs/streetcrafter/bin/python scripts/multiscene-configs.py
bash scripts/multiscene.sh insert 034 034_01
bash scripts/multiscene.sh reload 034 034_01
# 去掉最后的组号可生成这一条道路全部 10 组
bash scripts/multiscene.sh insert 109
bash scripts/multiscene.sh insert 110
```

迁入输出位于 `outputs/道路病害_多场景迁入/迁入场景/<道路_组号>/`。每组有五类近景、行车迁入前后、检查点和 `scene.json`。可用 `reload` 重新加载保存的原生检查点，比较五份近景与原 JPEG 的像素误差。

需要从公开原始数据重新训练新道路时：

```bash
bash scripts/multiscene.sh prepare 034
bash scripts/multiscene.sh train 034
bash scripts/multiscene.sh insert 034
```

将 034 换成 109 或 110 即可。下载脚本使用版本化 ZIP 索引，按成员检查原始压缩数据的 CRC32 与大小；天空分割模型也固定 revision。每个场景使用前视相机 80 帧、960×540、7,000 轮独立初始动态 3DGS 训练，未运行扩散蒸馏。重新训练的结果会受 GPU、随机初始化、CUDA 和恢复历史影响；需要重现发布结果时使用已提供的基础检查点。

## 迁入方法及自定义宿主

1. 宿主必须具有可加载的 StreetCrafter 动态高斯检查点、相机标定、时间戳和实例信息。单个场景 PLY 只是某一时刻的快照，不能替代动态检查点。
2. 用宿主道路 LiDAR 在候选点邻域内迭代拟合平面，排除残差过大、点数不足、覆盖不全和动态物体遮挡的位置。
3. 记录米制世界中心 `center_world` 与右手正交旋转基 `basis_columns`。局部 Z 轴沿道路法线，局部 X/Y 位于道路切平面。原尺度迁入使用 `x_world = R @ x_local + center`。
4. 同步变换高斯位置、四元数与球谐方向；尺度保持源资产原米制尺寸。若自行缩放，位置、三轴尺度、路面高度场及边界宽度必须同步修改，不能只缩放中心坐标。
5. 使用 `surface.npz` 的 `alpha` 和 `height` 计算替换区域，切掉可能覆盖病害的原道路高斯，并为其尺度扩张留出余量，避免原道路把坑填平。
6. 从标定相机与重建深度烘焙宿主道路颜色，用曝光调整和边缘过渡高斯融合路面，保留坑洞深度、井筒和盖板结构。实拍 RGB 已含采集光照，当前适配不是完整的物理重光照。
7. 合并背景高斯后保留原有动态对象、轨迹和天空参数。保存原生检查点，重新加载并比较近景与动态帧。

锚点样例见 `outputs/road_hazard_joint_scenes/reproduction/plan.json` 或 `tools/multiscene/plan.json`。新宿主的锚点必须重新估计，不能直接把 055/056 的世界坐标用于另一条道路。`tools/multiscene/road_plan.py` 提供基于 LiDAR、标定和动态掩膜的道路定位实现。
