# 道路病害资产工厂

**路线状态（2026-11-23）**　程序化造坑已封存，P160–P189 为最后一批。以下造坑命令供历史复现使用。新增坑洞优先从带尺度的参考照片、双目或 RGB-D 观测重建。[末批归档](#大面积坑洞) · [实拍重建入口](02-asset-reconstruction.md#多帧实拍重建同一组坑洞)

2026-11-20 已验证：30个新设计资产；五类各10条资产记录，迁入3条道路，共150处放置、30份原生检查点。五份综合视频各120秒，另有30个设计资产的两页4×4环绕。实拍记录尚未按物理坑洞去重。当前44个新增文件中8个已完成云端大小、SHA-256和公开只读权限核对并进入下载清单，另36个尚未完成；状态见 catalog/factory-v3-download-status.json。共享rclone客户端项目触发每分钟请求配额，当前暂停重试并调整传输通道。连接器单文件上限为100MiB。

按米制设计连续追加坑洞、缺盖井和异常井盖，并将五类资产迁入标定道路场景。双目与 RGB-D 继续使用原实拍库；放大演示资产不参与本流程。

## 生成与追加

在原项目根目录运行，output 放交付资产，work 放缓存和日志：

```bash
python scripts/hazard_factory/run.py --output /absolute/output/资产库 --work /absolute/work --seed 20261120 --start 0 --count 12
```

追加时保持种子、增加索引，例如 `--start 12 --count 3`，再使用 `--start 15 --count 30`。同一个 seed/index 总是对应同一配方；形状、材质、井沿状态分层轮换，尺寸、磨损和姿态扰动由种子决定。生成可指定任意正整数数量；五类联合迁入要求新的三类数量平衡，建议按 3 的倍数追加完整批次。

每个资产包含 GLB、3DGS PLY、米制切口场、预览、完整配方、质量记录和内容校验。网格、Gaussian、独立检查分为子进程阶段，使用现有 assets 和 streetcrafter 环境。检查通过才原子发布；失败目录不进入索引。重跑会核验实际文件哈希，损坏文件重建，旧版隔离。不同尝试使用各自的中间网格目录，避免被中断的任务污染后续尝试。

pipeline_provenance.json 记录生成源码与材质文件哈希。改变生成代码或素材后，旧资产不会被当作当前版本缓存。

## 迁入

任务 JSON 包含 output（迁入场景目录）、work（日志/掩膜目录）、libraries（原实拍库和新库）、plan（LiDAR 平面/放置点/分类顺序/源帧范围）、configs（场景 ID 对应的原生重建配置）、transformers_path（本机已准备的分割依赖目录）。

```bash
python scripts/hazard_factory/migrate_all.py --job /absolute/path/migration_job.json --workers 2
```

入口先核验或生成道路分割缓存，再运行各场景任务；失败任务单独重试一次。内容未变的已发布场景直接核验复用。case_plan.json 持久保存旧组编号，新种子即使数值较小也只向后追加；联合迁入按每个场景组单独生成道路补面与缓存签名。工作目录分组丢失时，从交付目录的场景分组计划.json 恢复。每组含双目、RGB-D、程序化坑洞、缺盖井、异常井盖各一个。

路面颜色取自标定源 RGB，要求像素被固定版本 SegFormer 识别为 road，且不在数据集动态物体掩膜内。道路分割是模型预测，并非人工真值。设置 annotated_road_cache 后，几何参考只选取原始 LiDAR 中索引严格对齐的 Road 与道路标记真值，拟合帧与核对帧分开；颜色采样仍使用模型预测的图像道路掩膜。几何由 LiDAR 道路平面约束，渲染深度用于软排序；资产实际占地观测覆盖率必须达到 80%。

material_contact 启用网格部件识别：只有实际路面部件混入宿主道路色，井盖背面、侧壁及井框保留金属材质。接触遮蔽来自三维半球射线，仅压暗宿主道路颜色，不改变几何。切除旧路面使用原高斯旋转、协方差在边界方向上的三倍标准差及井深范围。资产不缩放，保留车辆、轨迹和天空。每份检查点落盘后立即重新加载，复现五类近景。

### 低视角道路遮挡

迁入默认增加原生三维道路补面，覆盖低视角射向地下井壁的视线与路面的交点。直接观测区按 LiDAR 拟合地形；近处盲区只在三个相邻可靠道路平面的高度预测差小于 2.5 cm、距离已测地形小于 3 m、且源 RGB 确认为道路时延伸，并单独记录推断数量。所有资产的原切口区域继续排除，井筒保持完整。

补面是写入检查点的 Gaussian 几何，不依靠视频裁剪或图像遮挡。原始相机、LiDAR、资产、配置和补面源码参与缓存校验。显式设置 `road_occlusion_guard: false` 可复查旧版行为。`mesh_cache` 可指定生成阶段网格缓存路径，供独立验收使用。

## 展示与验收

```bash
HAZARD_FACTORY_JOB=/absolute/path/migration_job.json work/envs/streetcrafter/bin/python scripts/hazard_factory/present.py
work/envs/streetcrafter/bin/python scripts/hazard_factory/verify_release.py --job /absolute/path/migration_job.json --presentation
python scripts/hazard_factory/audit_all.py --job /absolute/path/migration_job.json --workers 2
work/envs/streetcrafter/bin/python scripts/hazard_factory/verify_media.py --job /absolute/path/migration_job.json
```

展示包含资产 360° 环绕、连续 40 源帧的全局前后、同相机局部前后。圆圈只提示位置，几何尺度始终为 1。独立验收读取 PLY/GLB 与全部原生检查点，核对位置、旋转、尺度、剩余背景及动态参数，解码近景/全局视频，并检查最终综合视频中的实际场景运动。

可见性检查对完整 40 个行车帧进行地下井壁颜色探针：只在内存中改变地下点的 SH 颜色，几何、透明度和原检查点不变。记录井口及其滤波覆盖区以外的颜色响应，同时保留原始响应，不把像素覆盖误判为实体穿出。视频专用检查分别把最终的全局前、后面板与原生视频匹配，验证实际画面及帧顺序。

视频先在私有临时目录全部生成，再逐文件原子替换并最后发布带内容哈希的播放索引；单次渲染中断时保留已有完整视频。多文件切换不是单一目录事务，读取者通过索引哈希检测切换。环绕缓存同时验证源码、PLY、NPY 内容、尺寸和类型，损坏时重新渲染。check_media_recovery.py 用隔离夹具中断真实渲染进程，并破坏/恢复 CUDA 环绕缓存。

完整恢复的81张图像道路掩膜在验证源图、动态掩膜和缓存哈希后可以离线使用，无需导入 transformers 或模型权重。缺失帧才加载固定版本模型；新道路先在 CUDA 环境运行 prepare_model.py 准备权重。

补面的高度另与未参与拟合的 LiDAR 帧比较。没有独立观测的盲区仍标记为推断，不能把这部分称为实测地形。

check_recovery.py 是首批验收夹具：要求专用目录恰好含种子 20261120 的首 12 个资产，会故意损坏预览、删除 PLY、中断追加，再恢复到 15 个。日常运行使用正常生成入口即可。

## 真实性边界

新资产是扫描材质驱动的合成几何，尺寸为设计值；实拍测量限制保留在原元数据中。井盖加入几何凸纹、磨损、提盖下凹槽，井壁/井底/背面/厚度也是实际几何。采样对井盖正面与井框增加密度。

目前仍为烘焙 AO 与一阶 SH，尚未完整物理重光照；单前视背景在近距虚拟视角下仍会模糊。文件、几何和运动通过不等于照片级真实。后续仍需改善材质过渡、光照一致性、道路细节、实拍去重和可感知质量评估。

## 换目录恢复

公开仓库的 catalog/factory-v3-*.json 与下载包共同保存配方、选择计划与分组编号。先恢复 factory-v3-assets、factory-v3-inputs、factory-v3-extra-material、原始20个实拍资产和三条道路运行包，再在 CUDA 环境运行 scripts/hazard_factory/configure.py。基础 PBR 包的内容会补到原生成器使用的 work/road_hazard_sources 路径；额外 Road008C 两个输入参与已发布生成版本的指纹，也需要恢复。配置器仅重定位 YAML 与相应完成记录中的配置哈希，原生检查点字节保持不变。迁入缓存签名包含本机 job/config，换目录后重新迁入会生成新缓存。

当前独立标注高度核对仅取每条道路第一组作为几何诊断，不能支持所有补面已被独立多帧核对的结论。110 道路的多帧有效观测不足，诊断保留 insufficient_support；可见性探针也保留待复查的微弱残留。

## 恢复本版数据的顺序

本版资产和原生场景包尚未上传完成，以下完整恢复流程须等所需分组在状态清单中全部可用后执行。先用 catalog/downloads.json 中的精确SHA-256核验下载包；基础环境与原始道路恢复参见 docs/03-environment-and-relocation.md。恢复 inputs-procedural、sample-multiscene-50、scenes-034、scenes-109、scenes-110，再恢复 factory-v3-assets、factory-v3-inputs、factory-v3-extra-material。需要直接加载本版原生场景时，再恢复三个 factory-v3-scenes-* 分组。

```bash
work/envs/streetcrafter/bin/python scripts/hazard_factory/configure.py
python scripts/hazard_factory/run.py --output outputs/hazard_factory_v3/资产库 \
  --work work/hazard_factory_v3 --seed 20261119 --start 15 --count 3
python scripts/hazard_factory/migrate_all.py --job work/hazard_factory_v3/migration_job.json --workers 2
```

追加从该种子未使用的索引15开始；固定seed/index重跑会检查内容哈希并复用。输入包保存的是派生的道路标签子集、图像道路掩膜与网格部件信息，不包含虚拟环境或账户配置。

只看视频可恢复 factory-v3-video-stereo/rgbd/pothole/missing/raised 和 factory-v3-metadata。验证最终视频与原生40帧的对应关系，还需三个原生场景分组及资产/道路输入；运行 verify_media.py --job work/hazard_factory_v3/migration_job.json。视频中的原生输入绑定使用场景及视频文件哈希，因此机器路径重定位不会导致画面证明失效。

恢复工具现在按扩展名及文件头区分ZIP与zstd；大型tar.zst中的NPZ末尾记录曾导致ZIP探测误识别，该问题已用本版实际归档恢复测试覆盖。

另一个源码目录的30份恢复检查点与原文件字节一致，重新迁入3份第10组场景得到30张像素完全一致的前后图；仍是同一物理机器，不能当作跨硬件验证。所有1200个行车帧已执行低视角井壁探针，最多6个微弱残留像素继续保留复查标记。部分竖起井盖的局部近景有边缘裁切，尚未修正。

## 完整井盖取景预览

camera_fit.py 根据完整井框、井盖的几何包围盒和15mm外扩量求解相机距离，保留28px画面边距。review_framing.py 已对60处井类放置、1440个观察角度检查几何范围，原生检查点保持不变；109_10的2秒实渲染对照中，原固定距离镜头裁切井盖上沿，新镜头显示完整井盖。此预览尚未集成到五类综合视频或联合迁入主入口，不能据此声称主入口已修正局部裁切。

运行时设置 HAZARD_FACTORY_JOB、ROAD_BATCH_WORK、FRAMING_REVIEW_OUTPUT；用相应配置运行 review_framing.py --config /absolute/pandaset_109.yaml mode trajectory。FRAMING_REVIEW_CASE 默认为109_10。该脚本只写入指定预览输出目录，不改变原生场景。

## 大面积坑洞

P160–P189 是程序化造坑路线封存时的最后一批设计资产，包含碗状坑、车辙坑、连通坑、月牙坑、群坑、浅层剥落、补丁破损、深层破损、板角破损和积水坑。每种形态建了三档面积。模型保留米制尺寸，坑沿、坑底和局部碎块由几何生成，使用现有路面扫描材质。积水坑单独保存水面和坑底。

用水平路面以下至少 1 厘米处的网格采样统计水平投影，新坑覆盖 3.33–14.44 平方米。按同一口径，原扩充库和 v3 工厂的 130 个程序化坑洞最大为 1.10 平方米。独立复核在同一高度裁切导出 GLB 的三角形，30 个面积与采样结果的相对差均小于 0.52%。[同尺度俯视图](figures/large-potholes.jpg) · [逐件面积与模型校验值](../validation/large-potholes-areas.json)

归档包按项目相对路径保存这 30 件成品、生成与审查源码、实际用到的扫描材质和环绕视频。可解压到新目录查看，或用项目已有的 assets、streetcrafter 环境复现。[本地封存包](../data/history/2026-11-23/procedural-potholes-P160-P189.tar.zst) · [归档 SHA-256](../catalog/procedural-potholes-P160-P189.sha256) · [原环绕视频](../outputs/large-potholes/orbits.mp4)

在仓库根目录复现时使用以下命令。材质取自 `inputs-procedural` 恢复的 `work/reconstruction-inputs/pbr`，也支持原库的 `work/road_hazard_sources`。构建程序向新目录发布已核验的资产。只复建一个编号可用 `--ids P181 --out runs/P181-large`。

~~~bash
work/envs/assets/bin/python scripts/build-large-potholes.py
work/envs/streetcrafter/bin/python scripts/review-large-potholes.py --video
python scripts/build-asset-atlas.py
~~~

本地资产在 `outputs/large-potholes/pothole/<编号>/`，每个目录有 GLB、高斯 PLY、米制高度场、尺寸记录和预览。面积对照图和 40 秒环绕视频位于 `outputs/large-potholes/`。图谱中搜索「大面积」可查看全部 30 件。
