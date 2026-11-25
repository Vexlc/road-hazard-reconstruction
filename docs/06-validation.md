# 代码发布与复现验证状态

两组路面散落物以 12 份实拍曝光复核 60 个产物，另外用四个全经度双目机位检查接缝、空背景与前景遮挡。[方法与逐资产记录](36-street-clutter.md)

两段 KITTI 左右图的三角化距离由 12,667 个独立激光投影点检验。另有四个 220° Cycles 机位检查非平行双目的资产轮廓、部件和近景遮挡。[检验方法与精度范围](35-stereo-drains.md#在-cpu-上重放实车与双目实验) · [逐机位记录](../validation/stereo-drains.json)

历史生成图两批共 109 件，干净目录从归档输入构建后，218 个 GLB、PLY 均与原发布指纹一致。可运行 `python3 gen2hole/gen2hole.py audit --rebuilt runs/all-generated` 直接复查文件。[逐件哈希](../validation/generated-rgb-replay.json) · [重建命令](13-reproduce-and-transfer.md#重建-109-件生成-rgb-资产)

两组混合资产用 8 份实拍曝光检查 RGB、实例、部件和径向距离。独立 Blender 几何对照另检查 21,706 个井内外表面样点的直接光。[新资产与迁入验收](34-sun-openings.md) · [逐机位记录](../validation/sun-openings.json)

本次发布核心重建、场景迁入、全部可视化源码及其说明。源码清单、语法、本地模块引用和统一视频执行顺序已检查；模型与视频不进入 Git。Google Drive 的 45 个包、17.54 GB 已全部上传，并完成云端大小与可用哈希核对，详见存储计划。

最新源码检出的资产覆盖与验证命令见[全仓手册](16-reproduction-cookbook.md)。20 个常见大坑的输入与原生输出核对记录在[双坑合集验收](../validation/reproduction-2026-11-25.json)，P250/P251 的留出深度、检查点、真实前后视和重采样机位记录在[双坑与相机验收](../validation/remapped-pits.json)。四件散落货物另有原始层级顶点、逐件接地、实拍前后视、独立 Cycles 对照和重载检查点记录。[散落货物验证](../validation/lost-cargo.json)

P272–P275 在三条实车轨迹中各放两件。两例使用独立 RGB-D 曝光验收深度，十二份实拍相机输入可逐文件重放。左右原始鱼眼另通过 C++ 正向投影检查可见域，四次曝光经 OpenGL 逐三角形核验。[重跑步骤](33-mei-pits.md) · [分资产与相机的误差报告](../validation/mei-pits.json)

四种方井和圆井状态分两组迁入指定检查点。六份独立正交及鱼眼曝光均与原生实例标记吻合，八份实拍曝光的颜色、部件、实例和距离可逐文件复放。错转方盖另用缺盖井的几何独立检查承托。[四种井盖状态的验收记录](../validation/utility-covers.json) · [从原材料复跑](32-utility-covers.md)

P266–P269 分两组接入指定 KITTI 检查点。验证覆盖八份实拍曝光与四个 OpenGL 六面转鱼眼机位。三件留出同帧深度块，另一件使用第二次实拍验收。原生栅格对照、实拍复核和资产原始深度误差分别列出。[四处沥青坑验证记录](../validation/cube-pits.json) · [重跑步骤](31-cube-pits.md)

下文的 1,199 件指五类历史库。

## 已重新执行

- 原库全部 1,199 个资产的 GLB/PLY 与已有版本化 SHA-256 清单逐项一致，surface 和元数据齐全。
- 在新目录解压原始输入，重新生成全部 1,199 个资产。
- 按发布时的精确 RGB/标签成员重建，822 个 RGB-D、120 个设计坑、64 个缺盖井、128 个异常井盖均已逐字节复现原 GLB、PLY。[先前全量哈希记录](../validation/full-reconstruction-exact-hashes.json)
- 本次单独从原始 XYZ、RGB 和轮廓重建 65 个历史双目样本，恢复主坑排除区的平面拟合规则。全部 GLB、PLY 与发布清单的 130 个 SHA-256 一致，高度场、纹理、主掩膜、采集记录共 260 个文件与发布文件相同。13 个曾有差异的样本在 39 个固定机位的原生 Gaussian 回放中，RGB、透明度和前景深度一致。两次验证合计覆盖五类历史库 1,199 件。[本次逐件报告](../validation/stereo-replay.json) · [重跑命令](02-asset-reconstruction.md#从干净克隆验证历史双目)
- 新目录重新生成 055 道路第 001 组迁入，加载检查点后复现动态第 44 帧；与保存 JPEG 的平均绝对差为 0.935/255。
- 新目录重新导出 056 全部 80 帧，PSNR 28.25214、SSIM 0.87072、LPIPS 0.12703。
- 五类 4×4 视频完成全帧解码、帧数和 1,199 个 ID 覆盖检查。
- 最新五类 1080p、2 分钟综合版完成全部 150 段全局场景运动检查。
- 旧云盘 fu-xi 已确认 trashed=true，没有永久清空。

## 本次代码完整性检查

- 为最终五类综合视频增加 scripts/presentation.sh，提供全流程与分阶段入口，GPU 作业默认顺序执行。
- 修复了移机后的 PandaSet SDK 导入路径、辅助构建目录未创建、旧辅助脚本等待本机已有日志的问题。
- 新目录恢复 034 运行包和选中 50 件资产后，重新生成 034_01；五类近景从保存的原生检查点重载，最大平均像素差为 0.455/255，记录见 current-new-road-reload.json。
- 使用统一 presentation.sh verify 入口重新验证最终五份视频全部 150 段运动，记录见 current-final-video-motion.json。
- 无 README、虚拟环境、缓存、模型、视频或授权配置进入代码提交。

## 跨机器验证范围

现有记录来自同一台 Linux/NVIDIA 主机的新输出目录。换另一台物理机完整重跑 30 组原尺度与 30 组放大版视频仍待验证。新机器的依赖和 CUDA 扩展需要按本机架构安装。[环境和换机](03-environment-and-relocation.md)

## 证据

- full-library-hashes.json：原库 1,199 项成品检查。
- full-reconstruction-exact-hashes.json：首次全量重建精确哈希比较，记录当时的 13 个双目差异。
- stereo-reconstruction-differences.json：13 个双目资产的参数和边界点数诊断。
- [stereo-replay.json](../validation/stereo-replay.json) 记录修正后的 65 件和 39 帧。
- original-* 为此前项目运行记录。

本次验证在当前 Linux/NVIDIA 主机的新目录执行，尚未访问另一台物理电脑。新机器仍需自行安装依赖与对应架构的 CUDA 扩展。
