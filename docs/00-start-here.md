# 下载与复现入口

从[全仓复现手册](16-reproduction-cookbook.md)开始。手头已有编号，查[逐系列重建与迁入代码表](asset-code-index.md)；已有新相机或道路，查[按宿主选择迁入入口](16-reproduction-cookbook.md#各类资产怎样迁入道路)。常见大坑、异常井盖和路面遗落物各有从原始输入到驶过后回看的示例。[20 件常见大坑](16-reproduction-cookbook.md#kitti-360-双坑与后视) · [实测双坑和鱼眼近景](24-asphalt-breakouts.md) · [缺盖井与通气盖](22-scanned-round-covers.md) · [轮胎、轮圈、筐和水泥袋](23-road-debris.md) · [行驶中的鱼眼与沉降、错位井盖](25-rolling-covers.md) · [双坑和裁切、去畸变相机](26-remapped-pits.md) · [干湿双坑、坡度与鱼眼反光](27-wet-pavement.md) · [三段道路与近物倒影](28-roadside-reflections.md) · [大坑搭配沉降盖或缺盖井](29-mixed-inlets.md) · [四件散落货物和逐件接地检查](30-lost-cargo.md) · [四处实测沥青坑与六面深度鱼眼](31-cube-pits.md) · [旋转方盖、隆起圆井与三维位置缓冲](32-utility-covers.md) · [四处实测剥落坑与原始鱼眼标定](33-mei-pits.md)

本文记录五类历史资产库的下载与复现方法，共 1,199 条资产记录。当前图生三维、场景与传感器的本地开发入口见[项目首页](../README.md)。

| 目的 | 文档 |
|---|---|
| 按用途选择下载内容 | [下载清单](01-downloads.md) |
| 从原始输入重建五类资产 | [五类重建](02-asset-reconstruction.md) |
| 换一台机器运行、配置环境 | [环境与移机](03-environment-and-relocation.md) |
| 将资产迁入道路场景 | [场景迁入](04-scene-insertion.md) |
| 查看和重新生成五类视频 | [可视化](05-visualization.md) |
| 查看本次实际验证范围 | [验证记录](06-validation.md) |
| 查询数据来源和上游许可 | [来源说明](07-sources.md) |
| 每份核心/可视化代码的作用与执行顺序 | [代码说明](08-code-map.md) |
| 云盘拟上传体积与尚未补传的文件 | [存储计划](09-storage-plan.md) |

五类不是五套互不相同的采集传感器：前两类为实拍重建，后三类共用程序化几何和 PBR 材质流程，按病害结构分别发布。

| 下载类别 | 数量 | 生成方法 |
|---|---:|---|
| `stereo` | 65 | 双目来源 XYZ + RGB + 病害分割 |
| `rgbd` | 822 | RealSense 深度 + RGB + 多边形标注 |
| `pothole` | 120 | 参数化坑洞高度场与裂损网格 |
| `missing` | 64 | 井口、井沿、井筒、井底的参数化网格 |
| `raised` | 128 | 参数化井口及具有开启、平移、偏转姿态的盖板 |

每个资产有纹理 GLB、3DGS PLY、`surface.npz`、来源与参数 `metadata.json`。GLB 是米制 Y-up；PLY 是米制 Z-up。两个文件是同一资产的两种表示，不重复计数。

最小验证路线：下载 `scenes-055-056` 与 `sample-scene-001` 两组包，安装环境，运行 [场景迁入文档](04-scene-insertion.md) 的第 001 组命令。无需下载全部 1,199 个资产。
