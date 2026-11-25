# 可选下载

[两组坑与雨水口的公开附件](https://github.com/Vexlc/road-hazard-reconstruction/releases/tag/stereo-drains-v1.0.0)提供四件模型、双目输入、两个独立联合检查点和五段视频。[逐文件下载和校验](35-stereo-drains.md#从公开版本开始)

两组实测坑与异常井口的四件模型、原始采集、紧凑曝光、两个联合检查点和四段前后视影片已按附件分别提供。[下载与校验](34-sun-openings.md#下载与校验)

新的道路资产按固定标签放在GitHub Releases。每个专题给出原始输入、成品、相机包或宿主检查点的下载命令和校验值。只看效果时可直接下载专题页开头的 MP4。[实测双坑与车尾鱼眼](24-asphalt-breakouts.md) · [扫描井盖及缺盖状态](22-scanned-round-covers.md) · [道路遗落物](23-road-debris.md) · [按资产类型查找其他专题](asset-code-index.md) · [行驶中的鱼眼与沉降、错位井盖](25-rolling-covers.md) · [双坑和裁切、去畸变相机](26-remapped-pits.md) · [干湿双坑、坡度与鱼眼反光](27-wet-pavement.md) · [三段道路与近物倒影](28-roadside-reflections.md) · [大坑搭配沉降盖或缺盖井](29-mixed-inlets.md) · [四件散落货物和逐件接地检查](30-lost-cargo.md) · [四处实测沥青坑与六面深度鱼眼](31-cube-pits.md) · [旋转方盖、隆起圆井与三维位置缓冲](32-utility-covers.md) · [四处实测剥落坑与原始鱼眼标定](33-mei-pits.md)

五类历史库的数据包位于 [Google Drive 项目目录](https://drive.google.com/drive/folders/1jPHeLvbyH7rWThMrXJZVWg3FIOKaVjm_)，按 `assets/`、`inputs/`、`scenes/`、`visualizations/`、`documents/` 分开存放。已设置为“知道链接的任何人可查看”，无需 Google 账号即可查看和下载；没有编辑权限，且未开启搜索发现。

[云盘校验文件](https://drive.google.com/file/d/1OgOUPyyvvPmWBUiLI6VqRJSa1V95gaNb/view?usp=drivesdk) · [云盘下载索引](https://drive.google.com/file/d/1dgF3qYTP-uydcLfzlJhzHPLJRp-FyLlC/view?usp=drivesdk)

`catalog/downloads.json` 是逐文件清单，记录分组、大小、SHA-256、云盘文件 ID 和下载页面。只下载需要的类别；RGB-D 与异常井盖有多个独立分卷，需要完整类别时请下载该类别所有分卷。

```bash
python scripts/packages.py list
python scripts/packages.py list --group assets-stereo
python scripts/packages.py list --group inputs-rgbd
```

| 分组 | 什么时候需要 |
|---|---|
| `assets-stereo` / `assets-rgbd` / `assets-pothole` / `assets-missing` / `assets-raised` | 使用已生成的某类资产；每卷包含对应的 GLB、PLY、surface 与元数据 |
| `inputs-stereo` | 从原始双目 RGB、分割和 XYZ 重新重建 65 条资产 |
| `inputs-rgbd` | 从 RGB、深度和标签重新重建 RGB-D 类别 |
| `inputs-procedural` | 三类程序化资产使用的精确材质输入，下载一份即可 |
| `scenes-055-056` | 原有两条道路的预处理相机、LiDAR 初始化与 7,000 轮检查点 |
| `sample-scene-001` | 最小迁入样例，包含第 001 组所需的 8 个资产 |
| `scenes-034` / `scenes-109` / `scenes-110` | 三条新增道路的预处理输入和基础重建检查点；是否已发布以清单为准 |
| `visualizations-<类别>` | 直接观看该类别 MP4，不需要 GPU 或模型文件 |
| `visualization-files-<类别>` | 该类别原文件名的视频和预览图，解压到项目布局，用于视频验证 |
| `visualization-index` | 所有视频的时间、页码、行列与资产 ID 检索 |
| `visualizations-scenes` | 原有 A/B 场景导出结果，可选；不影响重新渲染 |
| `visualizations-multiscene` | 新增道路的迁入前后与分类近景，可用性以清单为准 |

把下载文件保存到仓库根目录的 `downloads/`；不要修改压缩包文件名。

```bash
python scripts/packages.py check --downloads downloads --group assets-stereo
python scripts/packages.py restore --downloads downloads --group assets-stereo
python scripts/verify-assets.py --category stereo
```

`restore` 总是先核对整包 SHA-256，再解压。它会保留仓库中的全量资产清单，不会让某个分卷的局部清单覆盖 1,199 条记录。重复恢复会覆盖同路径文件，请在干净克隆中操作。

只看视频时直接下载 MP4 即可；直接 MP4 与 `visualization-files-*` 内的视频是同一内容的两种下载方式，不必同时下载。

浏览器可直接通过上述共享链接下载。如需使用 rclone 批量下载，先在自己的电脑运行 `rclone config` 授权 Google Drive，然后使用清单中的文件夹 ID。上传机器的授权文件、浏览器登录信息与令牌不包含在项目内。

## 生成 RGB 的原图和拟合输入

GitHub 的 generated-rgb-inputs.zip 包含 299 个输入文件和逐文件校验表，可以一次恢复 109 件的原图、标注与缓存拟合。generated-demo-records.zip 保留历史传感器报告和 PandaSet 034 的前视片段。[GitHub 下载和使用命令](13-reproduce-and-transfer.md#重建-109-件生成-rgb-资产) · [传感器记录包](https://github.com/Vexlc/road-hazard-reconstruction/releases/download/sun-openings-v1.0.0/generated-demo-records.zip)

这一批的最新分卷 ID 与校验值见仓库内的 [downloads.json](../catalog/downloads.json)。分组 `gen2hole-inputs-109` 包含两批生成 RGB 的原图、轮廓与缓存深度。运行 `python scripts/packages.py fetch --group gen2hole-inputs-109` 可自动获取、拼合和核对所有卷，再用 `restore` 解压至本机。[逐步复现](13-reproduce-and-transfer.md) · [Google Drive 分卷](https://drive.google.com/drive/folders/12y0agwLAAcPllCtBWqWOc2rctY19gG42)

## 2026-11-20 资产工厂更新

本版已在本机验证五类各10条资产记录、三条道路150处原尺度放置。云端同步尚未完成：44个新增文件中8个已通过服务器SHA-256、大小和公开只读权限检查，其余36个暂不进入可下载清单。准确状态见 catalog/factory-v3-download-status.json，复现说明见 docs/10-factory.md。不要把本机生成完成视为已全部上传。

当前可用：新道路与网格输入、额外PBR材料、元数据、30个设计资产4×4环绕、材质对照、完整井盖取景预览，以及缺盖井和异常井盖两类120秒视频。新增资产包、30份原生场景和另外三类综合视频仍待完成上传。

## 恢复旧版离线浏览页

首页中 2026-11-21 的三种离线浏览页使用相对路径加载模型、原图和传感器数据。完整归档包含 5,926 个文件，约 58.3 GB。使用已有版本目录时，将其放在 `data/releases/2026-11-21/`，然后运行 `python3 scripts/roadgs.py --release 2026-11-21 verify`。校验涵盖整个目录。原始 ZIP 名称、大小和 SHA-256 可在版本登记表逐项核对。[历史数据包登记](../catalog/roadgs-releases.json) · [目录结构与恢复范围](11-development.md#新旧版本如何管理)

| 需要查看的内容 | 在版本目录中打开 | 对应源码或重建方法 |
|---|---|---|
| 首批 9 件与场景对照 | `道路病害图生三维/资产与场景索引.md` | [九件重建和场景代码](../pipelines/image9) |
| 追加 100 件的原图、网格与 Gaussian | `道路病害图生三维_新增100/资产浏览.html` | [生成浏览页](../pipelines/image100/rebuild/gallery.py) |
| 相机、点云和剖面 | `道路病害_传感器与几何增强/传感器对照.html` | [传感器重放](../pipelines/sensors/tools/replay.py) |

手头只有源码时，可从上面的 `gen2hole-inputs-109` 分组恢复图生资产的原图和拟合输入，再跟随复现手册构建所选资产。该分组提供重建输入。完整的历史离线页面及其场景序列依赖对应的历史交付数据。[按编号重建与迁入](13-reproduce-and-transfer.md) · [早期包验证记录](../validation/roadgs-2026-11-21.json)
