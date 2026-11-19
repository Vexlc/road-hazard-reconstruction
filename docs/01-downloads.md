# 可选下载

所有数据包位于 [Google Drive 项目目录](https://drive.google.com/drive/folders/1jPHeLvbyH7rWThMrXJZVWg3FIOKaVjm_)，按 `assets/`、`inputs/`、`scenes/`、`visualizations/`、`documents/` 分开存放。默认保持私有，请使用有访问权限的 Google 账号。

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

如果使用 rclone 下载私有目录，先在自己的电脑运行 `rclone config` 授权 Google Drive，然后使用清单中的文件夹 ID。上传机器的授权文件、浏览器登录信息与令牌不包含在项目内。
