# 五类可视化与重新渲染

## 本次新增的综合版

全局动态确认版的五份视频均为 1920×1080、24 fps、120 秒。每类 10 个资产 × 3 条道路，30 段；每段 4 秒，包含资产环绕、连续行车全局前后、局部前后。双目三维 XYZ、Gaussian 尺度和路面高度场实际放大 10 倍，其他四类保持原尺度；原尺度版本另保留。它是演示放大版，不能当作原始实测尺寸。

五份视频的全部 150 段已重新执行全局运动检查。生成代码在 tools/multiscene；完整移机执行入口仍待最后验证。

## 全资产 4×4 巡览

五份 MP4 分别覆盖每类所有资产，4×4 同屏、每页 4 秒、每格旋转一周；不足 16 件的末页留空。分辨率 2560×1920，24 fps，H.264。

| 类别 | 资产 | 页数 | 时长 | 直接视频文件 |
|---|---:|---:|---:|---|
| 双目 | 65 | 5 | 20 秒 | `visualization-stereo.mp4` |
| RGB-D | 822 | 52 | 208 秒 | `visualization-rgbd.mp4` |
| 程序化坑洞 | 120 | 8 | 32 秒 | `visualization-pothole.mp4` |
| 缺盖窨井 | 64 | 4 | 16 秒 | `visualization-missing.mp4` |
| 异常井盖 | 128 | 8 | 32 秒 | `visualization-raised.mp4` |

下载页面见 `catalog/downloads.json` 或运行 `python scripts/packages.py list --group visualizations-rgbd`。GitHub Release 的独立视频附件待审阅确认后发布。

`catalog/visualization-index.json` 为每条资产记录视频原文件名、页码、行列、起止秒数和完整 ID。屏幕内每件资产独立取景，画面大小不能作为实际尺寸的比较依据，应读取尺寸标签或资产元数据。

## 从 PLY 重新渲染

准备所选类别的资产包和 CUDA 环境：

```bash
python scripts/packages.py restore --group assets-missing
work/envs/streetcrafter/bin/python tools/visualization/render_catalog.py --category missing
```

输出到 `outputs/道路病害_五类4x4视频/`，渲染日志与分段索引写入 `work/visualization/`。去掉 `--category` 会渲染所有五类，需要全部资产；`--test --category missing` 只生成该类别首件资产的渲染测试图。

视频合成需要中文字体，Linux 可安装 Noto CJK（例如 Ubuntu 的 `fonts-noto-cjk`）。FFmpeg 来自 `imageio-ffmpeg` Python 包，不依赖系统预装的 `ffmpeg` 命令。

## 校验下载的视频

恢复五类 `visualization-files-*` 与 `visualization-index` 后运行：

```bash
work/envs/streetcrafter/bin/python tools/visualization/verify_catalog.py
```

该脚本逐帧解码，检查帧数、分辨率、帧率、1,199 个 ID 的完整覆盖与不重复，并比较相隔半周的画面变化。它会把校验结果写到输出目录，不修改发布包。

完整场景视频与资产独立视频分开：独立视频用于检查形状，场景近景与行车视频用于检查迁入后遮挡、尺度和路面接缝。前视单相机训练的场景在近景新视角下可能出现背景模糊或拉伸，这属于宿主重建覆盖范围的限制。
