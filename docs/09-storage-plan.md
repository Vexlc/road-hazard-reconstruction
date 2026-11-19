# Google Drive 上传体积

当前选择性下载清单共 45 个文件，合计 **17,538,283,067 字节，即 17.54 GB / 16.33 GiB**。这包含所有可选包，使用者无需全部下载。

| 内容 | 计划体积（十进制 GB） |
|---|---:|
| 五类成品资产：GLB、Gaussian PLY、预览、surface、元数据 | 8.406 |
| 原始重建输入：双目、RGB-D、程序化 PBR | 3.030 |
| 五条道路的基础运行包：055/056、034、109、110 | 3.625 |
| 迁入最小样例与选中 50 件资产 | 0.351 |
| 可视化、可选原版导出与打包副本 | 2.127 |
| **合计** | **17.538** |

其中最新五份 1080p、2 分钟综合视频合计 **422,710,100 字节，约 423 MB**。直接 MP4 与某些可视化归档中的视频内容重复，用于直接观看和按项目布局解压两种方式；不必同时下载。

当前云端目录读回查到并核对文件大小的内容为 **39 个文件、14.49 GB**；以下六个文件仍待补传或完成云端确认，共 **3.05 GB**：

- `pandaset-034-runtime.tar.zst`
- `pandaset-109-runtime.tar.zst`
- `pandaset-110-runtime.tar.zst`
- `multiscene-50-selected-assets.tar.zst`
- `dynamic-presentation-metadata.tar.zst`
- `multiscene-original-scale-previews.tar.zst`

这个状态是云端文件存在与大小的读回核对，不冒充对所有文件重新下载后的全量 SHA-256 检查。每个本地发布包的 SHA-256 已写入 `catalog/downloads.json`，下载后由 `scripts/packages.py` 校验。

剩余云盘传输保持暂停；本次只发布已经检查完整的代码和文档。GitHub 中包含源码与小型索引/报告，模型、数据、虚拟环境、缓存和视频不进入 Git 历史。
