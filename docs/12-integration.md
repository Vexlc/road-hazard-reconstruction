# 四段工作的来源与去向

`I2hole2road` 是本机开发根目录，项目简称沿用最后一段对话选定的 **RoadGS**。当前版本依据 11 月 21 日最终交付中的资产清单、相机与雷达索引、原生场景及校验文件确定。11 月 18 日的实拍与程序化资产库、11 月 19 日开始的稳定工厂，各有独立的数据和复现入口。

| 来源与对话 | 归入本项目的内容 | 本地入口 |
| --- | --- | --- |
| `2026-11-18/fu-xi` · [复现 StreetCrafter 并选水泥路数据](codex://threads/01a1eb14-bb33-7c20-a9b1-abbf3ba2bbee) | PandaSet 055/056 的 7,000 轮初始 3DGS、沥青与水泥路面上的病害资产、环境和基础管线 | [场景复现](03-environment-and-relocation.md) · [早期说明](../data/history/2026-11-18/fu-xi-README.md) |
| `2026-11-18/3d-x20-3d-2` · [制作道路坑洞与缺失井盖资产](codex://threads/01a1eca1-a44e-7a23-a0c3-debfeb65379d) | 65 件双目、822 件 RGB-D、120 件建模坑洞、64 件缺盖井、128 件异常井盖，共 1,199 条记录；150 组联合场景、15 卷资产包和相邻机位对照 | [历史交付](../data/history/2026-11-18/capture) · [资产库](../outputs/road_hazard_expansion) · [联合场景](../outputs/road_hazard_joint_scenes) |
| `2026-11-19/home-xiaosz-documents-codex-2026-11` · [梳理现有资产类别](codex://threads/01a1f391-f6ce-70e0-94b7-45aed1f07d79) | 三段新道路、v1/v2 与 v3。采用 11 月 20 日收尾的 v3 记录，五类各 10 件、三条道路、150 次原尺度放置，另留取景优化预览 | [v3 成果](../outputs/道路病害_稳定生成与迁移_v3) · [v1/v2 与取景预览](../data/history/2026-11-19) · [工厂文档](10-factory.md) |
| `2026-11-21/home-xiaosz-documents-codex-2026-11` · [生成真实路面危险资产](codex://threads/01a1fbd8-09fd-7520-8a6a-c2301267dee6) | 首批 9 件加独立新增 100 件。传感器增强复用这 109 件资产，提供三路段的前视、后视、雷达和局部几何 | [完整本地交付](../data/releases/2026-11-21) · [维护源码](../pipelines) · [文件清单](../catalog/roadgs-2026-11-21.sha256) |

以上四个来源目录原位于 `/home/xiaosz/Documents/Codex/` 下。历史说明和实验日志保有生成时的上下文。从当前仓库运行时采用开发指南中的相对路径和配置命令。已归档的试验脚本可在 `data/history/<日期>/` 中检索。实际开发入口采用交付包中的最终重建和传感器源码，分批试验中用于改写生成脚本的一次性工具留作溯源。[开发指南](11-development.md) · [早期技术记录](../data/history/2026-11-18/capture/道路病害三维资产与场景验证技术记录.md) · [相邻机位交互对照](../data/history/2026-11-18/capture-work/multi_view/paired-road-cameras.html)

## Git 历史

`main` 接续 `Vexlc/road-hazard-reconstruction` 的公开提交 `17d8a9f`，本地标签 `archive/github-2026-11-20` 对应该次整合的起点。该提交已包含 v3 迁入、取景核对与当时的下载状态。[上游提交](https://github.com/Vexlc/road-hazard-reconstruction/commit/17d8a9f86b52d7a7d02cd4e0fc9d1926395545a5)

`fu-xi` 来自另一条提交历史。`archive/fu-xi-2026-11-20` 保存原 `18fe711` 及其先前提交，并收录原工作区尚未提交的 README、Gaussian 导出修订和工厂脚本。重合的运行脚本以公开仓库中可重定位的版本为基础。可用下列命令查阅任一旧版本。

~~~bash
git log --oneline archive/github-2026-11-20
git log --oneline archive/fu-xi-2026-11-20
git show archive/fu-xi-2026-11-20:README.md
~~~

## 本地数据和验收

11 月 21 日发布树包含三个原始压缩包、展开的两批资产、111 个场景检查点、前视序列和传感器包。三个 ZIP 的大小与 SHA-256 记在版本清单内，整棵树共 5,926 个文件、58,304,563,851 字节。整合时从旧交付计算哈希，再从新目录逐文件读取并核验。[版本清单](../catalog/roadgs-releases.json)

本机重新生成了 P04。GLB、Gaussian PLY、路面场、逐点材质和元数据均与交付逐字节相同。034_01 和 110_37 两个原生场景分别覆盖首批和新增批，重渲染帧的 RGB 哈希与原冷启动记录一致。A13 的雷达局部回放在前后两次导出中匹配了 5,624 个原始点行号，其中 351 个回波位置发生变化。其余场景沿用原交付验证记录，本次整合进行了文件级全量核验。[本机结果](../validation/roadgs-2026-11-21.json)

本阶段的 Git 提交、标签和新增数据交付均保存在本机。旧版本的公开下载记录保留了当时的状态。[既有存储记录](09-storage-plan.md)

## 11 月 22 日清理后的入口

从图像设计坑洞、缺盖井和异常井盖时，用仓库内的 [Gen2Hole 重建流程](../gen2hole/docs/pipeline.md)。从实拍 RGB-D、双目数据或程序化配方重建时，用本项目的[五类资产入口](02-asset-reconstruction.md)。两者共用本项目的道路环境、PandaSet 数据与发布目录。

11 月 18、19、21 日三个 Codex 目录保留了原位置的代码。道路实验里仅在旧目录找到的源码、配置和说明另收进[历史源码和配置](../data/history/retained-road-sources-2026-11-22/历史源码和配置.zip)。[文件来源索引](../data/history/retained-road-sources-2026-11-22/文件来源.jsonl.gz)按旧路径记录 SHA-256 和保存位置。旧发布包里单独保存的脚本和配置按日期另存，原始输入的补充副本也在[同一归档目录](../data/history/retained-road-sources-2026-11-22/)。

旧目录的模型、视频、检查点、重复打包和下载缓存按文件清理。可查阅[逐文件清单](../data/history/retained-road-sources-2026-11-22/已清理文件.jsonl.gz)和[清理与重建核验结果](../data/history/retained-road-sources-2026-11-22/清理核验.json)。

Gen2Hole 已连同参考图、深度权重、建模配方和已运行案例迁入本仓库的 [gen2hole](../gen2hole/README.md)。原同级目录已移除。`gen2hole/published` 在仓库内指向同一份发布树。按来源检索现有资产可打开[统一图谱](../tools/asset_atlas/index.html)。

本轮追加清理 73,167 个文件，按最后一个硬链接被删除时的已分配块统计，另释放 25.87 GiB。三个日期目录独立计量约为 1.34 GiB、165 MiB 和 38 MiB。剩余内容包括代码、配置、Git 历史及其他工作的原始文档。Fan Stereo 的 Git 对象保存在合并工程的历史目录，旧检出通过 gitdir 文件访问同一提交。原位置的 48,553 个源码文件逐个复核通过。[本轮逐文件记录](../data/history/retained-road-sources-2026-11-22/合并后清理文件.jsonl.gz) · [两轮清理汇总](../data/history/retained-road-sources-2026-11-22/清理核验.json)
