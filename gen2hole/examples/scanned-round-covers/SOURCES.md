# 两套实扫井盖的输入

| 来源 | 发布的整幅图覆盖宽度 | 采用的盖面边界（1024 像素图上的中心、半径） | 用途 |
|---|---:|---|---|
| [ambientCG Manhole Cover 005](https://ambientcg.com/view?id=ManholeCover005) | 约 0.80 m | (505, 512)，421 px | 铸铁盖、内外两圈共 24 个排水口、固定井圈 |
| [ambientCG Manhole Cover 008](https://ambientcg.com/view?id=ManholeCover008) | 约 0.90 m | (513, 525)，384 px | 开裂的混凝土填芯盖、铸铁边缘及固定井圈 |

两项均由 ambientCG 使用表面摄影测量制作，许可为 CC0。仓库保留了 1K 原档中的颜色、相对位移、金属度、粗糙度、OpenGL 法线与不透明度通道。可直接使用这些 PNG，也可以用原 ZIP 和准备脚本重新生成。两个输入 ZIP 的 SHA-256 固定在脚本中。[ambientCG 许可](https://docs.ambientcg.com/license/) · [准备脚本](prepare.py)

盖面直径由整图宽度和图中接缝估算，分界圆近似了实际不规则边缘。005 的通孔边界按已注册的颜色和凹陷共同提取，局部堵塞保留为较小的可见开口，两只矩形提手槽与中央商标保留为表面凹陷。008 的照片显示龟裂和剥蚀。项目没有测量裂纹以下的结构、井深、盖厚和井座，它们以及移位、缺失状态使用随库配方中的设计值。井壁混凝土与周边沥青沿用 CC0 的 Concrete 049、Asphalt 031。[既有衬壁和路面材料来源](../perforated-covers/sources.json)

实车底图及对应标定来自 KITTI-360 和 Audi A2D2。独立验证场景的道路纹理由 Rob Tuytel / Poly Haven 提供。[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) · [Audi A2D2](https://www.a2d2.audi/) · [Asphalt 02](https://polyhaven.com/a/asphalt_02)
