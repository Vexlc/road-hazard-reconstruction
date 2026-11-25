# 道路散落物的来源与尺度

四件资产使用 Poly Haven 的米制 glTF 网格、原始 UV、漫反射、法线、金属度及粗糙度。每件有五个原始文件，可在 original 目录离线重建。sources.json 登记作者、下载地址、字节数和 SHA-256。组合与落地姿态由本仓库指定。[原始文件与指纹](sources.json) · [Poly Haven 的 CC0 许可](https://polyhaven.com/license)

| 编号 | 原模型与作者 | 场景中的状态 |
|---|---|---|
| R282 | [Trashbag](https://polyhaven.com/a/trashbag)，Benny Weimer | 扎口垃圾袋，保留袋体褶皱 |
| R283 | [Plastic Crate 03](https://polyhaven.com/a/plastic_crate_03)，Pacific Penguin | 红色周转筐倒扣，格孔和把手保留贯通几何 |
| R284 | [Barrel 03](https://polyhaven.com/a/barrel_03)，Serhii Khromov | 直立的蓝色旧钢桶 |
| R285 | [Pipe Wrench](https://polyhaven.com/a/pipe_wrench)，Will Evarts | 钳体侧放，保留下颚、螺母和磨损涂层 |

从给定姿态附近的凸包支撑面中，选择能支撑面积加权壳体质心的一面。它给出接地初值。迁入目标路面后，单独检查每个物体的最低接触顶点、邻近测距和车辆扫过区域。RGB 和井/坑之外的原始道路背景来自 KITTI-360 或 Audi A2D2。这里展示的是把上述模型加到原场景后的合成画面。
