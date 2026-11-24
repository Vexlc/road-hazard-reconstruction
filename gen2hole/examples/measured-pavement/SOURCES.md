# 数据与衍生文件来源

P238 来自 PothRGBD 的 20250305_021453，P239 来自 20250305_124804。采集作者为 Mustafa Yurdakul 和 Şakir Tasdemir。仓库的输入索引保存了各曝光颜色、深度和原始标注成员的 SHA-256。发布模型对原始观测进行了路面坐标变换、高度重建、纹理投影和高斯采样。[原始数据](https://github.com/ymyurdakul/PData/tree/7596dc1c745e68b90403d7e82f89e7492fa95edc) · [原始成员索引](../../../catalog/capture-inputs.json)

实车回放的底图和标定分别来自 KITTI-360 与 Audi A2D2。使用原始数据和衍生文件时，沿用各数据源的使用条件。[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) · [Audi A2D2](https://www.a2d2.audi/a2d2/en.html)

Cycles 对照道路使用 Rob Tuytel / Poly Haven 的 Asphalt 02 沥青贴图，许可为 CC0；两处坑床来自上列 PothRGBD 采集。[沥青贴图](https://polyhaven.com/a/asphalt_02)
