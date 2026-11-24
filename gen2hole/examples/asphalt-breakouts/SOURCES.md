# 两处道路破损的数据来源

P246 取自 PothRGBD 的 20250305_135325，P247 取自 20250305_122119。两件分别使用不同曝光的颜色、毫米深度与发布标注，标称焦距取 462 px。运行时按采集时间及原档成员名读取，核验原始字节的 SHA-256。源录及每个输入角色见对应时间戳条目。[原始成员索引](../../../catalog/capture-inputs.json) · [Yurdakul 与 Tasdemir 的数据论文](https://arxiv.org/abs/2505.04207) · [仓库取数与来源说明](../../../docs/07-sources.md)

`layout.json` 为实车布局，`crown.json` 为独立的两坡控制场景。`kitti.json` 固定了指定 KITTI-360 0000 宿主中两件的中心、朝向和路面支撑。Audi A2D2、KITTI-360 的实拍曝光用作宿主。Cycles 另用 Poly Haven 的 Asphalt 02 制作控制路面。[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) · [Audi A2D2](https://www.a2d2.audi/a2d2/en.html) · [Asphalt 02](https://polyhaven.com/a/asphalt_02)
