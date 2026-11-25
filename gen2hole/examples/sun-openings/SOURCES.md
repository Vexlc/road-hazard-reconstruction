# 两处实测坑与十四通孔井盖的输入

| 资产 | 输入 | 实测量与设计量 |
|---|---|---|
| P276 | PothRGBD 的 20250227_185200，彩色、毫米深度、轮廓 | 分叉坑沿及坑内高差来自同一次曝光。 |
| P278 | PothRGBD 的 20250227_173336，同样三种输入 | 浅色集料和完整坑口来自另一次曝光。 |
| A277、M277 | ambientCG ManholeCover009、Asphalt031、Concrete049 | 颜色、相对浮雕和十四个孔口来自同一套扫描。盖径 0.62 m、井深 0.54 m 和安装姿态按配方设计。A277 的盖板整体平移，M277 共用井圈和井壁。 |

建坑时按标称焦距 462 px、图像中心主点换算米制几何。留出同帧的部分深度块，在拟合时剔除，复核坑内误差。发布的 captures.zip 保留原始成员名和文件内容。原始材质的配准派生图已随本仓库收录。[PothRGBD 固定修订](https://github.com/ymyurdakul/PData/tree/7596dc1c745e68b90403d7e82f89e7492fa95edc) · [逐曝光哈希](../../../catalog/capture-inputs.json) · [井盖原始扫描](https://ambientcg.com/a/ManholeCover009) · [共享的扫描派生图](../perforated-covers/split-iron) · [原始材质哈希](../perforated-covers/sources.json)

KITTI-360 0000 的太阳方向取自同一段道路上立柱及投影端点的三帧三角化。两个相机分别在原始照片的同类沥青上框选直射和阴影区。kitti-sun.json 中 xyxy 为原始像素的左、上、右、下，右下开区间，RGB 取各通道中位数。Audi A2D2 的左前和车尾各自用邻近路面登记曝光。[方向观测](../roadside-props/kitti-sun-observations.json) · [取样位置及整帧指纹](kitti-sun.json)
