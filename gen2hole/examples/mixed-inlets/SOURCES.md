# 两处坑床与十六孔填芯盖

P260 使用 PothRGBD 的 20250227_184048，P261 使用 20250305_053657。两帧均含实拍彩色、以毫米记录的光学深度和坑口标注。重建采用标称 462 像素焦距，拟合前留出原始深度像素块。两个坑的原始输入随 captures.zip 提供。[测量原始文件指纹](../../../catalog/capture-inputs.json) · [PothRGBD 原作者发布版本](https://github.com/ymyurdakul/PData/tree/7596dc1c745e68b90403d7e82f89e7492fa95edc)

井盖取自 ambientCG 的 Manhole Cover 004。发布者给出的画幅约 1.25 × 1.25 m。1K 扫描中盖板接缝半径取 285 px，中心 (527, 495)。盖板外径约 0.696 m，外围固定框带宽约 49 mm。原扫描有十六个孔位，左上一个孔位中有一道可见的遮挡，其材质和厚度未知。重建沿十七块可见暗区贯通，保留该遮挡。摄影测量提供盖面颜色、法线、粗糙度、金属度和相对位移。[扫描源](https://ambientcg.com/a/ManholeCover004) · [CC0 许可](https://docs.ambientcg.com/license/) · [提取与分割代码](prepare.py)

两种安装状态使用同一组固定井圈及井筒。配方设定井中心下沉 26 mm、坡角 3.2°、盖厚 40 mm 和井深 0.62 m。A259 安放盖板，M259 移除盖板。井壁和坑外过渡使用共用的混凝土及沥青纹理。原扫描记录盖面，井下结构与安装量由配方指定。[完整配方](004/survey.json) · [共用材料](../perforated-covers/sources.json)

KITTI-360 0000 提供前视、侧面鱼眼、激光扫描及车辆轨迹。Audi A2D2 提供左前与车尾鱼眼、激光与时间戳。迁入使用对应原始曝光；独立验证使用带度量深度、实例 ID 和有色遮挡杆的 Cycles 220° 鱼眼。[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) · [A2D2](https://www.a2d2.audi/a2d2/en.html) · [独立场景](../../../scripts/render-inlet-control.py)
