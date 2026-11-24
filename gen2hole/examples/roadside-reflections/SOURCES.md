# 坑床、道路与反光观测

P256 使用 PothRGBD 的 20250305_121628。照片中宽浅坑的破碎边露出集料。P257 使用 20250305_110328，较平滑的椭圆坑底一侧有碎石。两者各使用一份注册 RGB、毫米深度和原始轮廓，投影使用标称焦距 462 px。拟合前留出部分深度块，按同一曝光的剩余深度重建。P258 使用 P257 的坑床，注水点与 9 L 水量由配方指定。[采集文件和指纹](../../../catalog/capture-inputs.json) · [数据集发布者](https://github.com/ymyurdakul/PData/tree/7596dc1c745e68b90403d7e82f89e7492fa95edc)

KITTI-360 0000 和 0003 提供原始左右同步图像、右侧鱼眼、外参、轨迹及 Velodyne。Audi A2D2 提供实车左前与车尾的鱼眼曝光及激光测距。相机包里的近物颜色来自相应原图。控制实验使用独立 Cycles 的发光挡板、水平玻璃、黑色坑底与三个留出鱼眼视角。[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) · [Audi A2D2](https://www.a2d2.audi/a2d2/en.html) · [控制实验代码](../../../scripts/render-roadside-reflection-control.py)
