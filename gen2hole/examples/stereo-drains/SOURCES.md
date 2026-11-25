# 采集、尺度和设计状态

| 编号 | 原始输入 | 尺度与形状 |
|---|---|---|
| P279 | PothRGBD 图像、光学深度与轮廓 20250305_043912 | 长条形露底坑。用周围沥青拟合坑沿，深度换算为米。 |
| P281 | 同数据集另一处实拍 20250305_144032 | 圆弧坑沿和粗集料来自这次采集。 |
| A280、M280 | 横田基地雨水篦照片，DVIDS 9735953，VIRIN 260603-F-FN020-1083，摄影 Julian Batista，美国空军，公有领域 | 同一张照片上校正格栅透视并描出 60 个通孔。参考宽 0.45 m，另一边长和厚度由透视及可见侧边推算。设计井深 0.58 m。井框中心下沉 28 mm，坡角 3.8°。M280 移除整扇篦子。 |

坑洞使用标称焦距 462 px、图像中心主点和同帧毫米深度。两处采集分别留出一部分坑内深度块检验网格；该误差描述同帧表面复原。井框、井壁、支座及安装高差为补建几何。混凝土井壁使用 Concrete049 派生纹理。[固定 PothRGBD 修订](https://github.com/ymyurdakul/PData/tree/7596dc1c745e68b90403d7e82f89e7492fa95edc) · [输入哈希](../../../catalog/capture-inputs.json) · [原始雨水篦照片](https://www.dvidshub.net/image/9735953/374-ces-airmen-maintain-installation-readiness-through-storm-mitigating-flood) · [照片、轮廓与纹理](../perforated-covers/yokota) · [纹理来源](../perforated-covers/sources.json)

道路来自 KITTI-360 和 Audi A2D2。手工在两张 KITTI 原始左图中圈出接地用的沥青路面。左右匹配使用整张原始照片；激光扫描另外投影到左图检验径向距离。KITTI 0000 两台相机各自使用原路面的日照与遮阴观测。Audi 标注了车身遮挡。独立 220° 镜头与横杆前景由 Blender Cycles 构造。[接地多边形及帧号](road-windows.json) · [日照样本](../sun-openings/kitti-sun.json) · [Audi 车身边界](../traffic-cones/audi-body.json)
