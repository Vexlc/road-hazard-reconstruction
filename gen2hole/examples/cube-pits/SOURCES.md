# 四处实测沥青坑和立方体鱼眼对照

| 资产 | 用于重建的 PothRGBD 曝光 | 用于验收的数据 |
|---|---|---|
| P266 | 20250305_054900 | 同次曝光预留深度块 |
| P267 | 20250227_170442 | 同次曝光预留深度块 |
| P268 | 20250305_055452 | 同次曝光预留深度块 |
| P269 | 20250227_184349 | 独立曝光 20250227_184422 |

每份曝光含原始 RGB、毫米光学深度和坑口多边形。图像尺寸沿用数据集，相机焦距采用 462 px 标称值，尺度精度受此影响。captures.zip 收录上述五份曝光的十五个原始文件，其字节校验值取自仓库输入目录。P269 用坑外图像特征和深度注册第二张曝光，建模仅用第一张。[PothRGBD 固定上游版本](https://github.com/ymyurdakul/PData/tree/7596dc1c745e68b90403d7e82f89e7492fa95edc) · [原始文件指纹](../../../catalog/capture-inputs.json) · [捕获与注册](../../pit_captures.py)

KITTI-360 原轨迹提供前视、侧面鱼眼和观测点云；Audi A2D2 提供实车左前、车尾曝光及标定。OpenGL 对照道路使用随仓库分发的 Asphalt 02 纹理，路面、遮挡条和相机姿态由脚本生成。[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) · [Audi A2D2](https://www.a2d2.audi/a2d2/en.html) · [沥青纹理及来源](../traffic-cones/ATTRIBUTION.md) · [原生栅格渲染](../../../scripts/render-cube-pits.py)
