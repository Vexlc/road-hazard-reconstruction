# 道路遗落物的原始模型

四个输入来自 Poly Haven 的 1K glTF 发布档。每个目录保存原始 glTF、二进制网格、漫反射贴图、ARM 合并图和 OpenGL 法线贴图。三角面与 UV 来自原模型。仓库给定落地姿态、成对布局与道路位置。[原作者许可 CC0](https://polyhaven.com/license) · [逐文件来源、长度和 SHA-256](sources.json)

| 编号 | 发布模型 | 作者 |
|---|---|---|
| R242 | [Old Tyre](https://polyhaven.com/a/old_tyre) | MP |
| R243 | [Rusted Wheel Rim 01](https://polyhaven.com/a/rusted_wheel_rim_01) | John Hutcheson |
| R244 | [Plastic Crate 01](https://polyhaven.com/a/plastic_crate_01) | PierreB3D |
| R245 | [Cement Bag](https://polyhaven.com/a/cement_bag) | PierreB3D |

`original/` 内可离线核对全部 20 个文件。采用发布模型的米制尺度；接触面、误差与支撑裕量记录在构建产物的 `metadata.json`。KITTI 和 Audi 的原始相机、标定及道路数据按各数据集条件另行取得，实车曝光上的物体是迁入结果。
