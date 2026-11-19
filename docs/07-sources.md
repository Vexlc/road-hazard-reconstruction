# 来源与上游许可

## 资产

- 双目：Rui (Ranger) Fan 等人的 [Stereo Pothole Datasets](https://github.com/ruirangerfan/stereo_pothole_datasets)，固定提交 `a6cc8e7354da4503b0ee03d15058df131a645195`。来源许可随双目资产卷和原始输入包保存。
- RGB-D：Mustafa Yurdakul 和 Şakir Tasdemir 的 [PothRGBD/PData](https://github.com/ymyurdakul/PData)，本地数据快照提交 `7596dc1c745e68b90403d7e82f89e7492fa95edc`；另见 [数据论文](https://arxiv.org/abs/2505.04207)。源下载仓库未附明确许可文本，私有复现包保留来源信息，不将源数据改称本项目独有数据或授予额外使用权。公开再分发前应核实来源条款。
- 程序化材质：[ambientCG](https://ambientcg.com/) CC0 的 Asphalt031、Concrete034、Concrete049、ManholeCover009。`work/road_hazard_pipeline/material_sources.json` 固定原下载链接、文件大小与 SHA-256。几何由项目参数化代码生成，尺寸为设计值；扫描来源材质不能等同于整个资产的实物扫描。

## 宿主道路与代码

- [PandaSet](https://github.com/scaleapi/pandaset-devkit)：道路相机、LiDAR、实例与标定，数据来源为 Hesai Technology 与 Scale AI。项目使用 [压缩镜像](https://huggingface.co/datasets/georghess/pandaset)，保留版本化成员索引、CRC 与来源说明。使用应遵守 [PandaSet 数据条款](https://www.refonte.ai/legal/pandaset-tos)。
- [StreetCrafter](https://github.com/zju3dv/street_crafter)：提交 `9937775ba12f069d363c61ad00ff977900660f9c`，许可保存在 `streetcrafter/licenses/LICENSE`，本项目补丁与 overlay 单独保存。
- gsplat、nvdiffrast、EasyVolcap、PandaSet SDK 与天空分割模型的精确来源/提交见 `outputs/reproducibility/sources.json`。这些第三方组件遵守各自的许可证。

资产中可能存在同一实际坑洞的多个来源帧；1,199 是资产记录数量，并不声称有 1,199 个不同的实拍物理缺陷。
