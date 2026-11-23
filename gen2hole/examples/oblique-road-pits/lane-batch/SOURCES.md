# P198–P201 参考图和模型

P198 取自 Jennifer Petrie 于 2026 年 1 月 23 日拍摄的《Enormous Sinkhole at Lamer Street Dunbar》，© Jennifer Petrie。模型截取前景两堆土之间可辨认的路面开口，标注坑沿、从单张照片估计高程，并用扫描材料补充被遮挡的表面。该照片及其在 P198 模型、纹理和预览中的改编按 CC BY-SA 2.0 提供。[原照片](https://www.geograph.org.uk/photo/8238971) · [许可](https://creativecommons.org/licenses/by-sa/2.0/)

P199、P200、P201 各使用一张由 OpenAI ImageGen 独立生成的车道路面参考图。标注轮廓后，分别重建三维网格和 3D 高斯。设计长轴依次为 5.0、5.6 和 5.1 米。P201 还将深度先验的第 95 百分位设为 0.40 米。三张生成参考图随输入包保存，生成提示词见同目录的 `prompts.json`。

四个模型使用 Poly Haven 的 Gravelly Sand、Stony Dirt Path 与 ambientCG 的 Asphalt 031 扫描材质。车载背景来自 PandaSet 109。[砂砾](https://polyhaven.com/a/gravelly_sand) · [碎石土](https://polyhaven.com/a/stony_dirt_path) · [沥青](https://ambientcg.com/view?id=Asphalt031) · [PandaSet](https://github.com/scaleapi/pandaset-devkit)
