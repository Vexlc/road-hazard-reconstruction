# 道路散落货物的输入

四个输入均为 Poly Haven 发布的米制 glTF 网格，采用 CC0。保留发布者的形状、纹理和尺度。每件的 1K glTF、二进制几何、颜色、ARM 和法线贴图均在 `original/<源名称>/` 下。ARM 的 RGB 通道依次为环境遮蔽、粗糙度和金属度。[逐文件下载地址、字节数及 SHA-256](sources.json)

| 编号 | 来源 | 作者 | 使用方式 |
|---|---|---|---|
| R262 | [Cardboard Box 01](https://polyhaven.com/a/cardboard_box_01) | Rahul Chaudhary | 保留压痕、纸箱折口及胶带 |
| R263 | [Metal Toolbox](https://polyhaven.com/a/metal_toolbox) | Mateusz Sadek | 保留箱身、托盘、旋转的箱盖及其下属搭扣和提手 |
| R264 | [Compost Bag 02](https://polyhaven.com/a/compost_bag_02) | Satyaki Mandal | 将装有基质的园艺袋侧放，求解接地姿态 |
| R265 | [Wooden Crate 01](https://polyhaven.com/a/wooden_crate_01) | James Ray Cock | 保留箱身、箱盖和搭扣 |

`sources.json` 固定参与重建的节点、初始姿态和语义名称。两份成组配方记录平面偏移和朝向。每组的成员有各自的实例编号，可由后视画面追溯到原始网格。[第一组](R262-R263.json) · [第二组](R264-R265.json)
