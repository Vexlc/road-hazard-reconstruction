# 作者、来源与改编

- P546–P550 的几何取自 Tingrui Zhang 等人的 PointPSSN。依次采用训练集 492、58、105、26、396 号 XYZ 和对应的二元标签。每个文件的指纹随配方提供。发布包包含派生模型和材质预览，原始测点请从作者处取得。上游仓库未列出明确的数据许可，使用者应核对原作者条款。[论文](https://doi.org/10.1016/j.eswa.2025.129833) · [作者仓库](https://github.com/5rexi/PointPSSN/tree/1e4c122811066798db1dc0767a159dce05e48383) · [数据集](https://drive.google.com/file/d/1zmQ3IzpPbuiBKXMo-ttX1bqew8GSZ9to/view)
- 扫描件的坑床借用 Dimitrios Savva 拍摄的两组路面材质，路肩使用 ambientCG Asphalt031，均为 CC0。重建脚本对实拍骨料采样、拼接并调整坑内外亮度，测点本身没有 RGB。方法同时用到配方内记录的 PothRGBD 颜色统计量。[Road Damaged](https://polyhaven.com/a/road_damaged) · [Road Damaged 2](https://polyhaven.com/a/road_damaged_2) · [Asphalt031](https://ambientcg.com/a/Asphalt031)
- P551，Kenneth Allen，Hospital Road, Omagh，积水坑。改编包含轮廓标注、投影贴图、独立水面和补全坑床。照片及其改编采用 CC BY-SA 2.0。[现场照片](https://www.geograph.org.uk/photo/8272133) · [许可](https://creativecommons.org/licenses/by-sa/2.0/)
- M552，Part Time Security，Stolen Manhole。作者将原照置于公有领域。改编包含井口校正、砖墙补全、井底和井圈建模。[原照及权利声明](https://commons.wikimedia.org/wiki/File:Stolen_Manhole.jpg)
- A553，AKA MBG，破碎复合井盖。作者将原照置于公有领域。从照片描出两块松动碎片、加强筋和断面，转放到路面。M552 井外铺装和 A553 的宿主铺装复用仓库内的道路材质，详见上期来源。[原照及权利声明](https://commons.wikimedia.org/wiki/File:Reinforced_polymer-sand_manhole_broke_into_pieces,_must_withstand_5_tons.jpg) · [上期输入](https://github.com/Vexlc/road-hazard-reconstruction/tree/field-chambers-v1.0.1/releases/field-chambers/gen2hole/examples/perforated-covers)
- P554，Roger A Smith，Abbotsham Road。改编包含边缘标注和深度估计。照片及其改编采用 CC BY-SA 2.0。[现场照片](https://www.geograph.org.uk/photo/7472755) · [许可](https://creativecommons.org/licenses/by-sa/2.0/)
- P555，Mat Fascione，Spurn Head。改编包含分板、错台估计和外露断面。照片及其改编采用 CC BY-SA 2.0。[现场照片](https://www.geograph.org.uk/photo/5555078) · [许可](https://creativecommons.org/licenses/by-sa/2.0/)
- P556，Gary Dickson，Road washed away。照片记录了 Storm Frank 后 River Muick 边被冲毁的路段。改编包含两侧断口标注、路面投影、路基剖面估计和模型驾驶视角。照片及其改编采用 CC BY-SA 2.0。[现场照片](https://www.geograph.org.uk/photo/4910565) · [许可](https://creativecommons.org/licenses/by-sa/2.0/)
- 本地实拍宿主回放使用 Hesai Technology 与 Scale AI 的 PandaSet 109。拍摄时场景中没有这些新增资产。下载包提供合成的前后视短片、放置参数及数值校验，原始宿主影像按 PandaSet 条款单独获取。[PandaSet](https://github.com/scaleapi/pandaset-devkit)
