# 填芯井盖与十孔铸铁盖的来源

两张盖面来自 ambientCG 的高程场摄影测量。发布者为两幅素材给出的平面范围均约 0.80 × 0.80 m。仓库保留原版 1K ZIP 中的 Color、Displacement、Opacity、NormalGL、Roughness 和 Metalness，每套另有一张从扫描准备的孔口掩膜。原 ZIP 的 SHA-256 见准备程序。[Manhole Cover 003](https://ambientcg.com/a/ManholeCover003) · [Manhole Cover 006](https://ambientcg.com/a/ManholeCover006) · [ambientCG 的 CC0 许可](https://docs.ambientcg.com/license/) · [准备程序](prepare.py)

| 扫描 | 使用的图像证据 | 设计部分 |
|---|---|---|
| 003 | 灰色填芯、铸铁花边与外圈接缝。平面盖缝取中心 (506, 512)、半径 428 px。圆形暗部保留原表面起伏，其中是否通到井下没有测量。 | A248 的井圈及座面整体下沉 38 mm。M248 移除盖板。井深 0.64 m，盖厚 43 mm。 |
| 006 | 棕色铸铁、外圈八个开口和中央两个开口。盖缝取中心 (508, 510)、半径 458 px。用配准的颜色与相对位移圈定每个孔口，桥接两孔的实体铁条保留。 | A249 平移 (45, −13) mm，盖底与固定支撑首次接触确定高度。M249 移除盖板。井深 0.68 m，盖厚 38 mm。 |

相对起伏按配方缩放。井下内壁复用 Concrete 049 的材质，外侧使用 Asphalt 031。Audi、KITTI 和 Cycles 的资产外观均为后续重建与放置的结果。[共用材质来源](../perforated-covers/sources.json) · [真实道路和扫描复现](../../../docs/25-rolling-covers.md)
