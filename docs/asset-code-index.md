# 资产系列与迁入代码对照

本版登记 1,489 个资产编号、45 个重建系列和 15 条迁入或展示路线。同一次采集的不同盖板状态分别占一个编号。表中只列每个系列的代表编号，按任意编号运行 `python3 gen2hole/gen2hole.py locate 编号` 可以查看它的配方与迁入入口。[从干净克隆到首次运行](16-reproduction-cookbook.md) · [输入包与版本](01-downloads.md)

从“操作步骤”进入对应系列的文档；“重建”是实际程序入口，“配方”列出原始数据或已登记的参数。从“迁入路线”可跳到本页下方对应的宿主、实现代码和验收方法。

## 历史库与封存路线

| 系列及操作步骤 | 编号数 | 示例编号 | 重建、输入 | 迁入路线 |
|---|---:|---|---|---|
| [双目 XYZ](../docs/02-asset-reconstruction.md#1-%E5%8F%8C%E7%9B%AE%E5%AE%9E%E6%8B%8D%E5%9D%91%E6%B4%9E65-%E6%9D%A1) | 65 | stereo_pothole_fan_1_01 | [重建](../scripts/reconstruct-assets.py) · [配方](../outputs/road_hazard_expansion/asset_manifest.json) | [legacy](#route-legacy) |
| [单帧实拍 RGB-D](../docs/02-asset-reconstruction.md#2-rgb-d-%E5%AE%9E%E6%8B%8D%E5%9D%91%E6%B4%9E822-%E6%9D%A1) | 822 | rgbd_pothole_20250227_135438 | [重建](../scripts/reconstruct-assets.py) · [配方](../outputs/road_hazard_expansion/asset_manifest.json) | [legacy](#route-legacy) |
| [设计坑洞（程序化路线已封存）](../docs/02-asset-reconstruction.md#3-%E7%A8%8B%E5%BA%8F%E5%8C%96%E5%9D%91%E6%B4%9E120-%E6%9D%A1) | 120 | model_pothole_bowled_aged_asphalt_01 | [重建](../scripts/reconstruct-assets.py) · [配方](../outputs/road_hazard_expansion/asset_manifest.json) | [legacy](#route-legacy) |
| [设计缺盖井](../docs/02-asset-reconstruction.md#4-%E7%BC%BA%E7%9B%96%E7%AA%A8%E4%BA%9564-%E6%9D%A1) | 64 | model_missing_oval_broken_flange_aged_asphalt | [重建](../scripts/reconstruct-assets.py) · [配方](../outputs/road_hazard_expansion/asset_manifest.json) | [legacy](#route-legacy) |
| [倾斜、开启和移位盖板](../docs/02-asset-reconstruction.md#5-%E6%8E%80%E7%9B%96%E5%80%BE%E6%96%9C%E4%B8%8E%E7%A7%BB%E4%BD%8D%E7%AA%A8%E4%BA%95128-%E6%9D%A1) | 128 | model_raised_oval_ajar14_aged_asphalt | [重建](../scripts/reconstruct-assets.py) · [配方](../outputs/road_hazard_expansion/asset_manifest.json) | [legacy](#route-legacy) |
| [可追加的程序化资产](../docs/10-factory.md#%E7%94%9F%E6%88%90%E4%B8%8E%E8%BF%BD%E5%8A%A0) | 30 | factory_missing_20261119_000001_c4a7e0c5cd3d63f2 | [重建](../scripts/hazard_factory/run.py) · [配方](../catalog/factory-v3-assets.json) | [factory](#route-factory) |
| [大面积程序化坑洞（已封存）](../docs/10-factory.md#%E5%A4%A7%E9%9D%A2%E7%A7%AF%E5%9D%91%E6%B4%9E) | 30 | P160 | [重建](../scripts/build-large-potholes.py) · [配方](../scripts/build-large-potholes.py) | [archive](#route-archive) |

## 照片、双目及 RGB-D 路面破损

| 系列及操作步骤 | 编号数 | 示例编号 | 重建、输入 | 迁入路线 |
|---|---:|---|---|---|
| [首批 9 件生成 RGB](../gen2hole/README.md#%E5%A4%8D%E7%8E%B0%E4%B8%80%E4%B8%AA%E5%AE%8C%E6%95%B4%E6%A1%88%E4%BE%8B) | 9 | A01_leaning_iron | [重建](../gen2hole/gen2hole.py) · [配方](../gen2hole/catalog/assets.json) | [native](#route-native) |
| [后续 100 件生成 RGB](../gen2hole/README.md#%E5%A4%8D%E7%8E%B0%E4%B8%80%E4%B8%AA%E5%AE%8C%E6%95%B4%E6%A1%88%E4%BE%8B) | 100 | A04 | [重建](../gen2hole/gen2hole.py) · [配方](../gen2hole/catalog/assets.json) | [native](#route-native) |
| [单图 RGB 坑洞](../gen2hole/docs/new-assets.md#5-%E7%94%A8%E4%B8%80%E5%BC%A0%E5%AE%9E%E6%8B%8D-rgb-%E9%87%8D%E5%BB%BA%E5%B9%B2%E5%9D%91) | 1 | P90 | [重建](../gen2hole/gen2hole.py) · [配方](../gen2hole/examples/real-rgb/calibration/designs.json) | [native](#route-native) |
| [多帧 RGB-D 或双目坑洞](../docs/02-asset-reconstruction.md#%E5%A4%9A%E5%B8%A7%E5%AE%9E%E6%8B%8D%E9%87%8D%E5%BB%BA%E5%90%8C%E4%B8%80%E7%BB%84%E5%9D%91%E6%B4%9E) | 5 | P110 | [重建](../gen2hole/multiview_pits.py) · [配方](../gen2hole/examples/multiview-potholes/survey.json) | [native](#route-native) |
| [双张彩照坑洞](../docs/02-asset-reconstruction.md#%E4%B8%A4%E5%BC%A0%E5%BD%A9%E7%85%A7%E9%87%8D%E5%BB%BA%E5%9D%91%E5%BA%95%E8%B5%B7%E4%BC%8F) | 4 | P121 | [重建](../gen2hole/multiview_pits.py) · [配方](../gen2hole/examples/photo-parallax/survey.json) | [native](#route-native) |
| [双目深度缺口与照片补全](../docs/02-asset-reconstruction.md#%E7%94%A8%E5%BD%A9%E7%85%A7%E8%A1%A5%E5%9B%9E%E5%8F%8C%E7%9B%AE%E6%B7%B1%E5%BA%A6%E7%BC%BA%E5%8F%A3) | 2 | P142 | [重建](../gen2hole/multiview_pits.py) · [配方](../gen2hole/examples/pit-depth-completion/survey.json) | [native](#route-native) |
| [实拍大面积干坑的断边与场景光照](../docs/15-large-dry-pits.md#%E7%94%A8%E8%BE%93%E5%85%A5%E5%8C%85%E9%87%8D%E5%BB%BA) | 1 | P190 | [重建](../gen2hole/large_pits/__main__.py) · [配方](../gen2hole/examples/oblique-road-pits/P190-detail.json) · [输入准备](../gen2hole/large_pits/__main__.py) | [drive](#route-drive)、[kitti-pair](#route-kitti-pair) |
| [车行道大坑的实拍与生成参考图重建](../docs/15-large-dry-pits.md#%E8%BD%A6%E8%A1%8C%E9%81%93%E4%B8%8A%E7%9A%84%E5%9B%9B%E4%B8%AA%E5%9D%91) | 4 | P198 | [重建](../gen2hole/large_pits/__main__.py) · [配方](../gen2hole/examples/oblique-road-pits/lane-batch) · [输入准备](../gen2hole/large_pits/__main__.py) | [drive](#route-drive)、[kitti-pair](#route-kitti-pair) |
| [常见大型车行道坑洞](../docs/15-large-dry-pits.md#%E4%BA%8C%E5%8D%81%E4%BB%B6%E8%BD%A6%E8%A1%8C%E9%81%93%E5%9D%91%E6%B4%9E) | 15 | P205 | [重建](../gen2hole/large_pits/__main__.py) · [配方](../gen2hole/examples/oblique-road-pits/common-large) · [输入准备](../gen2hole/large_pits/__main__.py) | [drive](#route-drive)、[kitti-pair](#route-kitti-pair) |
| [实拍单图大坑与路缘塌陷](../docs/02-asset-reconstruction.md#%E4%BB%8E%E6%96%9C%E6%8B%8D%E7%85%A7%E7%89%87%E9%87%8D%E5%BB%BA%E5%A4%A7%E9%9D%A2%E7%A7%AF%E5%9D%91%E6%B4%9E) | 2 | P191 | [重建](../gen2hole/oblique_pits.py) · [配方](../gen2hole/examples/oblique-road-pits) · [输入准备](../gen2hole/infer_oblique_depth.py) | [native](#route-native) |
| [大面积实拍 RGB-D 坑洞](../docs/02-asset-reconstruction.md#%E4%BB%8E%E6%96%9C%E6%8B%8D%E7%85%A7%E7%89%87%E9%87%8D%E5%BB%BA%E5%A4%A7%E9%9D%A2%E7%A7%AF%E5%9D%91%E6%B4%9E) | 5 | P192 | [重建](../gen2hole/multiview_pits.py) · [配方](../gen2hole/examples/oblique-road-pits/rgbd.json) | [native](#route-native) |
| [路口横向沥青剥落带](../docs/02-asset-reconstruction.md#p203-%E8%B7%AF%E5%8F%A3%E6%A8%AA%E5%90%91%E5%89%A5%E8%90%BD%E5%B8%A6) | 1 | P203 | [重建](../gen2hole/oblique_pits.py) · [配方](../gen2hole/examples/oblique-road-pits/P203-spall.json) · [输入准备](../gen2hole/infer_oblique_depth.py) | [drive](#route-drive) |
| [实测沥青坑蚀与接缝掉块](../docs/17-mixed-road-transfer.md#%E7%94%A8%E5%8E%9F%E5%A7%8B%E8%BE%93%E5%85%A5%E9%87%8D%E5%BB%BA%E4%B9%9D%E4%BB%B6%E8%B5%84%E4%BA%A7) | 4 | P220 | [重建](../gen2hole/multiview_pits.py) · [配方](../gen2hole/examples/mixed-road) | [native](#route-native)、[mesh-view](#route-mesh-view) |
| [实测片状剥落与粗集料碗坑](../docs/21-measured-pavement.md#%E4%BB%8E%E5%8E%9F%E5%A7%8B-rgb-d-%E9%87%8D%E5%BB%BA%E4%B8%A4%E5%A4%84%E7%A0%B4%E6%8D%9F) | 2 | P238 | [重建](../gen2hole/multiview_pits.py) · [配方](../gen2hole/examples/measured-pavement) | [mesh-view](#route-mesh-view)、[pavement-layout](#route-pavement-layout)、[pavement-joint](#route-pavement-joint) |

## 裂缝、积水、铺装与修补

| 系列及操作步骤 | 编号数 | 示例编号 | 重建、输入 | 迁入路线 |
|---|---:|---|---|---|
| [裂缝照片与像素掩膜](../docs/02-asset-reconstruction.md#%E4%BB%8E%E5%AE%9E%E6%8B%8D%E8%A3%82%E7%BC%9D%E4%B8%8E%E5%83%8F%E7%B4%A0%E6%8E%A9%E8%86%9C%E9%87%8D%E5%BB%BA) | 3 | P101 | [重建](../gen2hole/gen2hole.py) · [配方](../gen2hole/examples/photo-cracks/calibration/designs.json) | [native](#route-native) |
| [道路扫描裂缝](../docs/02-asset-reconstruction.md#%E4%BB%8E%E9%81%93%E8%B7%AF%E6%89%AB%E6%8F%8F%E9%87%8D%E5%BB%BA%E8%A3%82%E7%BC%9D) | 4 | P134 | [重建](../gen2hole/gen2hole.py) · [配方](../gen2hole/examples/scanned-cracks/survey.json) · [输入准备](../gen2hole/examples/scanned-cracks/prepare.py) | [native](#route-native) |
| [可拆卸混凝土碎板](../docs/02-asset-reconstruction.md#%E4%BB%8E%E9%97%AD%E5%90%88%E7%BD%91%E8%A3%82%E4%B8%AD%E6%8F%90%E5%8F%96%E5%8F%AF%E6%8B%86%E5%8D%B8%E7%9A%84%E6%B7%B7%E5%87%9D%E5%9C%9F%E7%A2%8E%E6%9D%BF) | 4 | P144 | [重建](../gen2hole/gen2hole.py) · [配方](../gen2hole/examples/faulted-slab/survey.json) · [输入准备](../gen2hole/examples/faulted-slab/prepare.py) | [native](#route-native) |
| [实拍坑床的积水状态](../docs/02-asset-reconstruction.md#%E5%AE%9E%E6%8B%8D%E5%9D%91%E5%BA%95%E4%B8%8A%E7%9A%84%E7%A7%AF%E6%B0%B4) | 5 | P129 | [重建](../gen2hole/fill_pits.py) · [配方](../gen2hole/examples/standing-water/survey.json) | [wet](#route-wet)、[native](#route-native) |
| [单图路缘积水坑（观测水线、假设坑床，按目标坡度重新蓄水）](../docs/02-asset-reconstruction.md#p204-%E8%B7%AF%E7%BC%98%E7%A7%AF%E6%B0%B4%E5%9D%91) | 1 | P204 | [重建](../gen2hole/oblique_pits.py) · [配方](../gen2hole/examples/oblique-road-pits/P204-water.json) · [输入准备](../gen2hole/infer_oblique_depth.py) | [photo-water](#route-photo-water)、[drive](#route-drive) |
| [泥水坑与倾斜联锁砖](../docs/02-asset-reconstruction.md#%E6%B3%A5%E6%B0%B4%E5%9D%91%E6%97%81%E7%9A%84%E5%80%BE%E6%96%9C%E8%81%94%E9%94%81%E7%A0%96) | 1 | P154 | [重建](../gen2hole/multiview_pits.py) · [配方](../gen2hole/examples/displaced-paver/survey.json) | [native](#route-native) |
| [缺砖边缘与坑内碎片](../docs/02-asset-reconstruction.md#%E5%8D%95%E5%B8%A7-rgb-d-%E7%9A%84%E7%BC%BA%E7%A0%96%E8%BE%B9%E7%BC%98%E4%B8%8E%E5%9D%91%E5%86%85%E7%A2%8E%E7%89%87) | 2 | P149 | [重建](../gen2hole/multiview_pits.py) · [配方](../gen2hole/examples/single-depth-surfaces/survey.json) | [native](#route-native) |
| [双排凹槽](../docs/02-asset-reconstruction.md#%E4%BB%8E%E5%8F%8C%E6%8E%92%E5%87%B9%E7%97%95%E6%8F%90%E5%8F%96%E5%8D%95%E4%B8%AA%E6%A7%BD%E5%8F%A3) | 1 | P148 | [重建](../gen2hole/multiview_pits.py) · [配方](../gen2hole/examples/paired-road-slots/survey.json) | [native](#route-native) |
| [横向沥青修补带（本地样例，需原截图）](../gen2hole/examples/transverse-repair/README.md#%E9%87%8D%E5%BB%BA) | 1 | P202 | [重建](../gen2hole/repair_patch.py) · [配方](../gen2hole/examples/transverse-repair/P202.json) | [repair](#route-repair) |

## 井盖、雨水篦与敞井

| 系列及操作步骤 | 编号数 | 示例编号 | 重建、输入 | 迁入路线 |
|---|---:|---|---|---|
| [可动圆井盖](../docs/02-asset-reconstruction.md#%E4%B8%A4%E5%BC%A0%E5%AE%9E%E6%8B%8D%E9%87%8D%E5%BB%BA%E5%8F%AF%E5%8A%A8%E5%9C%86%E4%BA%95%E7%9B%96) | 6 | A91 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/photo-chamber/survey.json) | [native](#route-native) |
| [自驾相机和 LiDAR · 034](../docs/02-asset-reconstruction.md#%E4%BB%8E%E8%87%AA%E9%A9%BE%E7%9B%B8%E6%9C%BA%E5%92%8C%E5%8E%9F%E5%A7%8B%E6%BF%80%E5%85%89%E7%82%B9%E9%87%8D%E5%BB%BA%E8%90%BD%E5%BA%A7%E4%BA%95%E7%9B%96) | 3 | A104 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/pandaset-034/capture.json) · [输入准备](../gen2hole/fit_road_cover.py) | [native](#route-native) |
| [自驾相机和 LiDAR · 109](../docs/02-asset-reconstruction.md#%E4%BB%8E%E8%87%AA%E9%A9%BE%E7%9B%B8%E6%9C%BA%E5%92%8C%E5%8E%9F%E5%A7%8B%E6%BF%80%E5%85%89%E7%82%B9%E9%87%8D%E5%BB%BA%E8%90%BD%E5%BA%A7%E4%BA%95%E7%9B%96) | 3 | A107 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/pandaset-109/capture.json) · [输入准备](../gen2hole/fit_road_cover.py) | [native](#route-native) |
| [雨水篦通孔与断条](../docs/02-asset-reconstruction.md#%E4%BB%8E%E5%AE%9E%E6%8B%8D%E7%9B%96%E6%9D%BF%E9%87%8D%E5%BB%BA%E8%B4%AF%E7%A9%BF%E5%AD%94%E4%B8%8E%E6%96%AD%E6%9D%A1) | 4 | A115 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/perforated-covers/yokota/survey.json) | [native](#route-native) |
| [铺石穿孔圆盖](../docs/02-asset-reconstruction.md#%E4%BB%8E%E5%AE%9E%E6%8B%8D%E7%9B%96%E6%9D%BF%E9%87%8D%E5%BB%BA%E8%B4%AF%E7%A9%BF%E5%AD%94%E4%B8%8E%E6%96%AD%E6%9D%A1) | 4 | A118 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/perforated-covers/cobblestone/survey.json) | [native](#route-native) |
| [断片铸铁井盖](../docs/02-asset-reconstruction.md#%E4%BA%95%E7%9B%96%E6%96%AD%E7%89%87%E4%B8%8E%E4%BA%95%E5%86%85%E5%89%96%E8%A7%86) | 4 | A125 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/perforated-covers/split-iron/survey.json) | [native](#route-native) |
| [井盖安装高差](../docs/02-asset-reconstruction.md#%E4%BA%95%E7%9B%96%E4%B8%8E%E8%B7%AF%E9%9D%A2%E7%9A%84%E5%AE%89%E8%A3%85%E9%AB%98%E5%B7%AE) | 5 | A138 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/perforated-covers/grade/survey.json) | [native](#route-native) |
| [六边形井盖四态](../docs/02-asset-reconstruction.md#%E4%BB%8E%E5%8D%95%E5%BC%A0%E7%85%A7%E7%89%87%E9%87%8D%E5%BB%BA%E5%85%AD%E8%BE%B9%E5%BD%A2%E4%BA%95%E7%9B%96) | 4 | A151 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/polygon-cover/survey.json) | [native](#route-native) |
| [实扫方形检查井及异常安装状态](../docs/17-mixed-road-transfer.md#%E7%94%A8%E5%8E%9F%E5%A7%8B%E8%BE%93%E5%85%A5%E9%87%8D%E5%BB%BA%E4%B9%9D%E4%BB%B6%E8%B5%84%E4%BA%A7) | 5 | A224 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/mixed-road/square-hatch/survey.json) | [native](#route-native)、[mesh-view](#route-mesh-view) |
| [实扫双联检查井与半缺盖](../docs/18-twin-hatch-transfer.md#%E4%BB%8E%E6%89%AB%E6%8F%8F%E9%87%8D%E5%BB%BA%E4%BA%94%E4%BB%B6%E8%B5%84%E4%BA%A7) | 5 | A228 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/mixed-road/twin-hatch/survey.json) | [native](#route-native)、[mesh-view](#route-mesh-view) |
| [带双圈排水孔的铸铁盖与缺盖井](../docs/22-scanned-round-covers.md#%E4%BB%8E%E6%89%AB%E6%8F%8F%E9%87%8D%E5%BB%BA%E5%9B%9B%E4%B8%AA%E7%8A%B6%E6%80%81) | 2 | A240 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/scanned-round-covers/005/survey.json) · [输入准备](../gen2hole/examples/scanned-round-covers/prepare.py) | [pavement-layout](#route-pavement-layout)、[pavement-joint](#route-pavement-joint)、[metric-ray-origins](#route-metric-ray-origins) |
| [裂纹填芯盖与缺盖井](../docs/22-scanned-round-covers.md#%E4%BB%8E%E6%89%AB%E6%8F%8F%E9%87%8D%E5%BB%BA%E5%9B%9B%E4%B8%AA%E7%8A%B6%E6%80%81) | 2 | A241 | [重建](../gen2hole/survey.py) · [配方](../gen2hole/examples/scanned-round-covers/008/survey.json) · [输入准备](../gen2hole/examples/scanned-round-covers/prepare.py) | [pavement-layout](#route-pavement-layout)、[pavement-joint](#route-pavement-joint)、[metric-ray-origins](#route-metric-ray-origins) |

## 路侧与路面立体设施

| 系列及操作步骤 | 编号数 | 示例编号 | 重建、输入 | 迁入路线 |
|---|---:|---|---|---|
| [实物模型混凝土隔离墩](../docs/19-roadside-props.md#%E4%BB%8E%E5%8E%9F%E5%A7%8B%E6%A8%A1%E5%9E%8B%E9%87%8D%E5%BB%BA%E5%9B%9B%E4%BB%B6%E8%B5%84%E4%BA%A7) | 2 | B232 | [重建](../gen2hole/roadside_props.py) · [配方](../gen2hole/examples/roadside-props/sources.json) | [mesh-view](#route-mesh-view)、[joint-props](#route-joint-props) |
| [消防栓与风化涂层](../docs/19-roadside-props.md#%E4%BB%8E%E5%8E%9F%E5%A7%8B%E6%A8%A1%E5%9E%8B%E9%87%8D%E5%BB%BA%E5%9B%9B%E4%BB%B6%E8%B5%84%E4%BA%A7) | 2 | H234 | [重建](../gen2hole/roadside_props.py) · [配方](../gen2hole/examples/roadside-props/sources.json) | [mesh-view](#route-mesh-view)、[joint-props](#route-joint-props) |
| [直立与倒地交通锥](../docs/20-traffic-cones.md#%E4%BB%8E%E5%8E%9F%E5%A7%8B%E6%A8%A1%E5%9E%8B%E9%87%8D%E5%BB%BA%E5%B9%B6%E6%88%90%E5%AF%B9%E6%94%BE%E7%BD%AE) | 2 | C236 | [重建](../gen2hole/traffic_cones.py) · [配方](../gen2hole/examples/traffic-cones/source.json) | [mesh-view](#route-mesh-view)、[joint-props](#route-joint-props)、[cycles](#route-cycles) |

## 按宿主选迁入入口

| 路线 | 适用场景 | 代码 | 数据、运行和验收 |
|---|---|---|---|
| <a id="route-legacy"></a>legacy | 055/056 历史八件组合 | [run.sh](../outputs/road_hazard_joint_scenes/reproduction/run.sh) | [指南](../docs/04-scene-insertion.md) |
| <a id="route-factory"></a>factory | 034/109/110 五类工厂组合 | [migrate_all.py](../scripts/hazard_factory/migrate_all.py) | [指南](../docs/10-factory.md) |
| <a id="route-native"></a>native | 034/109/110 原生 Gaussian 场景、相机及 LiDAR | [place.py](../gen2hole/downstream/place.py) | [指南](../docs/13-reproduce-and-transfer.md) |
| <a id="route-drive"></a>drive | PandaSet 单坑原尺度行车回放 | [__main__.py](../gen2hole/large_pits/__main__.py) | [指南](../docs/15-large-dry-pits.md) |
| <a id="route-kitti-pair"></a>kitti-pair | KITTI-360 0000 两坑一组及虚拟后视 | [pit_pairs.py](../streetcrafter/joint/pit_pairs.py) | [指南](../streetcrafter/joint/README.md) |
| <a id="route-wet"></a>wet | 先按目标坡面求水位，再迁入湿态 | [fill_pits.py](../gen2hole/fill_pits.py) | [指南](../docs/02-asset-reconstruction.md) |
| <a id="route-photo-water"></a>photo-water | 按站点重建水面、坑床及折射回放 | [oblique_pits.py](../gen2hole/oblique_pits.py) | [指南](../docs/02-asset-reconstruction.md) |
| <a id="route-repair"></a>repair | 实拍沥青基色迁移及同机位投影 | [review-repair-transfer.py](../scripts/review-repair-transfer.py) | [指南](../gen2hole/examples/transverse-repair/README.md) |
| <a id="route-archive"></a>archive | 封存成品及环绕复核；独立 GLB 的宿主接入见手册 | [review-large-potholes.py](../scripts/review-large-potholes.py) | [指南](../docs/10-factory.md) |
| <a id="route-mesh-view"></a>mesh-view | 米制网格、标定光线与宿主遮挡合成 | [transfer-mesh-view.py](../scripts/transfer-mesh-view.py) | [指南](../docs/17-mixed-road-transfer.md) |
| <a id="route-joint-props"></a>joint-props | KITTI 0000 原生联合场景的成对立体设施及前后视 | [roadside_pairs.py](../streetcrafter/joint/roadside_pairs.py) | [指南](../docs/19-roadside-props.md) |
| <a id="route-cycles"></a>cycles | Blender Cycles 原生等距鱼眼、米制深度与双物体对照 | [render-cone-cycles.py](../scripts/render-cone-cycles.py) | [指南](../docs/20-traffic-cones.md) |
| <a id="route-pavement-layout"></a>pavement-layout | 独立支撑面上的成组开口、实车鱼眼与 Cycles 冠形路面 | [transfer-mesh-view.py](../scripts/transfer-mesh-view.py) | [指南](../docs/21-measured-pavement.md) |
| <a id="route-pavement-joint"></a>pavement-joint | 坑洞和检查井成对迁入 KITTI 联合检查点，按开口切分原路面高斯 | [pavement_pair.py](../streetcrafter/joint/pavement_pair.py) | [指南](../docs/21-measured-pavement.md) |
| <a id="route-metric-ray-origins"></a>metric-ray-origins | 逐像素起点的米制光线与独立 Cycles 正交、后向鱼眼验收 | [transfer-mesh-view.py](../scripts/transfer-mesh-view.py) | [指南](../docs/22-scanned-round-covers.md) |

更新目录后运行 `python3 scripts/check-reproduction-index.py --write-map`。常规检查运行同一命令并省略参数，检查编号、源文件、指南章节、迁入路线与本页是否一致。
