# 从一个新设计开始

本项目带有一份可直接填写的三件设计批次。P38 是坑洞，M38 是缺盖井，A38 是侧铰掀盖。它们是新编号的示例配方。已发布的 109 件模型和实际生成图可在[资产清单](../catalog/assets.json)中查询。

## 1. 先确定设计尺寸和取景

复制示例，再修改名称、材质、轮廓描述和尺寸。`show` 可单独查看一个提示词。`prompts` 给出本批次每张主图的完整提示词与目标 PNG 路径，也会列出掀盖所需的敞井参考。

~~~bash
mkdir -p experiments
cp -r examples/new-batch experiments/batch-38
mkdir -p experiments/batch-38/references
python gen2hole.py prompts --inputs experiments/batch-38
~~~

| 设计字段 | 含义 |
|---|---|
| `id / label / category` | P 对应 `pothole`，M 对应 `missing`，A 对应 `raised`。ID 使用类别字母加至少两位数字，如 `P38`。每张主图以 ID 命名。 |
| `model` | 坑洞支持 `pothole/wet/spall`，缺盖井支持 `missing/brick/rect`，异常盖板支持 `raised/hatch/offset`。 |
| `span / depth` | 坑洞的 `span` 是整张主图横向覆盖的米数。`depth` 是设计凹深，单位米。 |
| `shape / width / length / depth` | 井类取 `round` 或 `rect`。`width/length` 是井框外侧设计尺寸，`depth` 是井深，单位米。 |
| `angle / donor` | 掀盖的 `angle` 是提示词与初始拟合的设计角，单位度。侧铰和上翻盖的 `donor` 指定用于补全遮挡井壁的敞井 ID，该批次应包括其设计与主图。移位盖从其图中可见的井壁取材。 |

## 2. 用 GPT Image 生成参考图

在 Codex 中可直接使用下面的任务描述，沿用原实验的内置生图方式。每个编号对应一次独立生成。生成任务与几何构建之间的文件接口是三个命名 PNG。

> 请读取本项目 `experiments/batch-38/calibration/designs.json`，使用内置 GPT Image 按每条记录的 `prompt` 分别生成一张写实参考图。逐张检查物体完整入镜、俯视方向和周围的完好路面。把选定的原尺寸 PNG 按 `id` 保存为 `experiments/batch-38/references/P38.png`、`M38.png`、`A38.png`。

坑洞要看清坑沿与坑底，井口要露出内外圈。侧铰圆盖应呈压缩的椭圆，上翻方盖的铰边和四角应能定位。方图、正俯视、漫射光、主体四周有完好路面的约束同时服务于图像分割、纹理投影和边缘补面。保留被选中的实际 PNG 及提示词，同一段提示词再次生图可能得到不同轮廓。

共享的沥青断面、圆盖背面和方盖背面已经在 `inputs/shared/`。如果新批次需要特定材质，将 `fracture.png`、`back.png` 或 `squareback.png` 放入该批次的 `references/` 即可覆盖对应的共享补图。[共享材质参考图](../inputs/shared/)

## 3. 拟合并检查几何

~~~bash
python gen2hole.py build --inputs experiments/batch-38 P38 M38 A38 --mesh-only --preview --out runs/38-mesh
~~~

自定义批次会从 PNG 开始推理坑洞深度并重算拟合。可在 `runs/38-mesh/series/calibration/` 查看每件的 `*_fit.json`、坑洞轮廓和水线，在 `views/` 对比两个真实 GLB 视角。单独构建 `A38` 时，程序会准备 `M38` 的参考图和井框拟合，并只导出选中的资产。

当坑沿分割到了完好路面、水线穿过碎石，或盖板拟合偏离原图时，在输入批次的 `calibration/overrides.json` 中按 ID 修订，再写入新的运行目录。所有图像标记使用 `0…479` 坐标。原图像素可按 `u480 = u × 479 / (W − 1)`、`v480 = v × 479 / (H − 1)` 换算，原点在左上。

| 对象 | 可修订的图像标记 |
|---|---|
| 坑沿、积水 | `outline`、`water` 为沿边界依次排列的 `[u,v]` 多边形。`band` 调整 GrabCut 边界带，单位为归一化图像像素。图中没有可见水面时可设 `dry: true`。 |
| 井框和净开口 | `outer`、`inner` 均为 `[cx,cy,rx,ry]`，后两项是半宽和半高。`wall_end` 和 `bottom_photo` 控制井壁向下展开与底部的图像取样区。 |
| 侧铰或移位圆盖 | `ellipse` 为盖面投影的 `[cx,cy,rx,ry]`，侧铰可另标 `bow_x`。 |
| 上翻方盖 | `quad` 按左上、右上、右下、左下给出盖板四个对应点。保留与铰边的方向关系，才能区分过竖直姿态。 |

已有的 A13 给出了同一件侧铰盖的设计、标记、最终拟合与实物尺度。新主图需要按自身像素位置修订。[A13 原图](../inputs/series/references/A13.png) · [发布批的逐件修订](../inputs/series/calibration/overrides.json) · [A13 最终拟合](../inputs/series/calibration/A13_fit.json)

## 4. 导出、核验和迁入

~~~bash
python gen2hole.py build --inputs experiments/batch-38 P38 M38 A38 --preview --out runs/38-final
python gen2hole.py verify --run runs/38-final
python downstream/place.py --bundle runs/38-final/series --site 034 --group 38 --assets P38 M38 A38 --front
python downstream/simulate.py --run runs/38-final --out runs/38-final/sensors
~~~

建模时固定已选 RGB 可重复检查几何。`verify` 解码 GLB 与高斯 PLY，检查嵌入贴图、顶点有限值、三角面数、网格高低点、切口范围和逐点材质数组长度，并与首次构建记录核对。对于既有编号，`checks.json` 另记录与历史成品的数量和字节比较。对于新设计，两个 GLB 视角用于人工核对轮廓、露出的井壁、盖板厚度和接缝。[资产核验实现](../recipes/common/verify.py)

可以用已运行的 `034_13` 案例直接体验 LiDAR 导出。两条命令围绕 A13，从原始行号中取相同的空间裁切。`after.npz` 同时保留修改标记、回波类别和源语义。

~~~bash
python gen2hole.py scan --root runs/demo/sensors --case 034_13 --frame 46 --asset A13 --mode before --output runs/demo/replay/A13-before.ply
python gen2hole.py scan --root runs/demo/sensors --case 034_13 --frame 46 --asset A13 --mode after --output runs/demo/replay/A13-after.npz
~~~

## 5. 用一张实拍 RGB 重建干坑

在方形俯视照片上圈出包含完整坑沿的粗框，四周留出完好路面。参考[可运行的实拍批次](../examples/real-rgb/)设置 `depth_profile` 为 `designed_bowl`，在 `overrides.json` 中按 ID 提供 `box`，格式为 480 像素坐标系中的 `[左, 上, 右, 下]`。程序从 RGB 分割坑沿，按到坑沿的距离构造碗状剖面，并使用原图贴纹理。照片横宽 `span` 和凹深 `depth` 均由使用者提供。需要恢复现场深度时可使用[配准 RGB-D 入口](../../docs/02-asset-reconstruction.md#自带-rgb-d-文件)。

从仓库根目录复跑以下命令。选用新的输出目录即可与已存的 `runs/rgb-box-P90/` 比较。

~~~bash
python gen2hole/gen2hole.py build --inputs gen2hole/examples/real-rgb P90 --preview --out runs/my-field-pit
work/envs/assets/bin/python scripts/evaluate-rgb-outline.py --run runs/my-field-pit --out runs/my-field-pit/outline-eval.json
python gen2hole/downstream/place.py --bundle runs/my-field-pit/series --group 90 --site 034 --assets P90 --front
python gen2hole/downstream/simulate.py --run runs/my-field-pit --out runs/my-field-pit/sensors
~~~

P90 的粗框从照片上选取。重建后用独立的二维标注核对，轮廓 IoU 从无引导时的 0.0062 提高到 0.9205。设定坑深 0.11 米；同源 RGB-D 资产的 P95 凹深约 0.065 米。此例验证了单帧轮廓与迁入流程。[评估记录](../../validation/rgb-box-P90.json) · [模型与道路对照](../../docs/figures/reconstruction-evidence.jpg)
