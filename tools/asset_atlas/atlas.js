"use strict";
const el = id => document.getElementById(id);
const assets = window.roadAssets || [];
const byId = new Map(assets.map(row => [row.id,row]));
const pageSize = 12;
let page = 0, selected, filtered = [], reverse = false, before = false, spinning = false, lidSource = false, cutaway = false, gaussian = false;

function photo(id, path, title) {
  el(id).src = path;
  el(id).alt = title;
  el(id+"Link").href = path;
}
function metric(title, value) {
  const group = document.createElement("div");
  for (const [tag, content] of [["dt",title],["dd",value]]) {
    const node = document.createElement(tag);
    node.textContent = content;
    group.append(node);
  }
  el("metrics").append(group);
}
function download(title, url) {
  const a = document.createElement("a");
  a.href = url;
  a.textContent = title;
  el("downloads").append(a);
}
function showScene() {
  const view = selected.scenes[Number(el("scene").value)];
  el("roadCompare").hidden = !view?.note;
  el("roadCompare").textContent = view?.note || "";
  if (view) {
    photo("road", before ? view.before : view.after, view.label + (before ? (view.beforeLabel || " 迁入前") : (view.afterLabel || " 迁入后")));
    el("roadLink").href=(before?view.fullBefore:view.fullAfter) || el("road").src;
    el("before").hidden = !view.before;
    el("before").textContent = before ? (view.afterLabel || "查看迁入后") : (view.beforeLabel || "查看迁入前");
  }
  el("downloads").replaceChildren();
  if (selected.matchedRoad) {
    const group=selected.matchedRoad.find(g=>g.startsWith("109-")) || selected.matchedRoad[0];
    download("同位置对照其他资产","compare-road.html#"+new URLSearchParams({group,a:selected.id}));
  }
  download("交互三维","model.html#"+encodeURIComponent(selected.id));
  if (selected.components) download("逐槽尺寸与复位坐标",selected.componentIndex);
  if (selected.blockModel) download("独立原位碎板 GLB",selected.blockModel);
  if (selected.edgeSampling) {
    download("断边加密 PLY",selected.edgeSampling.ply);
    download("断边采样元数据",selected.edgeSampling.metadata);
  }
  download("3DGS 与网格对照","splats.html#"+encodeURIComponent(selected.id));
  for (const [name,key] of [["GLB","glb"],[selected.edgeSampling ? "原始面积采样 PLY" : "3DGS PLY","ply"],["路面场","surface"],["元数据","metadata"],["输入掩膜","mask"],[selected.auditLabel || (selected.family ? "几何一致性核验" : "轮廓评估"),"audit"],[selected.scanLabel || "雷达核验","scan"],["实拍拟合记录","fit"],[selected.reprojectionLabel || "留出帧核验","reprojection"]])
    if (selected[key]) download(name,selected[key]);
  const video = view?.video || selected.video;
  if (video) download("前视视频",video);
  if (selected.credit) download("照片出处",selected.credit);
  if (selected.related) download(selected.relatedLabel || (selected.method==="实拍 RGB" ? "同源 RGB-D" : "同源单目 RGB"),"#"+selected.related);
  if (selected.sensor && view) {
    const params = new URLSearchParams({asset:selected.id,site:view.site,camera:view.camera});
    download("雷达与剖面",selected.sensor+"#"+params);
  }
}
function showModel() {
  const sectionLabel = selected.blockPreview ? "独立原位碎板 · GLB 斜视" : "井壁剖开 · GLB 固定机位";
  const gaussianLabel=selected.edgeSampling?.gaussianPreview ? "断边加密 PLY · 高斯渲染" : "原始 PLY · 高斯渲染";
  const label = gaussian ? gaussianLabel : cutaway ? sectionLabel : spinning ? "GLB · 环绕渲染" : selected.reverse ? (reverse ? "GLB · 背向斜视" : "GLB · 正向斜视") : "资产预览";
  photo("model",gaussian ? (selected.edgeSampling?.gaussianPreview || selected.gaussianPreview) : cutaway ? (selected.blockPreview || selected.cutaway) : spinning ? selected.orbit : reverse ? selected.reverse : selected.preview, selected.label+" "+label);
  el("modelLabel").textContent = label;
  el("representation").hidden = !selected.gaussianPreview;
  el("representation").textContent = gaussian ? "看网格渲染" : "看高斯渲染";
  el("representation").setAttribute("aria-pressed",String(gaussian));
  el("rotate").hidden = !selected.reverse || spinning || cutaway || gaussian;
  el("spin").hidden = !selected.orbit || cutaway || gaussian;
  el("cutaway").hidden = !(selected.cutaway || selected.blockPreview) || gaussian;
  el("cutaway").textContent = selected.blockPreview ? (cutaway ? "查看整片路面" : "看独立碎板") : (cutaway ? "还原井体" : "剖开井体");
  el("cutaway").setAttribute("aria-pressed",String(cutaway));
  el("spin").textContent = spinning ? "停止环绕" : "360° 环绕";
  el("spin").setAttribute("aria-pressed",String(spinning));
}
const notes = {
  "实拍 RGB":"人工粗框引导分割。单张照片提供纹理与坑沿，坑底按设计深度生成。此样例可与同源 RGB-D 记录对照。",
  "生成 RGB":"参考图由模型生成。井深、坑深和遮挡部分采用设计约束，前后视图来自同一件三维资产。",
  "实拍 RGB-D":"使用实测深度与 RGB 纹理。尺寸由标称内参估算，多个采集帧可能来自同一处坑洞。",
  "实拍双目":"从双目 XYZ 拟合局部路面，使用实拍 RGB 贴图。多个采集帧可能来自同一处坑洞。",
  "程序化":"按形态、尺寸、材质与随机种子生成，尺度来自配方。"
};
function showSource() {
  el("sourcePanel").hidden = !selected.reference;
  if (!selected.reference) return;
  el("sourceView").hidden = !selected.sourceViews;
  if (selected.sourceViews) {
    const item = selected.sourceViews[Number(el("sourceView").value)];
    photo("source",item.image,item.label);
    el("sourceLabel").textContent = item.note || (item.label === "实测覆盖次数" ? "灰 0 · 黄 1 · 绿 2 · 紫 ≥3" : "输入与纹理");
    el("switchSource").hidden = true;
    return;
  }
  const labels = selected.sourceLabels || ["井圈原图","盖面原图"];
  const title = selected.lidReference ? labels[Number(lidSource)]
    : selected.method==="生成 RGB" ? "生成参考图" : "实拍 RGB 裁切";
  photo("source",lidSource ? selected.lidReference : selected.reference,title);
  el("sourceLabel").textContent = title;
  el("switchSource").hidden = !selected.lidReference;
  el("switchSource").textContent = "查看"+labels[Number(!lidSource)];
}
function showCapture() {
  const view = selected.captureViews[Number(el("captureView").value)];
  const mode = el("captureMode").value;
  photo("captureImage",view.images[mode],selected.captureAsset+" 落座实拍核验 · "+view.label);
  const legend = {photo:"原始相机裁切",native:"黄线为人工外圈，粉线为原始位姿下的 GLB 轮廓",registered:"黄线为人工外圈，青线为局部配准后的 GLB 轮廓"};
  el("captureLegend").textContent = legend[mode];
  const scores = "金属外缘 IoU　原始 "+view.native.toFixed(3)+" / 配准 "+view.registered.toFixed(3);
  el("captureScore").textContent = view.apronNative == null ? scores : scores+"　修补圈 IoU　原始 "+view.apronNative.toFixed(3)+" / 配准 "+view.apronRegistered.toFixed(3);
}
function showPitView(resetModes) {
  const view = selected.pitViews[Number(el("pitFrame").value)];
  if (resetModes) el("pitMode").replaceChildren(...view.modes.map((m,i)=>new Option(m.label,String(i))));
  const m = view.modes[Number(el("pitMode").value)];
  photo("pitImage",m.image,view.frame+" · "+m.label);
  el("pitLegend").textContent = m.note || (m.error ? "深度绝对误差　黑紫接近 0，橙黄接近 35 毫米，灰色为比较范围以外" :
    m.iou != null ? "黄线为原始标注，青线为模型回投" : m.texture ? "坑内像素取自 GLB 内嵌材质，周围为留出照片" : "独立观测的实拍照片");
  el("pitScore").textContent = m.iou != null ? "坑缘 IoU "+m.iou.toFixed(3) : m.texture && view.colorError != null ? "RGB 平均绝对误差 "+view.colorError.toFixed(1)+" / 255" : "";
  let text = view.description || ("导出 GLB 与留出实测深度的重合区有 "+view.samples.toLocaleString()+" 个像素。深度绝对误差中位数 "+view.median.toFixed(1)+" 毫米，P95 为 "+view.p95.toFixed(1)+" 毫米。");
  if (view.spacing != null) text += " 留出深度估计的双坑中心距为 "+view.spacing.toFixed(3)+" 米。";
  el("pitDepth").textContent = text;
}
function showStates() {
  const bar = el("states");
  bar.replaceChildren();
  bar.hidden = !selected.states;
  if (!selected.states) return;
  for (const key of selected.states) {
    const state = byId.get(key);
    if (!state) continue;
    const link = document.createElement("a");
    link.href = "#"+key;
    link.textContent = state.state;
    if (key === selected.id) link.setAttribute("aria-current","true");
    bar.append(link);
  }
}
function inspect(id) {
  selected = byId.get(id);
  if (!selected) return;
  reverse = before = spinning = lidSource = cutaway = gaussian = false;
  el("detail").hidden = false;
  el("assetKey").textContent = selected.id;
  el("assetTitle").textContent = selected.label;
  el("sourceType").textContent = selected.category+" / "+selected.method;
  el("note").textContent = selected.note || notes[selected.method];
  el("metrics").replaceChildren();
  if (selected.components) metric("整组槽口 / 独立网格",selected.componentCount+" / "+selected.components.length);
  if (selected.slab) {
    metric("原图闭合面积（平方厘米）",(selected.slab.area*10000).toFixed(1));
    metric("独立碎板体积（升）",(selected.slab.volume*1000).toFixed(3));
    metric("顶面网格间距（毫米）",selected.slab.pitch.toFixed(2));
  }
  if (selected.scanGeometry) {
    const g=selected.scanGeometry;
    metric("网格间距（毫米）",g.pitch.toFixed(2));
    metric("留出缝内距作者位移的中位误差（毫米）",g.median.toFixed(3));
    metric("原图闭合板块（≥25 cm²）",String(g.islands));
  }
  if (selected.water) {
    const w=selected.water;
    metric("加入 / 留存（升）",w.input_liters.toFixed(3)+" / "+w.retained_liters.toFixed(3));
    metric("溢出（升）",w.spilled_liters.toFixed(3));
    metric("最大水深（毫米）",(1000*w.max_water_depth_m).toFixed(1));
    metric("水面面积（平方米）",w.surface_area_m2.toFixed(3));
  }
  if (selected.captureViews) {
    metric(selected.lidDesigned ? "实测外圈 / 设计盖面直径（米）" : "实测外圈 / 盖面直径（米）",selected.frameDiameter.toFixed(3)+" / "+selected.lidDiameter.toFixed(3));
  }
  if (selected.apronExtent) metric("实测修补圈宽 × 长（米）",selected.apronExtent.map(v=>v.toFixed(3)).join(" × "));
  if (selected.panelSize) {
    if (selected.installationGrade) {
      const g = selected.installationGrade;
      metric("井框外缘相对路面（毫米）",g.rim_range_m.map(v=>(1000*v).toFixed(1)).join(" 至 "));
      metric("座圈倾角（度）",g.tilt_deg.toFixed(1));
    }
    if (selected.throughHoles > 0 || selected.retainedVolume != null) {
      metric("盖板宽 × 长（米）",selected.panelSize.map(v=>v.toFixed(3)).join(" × "));
      metric("盖板厚度（毫米）",(selected.lidThickness*1000).toFixed(1));
      if (selected.retainedVolume != null) {
        metric("余盖贯穿孔",String(selected.throughHoles));
        metric("留在座圈的金属体积",(100*selected.retainedVolume).toFixed(1)+"%");
        if (selected.fragmentBottom != null) metric("碎片底面相对路面（米）", (Math.round(1000*selected.fragmentBottom)/1000).toFixed(3));
      } else {
        metric("贯穿孔 / 盖面开孔率",selected.throughHoles+" / "+(100*selected.openFace).toFixed(1)+"%");
      }
    }
    metric("净井口俯视可见井壁或井底",(selected.wellVisible*100).toFixed(1)+"%");
  }
  metric(selected.captureViews || selected.panelSize ? "设计井口宽 × 长（米）" : selected.scanGeometry ? "裂缝外接框宽 × 长（米）" : "宽 × 长（米）",selected.width.toFixed(2)+" × "+selected.length.toFixed(2));
  metric(selected.depthLabel+"（米）",selected.depth.toFixed(3));
  metric("三角面",selected.triangles.toLocaleString());
  if (selected.edgeSampling) metric("高斯点（原始 / 断边加密）",selected.gaussians.toLocaleString()+" / "+selected.edgeSampling.gaussians.toLocaleString());
  else metric("高斯点",selected.gaussians.toLocaleString());
  showModel();
  el("sourceView").replaceChildren(...(selected.sourceViews || []).map((v,i)=>new Option(v.label,String(i))));
  showSource();
  el("pitPanel").hidden = !selected.pitViews;
  if (selected.pitViews) {
    el("pitHeading").textContent=selected.depthPanelTitle || "比较多帧与单帧重建在留出照片上的表现";
    if (selected.matchCoverage!=null) metric("坑内双向匹配覆盖",(100*selected.matchCoverage).toFixed(1)+"%");
    if (selected.spacing != null) metric("双坑中心距（米）",selected.spacing.toFixed(3));
    el("pitFrame").replaceChildren(...selected.pitViews.map((v,i)=>new Option(v.frame,String(i))));
    showPitView(true);
  }
  showStates();
  el("capturePanel").hidden = !selected.captureViews;
  if (selected.captureViews) {
    el("captureView").replaceChildren(...selected.captureViews.map((r,i)=>new Option(r.label,String(i))));
    el("captureMode").value="registered";
    el("captureInfo").textContent=selected.captureAsset+" 为原场景实拍落座状态。局部配准参数来自其他帧。"+selected.laserCount+" 个留出激光点相对盖面网格的高度绝对误差中位数为 "+selected.laserMedianMM.toFixed(1)+" 毫米。";
    showCapture();
  }
  window.showAssetRelief(selected);
  el("roadPanel").hidden = !selected.scenes.length;
  el("scene").replaceChildren(...selected.scenes.map((s,i) => new Option(s.label,String(i))));
  showScene();
}
function renderCards() {
  const total = Math.ceil(filtered.length/pageSize);
  el("count").textContent = filtered.length.toLocaleString()+" 件";
  el("page").textContent = total ? (page+1)+" / "+total : "没有匹配的资产";
  el("prev").disabled = page===0;
  el("next").disabled = page+1>=total;
  el("cards").replaceChildren();
  for (const r of filtered.slice(page*pageSize,(page+1)*pageSize)) {
    const btn = document.createElement("button"), img = new Image(), body = document.createElement("div");
    btn.className="card"; btn.setAttribute("aria-current",String(r===selected));
    img.loading="lazy"; img.alt=r.label; img.src=r.preview;
    const meta = document.createElement("small"), name = document.createElement("strong");
    meta.textContent=r.method+" / "+r.category;
    name.textContent=r.label;
    body.append(meta,name); btn.append(img,body);
    btn.onclick=() => {
      location.hash=encodeURIComponent(r.id);
      inspect(r.id); renderCards();
      el("detail").focus({preventScroll:true}); el("detail").scrollIntoView({block:"start"});
    };
    el("cards").append(btn);
  }
}
function filter() {
  const q=el("query").value.trim().toLowerCase();
  filtered=assets.filter(r => (!el("method").value || el("method").value===r.method)
    && (!el("category").value || el("category").value===r.category)
    && [r.id,r.label,r.method,r.category].join(" ").toLowerCase().includes(q));
  page=0; renderCards();
}
function showCoverage() {
  const table = el("coverageTable");
  const methods = [...new Set(assets.map(r=>r.method))];
  const head = table.createTHead().insertRow();
  for (const name of ["资产形态",...methods]) {
    const cell = document.createElement("th"); cell.scope = "col"; cell.textContent = name;
    head.append(cell);
  }
  const body = table.createTBody();
  for (const category of new Set(assets.map(r=>r.category))) {
    const tr = body.insertRow();
    const label = document.createElement("th"); label.scope="row"; label.textContent=category; tr.append(label);
    const counts = new Map();
    for (const row of assets) if (row.category===category) counts.set(row.method,(counts.get(row.method)||0)+1);
    for (const method of methods) {
      const count = counts.get(method)||0, cell = tr.insertCell();
      if (!count) { cell.textContent="—"; continue; }
      const btn=document.createElement("button"); btn.type="button"; btn.textContent=String(count);
      btn.setAttribute("aria-label",category+"，"+method+"，"+count+" 条");
      btn.onclick=()=>{el("query").value="";el("category").value=category;el("method").value=method;filter();};
      cell.append(btn);
    }
  }
}
if (assets.length) {
  for (const key of ["method","category"])
    for (const item of new Set(assets.map(r => r[key]))) el(key).add(new Option(item,item));
  el("compareRoadLink").hidden=!assets.some(row=>row.matchedRoad);
  el("summary").textContent=assets.length.toLocaleString()+" 件资产记录 · "+new Set(assets.map(r=>r.method)).size+" 种重建方式";
  inspect(byId.has(decodeURIComponent(location.hash.slice(1))) ? decodeURIComponent(location.hash.slice(1)) : assets[0].id);
  filter();
  showCoverage();
  for (const key of ["method","category","query"]) el(key).addEventListener("input",filter);
  el("prev").onclick=()=>{page--; renderCards();}; el("next").onclick=()=>{page++; renderCards();};
  el("before").onclick=()=>{before=!before; showScene();};
  el("scene").onchange=()=>{before=false; showScene();};
  el("representation").onclick=()=>{gaussian=!gaussian; spinning=cutaway=false; showModel();};
  el("spin").onclick=()=>{spinning=!spinning; showModel();};
  el("cutaway").onclick=()=>{cutaway=!cutaway; spinning=false; showModel();};
  el("sourceView").onchange=showSource;
  el("pitFrame").onchange=()=>showPitView(true);
  el("pitMode").onchange=()=>showPitView(false);
  el("captureView").onchange=el("captureMode").onchange=showCapture;
  el("switchSource").onclick=()=>{lidSource=!lidSource; showSource();};
  el("rotate").onclick=()=>{reverse=!reverse; showModel();};
  window.addEventListener("hashchange",()=>{const id=decodeURIComponent(location.hash.slice(1));if (byId.has(id)) {inspect(id); renderCards();}});
} else {
  el("summary").textContent="请在仓库根目录运行 python scripts/build-asset-atlas.py 生成本机索引。";
}
