"use strict";
const el = id => document.getElementById(id);
const assets = window.roadAssets || [];
const byId = new Map(assets.map(row => [row.id,row]));
const pageSize = 12;
let page = 0, selected, filtered = [], reverse = false, before = false, spinning = false;

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
  if (view) {
    photo("road", before ? view.before : view.after, view.label + (before ? " 迁入前" : " 迁入后"));
    el("before").hidden = !view.before;
    el("before").textContent = before ? "查看迁入后" : "查看迁入前";
  }
  el("downloads").replaceChildren();
  for (const [name,key] of [["GLB","glb"],["3DGS PLY","ply"],["路面场","surface"],["元数据","metadata"],["前视视频","video"],["轮廓评估","audit"],["雷达核验","scan"]])
    if (selected[key]) download(name,selected[key]);
  if (selected.related) download(selected.method==="实拍 RGB" ? "同源 RGB-D" : "同源单目 RGB","#"+selected.related);
  if (selected.sensor && view) {
    const params = new URLSearchParams({asset:selected.id,site:view.site,camera:view.camera});
    download("雷达与剖面",selected.sensor+"#"+params);
  }
}
function showModel() {
  const label = spinning ? "GLB · 环绕渲染" : selected.reverse ? (reverse ? "GLB · 背向斜视" : "GLB · 正向斜视") : "3DGS · 独立渲染";
  photo("model",spinning ? selected.orbit : reverse ? selected.reverse : selected.preview, selected.label+" "+label);
  el("modelLabel").textContent = label;
  el("rotate").hidden = !selected.reverse || spinning;
  el("spin").hidden = !selected.orbit;
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
function inspect(id) {
  selected = byId.get(id);
  if (!selected) return;
  reverse = before = spinning = false;
  el("detail").hidden = false;
  el("assetKey").textContent = selected.id;
  el("assetTitle").textContent = selected.label;
  el("sourceType").textContent = selected.category+" / "+selected.method;
  el("note").textContent = notes[selected.method];
  el("metrics").replaceChildren();
  metric("宽 × 长（米）",selected.width.toFixed(2)+" × "+selected.length.toFixed(2));
  metric(selected.depthLabel+"（米）",selected.depth.toFixed(3));
  metric("三角面",selected.triangles.toLocaleString());
  metric("高斯点",selected.gaussians.toLocaleString());
  showModel();
  el("sourcePanel").hidden = !selected.reference;
  if (selected.reference) {
    const title = selected.method==="生成 RGB" ? "生成参考图" : "实拍 RGB 裁切";
    photo("source",selected.reference,title);
    el("sourceLabel").textContent = title;
  }
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
if (assets.length) {
  for (const key of ["method","category"])
    for (const item of new Set(assets.map(r => r[key]))) el(key).add(new Option(item,item));
  el("summary").textContent=assets.length.toLocaleString()+" 件资产记录 · "+new Set(assets.map(r=>r.method)).size+" 种重建方式";
  inspect(byId.has(decodeURIComponent(location.hash.slice(1))) ? decodeURIComponent(location.hash.slice(1)) : assets[0].id);
  filter();
  for (const key of ["method","category","query"]) el(key).addEventListener("input",filter);
  el("prev").onclick=()=>{page--; renderCards();}; el("next").onclick=()=>{page++; renderCards();};
  el("before").onclick=()=>{before=!before; showScene();};
  el("scene").onchange=()=>{before=false; showScene();};
  el("spin").onclick=()=>{spinning=!spinning; showModel();};
  el("rotate").onclick=()=>{reverse=!reverse; showModel();};
  window.addEventListener("hashchange",()=>{const id=decodeURIComponent(location.hash.slice(1));if (byId.has(id)) {inspect(id); renderCards();}});
} else {
  el("summary").textContent="请在仓库根目录运行 python scripts/build-asset-atlas.py 生成本机索引。";
}
