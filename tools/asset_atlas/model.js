"use strict";
const get = id => document.getElementById(id);
const viewer = get("mesh");
const inventory = new Map((window.roadAssets || []).map(item=>[item.id,item]));
let localUrl;
function status(message) { get("status").textContent=message; }
function pose(orbit) {
  viewer.cameraTarget="auto auto auto";
  viewer.cameraOrbit=orbit;
  viewer.fieldOfView="auto";
}
get("reset").onclick=()=>pose("20deg 48deg auto");
get("top").onclick=()=>pose("0deg 0deg auto");
get("side").onclick=()=>pose("0deg 86deg auto");
get("rotateModel").onclick=()=>{
  viewer.autoRotate=!viewer.autoRotate;
  get("rotateModel").setAttribute("aria-pressed",String(viewer.autoRotate));
  get("rotateModel").textContent=viewer.autoRotate ? "停止旋转" : "自动旋转";
};
viewer.addEventListener("load",()=>{
  status("拖动旋转，滚轮或双指缩放。");
});
viewer.addEventListener("error",()=>{
  status("模型加载失败。可以选择本机 GLB 重新打开。");
  get("fileLabel").hidden=false;
});
get("file").onchange=()=>{
  const file=get("file").files[0];
  if (!file) return;
  if (localUrl) URL.revokeObjectURL(localUrl);
  localUrl=URL.createObjectURL(file);
  viewer.src=localUrl;
  get("key").textContent=file.name;
  get("title").textContent="本机三维模型";
  status("正在读取所选模型…");
};
function openSelection() {
  const requested = decodeURIComponent(location.hash.slice(1));
  const row = inventory.get(requested);
  get("states").replaceChildren();
  get("states").hidden = !row?.states;
  get("fileLabel").hidden = Boolean(row) && location.protocol!=="file:";
  get("glb").hidden = !row;
  get("title").textContent=row?.label || "三维模型";
  get("key").textContent=row?.id || "";
  get("back").href=row ? "index.html#"+encodeURIComponent(row.id) : "index.html";
  if (row) {
    get("glb").href=row.glb;
    viewer.poster=row.preview;
    for (const id of row.states || []) {
      const link=document.createElement("a");
      link.href="#"+id;
      link.textContent=inventory.get(id)?.state || id;
      if (id===row.id) link.setAttribute("aria-current","true");
      get("states").append(link);
    }
  }
  if (!row || location.protocol==="file:") {
    viewer.removeAttribute("src");
    status(row ? "直接打开网页时，请选择对应的 GLB。使用项目的本地预览服务可直接加载所选资产。" : "选择一个本机 GLB 查看。");
  } else {
    status("正在读取资产网格与纹理…");
    viewer.src=row.glb;
  }
}
window.addEventListener("hashchange",openSelection);
openSelection();
