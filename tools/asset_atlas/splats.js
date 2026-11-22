"use strict";
const el=id=>document.getElementById(id);
const viewer=el("mesh"), canvas=el("cloud");
const inventory=new Map((window.roadAssets || []).map(row=>[row.id,row]));
let renderer, worker, localUrls=[], row, selection=0, bounds, hasMesh=false;
let mode="gaussian", sortedDirection, sorting=false, orderReady=false, recycled, frame=0;
let roads=[], before=false, errorState=false, initialPose, meshFailed=false;
const workerUrl=URL.createObjectURL(new Blob(["("+gaussianWorker.toString()+")()"],{type:"text/javascript"}));

// A local PLY can use model-viewer's orbit and touch controls without a companion mesh.
// The bounding box only supplies camera limits and is never offered as a mesh comparison.
function boundsModel(lo,hi) {
  const positions=new Float32Array(24);
  for (let i=0;i<8;i++) {
    const x=(i&1?hi:lo)[0], y=(i&2?hi:lo)[1], z=(i&4?hi:lo)[2];
    positions.set([x,z,-y],i*3);
  }
  const indices=new Uint16Array([0,2,3,0,3,1,4,5,7,4,7,6,0,1,5,0,5,4,2,6,7,2,7,3,0,4,6,0,6,2,1,3,7,1,7,5]);
  const size=positions.byteLength+indices.byteLength;
  const doc={asset:{version:"2.0"},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],
    meshes:[{primitives:[{attributes:{POSITION:0},indices:1}]}],buffers:[{byteLength:size}],
    bufferViews:[{buffer:0,byteLength:positions.byteLength},{buffer:0,byteOffset:positions.byteLength,byteLength:indices.byteLength}],
    accessors:[{bufferView:0,componentType:5126,count:8,type:"VEC3",min:[lo[0],lo[2],-hi[1]],max:[hi[0],hi[2],-lo[1]]},
               {bufferView:1,componentType:5123,count:indices.length,type:"SCALAR"}]};
  const json=new TextEncoder().encode(JSON.stringify(doc)), length=Math.ceil(json.length/4)*4;
  const data=new Uint8Array(28+length+size);
  const view=new DataView(data.buffer);
  for (const [offset,value] of [[0,0x46546c67],[4,2],[8,data.length],[12,length],[16,0x4e4f534a],[20+length,size],[24+length,0x004e4942]]) view.setUint32(offset,value,true);
  data.fill(32,20,20+length); data.set(json,20);
  data.set(new Uint8Array(positions.buffer),28+length); data.set(new Uint8Array(indices.buffer),28+length+positions.byteLength);
  return new Blob([data],{type:"model/gltf-binary"});
}
function localUrl(file) {
  const url=URL.createObjectURL(file);
  localUrls.push(url);return url;
}
function message(s) {el("status").textContent=s;}
function fail(s) {
  errorState=true; message(s);
  el("splatPoster").hidden=!row?.gaussianPreview || mode==="meshMode";
  if (worker) worker.terminate();
  worker=null; sorting=false;
}
function pose(theta,phi) {
  if (!initialPose) return;
  viewer.cameraTarget=initialPose.target.map(n=>n+"m").join(" ");
  const t=theta==null?initialPose.theta:theta*Math.PI/180;
  const p=phi==null?initialPose.phi:phi*Math.PI/180;
  const aspect=viewer.clientWidth/Math.max(viewer.clientHeight,1);
  const radius=initialPose.radius*Math.max(1,(initialPose.aspect||1)/aspect);
  viewer.cameraOrbit=t+"rad "+p+"rad "+radius+"m";
  viewer.fieldOfView=initialPose.fov+"deg";
}
function camera() {
  const {theta,phi,radius}=viewer.getCameraOrbit(), target=viewer.getCameraTarget();
  const t=theta-viewer.turntableRotation, st=Math.sin(t), ct=Math.cos(t), sp=Math.sin(phi), cp=Math.cos(phi);
  const away=[sp*st,-sp*ct,cp];
  return {eye:[target.x+away[0]*radius,-target.z+away[1]*radius,target.y+away[2]*radius],
    right:[ct,st,0],up:[-cp*st,cp*ct,sp],forward:away.map(v=>-v),fov:viewer.getFieldOfView()};
}
function repaint() {
  if (!frame) frame=requestAnimationFrame(paint);
}
function paint() {
  frame=0;
  if (!renderer?.count || !viewer.loaded || mode==="meshMode" || errorState) return;
  const cam=camera();
  if (!Number.isFinite(cam.fov) || !cam.eye.every(Number.isFinite)) return;
  const offset=-cam.forward.reduce((sum,v,i)=>sum+v*cam.eye[i],0);
  const sortView=[...cam.forward,offset].map(Math.fround);
  const needsSort=!sortedDirection || sortView.some((v,i)=>Math.abs(v-sortedDirection[i])>1e-7);
  if (needsSort && !sorting && worker) {
    sorting=true;
    worker.postMessage({type:"sort",forward:sortView,recycle:recycled},recycled?[recycled]:[]);
    recycled=null;
  }
  if (!orderReady) return;
  const rect=viewer.getBoundingClientRect();
  const scale=Math.min(window.devicePixelRatio || 1,1600/Math.max(1,rect.width),1080/Math.max(1,rect.height));
  renderer.draw(cam,Math.max(1,Math.ceil(rect.width*scale)),Math.max(1,Math.ceil(rect.height*scale)));
}
function setMode(next) {
  if (next!=="gaussian" && !hasMesh) return;
  mode=next;
  for (const id of ["gaussian","meshMode","split"]) el(id).setAttribute("aria-pressed",String(id===mode));
  el("meshMode").disabled=el("split").disabled=!hasMesh;
  el("splatPoster").hidden=orderReady || !row?.gaussianPreview || mode==="meshMode";
  el("splitControl").hidden=el("splitLabels").hidden=el("boundary").hidden=mode!=="split";
  canvas.hidden=mode==="meshMode";
  el("hostControls").hidden=mode==="meshMode" || !renderer?.mask || !orderReady;
  const split=el("divider").value;
  el("splatStage").style.setProperty("--split",split+"%");
  canvas.style.clipPath=mode==="split"?"inset(0 "+(100-split)+"% 0 0)":"";
  repaint();
}
const auditNames={oblique:"斜视",top:"俯视",low:"低角度"};
const partNames={apron:"周边路面",cover:"盖板",wall:"井壁",floor:"井底",rim:"井圈",hinge:"铰链",water:"水面",pit_or_crack:"坑或裂缝",geometry_0:"表面网格"};
function showAudit(resetPart) {
  const data=row?.gaussianAudit?.[Number(el("auditPose").value)];
  if (!data) return;
  if (resetPart) {
    for (const [id,key] of [["auditMesh","mesh"],["auditSplat","gaussian"],["auditDepth","difference"]]) {
      el(id+"Link").href=el(id).src=data[key];
    }
    el("auditPart").replaceChildren(new Option("全部有效像素",""),...Object.keys(data.parts).map(k=>new Option(partNames[k] || k,k)));
  }
  const part=data.parts[el("auditPart").value] || {pixels:data.pixels,median_mm:data.median,p95_mm:data.p95};
  el("auditReadout").textContent=part.pixels.toLocaleString()+" 个共同像素的绝对差中位数为 "+part.median_mm.toFixed(2)+" mm，95 分位为 "+part.p95_mm.toFixed(2)+" mm。";
}
el("auditPose").onchange=()=>showAudit(true);
el("auditPart").onchange=()=>showAudit(false);
function showHostAudit() {
  const shot=row?.roadAudit?.[Number(el("hostAuditPose").value)];
  if (!shot) return;
  for (const [index,id] of ["rawRoad","roadProbe","hostColour"].entries()) {
    el(id+"Link").href=el(id).src=shot.images[index];
  }
  el("hostAuditReadout").textContent="地下高斯染色后，井口外的响应像素从 "+shot.exteriorRaw.toLocaleString()+" 个变为 "+shot.exteriorHost.toLocaleString()+
    " 个。井口内 "+shot.interior.toLocaleString()+" 个核验像素在铺路前后的 RGB 通道最大差为 "+shot.change+" / 255。";
}
el("hostAuditPose").onchange=showHostAudit;
el("roadAudit").ontoggle=()=>{if (el("roadAudit").open) showHostAudit();};
el("useAuditPose").onclick=()=>{
  const c=row?.gaussianAudit?.[Number(el("auditPose").value)]?.camera;
  if (!c || !viewer.loaded) return;
  viewer.cameraOrbit=c.theta+"rad "+c.phi+"rad "+c.radius+"m";
  viewer.cameraTarget=c.target.map(n=>n+"m").join(" ");
  viewer.fieldOfView=c.fov+"deg";viewer.jumpCameraToGoal();repaint();
};
function setRoads() {
  before=false;
  roads=(row?.scenes || []).filter(s=>s.camera==="front" || s.camera==="back");
  if (!roads.length) roads=(row?.scenes || []).filter(s=>s.before && !s.beforeLabel).slice(0,4);
  el("roadPanel").hidden=!roads.length;
  el("roadShot").replaceChildren(...roads.map((s,i)=>new Option(s.label,String(i))));
  showRoad();
}
function showRoad() {
  const shot=roads[Number(el("roadShot").value)];
  if (!shot) return;
  el("roadLink").href=el("road").src=before?shot.before:shot.after;
  el("before").hidden=!shot.before;el("before").textContent=before?"查看迁入后":"查看迁入前";
}
function start(selected,files={}) {
  const epoch=++selection;
  const family=Boolean(selected?.states?.includes(row?.id) && (!selected.scanGeometry || (selected.width===row.width && selected.length===row.length)));
  const previous=family && viewer.loaded ? {orbit:viewer.getCameraOrbit(),target:viewer.getCameraTarget(),fov:viewer.getFieldOfView()} : null;
  const hostRequested=family && renderer?.mask?el("hostRoad").checked:new URLSearchParams(location.search).get("road")==="1";
  const traceRequested=family && el("traceUnder").checked;
  row=selected; meshFailed=false; bounds=null; sortedDirection=null; sorting=orderReady=errorState=false; recycled=null; initialPose=null;
  if (worker) worker.terminate();
  worker=null; renderer?.clear();renderer?.setRoad(null);
  el("hostRoad").checked=el("traceUnder").checked=false;
  viewer.removeAttribute("src");
  localUrls.forEach(URL.revokeObjectURL); localUrls=[];
  if (row?.gaussianPreview) el("splatPoster").src=row.gaussianPreview; else el("splatPoster").removeAttribute("src");
  hasMesh=false;setMode("gaussian");
  el("title").textContent=row?.label || files.ply?.name || "高斯资产";
  el("key").textContent=row?row.id+"  /  "+row.method:"";
  el("count").textContent="";
  el("back").href=row?"index.html#"+encodeURIComponent(row.id):"index.html";
  el("states").replaceChildren();el("states").hidden=!row?.states;
  for (const key of row?.states || []) {
    const a=document.createElement("a");a.href="#"+encodeURIComponent(key);a.textContent=inventory.get(key)?.state || key;
    if (key===row.id) a.setAttribute("aria-current","true");
    el("states").append(a);
  }
  el("gaussianAudit").hidden=!row?.gaussianAudit;
  el("gaussianAudit").open=false;
  el("auditPose").replaceChildren(...(row?.gaussianAudit || []).map((a,i)=>new Option(auditNames[a.mode],String(i))));
  if (row?.gaussianAudit) showAudit(true);
  el("roadAudit").open=false;el("roadAudit").hidden=!row?.roadAudit;
  el("hostAuditPose").replaceChildren(...(row?.roadAudit || []).map((a,i)=>new Option(auditNames[a.mode],String(i))));
  for (const id of ["top","side","reset","useAuditPose"]) el(id).disabled=true;
  setRoads();
  if (!files.ply && (!row || location.protocol==="file:")) {
    message(row?"选择本机资产目录中的 asset.ply 即可查看。也可同时选择 asset.glb 做同机位对照。":"选择资产 PLY，或从资产图谱打开高斯视图。");
    return;
  }
  try {
    if (!renderer) renderer=new GaussianRenderer(canvas);
  } catch (err) { fail(err.message);return; }
  message("正在读取原始高斯参数…");
  const meshUrl=files.glb?localUrl(files.glb):(row && location.protocol!=="file:"?row.glb:null);
  if (meshUrl) viewer.src=meshUrl;
  worker=new Worker(workerUrl);
  worker.onerror=event=>{if (epoch===selection) fail(event.message || "无法读取高斯参数。");};
  worker.onmessage=({data})=>{
    if (epoch!==selection) return;
    if (data.type==="error") { fail(data.message);return; }
    if (data.type==="cloud") {
      try { renderer.upload(data); } catch (err) { fail(err.message);return; }
      const opening=window.gaussianRoad?.profiles[row?.id];
      if (opening && opening.glb===row?.gaussianSources?.glb) {
        renderer.setRoad(opening);
        el("hostRoad").checked=Boolean(hostRequested);el("traceUnder").checked=Boolean(traceRequested);
        updateRoadRendering();
      }
      bounds=[data.lo,data.hi];
      for (const id of ["top","side","reset","useAuditPose"]) el(id).disabled=false;
      if (!meshUrl || meshFailed) viewer.src=localUrl(boundsModel(...bounds));
      hasMesh=Boolean(meshUrl) && !meshFailed;setMode(mode);
      const center=data.lo.map((v,i)=>(v+data.hi[i])/2);
      const size=Math.max(...data.hi.map((v,i)=>v-data.lo[i]));
      const span=row ? Math.max(row.width,row.length,(data.hi[2]>0.05?data.hi[2]:0)) : size;
      initialPose=row?.gaussianCamera || {radius:Math.max(0.05,span*2.9),theta:.38,phi:.78,fov:30,aspect:1,target:[center[0],center[2],-center[1]]};
      if (previous) {
        const o=previous.orbit,t=previous.target;
        viewer.cameraOrbit=o.theta+"rad "+o.phi+"rad "+o.radius+"m";
        viewer.cameraTarget=t.x+"m "+t.y+"m "+t.z+"m";
        viewer.fieldOfView=previous.fov+"deg";
      } else pose();
      viewer.jumpCameraToGoal();
      el("count").textContent=data.count.toLocaleString()+" 个高斯";
      message("正在按当前机位合成高斯…");
      repaint();
    } else if (data.type==="order") {
      renderer.setOrder(data.order);recycled=data.order.buffer;sortedDirection=data.forward;sorting=false;
      if (!orderReady) message("拖动旋转，滚轮或双指缩放。");
      orderReady=true;el("splatPoster").hidden=true;
      el("hostControls").hidden=mode==="meshMode" || !renderer.mask;
      repaint();
    }
  };
  worker.postMessage({type:"load",url:row?new URL(row.ply,location.href).href:null,file:files.ply,
    textureLimit:renderer.limit,count:row?.gaussians});
}
async function updateRoadRendering() {
  const epoch=selection;
  renderer.traceUnder=el("traceUnder").checked;
  renderer.roadActive=el("hostRoad").checked;
  repaint();
  if (!renderer.roadActive) return;
  try {await renderer.loadRoadTexture(window.gaussianRoad);}
  catch(err) {
    if (epoch!==selection) return;
    renderer.roadActive=el("hostRoad").checked=false;
    message("无法读取路面材质。"+err.message);
  }
  repaint();
}
el("hostRoad").onchange=el("traceUnder").onchange=updateRoadRendering;
for (const id of ["gaussian","meshMode","split"]) el(id).onclick=()=>setMode(id);
el("divider").oninput=()=>setMode("split");
el("reset").onclick=()=>pose();
el("top").onclick=()=>pose(0,0);
el("side").onclick=()=>pose(0,82);
async function fileDigest(file) {
  const digest=await crypto.subtle.digest("SHA-256",await file.arrayBuffer());
  return Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,"0")).join("");
}
let fileRequest=0;
el("files").onchange=async()=>{
  const files=Array.from(el("files").files);
  const ply=files.filter(f=>/\.ply$/i.test(f.name)), glb=files.filter(f=>/\.glb$/i.test(f.name));
  el("files").value="";
  if (ply.length!==1 || glb.length>1) {message("请选择一个 PLY，可同时选择对应的一个 GLB。");return;}
  const request=++fileRequest, active=selection;
  message("正在核对所选文件与资产图谱…");
  try {
    const [plySha,glbSha]=await Promise.all([fileDigest(ply[0]),glb[0]?fileDigest(glb[0]):null]);
    if (request!==fileRequest || active!==selection) return;
    // Prefer the active record if two entries share an identical PLY.
    const match=row?.gaussianSources?.ply===plySha?row:Array.from(inventory.values()).find(r=>r.gaussianSources?.ply===plySha);
    if (glbSha && match && match.gaussianSources.glb!==glbSha) {
      message("所选网格与 "+match.id+" 的文件指纹不同。可以单独打开 PLY，或同时选择该资产的 GLB。");
      return;
    }
    history.replaceState(null,"",location.href.split("#")[0]+(match?"#"+encodeURIComponent(match.id):""));
    start(match,{ply:ply[0],glb:glb[0]});
  } catch(err) {if (request===fileRequest && active===selection) message("无法读取所选文件。"+err.message);}
};
el("roadShot").onchange=()=>{before=false;showRoad();};
el("before").onclick=()=>{before=!before;showRoad();};
viewer.addEventListener("camera-change",repaint);
viewer.addEventListener("load",()=>{viewer.jumpCameraToGoal();repaint();});
viewer.addEventListener("error",()=>{meshFailed=true;if (bounds && hasMesh) {
  hasMesh=false;viewer.src=localUrl(boundsModel(...bounds));setMode("gaussian");message("网格读取失败，继续显示高斯资产。");
}});
canvas.addEventListener("webglcontextlost",event=>{event.preventDefault();fail("显存上下文已释放，重新载入后可继续查看。");});
canvas.addEventListener("webglcontextrestored",()=>{renderer=null;openSelection();});
new ResizeObserver(repaint).observe(viewer);
function openSelection() {start(inventory.get(decodeURIComponent(location.hash.slice(1))));}
window.addEventListener("hashchange",openSelection);
openSelection();
