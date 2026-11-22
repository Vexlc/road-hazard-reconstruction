"use strict";
(() => {
  const groups=window.matchedRoad;
  const el=id=>document.getElementById(id);
  if (!groups || !Object.keys(groups).length) {
    el("compareStatus").textContent="请先从本项目构建同位置道路观测。";
    return;
  }
  const examples=[
    ["109-01","P112","A119","沿缝双坑与掀开的井盖"],
    ["109-01","A119","M118","同一井体的掀盖与缺盖"],
    ["109-01","P122","P130","同一坑底的干燥与积水"],
    ["034-01","A117","P135","断条雨水篦与混凝土网裂"],
    ["109-05","P143","M138","小型实拍坑洞与缺盖井"],
    ["109-05","A127","A128","断片在井底与井旁"],
    ["110-01","P02_wet_wheel_track","A93","生成车辙坑与实拍掀盖"],
    ["109-03","M11","M03_brick_lined","两种井壁的敞井"],
    ["109-01","P145","P147","同一碎板的凸起与脱落"],
    ["034-01","P144","P146","同一碎板的原位与下沉"]
  ].filter(([g,a,b])=>groups[g]?.assets[a] && groups[g].assets[b]);
  const partNames=["","","坑缘与坑底","坑壁","积水","井圈与座圈","井壁","井底","混凝土井肩","盖面","盖板底面","盖板边缘","铰链","碎板顶面","碎板断面","碎板底面","基层","坑槽侧壁"];
  const colors=["","","#dc803f","#ac5640","#3fa1be","#efbc53","#7659ae","#4a497a","#909589","#3b9c7f","#79c2a9","#316c61","#c35783","#409b89","#c76f45","#7b6491","#8f7149","#ae7d69"];
  const srcRoot="../../runs/road-comparisons/";
  const panels=["a","b"].map((side,i)=>{
    const panel=el("alternative").content.firstElementChild.cloneNode(true);
    const picker=document.createElement("select");picker.id=side;
    panel.querySelector(".pick").append("资产 "+(i+1),picker);
    el("alternatives").append(panel);
    picker.onchange=()=>setLocation({[side]:picker.value});
    return {panel,picker};
  });
  for(const key of Object.keys(groups)) {
    const [site,slot]=key.split("-");
    el("place").add(new Option("PandaSet "+site+" · 位置 "+slot,key));
  }
  examples.forEach((e,i)=>el("preset").add(new Option(e[3],String(i))));
  const modes={front:"前视",back:"后视",near:"近处",far:"远处"};
  const formatView=name=>name.split("-").map(m=>modes[m]).join(" · ");
  const fmt=n=>n.toLocaleString("zh-CN");
  let group,key,shotName,a,b;
  const observations=new Map(),loading=new Set();
  window.roadViewData=(name,views)=>{
    observations.set(name,views);
    if(name===key) draw();
  };
  function setLocation(change) {
    const params=new URLSearchParams({group:key,view:shotName,a,b,...change});
    if(el("zoom").checked) params.set("zoom","1");
    if(el("parts").checked) params.set("parts","1");
    if(el("rays").checked) {
      params.set("rays","1");
      if(selectedBeam!==null && params.get("group")===key && params.get("view")===shotName &&
         params.get("a")===a && params.get("b")===b) params.set("beam",selectedBeam);
    }
    if(location.hash.slice(1)===params.toString()) populate();
    else location.hash=params.toString();
  }
  function populate() {
    const q=new URLSearchParams(location.hash.slice(1));
    const initial=groups["109-01"] ? "109-01" : Object.keys(groups)[0];
    key=Object.hasOwn(groups,q.get("group")) ? q.get("group") : initial;
    group=groups[key];
    const sample=examples.find(e=>e[0]===key);
    const ids=Object.keys(group.assets);
    const valid=id=>id==="road" || Object.hasOwn(group.assets,id);
    a=valid(q.get("a")) ? q.get("a") : sample?.[1] || ids[0];
    b=valid(q.get("b")) ? q.get("b") : (sample?.[2]!==a && sample?.[2]) || ids.find(id=>id!==a) || "road";
    shotName=Object.hasOwn(group.shots,q.get("view")) ? q.get("view") : Object.keys(group.shots)[0];
    const row=q.get("beam");
    selectedBeam=row!==null && /^\d+$/.test(row) && Number.isSafeInteger(Number(row)) ? Number(row) : null;
    el("rays").checked=q.get("rays")==="1" || selectedBeam!==null;
    el("parts").checked=q.get("parts")==="1";
    el("zoom").checked=q.get("zoom")==="1";
    el("place").value=key;
    el("shot").replaceChildren(...Object.entries(group.shots).map(([k,v])=>new Option(formatView(k)+" · "+v.distance_m.toFixed(1)+" m",k)));
    el("shot").value=shotName;
    const match=examples.findIndex(e=>e[0]===key && e[1]===a && e[2]===b);
    el("preset").value=match===-1 ? "" : String(match);
    for(const [{picker},id] of panels.map((p,i)=>[p,i?b:a])) {
      picker.replaceChildren(new Option("原始路面","road"),...ids.map(k=>new Option(k+" · "+group.assets[k].label,k)));
      picker.value=id;
    }
    draw();
  }
  let selectedBeam=null, pairs=[];
  const beamDifference=pair=>pair[1][7]-pair[0][7];
  function beamOptions(views) {
    const section=el("beamComparison"), active=el("rays").checked;
    section.hidden=!active;
    pairs=[];
    if(!active) return null;
    const left=views[a]?.[shotName].laser.points || [], right=views[b]?.[shotName].laser.points || [];
    const byRow=new Map(right.map(p=>[p[5],p]));
    for(const p of left) {
      const q=byRow.get(p[5]);
      if(q && p.length===8 && p[3]===q[3] && p[6]===q[6]) pairs.push([p,q]);
    }
    pairs.sort((a,b)=>Math.abs(beamDifference(b))-Math.abs(beamDifference(a)) || a[0][5]-b[0][5]);
    const count=pairs.filter(p=>Math.abs(beamDifference(p))>10000).length;
    el("beamSummary").textContent=a==="road" || b==="road" ? "选择两件资产可比较它们对同一条原始光束的模拟返回距离。" :
      "同一帧中，有 "+fmt(pairs.length)+" 束激光在两种资产上均有返回并投到裁切内，其中 "+fmt(count)+" 束的距离变化超过 10 毫米。";
    el("beamChoice").hidden=!pairs.length;el("beamDetails").textContent="";
    if(!pairs.length) return null;
    const selected=pairs.find(p=>p[0][5]===selectedBeam) || pairs[0];
    selectedBeam=selected[0][5];
    const picks=pairs.slice(0,30);
    if(!picks.includes(selected)) picks.push(selected);
    const options=picks.map(p=>new Option("行 "+p[0][5]+" · 相差 "+(Math.abs(beamDifference(p))/1000).toFixed(1)+" mm",String(p[0][5])));
    el("beam").replaceChildren(...options);el("beam").value=String(selectedBeam);
    const [one,two]=selected;
    const reading=p=>partNames[p[2]]+" "+(p[7]/1e6).toFixed(3)+" m";
    el("beamDetails").textContent="雷达 "+one[3]+"，原始返回 "+(one[6]/1e6).toFixed(3)+" m。资产 1 命中"+reading(one)+"，资产 2 命中"+reading(two)+"。白圈标出所选回波。";
    return selected;
  }
  el("beam").onchange=()=>{selectedBeam=Number(el("beam").value);setLocation({});};
  function zoomWindow(views,shot,w,h) {
    if(!el("zoom").checked) return [0,0,w,h];
    const extents={};
    let span=Math.min(w,80);
    for(const [name,s] of Object.entries(group.shots)) {
      if(s.camera!==shot.camera) continue;
      let l=Infinity,t=Infinity,r=-Infinity,bottom=-Infinity;
      function include(x,y,right=x+1,lower=y+1) {
        l=Math.min(l,x);t=Math.min(t,y);r=Math.max(r,right);bottom=Math.max(bottom,lower);
      }
      for(const id of new Set([a,b])) {
        const obs=views[id]?.[name];
        if(obs?.bounds) include(...obs.bounds);
        for(const p of obs?.laser.points || []) include(p[0],p[1]);
      }
      if(!Number.isFinite(l)) {
        const world=group.anchor.center.map((v,i)=>v-s.pose[i][3]);
        const cam=[0,1,2].map(j=>world.reduce((sum,v,i)=>sum+v*s.pose[i][j],0));
        include(s.K[0][0]*cam[0]/cam[2]+s.K[0][2]-s.rect[0],s.K[1][1]*cam[1]/cam[2]+s.K[1][2]-s.rect[1]);
      }
      span=Math.max(span,(r-l+12)*1.13,(bottom-t+12)*1.13*w/h);
      extents[name]=[(l+r)/2,(t+bottom)/2];
    }
    const width=Math.min(w,Math.ceil(span)),height=Math.min(h,Math.ceil(width*h/w));
    const origin=(c,size,total)=>Math.max(0,Math.min(total-size,Math.round(c-size/2)));
    const [cx,cy]=extents[shotName];
    return [origin(cx,width,w),origin(cy,height,h),width,height];
  }
  function frameRect(element,x,y,width,height,parentWidth,parentHeight) {
    const sx=100/parentWidth,sy=100/parentHeight;
    Object.assign(element.style,{left:x*sx+"%",top:y*sy+"%",width:width*sx+"%",height:height*sy+"%"});
  }
  function draw() {
    const views=observations.get(key);
    if(!views) {
      el("compareStatus").textContent="正在载入该位置的道路观测…";el("compareStatus").hidden=false;
      el("alternatives").hidden=true;document.querySelector(".road-frame").hidden=true;
      el("beamComparison").hidden=true;
      if(!loading.has(key)) {
        const name=key,script=document.createElement("script");loading.add(name);
        script.src=srcRoot+name+"/views.js";
        script.onload=()=>{loading.delete(name);script.remove();};
        script.onerror=()=>{loading.delete(name);script.remove();if(name===key) el("compareStatus").textContent="该位置的道路观测文件未能读取。";};
        document.head.append(script);
      }
      return;
    }
    el("compareStatus").hidden=true;el("alternatives").hidden=false;document.querySelector(".road-frame").hidden=false;
    const shot=group.shots[shotName],folder=srcRoot+key+"/"+shotName+"/";
    const [l,t,r,bottom]=shot.rect,w=r-l,h=bottom-t;
    const [ox,oy,vw,vh]=zoomWindow(views,shot,w,h);
    const context=el("context"),contextUrl=folder+"road.jpg";
    context.src=contextUrl;el("contextLink").href=contextUrl;
    el("sceneTitle").textContent="PandaSet "+key.split("-")[0]+"　"+formatView(shotName);
    el("sceneCaption").textContent="实拍第 "+shot.frame+" 帧，相机距放置中心 "+shot.distance_m.toFixed(1)+" 米。";
    const zoomed=el("zoom").checked;
    el("focusWindow").hidden=!zoomed;
    if(zoomed) frameRect(el("focusWindow"),l+ox,t+oy,vw,vh,...shot.size);
    const area=zoomed ? "白框内放大区域 "+vw+" × "+vh+" 像素，同一相机近远共用此跨度。" : "黄框内原始裁切 "+w+" × "+h+" 像素。";
    el("sceneCrop").textContent=area+"此位置可对照 "+Object.keys(group.assets).length+" 件资产。";
    const selectedPair=beamOptions(views);
    const seen=new Set();
    panels.forEach(({panel},i)=>{
      const id=i?b:a,row=group.assets[id],v=views[id]?.[shotName];
      const at=sel=>panel.querySelector(sel);
      at(".crop").style.aspectRatio=vw+"/"+vh;
      const plane=at(".image-plane");
      frameRect(plane,-ox,-oy,w,h,vw,vh);
      const image=at(".appearance");image.src=folder+"before.webp";
      const overlayImage=at(".asset-layer");overlayImage.hidden=!row;
      if(row) overlayImage.src=folder+id+".webp";else overlayImage.removeAttribute("src");
      image.alt=(row?row.label:"原始路面")+"，"+formatView(shotName)+"，帧 "+shot.frame;
      const mask=at(".mask");
      mask.hidden=!(v && el("parts").checked);
      if (!mask.hidden) mask.src=folder+id+".png";else mask.removeAttribute("src");
      if (v) v.parts.forEach((n,code)=>{if(n) seen.add(code);});
      const overlay=at(".returns");
      overlay.hidden=!(v && el("rays").checked);overlay.width=w;overlay.height=h;
      if (!overlay.hidden) {
        const ctx=overlay.getContext("2d");
        for(const [x,y,,,err] of v.laser.points) {
          const agreed=err!==null && Math.abs(err)<=50;
          ctx.beginPath();ctx.arc(x+.5,y+.5,agreed?1.8:2.4,0,2*Math.PI);
          if(agreed){ctx.fillStyle="#44fae7";ctx.fill();}
          ctx.lineWidth=agreed ? .7 : 1.4;ctx.strokeStyle=agreed?"#183e42":"#ffb543";ctx.stroke();
        }
      }
      const crop=at(".crop");
      crop.classList.toggle("pick-beam",Boolean(selectedPair));crop.onclick=null;
      if(selectedPair) {
        const [x,y]=selectedPair[i],ctx=overlay.getContext("2d");
        ctx.beginPath();ctx.arc(x+.5,y+.5,6,0,2*Math.PI);
        ctx.strokeStyle="#192722";ctx.lineWidth=3;ctx.stroke();ctx.strokeStyle="#fff";ctx.lineWidth=1.5;ctx.stroke();
        crop.onclick=event=>{
          const rect=plane.getBoundingClientRect(),cx=(event.clientX-rect.left)*w/rect.width,cy=(event.clientY-rect.top)*h/rect.height;
          let choice,nearest=(9*w/rect.width)**2;
          for(const pair of pairs) {
            const pt=pair[i],d=(pt[0]+.5-cx)**2+(pt[1]+.5-cy)**2;
            if(d<=nearest){choice=pt[5];nearest=d;}
          }
          if(choice!==undefined){selectedBeam=choice;setLocation({});}
        };
      }
      at(".asset-id").textContent=row?id:"";
      at(".label").textContent=row?.label || "原始路面";
      const depth=Number.isFinite(row?.depth) && row.depthLabel ? "　"+row.depthLabel+" "+(1000*row.depth).toLocaleString("zh-CN",{maximumFractionDigits:1})+" mm" : "";
      at(".source").textContent=row ? row.method+"　"+row.width.toFixed(2)+" × "+row.length.toFixed(2)+" m"+depth : "";
      at(".counts").textContent=v ? "网格可见 "+fmt(v.visible)+" 像素，高斯不透明度超过 48% 的可见像素 "+fmt(v.opaque)+" 个。" : "实拍相机原始像素。";
      at(".lidar").textContent=v ? "当帧有 "+fmt(v.laser.total)+" 个资产模拟回波，"+fmt(v.laser.points.length)+" 个投到此裁切内。" : "";
      at(".asset-images").hidden=!row;
      for(const cls of [".reference-image",".model-preview"]) at(cls).hidden=!row;
      const link=at(".details-link");link.hidden=!row;
      if(row) {
        at(".reference-link").href=row.reference.image;
        at(".reference-link span").textContent=row.reference.label;
        at(".reference-image").src=row.reference.image;
        at(".model-preview").src=row.preview;
        const hasViewer=row.detail.startsWith("index.html#");
        at(".model-link").href=hasViewer ? "model.html#"+id : row.preview;
        at(".model-link span").textContent=hasViewer ? "旋转独立模型" : "放大独立模型";
        link.href=row.detail;
        link.textContent=hasViewer ? "查看资产与重建输入" : "查看标定样例";
      } else {
        at(".reference-image").removeAttribute("src");at(".model-preview").removeAttribute("src");
      }
    });
    const legend=el("partLegend");
    legend.hidden=!el("parts").checked;
    legend.replaceChildren(...[...seen].sort((x,y)=>x-y).map(code=>{
      const item=document.createElement("span"),swatch=document.createElement("i");
      swatch.style.background=colors[code];item.append(swatch,partNames[code]);return item;
    }));
    el("rayLegend").hidden=!el("rays").checked;
  }
  el("place").onchange=()=>setLocation({group:el("place").value});
  el("shot").onchange=()=>setLocation({view:el("shot").value});
  el("preset").onchange=()=>{if(el("preset").value!==""){
    const [group,a,b]=examples[Number(el("preset").value)];setLocation({group,a,b});
  }};
  for(const id of ["zoom","parts","rays"])el(id).onchange=()=>setLocation({});
  window.addEventListener("hashchange",populate);
  populate();el("comparison").hidden=false;
})();
