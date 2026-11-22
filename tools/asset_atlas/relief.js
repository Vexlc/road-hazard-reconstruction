"use strict";
(() => {
  const get=id=>document.getElementById(id);
  const panel=get("reliefPanel"), slider=get("reliefRow"), map=get("reliefMap"), chart=get("reliefChart");
  const cache=new Map(), pending=new Set();
  const records=new Map((window.roadAssets || []).map(row=>[row.id,row]));
  let current,grid,heat,rect,range,selectedCol=null,probe=null;
  const legendColor={paper:"#f5f4ed",below:"#254974",above:"#a34826",line:"#afbaae"};
  function decode(encoded,low,step,n) {
    const raw=atob(encoded), values=new Float32Array(n*n);
    if (raw.length!==values.length*2) throw Error("Invalid height grid length");
    for (let i=0;i<values.length;i++) {
      const q=raw.charCodeAt(2*i)+256*raw.charCodeAt(2*i+1);
      values[i]=q===0 ? NaN : low+(q-1)*step;
    }
    return values;
  }
  function metres(value) {
    return Number.isFinite(value) ? value.toFixed(3)+" m" : "无交点";
  }
  function color(z) {
    if (!Number.isFinite(z)) return [203,206,199];
    if (current.gradeReference && get("nearGrade").checked && z<range[0]) return [39,52,66];
    const end=z<0 ? [36,78,119] : [169,71,34];
    const mix=Math.sqrt(Math.min(1,Math.abs(z)/(z<0 ? -range[0] : range[1])));
    return [239,238,220].map((v,i)=>Math.round(v+(end[i]-v)*mix));
  }
  const sourceColors=[[230,145,68],[67,91,112],[59,193,188],[220,220,210]];
  const sourceLabels=["插值","原始距离","彩照补深","坑外路面"];
  function paintMap() {
    const {n,top,support}=grid;
    const showSource=support && get("pitSourceMap").checked;
    heat=document.createElement("canvas");heat.width=heat.height=n;
    const g=heat.getContext("2d"), image=g.createImageData(n,n);
    for (let i=0;i<top.length;i++) {
      const rgb=showSource ? sourceColors[support[i]] : color(top[i]);
      image.data.set([...rgb,255],4*i);
    }
    g.putImageData(image,0,0);
    let legend="俯视首个交点 · 蓝色低于路面，棕色高于路面，灰色无交点";
    if (showSource) legend="蓝灰为原始距离，青色为彩照补深，橙色为插值，浅灰为坑外路面";
    if (current.gradeReference && get("nearGrade").checked) legend="相对路面 −80 至 +45 mm · 深色为低于显示范围的交点";
    get("reliefLegend").textContent=legend;
  }
  function prepare() {
    const {n,top,under,rangeOnly,support}=grid;
    let lo=0,hi=0;
    for (const field of [top,under,rangeOnly].filter(Boolean)) {
      for (const z of field) if (Number.isFinite(z)) {lo=Math.min(lo,z);hi=Math.max(hi,z);}
    }
    const gradeZoom=current.gradeReference && get("nearGrade").checked;
    range=gradeZoom ? [-.08,.045] : [Math.min(lo,-.001),Math.max(hi,.001)];
    paintMap();
    slider.max=String(n-1);slider.value=String((n-1)/2);selectedCol=null;
    if (current.gradeReference && probe) {
      const sample=(value,extent)=>Math.max(0,Math.min(n-1,Math.round((value-extent[0])/(extent[1]-extent[0])*(n-1))));
      slider.value=String(sample(probe.y,grid.y));
      selectedCol=probe.x==null ? null : sample(probe.x,grid.x);
    }
    if (current.water && under) {
      let deepest=null, depth=0;
      for (let i=0;i<top.length;i++) {
        const column=top[i]-under[i];
        if (column>depth) {deepest=i;depth=column;}
      }
      if (deepest!==null) {slider.value=String(Math.floor(deepest/n));selectedCol=deepest%n;}
    }
    if (rangeOnly) {
      let peak=-1, largest=-1;
      for (let i=0;i<top.length;i++) {
        const delta=Math.abs(top[i]-rangeOnly[i]);
        if (support[i]===2 && delta>largest) {peak=i;largest=delta;}
      }
      if (peak!==-1) {slider.value=String(Math.floor(peak/n));selectedCol=peak%n;}
      else selectedCol=Math.floor(n/2);
      get("pitColumn").max=String(n-1);get("pitColumn").value=String(selectedCol);
    }
    get("reliefScale").textContent=current.water ? "实线为首个交点，虚线为实拍重建的坑底。高程沿资产本地竖轴读取。" : under ? "实线为首个交点，虚线为移除井盖后的固定部件。" : "实线为从上方投射到网格的首个交点。";
    if (rangeOnly) get("reliefScale").textContent="蓝色实线为彩照补深模型，橙色虚线为仅用距离图的模型。纵轴为相对路面的高程。";
    if (current.gradeReference) {
      get("reliefScale").textContent="蓝线为当前表面"+(current.id===current.gradeReference ? "" : "，橙色虚线为齐平状态的同一截线")+"。切换状态保留取样位置。";
    }
    get("reliefContent").hidden=false;
    const pitch=Math.max(Math.abs(grid.x[1]-grid.x[0]), Math.abs(grid.y[1]-grid.y[0]))/(n-1);
    get("reliefStatus").textContent="网格采样间隔约 "+(1000*pitch).toFixed(1)+" mm。点击俯视图或移动滑块可查看不同截线。";
    draw();
  }
  function draw() {
    if (!grid) return;
    const {n,top,under,x,y}=grid, row=Number(slider.value);
    const a=map.getContext("2d");a.fillStyle=legendColor.paper;a.fillRect(0,0,map.width,map.height);
    const factor=Math.min((map.width-28)/Math.abs(x[1]-x[0]),(map.height-20)/Math.abs(y[1]-y[0]));
    const w=factor*Math.abs(x[1]-x[0]),h=factor*Math.abs(y[1]-y[0]);
    rect=[(map.width-w)/2,(map.height-h)/2,w,h];
    const [left,t,wid,ht]=rect, yy=t+(row+.5)/n*ht;
    a.imageSmoothingEnabled=false;a.drawImage(heat,left,t,wid,ht);
    a.strokeStyle="#eea826";a.lineWidth=2;a.beginPath();a.moveTo(left,yy);a.lineTo(left+wid,yy);a.stroke();
    if (selectedCol!==null) {
      a.fillStyle="#181f1c";a.beginPath();a.arc(left+(selectedCol+.5)/n*wid,yy,4,0,2*Math.PI);a.fill();
    }
    const ctx=chart.getContext("2d"), L=70,R=chart.width-16,T=25,B=chart.height-41;
    ctx.fillStyle=legendColor.paper;ctx.fillRect(0,0,chart.width,chart.height);
    const floor=range[0],ceil=range[1],py=z=>B-(z-floor)/(ceil-floor)*(B-T);
    ctx.font="14px sans-serif";ctx.textAlign="right";
    for (const z of [...new Set([floor,0,ceil])]) {
      const ty=py(z);
      ctx.strokeStyle=z===0 ? "#829084" : "#d0d4c7";ctx.beginPath();ctx.moveTo(L,ty);ctx.lineTo(R,ty);ctx.stroke();
      if (z!==0 && Math.abs(ty-py(0))<15) continue;
      ctx.fillStyle="#435349";ctx.fillText(z.toFixed(3),L-8,ty+4);
    }
    ctx.textAlign="left";ctx.fillText("高程 m",8,16);ctx.fillText("X "+metres(x[0]),L,B+23);
    ctx.textAlign="right";ctx.fillText(metres(x[1]),R,B+23);
    function trace(field,stroke,dash) {
      ctx.strokeStyle=stroke;ctx.lineWidth=2.5;ctx.setLineDash(dash);ctx.beginPath();let pen=false;
      for (let i=0;i<n;i++) {
        const z=field[row*n+i],px=L+i/(n-1)*(R-L);
        if (!Number.isFinite(z) || z<floor || z>ceil) {pen=false;continue;}
        if (pen) ctx.lineTo(px,py(z));else ctx.moveTo(px,py(z));pen=true;
      }
      ctx.stroke();ctx.setLineDash([]);
    }
    const ref=cache.get(current.gradeReference);
    const aligned=ref && ref.n===n && ref.x.every((v,i)=>Math.abs(v-x[i])<1e-7) && ref.y.every((v,i)=>Math.abs(v-y[i])<1e-7);
    if (under && !current.gradeReference) trace(under,"#8a6b60",[6,4]);
    trace(top,legendColor.below,[]);
    if (grid.rangeOnly) trace(grid.rangeOnly,"#ab602d",[5,5]);
    if (aligned && current.id!==current.gradeReference) trace(ref.top,"#ab602d",[5,5]);
    let message="Y "+metres(y[0]+row/(n-1)*(y[1]-y[0]));
    if (selectedCol!==null) {
      const i=row*n+selectedCol;
      message+="　X "+metres(x[0]+selectedCol/(n-1)*(x[1]-x[0]))+"　表面 "+metres(top[i]);
      if (grid.rangeOnly) {
        const baseline=grid.rangeOnly[i],delta=1000*(top[i]-baseline);
        message+="　仅距离图 "+metres(baseline)+"　补深后高差 "+(delta>0 ? "+" : "")+delta.toFixed(1)+" mm　"+sourceLabels[grid.support[i]];
        const px=L+selectedCol/(n-1)*(R-L);
        ctx.strokeStyle="#666d63";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(px,T);ctx.lineTo(px,B);ctx.stroke();
        for (const [z,stroke] of [[baseline,"#ab602d"],[top[i],legendColor.below]]) {
          ctx.beginPath();ctx.arc(px,py(z),3.5,0,2*Math.PI);ctx.fillStyle=stroke;ctx.fill();
        }
      }
      if (under) message+=(current.water ? "　坑底 " : "　固定部件 ")+metres(under[i]);
      if (aligned && current.id!==current.gradeReference && Number.isFinite(top[i]-ref.top[i])) {
        message+="　相对齐平 "+((top[i]-ref.top[i])*1000).toFixed(1)+" mm";
      }
      if (current.water) message+="　铅直水深 "+((top[i]-under[i])*current.water.up_local[2]*1000).toFixed(1)+" mm";
    }
    get("reliefReadout").textContent=message;
  }
  function loadGrid(record) {
    if (!record?.relief || cache.has(record.id) || pending.has(record.id)) return;
    const id=record.id,s=document.createElement("script");pending.add(id);s.src=record.relief;
    s.onload=()=>{pending.delete(id);s.remove();};
    s.onerror=()=>{pending.delete(id);s.remove();if (id===current?.id) get("reliefStatus").textContent="此资产的高程文件读取失败。";};
    document.head.append(s);
  }
  function request() {
    if (!panel.open || !current?.relief) return;
    if (current.gradeReference) loadGrid(records.get(current.gradeReference));
    if (cache.has(current.id)) {grid=cache.get(current.id);prepare();return;}
    get("reliefStatus").textContent="正在读取网格高程…";
    loadGrid(current);
  }
  window.roadRelief=(id,data)=>{
    const value={...data,top:decode(data.top,data.low,data.step,data.n),
      under:data.under ? decode(data.under,data.low,data.step,data.n) : null};
    if (data.rangeOnly) {
      value.rangeOnly=decode(data.rangeOnly,data.low,data.step,data.n);
      value.support=Uint8Array.from(atob(data.support),c=>c.charCodeAt(0));
      if (value.support.length!==data.n*data.n || value.support.some(code=>code>3)) throw Error("Invalid support grid");
    }
    if (cache.size>=6 && !cache.has(id)) {
      const oldest=Array.from(cache.keys()).find(key=>key!==current?.id && key!==current?.gradeReference);
      cache.delete(oldest);
    }
    cache.set(id,value);
    if (id===current?.id && panel.open) {grid=value;prepare();}
    else if (id===current?.gradeReference && panel.open && grid) draw();
  };
  function remember() {
    if (!current?.gradeReference || !grid) return;
    const coordinate=(index,extent)=>extent[0]+index/(grid.n-1)*(extent[1]-extent[0]);
    probe={y:coordinate(Number(slider.value),grid.y),x:selectedCol==null ? null : coordinate(selectedCol,grid.x)};
  }
  window.showAssetRelief=row=>{
    if (!row.gradeReference || row.gradeReference!==current?.gradeReference) {
      probe=null;get("nearGrade").checked=true;
    } else remember();
    current=row;grid=null;get("reliefContent").hidden=true;get("reliefStatus").textContent="";
    get("pitColumnControl").hidden=!row.completionModels;
    get("pitSourceToggle").hidden=!row.completionModels;
    get("pitSourceMap").checked=true;
    get("gradeRange").hidden=!row.gradeReference;
    panel.hidden=!row.relief;request();
  };
  panel.addEventListener("toggle",request);
  get("pitSourceMap").addEventListener("change",()=>{if (grid?.support) {paintMap();draw();}});
  get("nearGrade").addEventListener("change",()=>{if (grid) {remember();prepare();}});
  slider.addEventListener("input",()=>{if (!grid?.rangeOnly) selectedCol=null;remember();draw();});
  get("pitColumn").addEventListener("input",()=>{if (grid?.rangeOnly) {selectedCol=Number(get("pitColumn").value);draw();}});
  map.addEventListener("pointerdown",e=>{
    if (!grid || !rect) return;
    const box=map.getBoundingClientRect(), px=(e.clientX-box.left)*map.width/box.width, py=(e.clientY-box.top)*map.height/box.height;
    const [l,t,w,h]=rect;
    if (px<l || px>=l+w || py<t || py>=t+h) return;
    slider.value=String(Math.floor((py-t)/h*grid.n));selectedCol=Math.floor((px-l)/w*grid.n);
    if (grid.rangeOnly) get("pitColumn").value=String(selectedCol);
    remember();draw();
  });
})();
