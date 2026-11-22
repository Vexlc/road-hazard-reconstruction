"use strict";
(() => {
  const get=id=>document.getElementById(id);
  const panel=get("reliefPanel"), slider=get("reliefRow"), map=get("reliefMap"), chart=get("reliefChart");
  const cache=new Map(), pending=new Set();
  let current,grid,heat,rect,range,selectedCol=null;
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
    const end=z<0 ? [36,78,119] : [169,71,34];
    const mix=Math.sqrt(Math.min(1,Math.abs(z)/(z<0 ? -range[0] : range[1])));
    return [239,238,220].map((v,i)=>Math.round(v+(end[i]-v)*mix));
  }
  function prepare() {
    const {n,top,under}=grid;
    let lo=0,hi=0;
    for (const field of under ? [top,under] : [top]) {
      for (const z of field) if (Number.isFinite(z)) {lo=Math.min(lo,z);hi=Math.max(hi,z);}
    }
    range=[Math.min(lo,-.001),Math.max(hi,.001)];
    heat=document.createElement("canvas");heat.width=heat.height=n;
    const g=heat.getContext("2d"), image=g.createImageData(n,n);
    for (let i=0;i<top.length;i++) image.data.set([...color(top[i]),255],4*i);
    g.putImageData(image,0,0);
    slider.max=String(n-1);slider.value=String((n-1)/2);selectedCol=null;
    if (current.water && under) {
      let deepest=null, depth=0;
      for (let i=0;i<top.length;i++) {
        const column=top[i]-under[i];
        if (column>depth) {deepest=i;depth=column;}
      }
      if (deepest!==null) {slider.value=String(Math.floor(deepest/n));selectedCol=deepest%n;}
    }
    get("reliefLegend").textContent="俯视首个交点 · 蓝色低于路面，棕色高于路面，灰色无交点";
    get("reliefScale").textContent=current.water ? "实线为首个交点，虚线为实拍重建的坑底。高程沿资产本地竖轴读取。" : under ? "实线为首个交点，虚线为移除井盖后的固定部件。" : "实线为从上方投射到网格的首个交点。";
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
        if (!Number.isFinite(z)) {pen=false;continue;}
        if (pen) ctx.lineTo(px,py(z));else ctx.moveTo(px,py(z));pen=true;
      }
      ctx.stroke();ctx.setLineDash([]);
    }
    if (under) trace(under,"#8a6b60",[6,4]);
    trace(top,legendColor.below,[]);
    let message="Y "+metres(y[0]+row/(n-1)*(y[1]-y[0]));
    if (selectedCol!==null) {
      const i=row*n+selectedCol;
      message+="　X "+metres(x[0]+selectedCol/(n-1)*(x[1]-x[0]))+"　表面 "+metres(top[i]);
      if (under) message+=(current.water ? "　坑底 " : "　固定部件 ")+metres(under[i]);
      if (current.water) message+="　铅直水深 "+((top[i]-under[i])*current.water.up_local[2]*1000).toFixed(1)+" mm";
    }
    get("reliefReadout").textContent=message;
  }
  function request() {
    if (!panel.open || !current?.relief) return;
    if (cache.has(current.id)) {grid=cache.get(current.id);prepare();return;}
    if (pending.has(current.id)) return;
    get("reliefStatus").textContent="正在读取网格高程…";
    const id=current.id,s=document.createElement("script");pending.add(id);s.src=current.relief;
    s.onload=()=>{pending.delete(id);s.remove();};
    s.onerror=()=>{pending.delete(id);s.remove();if (id===current?.id) get("reliefStatus").textContent="此资产的高程文件读取失败。";};
    document.head.append(s);
  }
  window.roadRelief=(id,data)=>{
    const value={...data,top:decode(data.top,data.low,data.step,data.n),
      under:data.under ? decode(data.under,data.low,data.step,data.n) : null};
    if (cache.size>=6) cache.delete(cache.keys().next().value);
    cache.set(id,value);
    if (id===current?.id && panel.open) {grid=value;prepare();}
  };
  window.showAssetRelief=row=>{
    current=row;grid=null;get("reliefContent").hidden=true;get("reliefStatus").textContent="";
    panel.hidden=!row.relief;request();
  };
  panel.addEventListener("toggle",request);
  slider.addEventListener("input",()=>{selectedCol=null;draw();});
  map.addEventListener("pointerdown",e=>{
    if (!grid || !rect) return;
    const box=map.getBoundingClientRect(), px=(e.clientX-box.left)*map.width/box.width, py=(e.clientY-box.top)*map.height/box.height;
    const [l,t,w,h]=rect;
    if (px<l || px>=l+w || py<t || py>=t+h) return;
    slider.value=String(Math.floor((py-t)/h*grid.n));selectedCol=Math.floor((px-l)/w*grid.n);draw();
  });
})();
