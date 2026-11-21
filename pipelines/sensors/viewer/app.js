'use strict';
const find = id => document.getElementById(id);
const catalog = window.ROAD;
const query = new URLSearchParams(location.hash.slice(1));
let asset = catalog[query.get('asset')] ? query.get('asset') : 'A13';
let site = ['034','109','110'].includes(query.get('site')) ? query.get('site') : '034';
let camera = query.get('camera') === 'back' ? 'back' : 'front';
let after = true, azimuth = 0.72, elevation = 0.46, revision = 0;
let cloud = null, grid = null, lastHeight = '', drag = null;
const files = new Map();
function dataset(path) {
  if (!files.has(path)) files.set(path,new Promise((resolve,reject) => {
    const script = document.createElement('script');
    script.onload = resolve; script.onerror = () => reject(new Error('无法读取 '+path));
    script.src = path; document.head.append(script);
  }));
  return files.get(path);
}
function shorts(data) {
  const raw=atob(data), buf=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++) buf[i]=raw.charCodeAt(i);
  return new Int16Array(buf.buffer);
}
function pressed(id,yes) { find(id).setAttribute('aria-pressed',String(yes)); }
function assetList(category, preferred) {
  const ids=Object.keys(catalog).filter(k=>catalog[k].category===category).sort();
  const sel=find('asset'); sel.replaceChildren();
  ids.forEach(id=>sel.add(new Option(id+' · '+catalog[id].label,id)));
  asset=ids.includes(preferred) ? preferred : ids[0]; sel.value=asset;
}
function setupCanvas(id) {
  const c=find(id), w=c.clientWidth, h=c.clientHeight, scale=Math.min(devicePixelRatio||1,2);
  c.width=Math.round(w*scale); c.height=Math.round(h*scale);
  const ctx=c.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);
  return [ctx,w,h];
}
function drawCloud() {
  const [g,w,h]=setupCanvas('cloud');g.fillStyle='#172521';g.fillRect(0,0,w,h);
  if(!cloud) return;
  const scale=Math.min(w*.235,h*.49),cx=w*.50,cy=h*.52;
  function project(x,y,z) {
    const u=x*Math.cos(azimuth)-y*Math.sin(azimuth);
    const v=x*Math.sin(azimuth)+y*Math.cos(azimuth);
    return [cx+u*scale,cy+(v*Math.sin(elevation)-z*Math.cos(elevation))*scale];
  }
  function ln(p,q,col,width=1) {
    g.strokeStyle=col;g.lineWidth=width;g.beginPath();g.moveTo(...p);g.lineTo(...q);g.stroke();
  }
  for(let j=-3;j<=3;j++){
    const k=j*.5;ln(project(-1.5,k,0),project(1.5,k,0),'#34493e');
    ln(project(k,-1.5,0),project(k,1.5,0),'#34493e');
  }
  const dots=[];
  for(let i=0;i<cloud.length;i+=9) {
    const code=cloud[i+6], off=after?3:0;
    if(!after && !cloud[i+8]) continue;
    const x=cloud[i+off]*.001,y=cloud[i+off+1]*.001,z=cloud[i+off+2]*.001;
    const [px,py]=project(x,y,z), key=x*Math.sin(azimuth)+y*Math.cos(azimuth)+z*Math.tan(elevation);
    const color=code===0?'#a1ada3':!after?'#e7ece9':code>=9?'#ee9064':code===5||code===8?'#dfbf6a':'#56bdd3';
    dots.push([key,px,py,code===0?1.35:1.9,color]);
  }
  dots.sort((a,b)=>a[0]-b[0]);
  for(const [,x,y,r,col] of dots){g.fillStyle=col;g.fillRect(x-r,y-r,r*2,r*2);}
}
function drawProfile(){
  const [g,w,h]=setupCanvas('profile');g.fillStyle='#fffefa';g.fillRect(0,0,w,h);
  if(!grid) return;
  const row=Number(find('slice').value), data=grid.values, count=grid.w;
  const y=grid.y0-grid.step*row;
  find('sliceTitle').textContent='Y = '+y.toFixed(2)+' m';
  const zvals=[];
  for(let j=0;j<data.length;j+=3){if(data[j]!==-32768)zvals.push(data[j]*.001);if(data[j+1]!==-32768)zvals.push(data[j+1]*.001);}
  let lo=Math.min(-.06,zvals.reduce((a,b)=>Math.min(a,b),0)), hi=Math.max(.04,zvals.reduce((a,b)=>Math.max(a,b),0));
  const gap=(hi-lo)*.09;lo-=gap;hi+=gap;
  const left=47,right=w-16,top=16,bottom=h-30;
  const hx=j=>left+(right-left)*j/(count-1), vz=z=>bottom-(z-lo)/(hi-lo)*(bottom-top);
  g.font='11px sans-serif';g.textAlign='right';g.textBaseline='middle';
  const labels=[0, Math.round((hi-gap)*100)/100,Math.round((lo+gap)*100)/100];
  [...new Set(labels)].forEach(z=>{g.strokeStyle=z===0?'#9caf9f':'#e0e3d9';g.setLineDash(z===0?[4,4]:[]);g.beginPath();g.moveTo(left,vz(z));g.lineTo(right,vz(z));g.stroke();g.fillStyle='#516258';g.fillText(z.toFixed(2),left-5,vz(z));});
  g.setLineDash([]);g.textAlign='left';g.fillText('Z (m)',4,10);
  g.textAlign='center';for(const j of [0,Math.floor(count/2),count-1])g.fillText((grid.x0+j*grid.step).toFixed(2),hx(j),bottom+15);
  g.textAlign='right';g.fillText('X (m)',right,bottom+26);
  function line(field,color,dash,width){
    g.strokeStyle=color;g.lineWidth=width;g.setLineDash(dash);g.beginPath();let active=false;
    for(let j=0;j<count;j++){const z=data[(row*count+j)*3+field];if(z===-32768){active=false;continue;}
      const x=hx(j),yp=vz(z*.001);if(active)g.lineTo(x,yp);else{g.moveTo(x,yp);active=true;}}
    g.stroke();g.setLineDash([]);
  }
  line(1,'#b96f43',[5,3],2.7);line(0,'#216d5c',[],1.8);
}
async function refresh(){
  const token=++revision, a=catalog[asset], placement=a.sites[site], shot=placement[camera];
  const key=asset+'_'+camera;
  pressed('front',camera==='front');pressed('back',camera==='back');pressed('old',!after);pressed('new',after);
  find('ttl').textContent=a.label;find('frame').textContent='路段 '+site+'  /  '+(camera==='front'?'前视':'后视')+'  /  源帧 '+String(shot.frame).padStart(3,'0');
  find('before').src=shot.before;find('after').src=shot.after;find('before-a').href=shot.before;find('after-a').href=shot.after;
  find('capture').textContent='两张图取自相同像素范围。病害可见区域 '+shot.pixels.toLocaleString()+' 像素。';
  find('rayStats').textContent='同编号扫描命中本资产 '+shot.returns.toLocaleString()+' 束，其中旋转雷达 '+shot.by_sensor[0]+' 束，前向雷达 '+shot.by_sensor[1]+' 束。显示 '+shot.draw_changed+' 个病害点。';
  find('sourceLink').href=a.photo;find('plyLink').href=a.gaussians;find('meshLink').href=a.glb;
  find('depthLink').href=shot.metric;find('deltaLink').href='cases/'+placement.case+'.npz';find('heightLink').href='geometry/'+asset+'.npz';
  history.replaceState(null,'','#asset='+encodeURIComponent(asset)+'&site='+site+'&camera='+camera);
  cloud=null;drawCloud();
  try{
    await Promise.all([dataset('viewer/cloud/'+placement.case+'.js'),dataset('viewer/height/'+asset+'.js')]);
    if(token!==revision)return;
    cloud=shorts(window.RDcloud[placement.case][key]);
    if(lastHeight!==asset){
      grid=Object.assign({},window.RDheight[asset]);grid.values=shorts(grid.data);
      find('slice').max=String(grid.h-1);find('slice').value=String(grid.row);lastHeight=asset;
    }
    drawCloud();drawProfile();document.documentElement.dataset.ready=String(token);
  }catch(e){find('rayStats').textContent=e.message;}
}
find('category').value=catalog[asset].category;assetList(find('category').value,asset);find('site').value=site;
find('category').onchange=e=>{assetList(e.target.value,null);refresh();};
find('asset').onchange=e=>{asset=e.target.value;refresh();};find('site').onchange=e=>{site=e.target.value;refresh();};
for(const cam of ['front','back'])find(cam).onclick=()=>{camera=cam;refresh();};
function cloudMode(mode){after=mode;pressed('new',mode);pressed('old',!mode);drawCloud();}
find('old').onclick=()=>cloudMode(false);find('new').onclick=()=>cloudMode(true);find('slice').oninput=drawProfile;
const cv=find('cloud');
cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId);};
cv.onpointermove=e=>{if(!drag)return;azimuth+=(e.clientX-drag[0])*.009;elevation=Math.max(-.15,Math.min(1.45,elevation+(e.clientY-drag[1])*.009));drag=[e.clientX,e.clientY];drawCloud();};
cv.onpointerup=cv.onpointercancel=()=>{drag=null;};
cv.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')azimuth-=.1;if(e.key==='ArrowRight')azimuth+=.1;if(e.key==='ArrowUp')elevation=Math.min(1.45,elevation+.1);if(e.key==='ArrowDown')elevation=Math.max(-.15,elevation-.1);drawCloud();};
new ResizeObserver(()=>{drawCloud();drawProfile();}).observe(document.querySelector('main'));refresh();
