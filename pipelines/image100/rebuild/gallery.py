from collections import defaultdict
from pathlib import Path
import json,sys,html
from PIL import Image,ImageOps,ImageFont,ImageDraw
root=Path(__file__).resolve().parents[1]
assets=sorted(json.loads((root/'assets/asset_manifest.json').read_text()),key=lambda a:('PMA'.index(a['id'][0]),a['id']))
assert len(assets)==100
groups=json.loads((root/'scenes/placement_plan.json').read_text())['groups'];by_id={k:g['number'] for g in groups for k in g['assets']}
sites=['034','109','110']
rows=[]
for a in assets:
    k=a['id'];c=a['category'];row={n:a[n] for n in ['id','label','category','glb','ply','surface','width_m','length_m','depth_design_m','state','gaussians']}
    row['angle']=round(a['lid_angle_degrees']) if 'lid_angle_degrees' in a else None
    row['case_number']=by_id[k]
    rows.append(row)
css="""
:root{color-scheme:light;--paper:#f2f1eb;--ink:#222a27;--muted:#59675f;--line:#c8d0c5;--green:#234b3d}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}main{max-width:1560px;margin:auto;padding:36px clamp(18px,4vw,64px) 56px}header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;padding:14px 0 26px;border-bottom:2px solid var(--ink)}h1{font-size:clamp(30px,4vw,56px);line-height:1.14;margin:0 0 12px;letter-spacing:-.04em}p{margin:0;line-height:1.65}.intro{color:var(--muted);font-size:15px}a{color:var(--green);text-decoration:underline;text-underline-offset:3px}a:hover{color:#92552d}a:focus-visible,button:focus-visible,input:focus-visible{outline:3px solid #e1ad49;outline-offset:2px}header nav{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:11px 18px;font-size:14px}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px 22px;padding:21px 0 18px}.cluster{display:flex;gap:5px;flex-wrap:wrap}.controls button{border:1px solid var(--line);padding:8px 12px;background:transparent;color:var(--ink);font:inherit;font-size:14px;border-radius:4px;cursor:pointer}.controls button[aria-pressed=true]{color:#fff;background:var(--green);border-color:var(--green)}input{min-width:140px;width:165px;border:1px solid var(--line);padding:9px 10px;font:inherit;background:#fafbf7;border-radius:4px}.views{margin-left:auto}.hint{font-size:13px;color:var(--muted);margin-bottom:15px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,275px),1fr));gap:18px 15px}article{background:#fbfbf7;border:1px solid var(--line);border-radius:5px;overflow:hidden}article>img{display:block;object-fit:contain;width:100%;height:auto;aspect-ratio:840/568;background:#dce0d8}section.info{padding:14px 16px 16px}.name{display:flex;align-items:baseline;gap:9px}.id{font:600 13px ui-monospace,monospace;color:#4d6d59}h2{font-size:18px;line-height:1.35;margin:0}.metric{font-size:13px;color:var(--muted);margin:7px 0 10px;font-variant-numeric:tabular-nums}.downloads{display:flex;gap:15px;align-items:center;font-size:13px}.downloads a:first-child{font-weight:700}details{font-size:13px;border-top:1px solid var(--line);margin-top:12px;padding-top:9px}summary{cursor:pointer;color:var(--green)}details p{margin-top:6px}details a{margin-left:10px}#none{padding:34px 0}footer{margin:28px 0 0;font-size:13px;color:var(--muted);line-height:1.75}.blind{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:780px){header{display:block}header nav{justify-content:flex-start;margin-top:15px}.views{margin:0}main{padding-top:18px}.controls{gap:10px}input{width:100%}}"""
template="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>路面病害三维资产 · 新增 100 件</title><style>CSS_HERE</style>
<main><header><div><h1>路面病害三维资产</h1><p class="intro">新增 100 件，共 34 件坑洞、33 件缺盖井和 33 件异常井盖。可按编号或名称检索。</p></div><nav aria-label="文件"><a href="三维资产总览.jpg">百件总览</a><a href="assets/asset_manifest.json">模型清单</a><a href="使用说明.txt">使用说明</a>SCENE_NAV</nav></header>
<div class="controls"><div class="cluster" id="cats" aria-label="筛选类别"></div><label><span class="blind">查找资产</span><input id="query" type="search" placeholder="查找编号或名称"></label><div class="cluster views" id="views" aria-label="显示视角"></div></div>
<p role="status" class="hint" id="count"></p><div class="grid" id="grid"></div><p id="none" hidden>没有匹配的资产。可更换关键字或类别。</p>
<footer>GLB 使用米和 Y 向上。3DGS PLY 使用米和 Z 向上。开口尺寸、井深和隐藏结构采用建模设定。</footer></main><script>
const rows=DATA_HERE,hasScenes=SCENE_FLAG;
let cat='all',view='inspection',needle='';
const g=document.querySelector('#grid'), types={all:'全部',pothole:'坑洞',missing:'缺盖井',raised:'异常井盖'},cams={inspection:'网格斜视',reverse:'反向斜视',source:'生成参考图'};
function link(path,text){const a=document.createElement('a');a.href=path;a.textContent=text;return a;}
function buttons(id,options,current,select){
 const node=document.querySelector(id);node.replaceChildren();
 for(const [key,label] of Object.entries(options)){const b=document.createElement('button');b.type='button';b.textContent=label;b.setAttribute('aria-pressed',key===current);b.onclick=()=>select(key);node.append(b);}
}
function draw(){
 buttons('#cats',types,cat,v=>{cat=v;draw()});buttons('#views',cams,view,v=>{view=v;draw()});
 const shown=rows.filter(r=>(cat==='all'||r.category===cat)&&((r.id+' '+r.label).toLowerCase().includes(needle)));
 g.replaceChildren();document.querySelector('#none').hidden=shown.length>0;document.querySelector('#count').textContent=shown.length+' 件资产';
 const group=document.createDocumentFragment();
 for(const [i,r] of shown.entries()){const card=document.createElement('article'),pic=document.createElement('img');pic.src=view==='source'?'references/'+r.id+'.png':'views/'+r.id+'_'+view+'.jpg';pic.alt=r.id+' '+r.label+' · '+cams[view];pic.loading=i<8?'eager':'lazy';card.append(pic);
 const info=document.createElement('section');info.className='info';let dims='净开口 ';if(r.category==='pothole')dims='坑口 ';
 dims+=r.width_m.toFixed(2)+' × '+r.length_m.toFixed(2)+' m · '+(r.category==='pothole'?'设计坑深 ':'设计井深 ')+r.depth_design_m.toFixed(2)+' m';if(r.angle>0)dims+=' · '+r.angle+'°';
 const title=document.createElement('div');title.className='name';const code=document.createElement('span');code.className='id';code.textContent=r.id;const h=document.createElement('h2');h.textContent=r.label;title.append(code,h);
 const dim=document.createElement('p');dim.className='metric';dim.textContent=dims;
 const files=document.createElement('div');files.className='downloads';files.append(link('assets/'+r.glb,'GLB 网格'),link('assets/'+r.ply,'3DGS PLY'),link('assets/'+r.surface,'路面切口'));info.append(title,dim,files);
 if(hasScenes){const more=document.createElement('details'),sum=document.createElement('summary');sum.textContent='查看三个路段的迁入结果';more.append(sum);
 for(const s of ['034','109','110']){const key=s+'_'+String(r.case_number).padStart(2,'0');const line=document.createElement('p');line.append('路段 '+s,link('sensor_views/'+key+'/front_after.mp4','车载视频'),link('sensor_views/'+key+'/classes.npz','掩膜'),link('scenes/'+key+'/render_config.yaml','场景配置'));more.append(line)}info.append(more)}
 card.append(info);group.append(card)}g.append(group)
}
document.querySelector('#query').addEventListener('input',e=>{needle=e.target.value.trim().toLowerCase();draw()});draw();
</script></html>"""
def gallery(has_scenes):
    nav='<a href="车载迁入演示.mp4">车载样例</a><a href="场景映射.json">场景映射</a>' if has_scenes else ''
    return template.replace('CSS_HERE',css).replace('SCENE_NAV',nav).replace('DATA_HERE',json.dumps(rows,ensure_ascii=False)).replace('SCENE_FLAG','true' if has_scenes else 'false')
(root/'资产浏览.html').write_text(gallery(True))
(root/'asset_only.html').write_text(gallery(False))
font=lambda n,b=False:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-'+('Bold' if b else 'Regular')+'.ttc',n)
mono=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',n)
def board(keys,cols,cw,title,name):
    a={r['id']:r for r in assets};m,g=32,9;h=round(cw*568/840);cap=40 if cw<300 else 52;top=107;nr=(len(keys)+cols-1)//cols
    out=Image.new('RGB',(2*m+cw*cols+g*(cols-1),top+(h+cap+g)*nr+17),'#F1F2EC');d=ImageDraw.Draw(out)
    d.text((m,15),title,font=font(39,True),fill='#202a25',anchor='lt')
    d.text((m,70),'34 件坑洞    /    33 件缺盖井    /    33 件异常井盖',font=font(19),fill='#486151',anchor='lt')
    for i,k in enumerate(keys):
        x=m+(i%cols)*(cw+g);y=top+(i//cols)*(h+cap+g)
        im=Image.open(root/'views'/(k+'_inspection.jpg')).resize((cw,h),Image.Resampling.LANCZOS);out.paste(im,(x,y))
        d.text((x+1,y+h+9),k,font=mono(17 if cw<300 else 21),fill='#496955',anchor='lt')
        d.text((x+(43 if cw<300 else 55),y+h+7),a[k]['label'],font=font(17 if cw<300 else 23,True),fill='#222a26',anchor='lt')
    out.save(root/name,quality=93,subsampling=0)
board([a['id'] for a in assets],10,215,'路面病害三维资产 · 新增 100 件','三维资产总览.jpg')
board(['P04','P23','P30','M16','A13','A17'],3,390,'三维网格 · 斜视选览','三维资产选览.jpg')
