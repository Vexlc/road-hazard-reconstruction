from pathlib import Path
import json, shutil, hashlib, subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)
def label(im,text,xy,size=24):ImageDraw.Draw(im).text(xy,text,font=font(size),fill='white')
def main():
    preview=Image.new('RGB',(1600,690),'#151b27')
    for col,name in enumerate(['manhole_scan_010','pothole_scan_road2']):
        d=OUT/'road_assets'/name;im=Image.open(d/'preview.png').resize((800,629));preview.paste(im,(col*800,52));meta=json.loads((d/'metadata.json').read_text())
        text='Manhole | 1.00 m | 884k triangles' if col==0 else 'Pothole | 1.26 x 0.81 m | 696k triangles'
        label(preview,text,(col*800+16,12),23)
    preview.save(OUT/'资产重建预览.jpg',quality=95)
    compare=Image.new('RGB',(1920,1160),'#151b27')
    detail=Image.new('RGB',(1600,740),'#151b27')
    for row,(lab,sid) in enumerate([('A','056'),('B','055')]):
        d=OUT/f'inserted_{lab}_{sid}';im=Image.open(d/'comparison_040.jpg');compare.paste(im,(0,row*580+40))
        label(compare,f'{lab} / PandaSet {sid} | Before',(16,row*580+7),25);label(compare,'After | fixed 3D scan assets',(976,row*580+7),25)
        after=im.crop((960,0,1920,540));crop=after.crop((315,395,660,525)).resize((1380,520))
        # Letterbox two inspection crops. This is an enlargement of recorded pixels.
        crop.thumbnail((1580,320));detail.paste(crop,((1600-crop.width)//2,row*370+45));label(detail,f'{lab} / {sid} | Asset area enlarged from the 960 x 540 render',(16,row*370+8),24)
    compare.save(OUT/'资产插入_AB对比.jpg',quality=96);detail.save(OUT/'资产插入细节.jpg',quality=96)
    writer=imageio.get_writer(str(OUT/'资产插入_AB对比.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2)
    for frame in range(80):
        canvas=Image.new('RGB',(1920,1160),'#151b27')
        for row,(lab,sid) in enumerate([('A','056'),('B','055')]):
            before=Image.open(OUT/f'scene_{lab}_{sid}'/'frames'/f'{frame:03d}.jpg');after=Image.open(OUT/f'inserted_{lab}_{sid}'/'frames'/f'{frame:03d}.jpg')
            canvas.paste(before,(0,row*580+40));canvas.paste(after,(960,row*580+40))
            label(canvas,f'{lab} / {sid} | Before',(16,row*580+7),25);label(canvas,'After | manhole + pothole',(976,row*580+7),25)
        writer.append_data(np.asarray(canvas))
    writer.close()
    repro=OUT/'reproducibility/asset_pipeline';repro.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,repro/p.name)
    shutil.copy2(ROOT/'work/street_crafter/insert_road_assets.py',repro/'insert_road_assets.py')
    for name in ['manhole_scan_010','pothole_scan_road2']:
        d=OUT/'road_assets'/name
        (d/'LICENSE.txt').write_text('CC0-1.0\n\nSource and author are recorded in metadata.json.\nhttps://creativecommons.org/publicdomain/zero/1.0/\n')
        (d/'README.md').write_text('GLB uses metres and standard glTF Y-up axes. Blend and mesh.npz use Z-up.\n\nHigh, medium, low files share the same 4K source material and differ in geometry density. All GLBs embed textures; Blend textures are packed.\n\nThe geometry is reconstructed from the published scan height field. Vertical scale is an authoring choice, not a surveyed measurement. See metadata.json.\n')
    shutil.copy2(ROOT/'work/asset_sources/ManholeCover010_4K-PNG/ManholeCover010_4K-PNG_Displacement.png',OUT/'road_assets/manhole_scan_010/source_height16.png')
    shutil.copy2(ROOT/'work/asset_sources/road_damaged_2_disp_4k.exr',OUT/'road_assets/pothole_scan_road2/source_height_full.exr')
    validation={'videos':{},'assets':{},'placements':{}}
    for lab,sid in [('A','056'),('B','055')]:
        d=OUT/f'inserted_{lab}_{sid}';m=json.loads((d/'placement.json').read_text());validation['placements'][lab]=[{k:p[k] for k in ['asset','local_road_fit']} for p in m['placements']]
        for filename,count in [('inserted.mp4',80),('before_after.mp4',80),('parallax_0.6m.mp4',48)]:
            p=d/filename
            reader=imageio.get_reader(str(p))
            metadata=reader.get_meta_data();actual=reader.count_frames();reader.close()
            r={'width':metadata['size'][0],'height':metadata['size'][1],'nb_frames':actual,'fps':metadata['fps'],'duration':metadata['duration']}
            assert actual==count,(p,r)
            validation['videos'][str(p.relative_to(OUT))]=r
    for d in (OUT/'road_assets').iterdir():
        if not d.is_dir():continue
        a=np.load(d/'mesh.npz');assert all(np.isfinite(a[k]).all() for k in a.files)
        assert a['faces'].min()>=0 and a['faces'].max()<len(a['vertices'])
        assert np.min(a['uv'])>=0 and np.max(a['uv'])<=1
        validation['assets'][d.name]={'medium_vertices':len(a['vertices']),'medium_triangles':len(a['faces']),'finite':True,'uv_in_range':True}
    (repro/'validation.json').write_text(json.dumps(validation,indent=2))
    freeze=subprocess.check_output([str(ROOT/'work/envs/assets/bin/python'),'-m','pip','freeze'],text=True);(repro/'asset_environment.txt').write_text(freeze)
    files=[p for p in (OUT/'road_assets').rglob('*') if p.is_file()]+[p for lab,sid in [('A','056'),('B','055')] for p in (OUT/f'inserted_{lab}_{sid}').glob('*') if p.is_file()]
    hashes={str(p.relative_to(OUT)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files}
    (repro/'sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(validation,indent=2),flush=True)

if __name__=='__main__':main()
