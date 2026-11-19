from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def label(im,text,xy):ImageDraw.Draw(im).text(xy,text,font=ImageFont.truetype(FONT,24),fill='white')
def main():
    repro=OUT/'reproducibility/gaussian_insertion';repro.mkdir(exist_ok=True)
    for name in ['gaussian_assets.py','gaussian_road.py','finalize_gaussians.py']:shutil.copy2(Path(__file__).parent/name,repro/name)
    for name in ['insert_gaussian_assets.py','check_gaussian_occlusion.py']:shutil.copy2(ROOT/'work/street_crafter'/name,repro/name)
    validation={'assets':{},'scenes':{}}
    preview=Image.new('RGB',(1408,600),'#151b27')
    for i,name in enumerate(['manhole_scan_010','pothole_scan_road2']):
        d=OUT/'gaussian_assets'/name;meta=json.loads((d/'metadata.json').read_text());source=json.loads((OUT/'road_assets'/name/'metadata.json').read_text())
        meta.update(license='CC0-1.0',source_url=source['source'],author=source['author']);(d/'metadata.json').write_text(json.dumps(meta,indent=2))
        ply=PlyData.read(d/'asset.ply')['vertex'];names=ply.data.dtype.names
        assert all(np.isfinite(ply[k]).all() for k in names)
        assert {'opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','f_dc_0','f_rest_8'}.issubset(names)
        q=np.stack([ply[f'rot_{k}'] for k in range(4)],1);err=float(np.max(abs(np.linalg.norm(q,axis=1)-1)));assert err<1e-5
        xyz=np.stack([ply[k] for k in ['x','y','z']],1);extent=np.ptp(xyz,axis=0)
        assert extent[2]>.04,extent
        validation['assets'][name]={'gaussians':len(ply),'finite':True,'quaternion_norm_max_error':err,'extent_m':extent.tolist(),'representation_fields':list(names)}
        preview.paste(Image.open(d/'preview.png'),(i*704,48));label(preview,('Manhole' if i==0 else 'Pothole')+f" | {len(ply):,} Gaussians",(i*704+12,12))
        (d/'README.md').write_text('Actual anisotropic 3D Gaussian asset, not an RGB point cloud.\n\nasset.ply contains position, opacity logits, logarithmic scales, WXYZ quaternion, SH DC and 9 SH rest properties (degree 1). Units: metres; local Z-up. Use a 3DGS-aware viewer or load with SH degree 1. asset.pth stores the same parameters.\n\nConverted from the scan height field and PBR maps using surface-oriented sampling and analytic appearance baking; this was not gradient-trained from original photographs. Vertical metric scale follows the existing authored asset.\n\nScene-specific world transforms and appearance are saved in gs_inserted_*/<asset>_world.ply and in the merged dynamic checkpoint. Source and CC0 licence are in metadata.json.\n')
    preview.save(OUT/'纯3DGS资产预览.jpg',quality=96)
    comparison=Image.new('RGB',(1920,1160),'#151b27')
    for row,(lab,sid) in enumerate([('A','056'),('B','055')]):
        d=OUT/f'gs_inserted_{lab}_{sid}';s=json.loads((d/'summary.json').read_text());assert s['validation']['checkpoint_reload_max_rgb_error']<1e-6
        checks={}
        for name,count in [('pure_3dgs.mp4',80),('hybrid_vs_3dgs.mp4',80),('parallax_0.6m.mp4',48)]:
            reader=imageio.get_reader(d/name);metadata=reader.get_meta_data();n=reader.count_frames();reader.close();assert n==count
            checks[name]={'frames':n,'fps':metadata['fps'],'size':metadata['size']}
        validation['scenes'][lab]={'gaussians_frame040':s['gaussians_frame040'],'checkpoint_reload_max_rgb_error':s['validation']['checkpoint_reload_max_rgb_error'],'videos':checks,'occlusion':json.loads((d/'occlusion_check.json').read_text())}
        for col,folder in enumerate([f'scene_{lab}_{sid}',f'gs_inserted_{lab}_{sid}']):comparison.paste(Image.open(OUT/folder/'frames/040.jpg'),(col*960,row*580+40))
        label(comparison,f'{lab} / {sid} | Original Gaussian scene',(16,row*580+7));label(comparison,'Merged 3DGS | one joint Gaussian pass',(976,row*580+7))
    comparison.save(OUT/'纯3DGS插入_AB对比.jpg',quality=96)
    writer=imageio.get_writer(str(OUT/'纯3DGS插入_AB对比.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=2)
    for frame in range(80):
        canvas=Image.new('RGB',(1920,1160),'#151b27')
        for row,(lab,sid) in enumerate([('A','056'),('B','055')]):
            for col,folder in enumerate([f'scene_{lab}_{sid}',f'gs_inserted_{lab}_{sid}']):canvas.paste(Image.open(OUT/folder/'frames'/f'{frame:03d}.jpg'),(col*960,row*580+40))
            label(canvas,f'{lab} / {sid} | Original Gaussian scene',(16,row*580+7));label(canvas,'Merged 3DGS | one joint Gaussian pass',(976,row*580+7))
        writer.append_data(np.asarray(canvas))
    writer.close()
    (repro/'validation.json').write_text(json.dumps(validation,indent=2))
    files=[p for p in (OUT/'gaussian_assets').rglob('*') if p.is_file()]
    for lab,sid in [('A','056'),('B','055')]:files.extend(p for p in (OUT/f'gs_inserted_{lab}_{sid}').rglob('*') if p.is_file() and 'frames' not in p.parts)
    hashes={str(p.relative_to(OUT)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files};(repro/'sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
