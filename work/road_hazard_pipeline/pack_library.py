from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs';LIB=OUT/'road_hazards'
def main():
    manifest=json.loads((LIB/'asset_manifest.json').read_text());packed=[]
    for a in manifest:
        p=dict(a);name=a['id'];p['gaussian_ply']=name+'/asset.ply';p['preview']=name+'/preview.png';p['geometry_metadata']=name+'/geometry.json';p.pop('gaussian_parameters',None);p.pop('scene_directory',None);packed.append(p)
    readme='''# 道路病害 3DGS 资产库

共18个独立高斯资产：9种结构/状态 × 沥青、水泥两种路面。

每个ID目录含 asset.ply、preview.png、metadata.json、geometry.json。PLY采用米制、Z-up、1阶SH；包含位置、opacity logit、log scale、WXYZ四元数与SH颜色。请使用支持Gaussian Splatting的查看器，配置sh_degree=1。

asset_manifest.json里的文件路径均相对于本压缩包根目录。场景视频、动态检查点、原始网格和重复的PTH参数文件未装入本包。

几何为道路病害建模，结合CC0来源材质及高度，不是18处现场病害的测量重建。尺寸与深度是建模设定。

素材来源：https://ambientcg.com/view?id=Asphalt031 、Concrete034、Concrete049、ManholeCover009；均为CC0。衍生资产按CC0-1.0提供：https://creativecommons.org/publicdomain/zero/1.0/
'''
    dest=OUT/'道路病害_18种高斯资产.zip';tmp=dest.with_suffix('.zip.part')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        z.writestr('README.md',readme);z.writestr('asset_manifest.json',json.dumps(packed,ensure_ascii=False,indent=2));z.write(OUT/'道路病害_18种高斯资产总览.jpg','preview.jpg')
        for a in manifest:
            name=a['id'];d=LIB/name;g=d/'gaussians'
            for file in ['asset.ply','metadata.json','preview.png']:z.write(g/file,name+'/'+file)
            z.write(d/'metadata.json',name+'/geometry.json')
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None;names=z.namelist();assert len(names)==len(set(names))
    tmp.replace(dest)
    h=OUT/'reproducibility/road_hazards/sha256.json';hashes=json.loads(h.read_text()) if h.exists() else {}
    with dest.open('rb') as f:hashes[str(dest.relative_to(OUT))]=hashlib.file_digest(f,'sha256').hexdigest()
    h.write_text(json.dumps(hashes,indent=2));print('PACKED',dest.stat().st_size,len(manifest),flush=True)
if __name__=='__main__':main()
