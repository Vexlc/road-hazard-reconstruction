"""Index local mesh, Gaussian and camera evidence by asset ID for the offline atlas."""
import json
import os
from tempfile import NamedTemporaryFile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "tools/asset_atlas"
GEN = ROOT / "gen2hole"
RELEASE = (GEN / "published").resolve()
SENSORS = RELEASE / "道路病害_传感器与几何增强"
EXPANSION = ROOT / "outputs/road_hazard_expansion"
FACTORY = ROOT / "outputs/道路病害_稳定生成与迁移_v3"
JOINT = ROOT / "outputs/road_hazard_joint_scenes"
CATEGORIES = {"stereo": "坑洞", "rgbd": "坑洞", "pothole": "坑洞", "missing": "缺盖井", "raised": "异常井盖"}
METHODS = {"stereo": "实拍双目", "rgbd": "实拍 RGB-D"}
TERMS = dict(round="圆形", oval="椭圆", square="方形", rectangular="长方形", bowled="碗状坑",
             cluster="群坑", crescent="月牙坑", deep_break="深坑", joined="连通坑", patch_break="补丁破损",
             slab_corner="板角破损", spall="剥落", waterlogged="积水坑", wheel_rut="车辙坑",
             ajar14="微开", broken_flange="破损井沿", flush="齐平井沿", masonry="砖砌井壁",
             propped43="支起", raised62="掀开", removed_side="移至井旁", settled="沉降井框",
             skewed="斜移", slid="平移", tilted29="倾斜", upright82="近直立", aged_asphalt="老化沥青",
             chip_seal="碎石封层", concrete="水泥路面", fine_asphalt="细粒沥青")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def link(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return Path(os.path.relpath(path, ATLAS)).as_posix()


def entry(row, base, method, label, preview):
    shape = "P95 凹深" if row["category"] in METHODS else "设计深度"
    depth = row["depth_p95_m"] if row["category"] in METHODS else row.get("depth_measured_m", row.get("depth_design_m"))
    if "depth_measured_m" in row:
        shape = "可见井底深度"
    return dict(id=row["id"], label=label, category=CATEGORIES[row["category"]], method=method,
                preview=link(preview), width=row["width_m"], length=row["length_m"], depth=depth,
                depthLabel=shape, gaussians=row["gaussians"],
                glb=link(base / row["glb"]), ply=link(base / row["ply"]),
                metadata=link(base / row["metadata"]), surface=link(base / row["surface"]), scenes=[])


def camera_views(root):
    shots = {}
    for path in sorted((root / "camera").glob("*/views.json")):
        for shot in read(path):
            if shot["available"]:
                shots.setdefault(shot["asset_id"], []).append(dict(
                    label=f"{shot['site']} · {'前视' if shot['camera']=='front_camera' else '后视'} · 帧 {shot['frame']}",
                    after=link(path.parent / shot["after_stem"]), before=link(path.parent / shot["before_stem"]),
                    site=shot["site"], camera=shot["camera"].split("_")[0]))
    return shots


def generated():
    shots = camera_views(SENSORS)
    rows = {}
    for spec in read(GEN / "catalog/assets.json").values():
        folder = RELEASE / spec["published_dir"]
        bundle = folder.parents[2]
        meta = read(folder / "metadata.json")
        name = meta["id"]
        assert name == spec["id"] and meta["source_sha256"] == spec["source_sha256"]
        paths = dict(glb="asset.glb", ply="asset.ply", metadata="metadata.json", surface="surface.npz")
        row = entry(meta | paths, folder, "生成 RGB", spec["label"], bundle / "views" / (name+"_inspection.jpg"))
        row.update(reference=link(bundle / meta["source_image"]), reverse=link(bundle / "views" / (name+"_reverse.jpg")),
                   triangles=spec["triangles"], scenes=shots[name], sensor=link(SENSORS / "传感器对照.html"))
        rows[name] = row
    return rows


def captured_and_designed():
    rows = {}
    for folder in (EXPANSION, FACTORY / "资产库"):
        for r in read(folder / "asset_manifest.json"):
            cat = r["category"]
            if cat in METHODS:
                label = ("双目坑洞 · " if cat=="stereo" else "深度相机坑洞 · ") + str(r["family"])
            else:
                label = " · ".join(TERMS[r[k]] for k in ("family","state","material") if r.get(k))
            row = entry(r, folder, METHODS.get(cat,"程序化"), label, folder / r["preview"])
            row["triangles"] = r.get("glb_triangles") or sum(r["geometry_parts"].values())
            if cat in METHODS:
                row["reference"] = link((folder / r["metadata"]).parent / "source_crop.png")
            if r["id"] in rows:
                raise ValueError(f"Duplicate ID: {r['id']}")
            rows[r["id"]] = row
    for folder in (JOINT, FACTORY / "迁入场景"):
        for path in sorted(folder.glob("*/scene.json")):
            scene = read(path)
            for p in scene["placements"]:
                row = rows[p["asset_id"]]
                view = dict(label=f"{scene['site']} · {path.parent.name} · 迁入近景")
                if folder == JOINT:
                    view["after"] = link(path.parent / p["preview_file"])
                else:
                    category = p["category"]
                    view.update(after=link(path.parent / (category+"_after.png")),
                                before=link(path.parent / (category+"_before.png")))
                row["scenes"].append(view)
    return rows


def real_rgb():
    bundle = ROOT / "runs/rgb-box-P90/series"
    if not bundle.exists():
        return {}
    r, = read(bundle / "assets/asset_manifest.json")
    name = r["id"]
    row = entry(r, bundle / "assets", "实拍 RGB", r["label"], bundle / "views" / (name+"_inspection.jpg"))
    row.update(triangles=sum(r["geometry_parts"].values()), reference=link(bundle / r["source_image"]),
               reverse=link(bundle / "views" / (name+"_reverse.jpg")), audit=link(ROOT / "validation/rgb-box-P90.json"))
    case = bundle / "scenes/034_90"
    if (case / "scene.json").is_file():
        assert read(case/"scene.json")["assets"] == [name]
        row["scenes"].append(dict(label="034 · 034_90 · 迁入近景", after=link(case/"pothole_after.png"),
                                  before=link(case/"pothole_before.png")))
        row["video"] = link(bundle / "sensor_views/034_90/front_after.mp4")
    sensor_root = bundle.parent / "sensors"
    if sensor_root.is_dir():
        row["scenes"] = camera_views(sensor_root)[name] + row["scenes"]
        row["scan"] = link(sensor_root / "checks/scan_audit.json")
    row["related"] = read(ROOT / "validation/rgb-box-P90.json")["source_asset"]
    return {name: row}


def add_perforated_views(row, run, profile, meta, family_audit, seated, visibility):
    key = meta["id"]
    audit = family_audit['states'][key]
    source = run / 'evaluation'
    row.update(
        panelSize=meta['panel_size_m'],
        lidThickness=meta['lid_thickness_m'],
        throughHoles=meta['through_holes'],
        wellVisible=audit['views'][0]['visible_well_fraction'],
        openFace=meta['open_face_fraction'],
        mask=link(run / 'series/references' / profile['mask']),
    )
    views = [
        ('实拍原图', run / 'series/references/frame.png', '原始参考图'),
        ('盖面正射图', run / 'series/references/lid_ortho.png', '盖面在正射坐标下的源纹理'),
    ]
    footprint = key if 'solid' in audit else seated
    if footprint:
        note = '黄线为标注，青线为 GLB 孔口投影。'
        if audit.get('fracture'):
            note += '此处显示断裂前的完整盖板。'
        elif not meta['through_holes']:
            note += '缺盖状态显示参考盖板。'
        views.append(('逐孔回投', source / footprint / 'openings.jpg', note))
    if seated and 'same_photo_side_check' in family_audit:
        views.extend(
            [
                (
                    '原机位侧壁',
                    source / seated / 'source-pose.jpg',
                    '原照机位的 GLB，金色为盖面，蓝绿色为可见侧壁',
                ),
                (
                    '原图下缘复核',
                    source / seated / 'side-fit.jpg',
                    '青线为 GLB 底边，黄点为同图留出的边缘标记',
                ),
            ]
        )
    if 'break_comparison' in audit:
        views.append(
            ('断口放大', source / key / 'broken-web.png', '左为完整，右为断条。金色为金属，深蓝为贯穿孔')
        )
    if 'section' in audit:
        views.extend([
            ('完整原盖与设计断线', source / seated / 'fracture-line.jpg', '青线为加在完整实拍盖面上的设计断线'),
            ('俯视与井内截面', source / key / 'fracture-section.png', '金色为余盖，红色为碎片，蓝色为井壁和井底。右图沿左图黄线剖开。'),
        ])
        row.update(preview=link(source / key / 'aligned.jpg'), cutaway=link(source / key / 'cutaway.jpg'))
        if (source / key / 'cutaway.glb').is_file():
            row['cutawayModel'] = link(source / key / 'cutaway.glb')
        fracture = audit.get('fracture')
        if fracture:
            row.update(retainedVolume=fracture['retained_metal_volume_fraction'],
                       fragmentBottom=fracture.get('fragment_bottom_z_m'))
    for angle in (0, 55):
        views.append(
            (
                f'射线视图 · {angle}°',
                source / key / f'rays-{angle}.png',
                '金色为盖面，棕色为孔壁，蓝绿色为井壁，深蓝为井底',
            )
        )
    row['sourceViews'] = [dict(label=t, image=link(p), note=caption) for t, p, caption in views]
    if visibility:
        for case, record in visibility['cases'].items():
            if record['asset'] != key:
                continue
            site = family_audit['placements'][case]['site']
            cams = record['cameras']
            frames = cams['front_camera']['frames']
            caption = (f"共 {frames} 帧。扣除前景后，前视有 {cams['front_camera']['frames_with_unobstructed_pixel']} 帧可见碎片，"
                       f"后视有 {cams['back_camera']['frames_with_unobstructed_pixel']} 帧可见。灰线为只投碎片，棕线含井体遮挡，绿线含场景前景遮挡。")
            row['scenes'].append(dict(label=f'{site} · 碎片逐帧可见性', after=link(source / key / (site+'-visibility.png')),note=caption))
            for cam, counts in cams.items():
                peak = counts['peak_frame']
                if peak is not None:
                    label = '前视' if cam == 'front_camera' else '后视'
                    row['scenes'].append(dict(label=f'{site} · {label}碎片投影 · 帧 {peak}',
                        after=link(source / key / (site+'-'+cam+'-visible-piece.png')),
                        note=f"橙色为该帧可见碎片的 {counts['peak_visible_pixels']:,} 个像素，背景为原始相机照片。"))
    for case, diff in family_audit.get('relative_to_missing_fragment', {}).items():
        placed = family_audit['placements'][case]
        if placed['asset'] != key:
            continue
        for camera, shot in diff['cameras'].items():
            title = '前视' if camera == 'front_camera' else '后视'
            row['scenes'].append(dict(label=f"{placed['site']} · {title} · 与断片缺失对照 · 帧 {shot['frame']}",
                after=link(source / key / shot['image']),
                note=f"左为断片缺失，右为{row['state']}。黄色标出部件或深度改变超过阈值的 {shot['changed_pixels']:,} 个像素"
                     f"（深度阈值 1 厘米）。同路段有 {diff['lidar']['range_change_over_1cm']:,} 束模拟回波距离差超过 1 厘米。"))
    for case, diff in family_audit.get('relative_to_seated', {}).items():
        placed = family_audit['placements'][case]
        if placed['asset'] != key:
            continue
        laser = diff['lidar']
        for camera, image in diff['cameras'].items():
            title = '前视' if camera == 'front_camera' else '后视'
            note = (
                f"左为落座，右为{row['state']}。黄色标出部件改变或光轴深度差超过 1 厘米的 "
                f"{image['changed_pixels']:,} 个像素。该路段 {placed['frames']} 帧中有 "
                f"{laser['range_change_over_1cm']:,} 个原始激光束的模拟回波距离差超过 1 厘米。"
            )
            row['scenes'].append(
                dict(
                    label=f"{placed['site']} · {title} · 与落座对照 · 帧 {image['frame']}",
                    after=link(source / key / image['image']),
                    note=note,
                )
            )
        if laser.get('plan_view'):
            distances = '、'.join(f"{1000 * h['delta_m']:.1f}" for h in laser['strongest_paired_hits'])
            row['scenes'].append(
                dict(
                    label=f"{placed['site']} · 断条回波落点俯视",
                    after=link(source / key / laser['plan_view']),
                    note=f"黄线为断条区域。青点为完整时的命中点，橙点为断条后的命中点。图中配对激光束的模拟距离差为 {distances} 毫米。",
                )
            )


def photo_families():
    rows = {}
    for info in read(ROOT / 'catalog/chamber-families.json'):
        run = ROOT / info['run']
        if not run.exists():
            continue
        bundle = run / 'series'
        spec = read(bundle / 'calibration/survey.json')
        state_ids = [s['id'] for s in spec['states']]
        perforated = 'perforations' in spec
        if perforated:
            perforation_audit = read(ROOT / info['audit'])
            visibility = read(ROOT / info['visibility']) if 'visibility' in info else None
            seated = next((s['id'] for s in spec['states'] if not s.get('absent') and
                           not s.get('angle_deg', 0) and not any(s.get('offset_m', [])) and not s.get('break_polygon_uv') and not s.get('fragment')), None)
        scenes = {}
        street_dir=run/'evaluation/streets'
        street_frames=read(street_dir/'frames.json') if (street_dir/'frames.json').exists() else {}
        driving={}
        for path in sorted((bundle / 'scenes').glob('*/scene.json')):
            for placed in read(path)['placements']:
                cat = placed['category']
                scenes.setdefault(placed['asset_id'],[]).append(dict(label=path.parent.name+" · 迁入近景",
                    after=link(path.parent/(cat+'_after.png')),before=link(path.parent/(cat+'_before.png'))))
                case=path.parent.name
                if case in street_frames:
                    frame=street_frames[case]
                    driving.setdefault(placed['asset_id'],[]).append(dict(
                        label=f"{frame['site']} · 完整前视 · 帧 {frame['frame']}",site=frame['site'],camera='front',
                        before=link(street_dir/(case+'-before.jpg')),after=link(street_dir/(case+'-after.jpg')),
                        video=link(bundle/'sensor_views'/case/'front_after.mp4')))
        sensor_root = run/'sensors'
        if (run/'sensors-multisite/checks/camera_audit.json').is_file():
            sensor_root = run/'sensors-multisite'
        shots = camera_views(sensor_root)
        paired = {}
        for path in sorted((bundle/'comparison').glob('*/comparison.json')):
            match = read(path)
            for key in match['states']:
                paired.setdefault(key,[]).append(dict(label=f"{match['site']} · 前视 · 同帧 {match['frame']}",
                    site=match['site'],camera='front',before=link(path.parent/'before.png'),after=link(path.parent/(key+'.png'))))
        capture_views=[]
        if 'reprojection' in info:
            repro=read(ROOT/info['reprojection'])
            capture_fit=read(bundle/'calibration/capture-fit.json')
            for frame,methods in repro['camera_mask_iou'].items():
                prefix,number=frame.rsplit('_',1)
                camera={'front_camera':'前视','back_camera':'后视'}.get(prefix,prefix)
                images={name:link(run/'reprojection'/(frame+'-'+name+'.png')) for name in ('photo','native','registered')}
                capture_views.append(dict(label=f"{camera} · 帧 {int(number)}",images=images,
                    native=methods['native_pose']['frame_edge'],registered=methods['locally_registered']['frame_edge'],
                    apronNative=methods['native_pose'].get('apron_edge'),apronRegistered=methods['locally_registered'].get('apron_edge')))
        for meta in read(bundle / 'assets/asset_manifest.json'):
            key = meta['id']
            if key in rows:
                raise ValueError(f"Asset ID collision: {key}")
            row = entry(meta, bundle/'assets', info['method'], meta['label'], bundle/'views'/(key+'_inspection.jpg'))
            row.update(family=spec['family'], states=state_ids, label=info.get('asset_label','圆井')+' · '+meta['label'], state=meta['label'],
                       reference=link(bundle/'references/frame.png'),
                       audit=link(ROOT / info['audit']), reverse=link(bundle/'views'/(key+'_reverse.jpg')),
                       credit=spec.get('source_url'), note=info['note'], triangles=sum(meta['geometry_parts'].values()),
                       scenes=driving.get(key,[])+paired.get(key,[])+shots.get(key,[])+scenes.get(key,[]))
            if perforated:
                row['category'] = info['asset_label']
                add_perforated_views(row, run, spec['perforations'], meta, perforation_audit, seated, visibility)
            else:
                row['lidReference']=link(bundle/'references/lid.png')
            if capture_views:
                row.update(reference=link(bundle/'references/capture.png'),lidReference=link(bundle/'references/frame.png'),
                           sourceLabels=['前视实拍裁切','两帧融合的正射纹理'],captureViews=capture_views,
                           fit=link(bundle/'calibration/capture-fit.json'),reprojection=link(ROOT/info['reprojection']),
                           captureAsset=repro['asset'],frameDiameter=capture_fit['frame_diameter_m'],
                           lidDiameter=capture_fit['lid_diameter_m'],lidDesigned=bool(capture_fit.get('lid_diameter_source')),
                           apronExtent=capture_fit.get('apron_xy_extent_m'),laserCount=repro['heldout_lid_returns'],
                           laserMedianMM=repro['lidar_abs_error_median_m']*1000)
            if not perforated and meta['lid_angle_degrees'] == 0 and meta['lid_offset_m'] == [0,0]:
                row['category']='落座参照'
            if (sensor_root/'checks/scan_audit.json').is_file():
                row['scan']=link(sensor_root/'checks/scan_audit.json')
            rows[key]=row
    return rows


def crack_examples():
    root = ROOT/'runs/photo-cracks'
    if not root.exists():
        return {}
    bundle = root/'series'
    provenance = read(GEN/'examples/photo-cracks/sources.json')
    shots = camera_views(root/'sensors')
    rows = {}
    for meta in read(bundle/'assets/asset_manifest.json'):
        key=meta['id']
        row=entry(meta,bundle/'assets',"实拍 RGB",meta['label'],bundle/'views'/(key+'_inspection.jpg'))
        row.update(category="裂缝",depthLabel="设计深度上限",triangles=sum(meta['geometry_parts'].values()),
                   note="实拍 RGB 和人工像素掩膜给出缝迹与分叉。裁切图按设计跨度映射到局部平面，沟槽剖面与深度由设计给定。",
                   reference=link(bundle/meta['source_image']),mask=link(bundle/'calibration'/(key+'_outline.png')),
                   reverse=link(bundle/'views'/(key+'_reverse.jpg')),credit=provenance['source'],scenes=shots.get(key,[]))
        for path in sorted((bundle/'scenes').glob('*/scene.json')):
            if key in read(path)['assets']:
                row['scenes'].append(dict(label=path.parent.name+' · 迁入近景',after=link(path.parent/'pothole_after.png'),
                    before=link(path.parent/'pothole_before.png')))
        rows[key]=row
    return rows


def pit_road_views(run):
    bundle=run/'series'
    shots=camera_views(run/'sensors')
    street_dir=run/'evaluation/streets'
    street_frames=read(street_dir/'frames.json') if (street_dir/'frames.json').exists() else {}
    for path in sorted((bundle/'scenes').glob('*/scene.json')):
        scene=read(path)
        for placement in scene['placements']:
            key=placement['asset_id']
            video=link(path.parent/'global_before_after.mp4')
            for shot in shots.get(key,[]):
                if shot.get('site')==scene['site']:
                    shot['video']=video
            shots.setdefault(key,[]).append(dict(label=path.parent.name+' · 迁入近景',video=video,
                after=link(path.parent/'pothole_after.png'),before=link(path.parent/'pothole_before.png')))
            case=scene['id']
            if case in street_frames:
                frame=street_frames[case]['frame']
                shots[key].insert(0,dict(label=f"{scene['site']} · 行车全景 · 帧 {frame}",video=video,
                    after=link(street_dir/(case+'-after.jpg')),before=link(street_dir/(case+'-before.jpg'))))
    return shots


def shared_profile_modes():
    run = ROOT / 'runs/rgb-parallax/evaluation'
    measurements = run / 'measurements.json'
    modes = {}
    if not measurements.is_file():
        return modes
    for key, case in read(measurements).items():
        for name, frame in case['frames'].items():
            result = frame.get('same_pixel_comparison')
            if not result:
                continue
            a, b = result['models']
            errors = result['median_abs_optical_error_mm']
            mode = dict(label='双照片与多帧双目 · 同像素剖面',
                        image=link(run / key / name / 'same-pixels.png'),
                        note=(f"黄线为照片标注，青线为 {a}，紫线为 {b}。在相同的 "
                              f"{result['common_interior_pixels']:,} 个坑内像素上，两模型的光轴深度绝对误差中位数"
                              f"分别为 {errors[a]:.1f} 和 {errors[b]:.1f} 毫米。"))
            for model in [a, b]:
                modes[model, name] = mode
    return modes


def multiview_potholes(shared):
    run=ROOT/'runs/multiview-pit-survey'
    if not run.is_dir():
        return {}
    bundle=run/'series'
    report=read(run/'evaluation/measurements.json')
    shots=pit_road_views(run)
    rows={}
    for meta in read(bundle/'assets/asset_manifest.json'):
        key=meta['id']
        case=report[key]
        directory=run/'evaluation'/key
        source=case['source']
        method='多帧 RGB-D' if source=='PothRGBD' else '多帧双目'
        row=entry(meta,bundle/'assets',method,meta['label'],bundle/'views'/(key+'_inspection.jpg'))
        inputs=[dict(label='拟合帧 '+k,image=link(directory/'input'/(k+'.jpg'))) for k in case['fit']]
        inputs.extend([dict(label='融合正射纹理',image=link(bundle/'references'/(key+'_ortho.png'))),
                       dict(label='实测覆盖次数',image=link(directory/'input/support.png'))])
        evidence=[]
        for frame,data in case['heldout'].items():
            dest=directory/frame
            stats=data['exported_glb']
            modes=[dict(label='融合 GLB 回投',image=link(dest/'contours.jpg'),iou=stats['mask_iou']),
                   dict(label='实拍原图',image=link(dest/'photo.jpg')),
                   dict(label='GLB 材质底色回投',image=link(dest/'base-color.jpg'),texture=True),
                   dict(label='光轴深度误差',image=link(dest/'errors.jpg'),error=True)]
            for k,single in data['single_fit_frame_controls'].items():
                modes.append(dict(label='仅用帧 '+k,image=link(dest/('single-'+k+'.jpg')),iou=single['mask_iou']))
            if (key, frame) in shared:
                modes.append(shared[key, frame])
            evidence.append(dict(frame=frame,modes=modes,median=stats['depth_abs_median_mm'],p95=stats['depth_abs_p95_mm'],
                                 samples=stats['visible_pit_depth_samples'],spacing=data['heldout_center_spacing_m'],
                                 colorError=data['material_reprojection'].get('raw_rgb_mae_0_255')))
        note='仅用坑外路面纹理和深度配准相机。坑内几何与颜色来自拟合帧，留出帧用于复核。外围缺少观测的部分用邻近路面纹理补齐。'
        appearance=read(bundle/'calibration'/(key+'_capture.json'))['photographic_texture']
        note+=f" 坑内 {appearance['observed_pit_fraction']:.1%} 的纹理像素可直接投影到有效照片，其余使用基础表面颜色。"
        if source=='PothRGBD':note+=' PothRGBD 的米制结果基于标称内参。'
        row.update(depth=meta['depth_p95_m'],depthLabel='实测 P95 凹深',triangles=sum(meta['geometry_parts'].values()),
                   reverse=link(bundle/'views'/(key+'_reverse.jpg')),reference=inputs[0]['image'],sourceViews=inputs,
                   pitViews=evidence,spacing=case['glb_center_spacing_m'],sourceDataset=source,note=note,
                   fit=link(bundle/'calibration'/(key+'_capture.json')),reprojection=link(run/'evaluation/measurements.json'),
                   credit='https://github.com/ymyurdakul/PData' if source=='PothRGBD' else 'https://github.com/ruirangerfan/stereo_pothole_datasets',
                   scenes=shots.get(key,[]))
        if (run/'sensors/checks/scan_audit.json').is_file():
            row['scan']=link(run/'sensors/checks/scan_audit.json')
        rows[key]=row
    return rows


def parallax_potholes(shared):
    run = ROOT / 'runs/rgb-parallax'
    if not run.is_dir():
        return {}
    bundle = run / 'series'
    specs = {s['id']: s for s in read(bundle / 'calibration/survey.json')}
    report = read(run / 'evaluation/measurements.json')
    scenes = pit_road_views(run)
    rows = {}
    for meta in read(bundle / 'assets/asset_manifest.json'):
        key = meta['id']
        case = report[key]
        spec = specs[key]
        dest = run / 'evaluation' / key
        row = entry(
            meta,
            bundle / 'assets',
            '双张实拍 RGB',
            meta['label'],
            bundle / 'views' / (key + '_inspection.jpg'),
        )
        inputs = [
            dict(
                label='主照片 · ' + case['fit'][0],
                image=link(bundle / 'references' / (key + '.png')),
                note='此照片提供网格表面的颜色',
            ),
            dict(
                label='第二张 · ' + case['fit'][1],
                image=link(bundle / 'references' / (key + '_second.png')),
                note='从另一机位拍摄同一处，灰度纹理用于双向匹配',
            ),
            dict(
                label='尺度参考',
                image=link(dest / 'scale.jpg'),
                note=f"黄色两点的路面距离为 {spec['ruler']['length_m']:.3f} 米",
            ),
            dict(
                label='双向匹配覆盖',
                image=link(dest / 'support.jpg'),
                note='绿色有双向纹理匹配，橙色为插值，灰色为周边路面',
            ),
            dict(
                label='正射纹理',
                image=link(bundle / 'references' / (key + '_ortho.png')),
                note='按恢复的曲面将主照片投到资产坐标系',
            ),
        ]
        evidence = []
        for name in case['unseen_rgb'] + case['fit']:
            data = case['frames'][name]
            directory = dest / name
            kind = {'new_view': '留出照片', 'texture_photo': '主照片', 'matching_photo': '第二张'}[
                data['role']
            ]
            modes = [
                dict(label='网格回投', image=link(directory / 'contours.jpg'), iou=data['mask_iou']),
                dict(label='实拍原图', image=link(directory / 'photo.jpg'), note=kind),
                dict(
                    label='米制剖面对照',
                    image=link(directory / 'section.png'),
                    note='黄线为剖面位置，绿线为模型，棕点为传感器深度。两者各以周边路面为零。',
                ),
                dict(
                    label='相对路面高度误差',
                    image=link(directory / 'relief-error.png'),
                    note='模型与传感器的相对高度绝对差，黑紫接近 0，橙黄接近 35 毫米。灰色为比较范围以外。',
                ),
                dict(label='光轴深度误差', image=link(directory / 'errors.jpg'), error=True),
                dict(
                    label='模型材质回投',
                    image=link(directory / 'material.jpg'),
                    texture=True,
                    note='坑内颜色取自 GLB 材质，周围为同机位照片',
                ),
            ]
            if (key, name) in shared:
                modes.append(shared[key, name])
            detail = (
                f"同机位共比较 {data['overlap_points']:,} 个坑内像素。相对路面高度的绝对误差中位数为 "
                f"{data['relief_abs_median_mm']:.1f} 毫米，同处平路对照为 {data['flat_relief_abs_median_mm']:.1f} 毫米。"
                f"光轴深度的绝对误差中位数为 {data['depth_abs_median_mm']:.1f} 毫米。"
            )
            evidence.append(
                dict(
                    frame=kind + ' · ' + name,
                    modes=modes,
                    description=detail,
                    colorError=data['material'].get('raw_rgb_mae_0_255'),
                )
            )
        note = (
            '两张不同机位的 RGB 照片提供几何与纹理。坑外平面和一段已知长度确定尺度，坑内按视差恢复，匹配不足处插值。'
            '示例参考长度取自两个坑外小区的传感器深度，坑内深度用于事后复核。'
        )
        if case['unseen_rgb']:
            note += f" 另有 {len(case['unseen_rgb'])} 张照片在建模完成后仅用坑外纹理定位。"
        row.update(
            category=spec['category_label'],
            depth=meta['depth_p95_m'],
            depthLabel='视差 P95 凹深',
            triangles=sum(meta['geometry_parts'].values()),
            note=note,
            depthPanelTitle='对照双照片模型、深度观测和米制剖面',
            reference=inputs[0]['image'],
            sourceViews=inputs,
            reverse=link(bundle / 'views' / (key + '_reverse.jpg')),
            matchCoverage=meta['rgb_supported_target_fraction'],
            pitViews=evidence,
            fit=link(bundle / 'calibration' / (key + '_capture.json')),
            reprojection=link(run / 'evaluation/measurements.json'),
            reprojectionLabel='独立深度核验',
            scenes=scenes.get(key, []),
            credit='https://github.com/ymyurdakul/PData'
            if spec['source'] == 'PothRGBD'
            else 'https://github.com/ruirangerfan/stereo_pothole_datasets',
        )
        if 'related' in spec:
            row.update(
                related=spec['related'],
                relatedLabel='同处含深度的重建',
                reverseRelatedLabel='同处的双照片重建',
            )
        if (run / 'sensors/checks/scan_audit.json').is_file():
            row['scan'] = link(run / 'sensors/checks/scan_audit.json')
        rows[key] = row
    return rows


def main():
    rows = {}
    shared = shared_profile_modes()
    for group in (generated(), captured_and_designed(), real_rgb(), photo_families(), crack_examples(), multiview_potholes(shared), parallax_potholes(shared)):
        if group.keys() & rows.keys():
            raise ValueError("Asset ID collision")
        rows.update(group)
    for row in rows.values():
        relief = ATLAS / "relief" / (row["id"] + ".js")
        if relief.is_file():
            row["relief"] = link(relief)
        orbit = ATLAS / "orbits" / (row["id"] + ".webp")
        if orbit.is_file():
            row["orbit"] = link(orbit)
        if row.get("related"):
            rows[row["related"]]["related"] = row["id"]
            if row.get("reverseRelatedLabel"):
                rows[row["related"]]["relatedLabel"] = row["reverseRelatedLabel"]
    featured = ["A127","P121","A128","A117","P124","A119","P122","P112","A116","M118","A104","A108","A93","P90","P102","A13","M11","P02_wet_wheel_track","rgbd_pothole_20250227_135438","A20",
                "model_raised_rectangular_removed_side_chip_seal","stereo_pothole_fan_1_03","P20",
                "M03_brick_lined","A11","model_pothole_cluster_chip_seal_01","M13","P13"]
    ordered = [rows[k] for k in dict.fromkeys(featured + sorted(rows)) if k in rows]
    # A local script works when opened directly with file://, including in offline browsers.
    data = json.dumps(ordered, ensure_ascii=False, separators=(",",":"), allow_nan=False)
    target = ATLAS / "catalog.js"
    with NamedTemporaryFile(mode="w", dir=ATLAS, encoding="utf-8", delete=False) as f:
        f.write("window.roadAssets="+data+";\n")
    os.replace(f.name,target)
    print(f"{len(rows)} assets, {sum(len(r['scenes']) for r in ordered)} views. {ATLAS / 'index.html'}")


if __name__ == "__main__":
    main()
