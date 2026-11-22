"""Publish the source trace, textured solids, cutaways and matched street observations of a split cover."""

import argparse
import importlib.util
from pathlib import Path

from road_camera_frames import camera_frames, read


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


Plate = load_script('render-pit-survey').Plate
family = load_script('render-chamber-family')


def render(run, report, visibility, site, asset_out, road_out):
    bundle = run / 'series'
    ev = run / 'evaluation'
    survey = read(bundle / 'calibration/survey.json')
    states = survey['states']
    keys = [s['id'] for s in states]
    groups = {s.get('fragment', {}).get('location', 'intact'): s['id'] for s in states}
    intact, missing, fallen, road = [groups[k] for k in ('intact', 'missing', 'floor', 'road')]
    check = read(report)
    audited = check['states']
    broken = audited[fallen]['fracture']
    split = broken['retained_metal_volume_fraction']
    directory, front = family.shared_frame(bundle, site, keys)
    processed = Path(read(family.ROOT / 'gen2hole/config/local.json')['pandaset_processed'])
    streets = camera_frames(run, processed)
    cols = [27, 394, 761, 1128]
    width = 347
    plate = Plate(1502, 1315)
    plate.text(28, 20, '同一块铸铁井盖，断片去了哪里', 42, True)
    plate.text(28, 82, '完整盖面的花纹、相对起伏和 14 个孔来自摄影测量。断线和碎片位置按设计给定。', 22)
    for col, title, path in zip(
        cols,
        ['完整井盖原照', '青线为设计断线', '实拍孔缘与模型回投', '碎片落井后的实际截面'],
        [
            bundle / 'references/frame.png',
            ev / intact / 'fracture-line.jpg',
            ev / intact / 'openings.jpg',
            ev / fallen / 'fracture-section.png',
        ],
    ):
        plate.text(col, 143, title, 20, True)
        plate.picture(path, col, 177, width, 174)
    plate.text(
        28,
        383,
        f'盖径按 {survey["perforations"]["size_m"][0]:.2f} 米赋值。断片占原盖金属体积的 {1 - split:.1%}，余盖由原座圈承托。',
        23,
    )
    for col, state in zip(cols, states):
        key = state['id']
        plate.text(col, 460, key + '   ' + state['label'], 23, True)
        plate.picture(ev / key / 'aligned.jpg', col, 503, width, 224)
    plate.text(28, 761, '剖开近侧井壁，保留完整盖面、碎片和井底', 26, True)
    for col, key in zip(cols, keys):
        plate.picture(ev / key / 'cutaway.jpg', col, 807, width, 244)
    plate.text(28, 1080, f'PandaSet {site} · 同一世界位置、同一前视相机，第 {front["frame"]} 帧', 23, True)
    for col, key in zip(cols, keys):
        plate.picture(directory / (key + '.png'), col, 1121, width, 139)
    plate.text(
        27,
        1282,
        'ambientCG · Manhole Cover 009 / Asphalt 031 / Concrete 049 · CC0     |     PandaSet · Scale AI and Hesai',
        15,
    )
    plate.save(asset_out)

    placement = check['placements']
    diffs = check['relative_to_missing_fragment']
    cases = {v['asset']: k for k, v in placement.items() if v['site'] == site}
    streets_for_piece = sorted((k, v) for k, v in streets.items() if placement[k]['asset'] == fallen)
    if len(streets_for_piece) < 2:
        raise ValueError('Expected the fallen fragment in two sites')
    page = Plate(1480, 1340)
    page.text(26, 20, '自驾相机能看见哪一块断片', 43, True)
    page.text(
        27,
        81,
        '同一井口、同一机位对照。左图断片缺失，右图有碎片。黄色标出可见部件改变或光轴深度差超过 1 厘米的像素。',
        20,
    )
    for x, (case, view) in zip([27, 758], streets_for_piece):
        page.text(x, 142, f'碎片在井底 · 路段 {view["site"]} · 前视第 {view["frame"]} 帧', 22, True)
        page.picture(ev / 'streets' / (case + '-after.jpg'), x, 180, 696, 359)
    for j, (key, label) in enumerate([(fallen, '碎片在井底'), (road, '碎片在井旁')]):
        y = 583 + j * 318
        pair = diffs[cases[key]]
        page.text(27, y, label + '  /  ' + site, 25, True)
        for x, cam in [(27, 'front_camera'), (502, 'back_camera')]:
            v = pair['cameras'][cam]
            text = (
                '前视' if cam == 'front_camera' else '后视'
            ) + f'第 {v["frame"]} 帧 · {v["changed_pixels"]:,} 个像素改变'
            page.text(x, y + 53, text, 18)
            page.picture(ev / key / v['image'], x, y + 87, 452, 154)
        vis = read(visibility)['cases'][cases[key]]['cameras']
        page.text(980, y + 53, '原始 80 帧中可见碎片的帧数', 19, True)
        for i, (cam, word) in enumerate([('front_camera', '前视'), ('back_camera', '后视')]):
            v = vis[cam]
            page.text(
                980,
                y + 96 + 39 * i,
                f'{word}   {v["frames_with_unobstructed_pixel"]} / {v["frames"]} 帧　最多 {v["peak_visible_pixels"]:,} 像素',
                19,
            )
        page.text(
            27,
            y + 261,
            f'同一批原始激光束上，{pair["lidar"]["range_change_over_1cm"]:,} 束的模拟距离差超过 1 厘米。',
            19,
        )
    page.text(27, 1260, '可见帧数按导出三角网格逐像素求交，扣除井壁、余盖、车辆和已知场景前景的遮挡。', 20)
    page.text(
        27,
        1308,
        'PandaSet · Scale AI and Hesai · 相机与激光雷达标定、原始照片和扫描     |     资产 A126 / A127 / A128',
        15,
    )
    page.save(road_out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--visibility', required=True, type=Path)
    p.add_argument('--site', default='109')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--road-out', required=True, type=Path)
    a = p.parse_args()
    render(a.run, a.report, a.visibility, a.site, a.out, a.road_out)
