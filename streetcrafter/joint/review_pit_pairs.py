"""Compare reloaded pair checkpoints against the original KITTI-360 checkpoint at matched exposures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from contextlib import contextmanager

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare import load_rig

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/streetgs-joint/kitti360-0000/30000"
DEST = ROOT / "outputs/streetgs-joint/kitti360-0000/30000/pit-pairs"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONTS = {size: ImageFont.truetype(FONT, size, index=2) for size in (14,16,18,20,22,26)}
HD_SIZE, BOARD = (1280, 342), (1280, 780)
PANEL = (612, 141)
INK = "#1e2a2c"
ACCENTS = ("#ffe0a2", "#9ee5df")


@contextmanager
def video(path, size):
    stage = path.with_suffix(".stage.mp4")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size[0]}x{size[1]}", "-r", "10", "-i", "-", "-an", "-c:v", "libx264",
           "-threads", "3", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(stage)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        try:
            yield proc.stdin
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"Video encoding failed: {path}")
    stage.replace(path)


def check_clip(path, frames, size):
    cap = cv2.VideoCapture(str(path))
    decoded = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if image.shape[1::-1] != size:
            raise ValueError(f"Frame dimensions changed: {path}")
        decoded += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if decoded != frames or fps != 10:
        raise ValueError(f"Cannot decode entire movie: {path}")
    return decoded


def label(image, xy, text, size, fill=INK):
    ImageDraw.Draw(image).text(xy, text, font=FONTS[size], fill=fill, anchor="lt")


def read_image(path):
    with Image.open(path) as im:
        if im.size != HD_SIZE:
            raise ValueError(f"Unexpected front-camera resolution: {path}")
        return im.convert("RGB")


def boundary(record):
    asset = ROOT / record["source_ply"]
    with np.load(asset.with_name("surface.npz")) as f:
        mask = (f["defect_mask"]>.5).astype("u1")
        width, length = float(f["width"]), float(f["length"])
    outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xy = max(outlines, key=cv2.contourArea)[:, 0]
    xyz = np.column_stack(((xy[:, 0]/(mask.shape[1]-1)-.5)*width,
                           (.5-xy[:, 1]/(mask.shape[0]-1))*length, np.zeros(len(xy))))
    return xyz @ np.array(record["basis_columns"]).T + record["center_world"]


def frame_metrics(vertices, w2c, k, after, before):
    _, w, h, fx, fy, cx, cy = k
    width, height = after.size
    scale = width / w
    if before.size != after.size or not np.isclose(scale*h, height):
        raise ValueError("Calibration resolution mismatch")
    delta = abs(np.asarray(after, dtype="i2")-np.asarray(before, dtype="i2")).max(axis=2)
    output = []
    for world in vertices:
        cam = world @ w2c[:3,:3].T + w2c[:3,3]
        if cam[:, 2].min() < 1:
            output.append(None)
            continue
        uv = np.rint(scale*(cam[:, :2]/cam[:, 2,None]*[fx, fy]+[cx, cy])).astype("i4")
        l, t = uv.min(0)
        r, b = uv.max(0)
        mask = np.zeros(delta.shape,"u1")
        cv2.fillPoly(mask, [uv], 1)
        difference = delta[mask>0]
        if len(difference) == 0:
            output.append(None)
            continue
        output.append(dict(bbox=[int(l),int(t),int(r),int(b)], pixels=len(difference),
                           changed=int((difference>6).sum()), mean_delta=round(float(difference.mean()),2),
                           complete=bool(l>=2 and t>=2 and r<width-2 and b<height-2)))
    return output


def select(metrics):
    def complete(v):
        return v is not None and v["complete"] and v["changed"] > 65
    best = []
    for pos in range(2):
        choices = [i for i, m in enumerate(metrics) if complete(m[pos])]
        if not choices:
            raise ValueError("Pit did not produce a complete visible change")
        best.append(max(choices, key=lambda i: metrics[i][pos]["pixels"]))
    both = [i for i, m in enumerate(metrics) if all(complete(v) for v in m)]
    if not both:
        raise ValueError("Both pits must be visible at the same exposure")
    joint = max(both, key=lambda i:min(v["pixels"] for v in metrics[i]))
    return best, joint, both


def crop_box(box):
    l,t,r,b = box
    aspect = PANEL[0]/PANEL[1]
    width = min(HD_SIZE[0], max((r-l)*1.35, (b-t)*aspect*1.22, 180))
    height = width/aspect
    x = int(np.clip((l+r-width)/2,0,HD_SIZE[0]-width))
    y = int(np.clip((t+b-height)/2,0,HD_SIZE[1]-height))
    return (x, y, int(x+width), int(y+height))


def badge(img, xy, text):
    x, y = xy
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((x, y, x+65, y+23), radius=3, fill=INK)
    label(img, (x+7, y+3), text, 14, "white")


def fixed_panels(records, best, metrics, original, migrated):
    board = Image.new("RGB", BOARD, "#f3f2ed")
    label(board, (20, 15), records[0]["id"]+"  +  "+records[1]["id"], 26)
    label(board, (326, 20), "KITTI-360  0000   /   30,000 步", 18)
    for i,(rec,g) in enumerate(zip(records,best)):
        x = 18+i*632
        box = crop_box(metrics[g][i]["bbox"])
        label(board, (x,422), rec["id"]+"   "+rec["label"], 22)
        label(board, (x+410,429), f"近景  ·  第 {200+g} 帧", 16)
        for y, frames, tag in [(458,original,"迁入前"), (604,migrated,"迁入后")]:
            panel = frames[g].crop(box).resize(PANEL,Image.Resampling.LANCZOS)
            board.paste(panel, (x,y))
            badge(board,(x+6,y+5),tag)
    label(board, (18,754), "上方为连续前视回放，下方各用同一机位对照迁入前后。", 16)
    return board


def pins(frame, ids, metrics):
    d = ImageDraw.Draw(frame)
    for key, info, color in zip(ids, metrics, ACCENTS):
        l,t,r,b = info["bbox"]
        x = int((l+r)/2)
        top = max(4, t-35)
        d.line((x,top+22,x,t-2), fill=color, width=2)
        d.rounded_rectangle((x-32,top,x+32,top+22),radius=4,fill=INK,outline=color,width=1)
        label(frame, (x-23,top+3),key,16,color)


def main():
    split,cams = load_rig(ROOT/"work/streetcrafter-rig/kitti360-0000")
    names = [f"{g:03}00.png" for g in range(len(split["frames"]))]
    original = [read_image(DEST/"original-hd"/n) for n in names]
    with (BASE/"joint.pt").open("rb") as f:
        parent_hash = hashlib.file_digest(f,"sha256").hexdigest()
    out = []
    folders = sorted((BASE/"pit-pairs").glob("[0-9][0-9]-P*-P*"))
    if len(folders) != 10:
        raise ValueError("Expected ten pair checkpoints")
    group_ids = set()
    full_count = 0
    poster = Image.new("RGB", (1240, 860), "#f3f2ed")
    label(poster,(16,12),"KITTI-360 0000    /    十组双坑同帧回看",26)
    with video(DEST/"all-pairs.mp4", BOARD) as full:
        for order,run in enumerate(folders,1):
            spec = json.loads((run/"placement.json").read_text())
            records = spec["placements"]
            keys = [r["id"] for r in records]
            if len(keys)!=2 or group_ids.intersection(keys) or spec["source_sha256"]!=parent_hash:
                raise ValueError("Unexpected group or parent")
            group_ids.update(keys)
            folder = DEST/run.name
            migrated = [read_image(folder/"frames-hd"/n) for n in names]
            vertices = [boundary(r) for r in records]
            metrics = [frame_metrics(vertices,cams[n],split["projections"]["eval"][split["views"][n]["projection"]],
                                     a,b) for n,a,b in zip(names,migrated,original)]
            best, joint, both = select(metrics)
            board = fixed_panels(records,best,metrics,original,migrated)
            annotated = migrated[joint].copy()
            pins(annotated,keys,metrics[joint])
            annotated.save(folder/"both-visible.jpg",quality=96)
            boxes = np.asarray([v["bbox"] for v in metrics[joint]])
            overview_box = crop_box([*boxes[:,:2].min(0),*boxes[:,2:].max(0)])
            shot = migrated[joint].crop(overview_box).resize((595,131),Image.Resampling.LANCZOS)
            x, y = 14+610*((order-1)%2), 54+159*((order-1)//2)
            poster.paste(shot,(x,y))
            label(poster,(x+1,y+134),f"{order:02}    {keys[0]}  +  {keys[1]}     ·     第 {200+joint} 帧",16)
            count = 0
            with video(folder/"comparison.mp4",BOARD) as comp, video(folder/"drive.mp4",HD_SIZE) as clean:
                for g,im in enumerate(migrated):
                    clean.write(im.tobytes())
                    freeze = g == joint
                    page = board.copy()
                    page.paste(annotated if freeze else im, (0,59))
                    label(page,(1020,21),f"第 {200+g} 帧   ·   {order:02} / 10",18)
                    raw = page.tobytes()
                    for _ in range(15 if freeze else 1):
                        comp.write(raw)
                        full.write(raw)
                        count += 1
            check_clip(folder/"drive.mp4",len(names),HD_SIZE)
            check_clip(folder/"comparison.mp4",count,BOARD)
            full_count += count
            out.append(dict(pair=run.name,ids=keys,matched_source_frames=[200+g for g in best],
                            simultaneous_complete_frames=[200+g for g in both],
                            comparison_frame=200+joint, changes_at_comparison=metrics[joint],
                            separation_m=spec["boundary_separation_m"], frames=count, checkpoint_sha256=spec["checkpoint_sha256"]))
            print(run.name, "joint frame", 200+joint, "changed pixels", [v["changed"] for v in metrics[joint]], flush=True)
    if len(group_ids) != 20:
        raise ValueError("Expected 20 distinct assets")
    check_clip(DEST/"all-pairs.mp4",full_count,BOARD)
    poster.save(DEST/"overview.jpg", quality=95)
    audit=dict(scene=split["scene"],source_checkpoint_sha256=parent_hash, groups=out, frames=full_count,
               fps=10, seconds=full_count/10, source_frames_per_drive=len(names), resolution=list(HD_SIZE))
    (DEST/"index.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    rows = ["# KITTI-360 0000 双坑迁入", "", "[十组对比视频](all-pairs.mp4) · [十组同帧近景](overview.jpg)", "",
            "每个子场景使用同一个 30,000 步联合检查点和两个独立的米制坑洞。纯前视保留原相机的 48 个时刻，对比视频在双坑同帧可见时停留，并列出每个坑的同帧原路面及迁入效果。", "",
            "| 组 | 坑洞 | 双坑对比 | 纯前视 | 场景文件 |", "|---:|---|---|---|---|"]
    for i,v in enumerate(out,1):
        folder=v["pair"]
        rel=os.path.relpath(BASE/"pit-pairs"/folder/"joint.pt",DEST)
        rows.append(f"| {i} | {'、'.join(v['ids'])} | [播放]({folder}/comparison.mp4) | [播放]({folder}/drive.mp4) | [joint.pt]({rel}) |")
    (DEST/"list.md").write_text("\n".join(rows)+"\n")


if __name__=="__main__":
    main()
