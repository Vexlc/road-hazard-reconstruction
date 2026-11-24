"""Show the ten paired scenes from a vehicle-mounted virtual rear camera after passing the pits."""
import hashlib
import json

import numpy as np
from PIL import Image

from prepare import load_rig
from review_pit_pairs import (BASE, DEST, ROOT, boundary, check_clip, frame_metrics, label, pins, video)


VIEW = (1280, 720)
BOARD = (1280, 784)
HOLD = 12


def image(path, size=VIEW):
    with Image.open(path) as src:
        if src.size != size:
            raise ValueError(f"Unexpected frame size at {path}")
        return src.convert("RGB")


def panel(frame, pair, order, title, source_frame, speed):
    sheet = Image.new("RGB", BOARD, "#f3f2ed")
    label(sheet, (18,17), f"{order:02} / 10    {pair[0]}  +  {pair[1]}", 22)
    label(sheet, (420,17), title, 22)
    label(sheet, (1000,21), f"第 {source_frame} 帧   ·   {speed}", 18)
    sheet.paste(frame, (0,64))
    return sheet.tobytes()


def clearances(edges, front_view, rear_view):
    front = np.linalg.inv(front_view)
    rear_center = np.linalg.inv(rear_view)[:3,3]
    heading = front[:3,2].copy()
    heading[2] = 0
    heading /= np.linalg.norm(heading)
    return [float(np.min((rear_center-rim) @ heading)) for rim in edges]


def identifiable(metric, clearance):
    return clearance > 0 and metric is not None and metric["complete"] and metric["changed"] >= 120


def select_frames(observations):
    identified = {g: [identifiable(v,d) for v,d in zip(row["metrics"],row["clearance_m"])]
                  for g,row in observations.items()}
    first = [g for g,visible in identified.items() if visible[0]]
    both = [g for g,visible in identified.items() if all(visible)]
    if not first or not both:
        raise ValueError("No verified post-pass view of both pits")
    # Use the frame with the clearest far pit while both complete rims still fit.
    best = max(both,key=lambda g:min(v["changed"] for v in observations[g]["metrics"]))
    return first[0], best, both


def main():
    split, front = load_rig(ROOT/"work/streetcrafter-rig/kitti360-0000")
    trace = json.loads((DEST/"original-rear/camera.json").read_text())
    with (BASE/"joint.pt").open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest() != trace["parent_sha256"]:
            raise ValueError("Source scene changed")
    rear = {int(n[:3]):np.array(v) for n,v in trace["w2c"].items()}
    original = {g:image(DEST/"original-rear"/f"{g:03}00.png") for g in rear}
    output = []
    counts = [0, 0]
    runs = sorted((BASE/"pit-pairs").glob("[0-9][0-9]-P*-P*"))
    if len(runs) != 10:
        raise ValueError("Expected ten pairs")
    with video(DEST/"all-pairs-front-to-rear.mp4", BOARD) as combined, video(DEST/"all-pairs-rear.mp4", BOARD) as rear_only:
        for order, run in enumerate(runs, 1):
            place = json.loads((run/"placement.json").read_text())
            keys = [p["id"] for p in place["placements"]]
            if len(keys) != 2 or place["source_sha256"] != trace["parent_sha256"]:
                raise ValueError("Unexpected checkpoint lineage")
            edges = [boundary(p) for p in place["placements"]]
            folder = DEST/run.name
            frames = {}
            observations = {}
            for g,w2c in rear.items():
                name = f"{g:03}00.png"
                after = image(folder/"rear-frames"/name)
                frames[g] = after
                observations[g] = dict(metrics=frame_metrics(edges,w2c,trace["pinhole"],after,original[g]),
                                       clearance_m=clearances(edges,front[name],w2c))
            start, best, both = select_frames(observations)
            # Lead-in is chronological and ends at the exposure immediately preceding the switch.
            for g in range(start):
                n = f"{g:03}00.png"
                current = image(folder/"frames-hd"/n,(1280,342))
                crop = current.crop((316,0,924,342)).resize(VIEW,Image.Resampling.LANCZOS)
                combined.write(panel(crop,keys,order,"前视  ·  驶向坑洞",split["frames"][g],"1×"))
                counts[0] += 1
            clean_frames = 0
            rear_frames = 0
            with video(folder/"rear-drive.mp4",VIEW) as clean:
                for g in range(start,len(split["frames"])):
                    im = frames[g]
                    clean.write(im.tobytes())
                    clean_frames += 1
                    passed = min(observations[g]["clearance_m"]) > 0
                    caption = "虚拟后视  ·  已驶过两处" if passed else "虚拟后视  ·  已驶过第一处"
                    live = panel(im,keys,order,caption,split["frames"][g],"0.5×")
                    # Two video frames per source time for easier inspection while driving away.
                    for _ in range(2):
                        combined.write(live)
                        rear_only.write(live)
                    rear_frames += 2
                    if g == best:
                        marked = im.copy()
                        pins(marked,keys,observations[g]["metrics"])
                        marked.save(folder/"rear-both.jpg",quality=96)
                        freeze = panel(marked,keys,order,caption,split["frames"][g],"定格")
                        for _ in range(HOLD):
                            combined.write(freeze)
                            rear_only.write(freeze)
                        rear_frames += HOLD
            check_clip(folder/"rear-drive.mp4",clean_frames,VIEW)
            counts[0] += rear_frames
            counts[1] += rear_frames
            output.append(dict(pair=run.name,ids=keys,checkpoint_sha256=place["checkpoint_sha256"],
                               first_rear_frame=split["frames"][start],last_rear_frame=split["frames"][-1],
                               simultaneous_post_pass_frames=[split["frames"][g] for g in both],
                               hold_frame=split["frames"][best],post_pass_check=observations[best],
                               clean_frames=clean_frames))
            print(run.name, "rear from",split["frames"][start],"both passed", [split["frames"][g] for g in both], flush=True)
    if len({k for group in output for k in group["ids"]}) != 20:
        raise ValueError("Expected 20 unique pits")
    for file, count in zip(("all-pairs-front-to-rear.mp4","all-pairs-rear.mp4"),counts):
        check_clip(DEST/file,count,BOARD)
    index = dict(camera="original-rear/camera.json",source_checkpoint_sha256=trace["parent_sha256"],
                 fps=10,front_to_rear_seconds=counts[0]/10,rear_seconds=counts[1]/10,groups=output)
    (DEST/"rear-index.json").write_text(json.dumps(index,ensure_ascii=False,indent=2)+"\n")
    lines = ["# 驶过坑洞后的回望", "", "[十组后视](all-pairs-rear.mp4) · [前视驶入、后视驶离](all-pairs-front-to-rear.mp4)", "",
             "后视由固定在车尾的虚拟相机沿原车轨迹渲染。每组的原速视频从车尾越过第一处坑沿开始。合集中后视慢放到半速，两处坑洞都已越过并在画面内时定格标出编号。",
             "", "| 组 | 坑洞 | 后视原速 | 双坑同帧 |", "|---:|---|---|---|"]
    for order,v in enumerate(output,1):
        d = v["pair"]
        lines.append(f"| {order} | {'、'.join(v['ids'])} | [播放]({d}/rear-drive.mp4) | [第 {v['hold_frame']} 帧]({d}/rear-both.jpg) |")
    (DEST/"rear-list.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
