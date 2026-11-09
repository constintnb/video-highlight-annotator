# -*- coding: utf-8 -*-
"""无 GUI 核心逻辑自测（仅 stdlib，可用系统 python 运行）。"""
import sys
from model import (
    Keyframe, Segment, interp_box, box_height, default_keyframe,
    clamp_box, predictions_to_model, segments_to_predictions,
)
from config import load_ratio, load_prefill

ok = True


def check(name, cond, extra=""):
    global ok
    print(("PASS" if cond else "FAIL"), name, extra)
    if not cond:
        ok = False


# 1. 线性插值 + 保持
kfs = [Keyframe(0, 0, 0, 100), Keyframe(100, 100, 50, 200)]
check("interp before", interp_box(kfs, -5) == (0, 0, 100))
check("interp mid", interp_box(kfs, 50) == (50, 25, 150))
check("interp after", interp_box(kfs, 300) == (100, 50, 200))
check("interp single", interp_box([Keyframe(10, 5, 6, 7)], 99) == (5, 6, 7))

# 2. box_height
check("height 16:9", box_height(720, 16, 9) == 405)
check("height 9:16", box_height(607, 9, 16) == 1079)

# 3. default_keyframe
k = default_keyframe(0, 720, 1280, 16, 9)   # 竖屏 -> 全宽
check("default portrait 16:9", (k.x, k.y, k.w) == (0, 437, 720), f"got {k}")
k = default_keyframe(0, 1920, 1080, 9, 16)  # 横屏 -> 全高
check("default landscape 9:16", (k.x, k.y, k.w) == (656, 0, 607), f"got {k}")

# 4. 分组与往返
preds = [{"frame": 90, "bboxes": [0, 393, 720]}, {"frame": 91, "bboxes": [0, 392, 720]},
         {"frame": 95, "bboxes": [10, 300, 720]}]   # 91->95 间隔 4 > gap=2，应拆成两段
segs = predictions_to_model(preds, gap=2, max_kf=20)
check("gap split -> 2 segments", len(segs) == 2, f"got {len(segs)}")
check("seg[0] range", (segs[0].start, segs[0].end) == (90, 91))
check("seg[1] range", (segs[1].start, segs[1].end) == (95, 95))

preds2 = [{"frame": 90, "bboxes": [0, 393, 720]}, {"frame": 91, "bboxes": [0, 392, 720]},
          {"frame": 92, "bboxes": [0, 391, 720]}]
segs2 = predictions_to_model(preds2, gap=2, max_kf=20)
check("consecutive -> 1 segment", len(segs2) == 1, f"got {len(segs2)}")
back = segments_to_predictions(segs2, 16, 9)
check("roundtrip frame count", len(back) == 3, f"got {len(back)}")
check("roundtrip bbox exact",
      back[0]["bboxes"] == [0, 393, 720] and back[2]["bboxes"] == [0, 391, 720])

# 5. clamp
check("clamp", clamp_box(-5, -5, 100, 100, 720, 1280) == (0, 0, 100, 100))

# 6. 读官方比例
r = load_ratio(r"C:\Users\luo\OneDrive\Desktop\jsonl", 0)
check("ratio vid0", r == [16, 9], f"got {r}")
r = load_ratio(r"C:\Users\luo\OneDrive\Desktop\jsonl", 20)
check("ratio vid20", r == [9, 16], f"got {r}")

# 7. 读预填
pre = load_prefill(r"C:\Users\luo\OneDrive\Desktop\比赛\426\github_ours_0-425.jsonl")
check("prefill loaded", len(pre) > 0, f"got {len(pre)}")
if pre:
    v0 = pre.get(0)
    check("prefill vid0 has predictions", v0 is not None and bool(v0.get("predictions")))
    print("    prefill vid0 ratio:", v0.get("targetRatioWH") if v0 else None,
          "n_pred:", len(v0.get("predictions", [])) if v0 else 0)

print("\nRESULT:", "ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
