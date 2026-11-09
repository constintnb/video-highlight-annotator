# -*- coding: utf-8 -*-
"""数据模型：高光段落、裁剪关键帧、线性插值、与官方格式互转。"""
from dataclasses import dataclass, field


@dataclass
class Keyframe:
    frame: int   # 帧号
    x: int       # 裁剪框左上角 x
    y: int       # 裁剪框左上角 y
    w: int       # 裁剪框宽度（高度 = w * rh / rw）


@dataclass
class Segment:
    start: int
    end: int
    keyframes: list = field(default_factory=list)  # list[Keyframe]

    def sorted_kf(self):
        return sorted(self.keyframes, key=lambda k: k.frame)


def interp_box(kfs, frame):
    """在关键帧序列中按帧号线性插值得到 (x, y, w)；首帧前保持首个、末帧后保持末个。"""
    if not kfs:
        return None
    kfs = sorted(kfs, key=lambda k: k.frame)
    if frame <= kfs[0].frame:
        k = kfs[0]
        return (k.x, k.y, k.w)
    if frame >= kfs[-1].frame:
        k = kfs[-1]
        return (k.x, k.y, k.w)
    for a, b in zip(kfs, kfs[1:]):
        if a.frame <= frame <= b.frame:
            if b.frame == a.frame:
                return (b.x, b.y, b.w)
            t = (frame - a.frame) / (b.frame - a.frame)
            x = round(a.x + (b.x - a.x) * t)
            y = round(a.y + (b.y - a.y) * t)
            w = round(a.w + (b.w - a.w) * t)
            return (x, y, w)
    k = kfs[-1]
    return (k.x, k.y, k.w)


def box_height(w, rw, rh):
    return max(1, int(round(w * rh / rw)))


def default_keyframe(frame, orig_w, orig_h, rw, rh):
    """默认裁剪框：能放入画面的、比例锁定的最大居中框。"""
    if orig_w / max(orig_h, 1) >= rw / rh:
        # 源比目标更宽 -> 高度填满，宽度向下取整保证派生高度不越界
        h = orig_h
        w = int(h * rw / rh)
    else:
        w = orig_w
        h = int(w * rh / rw)
    w = max(1, min(w, orig_w))
    h = max(1, min(h, orig_h))
    x = (orig_w - w) // 2
    y = (orig_h - h) // 2
    return Keyframe(frame, x, y, w)


def clamp_box(x, y, w, h, W, H):
    """把框钳制在画面内，并保持 w,h 不越界。"""
    w = max(1, min(w, W))
    h = max(1, min(h, H))
    x = max(0, min(x, W - w))
    y = max(0, min(y, H - h))
    return x, y, w, h


def group_frames(frames, gap=2):
    """把离散帧号按连续(间隔<=gap)分组 -> [(start, end), ...] 闭区间。"""
    if not frames:
        return []
    frames = sorted(set(frames))
    groups = []
    s = prev = frames[0]
    for f in frames[1:]:
        if f - prev <= gap:
            prev = f
        else:
            groups.append((s, prev))
            s = prev = f
    groups.append((s, prev))
    return groups


def simplify_keyframes(preds, max_kf=20):
    """把逐帧 predictions 抽稀成 <=max_kf 个关键帧（保留首尾 + 均匀采样）。"""
    preds = sorted(preds, key=lambda p: p["frame"])
    n = len(preds)
    if n == 0:
        return []
    if n <= max_kf:
        idxs = list(range(n))
    else:
        step = (n - 1) / (max_kf - 1)
        idxs = sorted(set([0, n - 1] + [int(round(i * step)) for i in range(1, max_kf)]))
    out = []
    for i in idxs:
        b = preds[i]["bboxes"]
        out.append(Keyframe(preds[i]["frame"], int(b[0]), int(b[1]), int(b[2])))
    return out


def predictions_to_model(preds, gap=2, max_kf=20):
    """把官方 predictions -> [Segment...]（每段带抽稀后的关键帧）。"""
    segs = []
    for s, e in group_frames([p["frame"] for p in preds], gap):
        seg_preds = [p for p in preds if s <= p["frame"] <= e]
        kfs = simplify_keyframes(seg_preds, max_kf)
        if not kfs:
            continue
        segs.append(Segment(s, e, kfs))
    return segs


def segments_to_predictions(segments, rw, rh):
    """把 [Segment...] 展开为官方格式 predictions（每帧一个 bbox）。"""
    preds = []
    for seg in segments:
        kfs = seg.sorted_kf()
        for f in range(seg.start, seg.end + 1):
            box = interp_box(kfs, f)
            if box is None:
                continue
            preds.append({"frame": f, "bboxes": [box[0], box[1], box[2]]})
    return preds
