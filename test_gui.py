# -*- coding: utf-8 -*-
"""无显示集成冒烟测试：导入全部模块、实例化控件、读视频、导出 json + 对比视频。"""
import os
import numpy as np
import cv2

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp")
os.makedirs(tmp, exist_ok=True)

# 1. 合成测试视频 160x90 @30fps, 60 帧
vid_path = os.path.join(tmp, "0.mp4")
vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 90))
for i in range(60):
    f = np.full((90, 160, 3), (i * 4 % 255, 90, 90), np.uint8)
    cv2.putText(f, str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    vw.write(f)
vw.release()

# 2. 导入所有模块
import config, model, video_reader, exporter, widgets, main_window, main  # noqa
print("ALL_IMPORTS_OK")

# 3. QApplication + 控件
from PySide6.QtWidgets import QApplication  # noqa
app = QApplication([])
from widgets import VideoView, TimelineView  # noqa
v = VideoView()
t = TimelineView()
t.set_data(60, 30.0, [(10, 20), (30, 40)], [10, 20, 30], 15, 0)
print("WIDGETS_OK")

# 4. VideoReader
r = video_reader.VideoReader(vid_path)
assert r.total == 60, r.total
fr = r.read(30)
assert fr is not None and fr.shape == (90, 160, 3)
print(f"READER_OK total={r.total} fps={r.fps}")

# 5. 导出
from model import Segment, Keyframe  # noqa
seg = Segment(10, 40, [Keyframe(10, 0, 20, 80), Keyframe(40, 40, 20, 80)])
preds = model.segments_to_predictions([seg], 16, 9)
out_mp4 = os.path.join(tmp, "0_out.mp4")
n = exporter.render_comparison(vid_path, [seg], 16, 9, out_mp4)
print(f"RENDER_OK frames={n} exists={os.path.exists(out_mp4)} "
      f"size={os.path.getsize(out_mp4) if os.path.exists(out_mp4) else 0}")
exporter.save_video_json(0, 16, 9, preds, tmp)
exporter.append_jsonl(0, 16, 9, preds, tmp)
print("EXPORT_OK json=", os.path.exists(os.path.join(tmp, "0.json")),
      "jsonl=", os.path.exists(os.path.join(tmp, "annotations.jsonl")))

# 6. MainWindow 实例化（离线）
from config import Config  # noqa
cfg = Config(source_dir=tmp, ratio_dir="", out_dir=tmp, prefill_jsonl="", default_ratio=[16, 9])
win = main_window.MainWindow(cfg, [(0, "0.mp4", vid_path)], {})
print(f"MAINWINDOW_OK vid={win.vid} ratio=({win.rw},{win.rh}) "
      f"total={win.reader.total} segments={len(win.segments)}")

print("\nALL_GUI_TESTS_DONE")
