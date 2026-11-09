# -*- coding: utf-8 -*-
"""主窗口：视频打标界面，按钮/时间轴/裁剪框/预览/保存下一个。"""
import os
import threading

import cv2

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QMessageBox, QProgressDialog, QSizePolicy, QFrame,
)

from video_reader import VideoReader
from widgets import VideoView, TimelineView, bgr_to_qimage
from model import (
    Segment, Keyframe, interp_box, box_height, default_keyframe,
    predictions_to_model, segments_to_predictions,
)
from config import load_ratio
from exporter import save_video_json, append_jsonl, render_comparison


class MainWindow(QMainWindow):
    settingsRequested = Signal()

    _progress = Signal(int, int)
    _done = Signal(str)
    _error = Signal(str)

    def __init__(self, cfg, video_list, prefill):
        super().__init__()
        self.cfg = cfg
        self.video_list = video_list   # [(vid, fn, path)]
        self.prefill = prefill         # {vid: {'targetRatioWH':[..], 'predictions':[..]}}

        self.reader = None
        self.vid = None
        self.index = -1
        self.rw, self.rh = 16, 9
        self.segments = []
        self.open_start = None
        self.playhead = 0
        self.selected = -1

        self._playing = False
        self._prog = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._connect()

        self._progress.connect(self._on_progress)
        self._done.connect(self._on_done)
        self._error.connect(self._on_error)

        if not self.video_list:
            QMessageBox.critical(self, "错误", "源视频文件夹里没有找到视频。")
            return
        start = self._first_unannotated()
        self.load_video(start)

    # ---------------- UI ----------------
    def _build_ui(self):
        self.setWindowTitle("高光打标工具")
        self.resize(1280, 860)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # 顶栏
        top = QHBoxLayout()
        self.title_lbl = QLabel("视频编号 -")
        self.title_lbl.setStyleSheet("font-size:18px; font-weight:bold;")
        self.progress_lbl = QLabel("")
        self.progress_lbl.setStyleSheet("color:#888;")
        btn_set = QPushButton("设置…")
        btn_set.clicked.connect(lambda: self.settingsRequested.emit())
        self.btn_save = QPushButton("✓ 保存并下一个")
        self.btn_save.setStyleSheet("background:#1e8e3e; color:white; padding:6px 14px;")
        top.addWidget(self.title_lbl)
        top.addSpacing(16)
        top.addWidget(self.progress_lbl)
        top.addStretch(1)
        top.addWidget(btn_set)
        top.addWidget(self.btn_save)
        root.addLayout(top)

        # 中部：左视频 + 右预览
        mid = QHBoxLayout()
        self.video_view = VideoView()
        self.video_view.setMinimumSize(420, 480)
        self.video_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#0f1012; color:#777; font-size:16px;")
        self.preview.setMinimumSize(260, 480)
        mid.addWidget(self.video_view, 1)
        mid.addWidget(self.preview, 0)
        root.addLayout(mid, 1)

        # 底部按钮
        btns = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_mark = QPushButton("标记高光(起点)")
        self.btn_add_kf = QPushButton("添加关键帧")
        self.btn_del_kf = QPushButton("删除关键帧")
        self.btn_undo = QPushButton("撤销标记")
        self.btn_prev = QPushButton("上一个")
        self.btn_next = QPushButton("下一个")
        for b in (self.btn_play, self.btn_mark, self.btn_add_kf, self.btn_del_kf,
                  self.btn_undo, self.btn_prev, self.btn_next):
            b.setMinimumHeight(30)
        btns.addWidget(self.btn_play)
        btns.addWidget(self.btn_mark)
        btns.addWidget(self.btn_add_kf)
        btns.addWidget(self.btn_del_kf)
        btns.addWidget(self.btn_undo)
        btns.addStretch(1)
        btns.addWidget(self.btn_prev)
        btns.addWidget(self.btn_next)
        root.addLayout(btns)

        # 时间轴
        self.timeline = TimelineView()
        root.addWidget(self.timeline)

        self.setCentralWidget(central)

        # 快捷键
        sc_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc_space.activated.connect(self.toggle_play)
        sc_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_left.activated.connect(lambda: self.step(-1))
        sc_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_right.activated.connect(lambda: self.step(1))

    def _connect(self):
        self.btn_save.clicked.connect(self.save_and_next)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_mark.clicked.connect(self.toggle_mark)
        self.btn_add_kf.clicked.connect(self.add_keyframe)
        self.btn_del_kf.clicked.connect(self.del_keyframe)
        self.btn_undo.clicked.connect(self.undo_mark)
        self.btn_prev.clicked.connect(lambda: self.load_video(self.index - 1))
        self.btn_next.clicked.connect(lambda: self.load_video(self.index + 1))
        self.timeline.seekRequested.connect(self.on_seek)
        self.timeline.segmentClicked.connect(self.on_segment_clicked)
        self.video_view.boxChanged.connect(self.on_box_changed)

    # ---------------- 载入视频 ----------------
    def _first_unannotated(self):
        out = self.cfg.out_dir
        for i, (vid, fn, path) in enumerate(self.video_list):
            if not os.path.exists(os.path.join(out, f"{vid}.json")):
                return i
        return 0

    def load_video(self, index):
        if self.reader:
            self.reader.release()
        index = max(0, min(index, len(self.video_list) - 1))
        self.index = index
        self.vid, fn, path = self.video_list[index]
        self.title_lbl.setText(f"视频编号 {self.vid}")
        self.progress_lbl.setText(f"第 {index + 1} / {len(self.video_list)} 个 · {fn}")

        try:
            self.reader = VideoReader(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            return

        # 比例：官方比例文件夹(必填) > 预填文件 > 默认 16:9
        r = load_ratio(self.cfg.ratio_dir, self.vid)
        if r is None and self.vid in self.prefill:
            r = self.prefill[self.vid].get("targetRatioWH")
        if r is None:
            r = self.cfg.default_ratio
        self.rw, self.rh = int(r[0]), int(r[1])
        self.video_view.set_ratio(self.rw, self.rh)

        # 预填
        self.segments = []
        self.open_start = None
        self.selected = -1
        if self.vid in self.prefill and self.prefill[self.vid].get("predictions"):
            self.segments = predictions_to_model(self.prefill[self.vid]["predictions"])
            # 确保每段有关键帧
            for seg in self.segments:
                if not seg.keyframes:
                    seg.keyframes = [default_keyframe(seg.start, self.reader.w,
                                                      self.reader.h, self.rw, self.rh)]

        self._set_preview_size()
        self.playhead = 0
        self.refresh()

    def _set_preview_size(self):
        pw = int(360 * self.rw / self.rh)
        pw = max(180, min(820, pw))
        self.preview.setFixedSize(pw, 360)

    # ---------------- 刷新 ----------------
    def _segment_at(self, frame):
        for seg in self.segments:
            if seg.start <= frame <= seg.end:
                return seg
        return None

    def refresh(self):
        if not self.reader:
            return
        frame = self.reader.read(self.playhead)
        self.video_view.set_frame(frame)

        seg = self._segment_at(self.playhead)
        if seg is not None and frame is not None:
            box = interp_box(seg.sorted_kf(), self.playhead)
            if box is not None:
                x, y, w = box
                h = box_height(w, self.rw, self.rh)
                self.video_view.set_box((x, y, w, h))
                self.video_view.interactive = True
                self._update_preview(frame, x, y, w, h)
            else:
                self.video_view.set_box(None)
                self.video_view.interactive = False
                self._preview_blank()
        else:
            self.video_view.set_box(None)
            self.video_view.interactive = False
            self._preview_blank()

        seg_ranges = [(s.start, s.end) for s in self.segments]
        kf_frames = [k.frame for s in self.segments for k in s.keyframes]
        self.timeline.set_data(self.reader.total, self.reader.fps, seg_ranges,
                               kf_frames, self.playhead, self.selected)
        self._update_mark_btn()

    def _update_preview(self, frame, x, y, w, h):
        x = max(0, min(x, self.reader.w - 1)); y = max(0, min(y, self.reader.h - 1))
        x2 = min(self.reader.w, x + w); y2 = min(self.reader.h, y + h)
        crop = frame[y:y2, x:x2] if (x2 > x and y2 > y) else frame
        # 先缩放到预览尺寸再转 QImage，避免对整幅裁剪大图做转换
        pw, ph = self.preview.width(), self.preview.height()
        if pw > 0 and ph > 0 and (crop.shape[1] > pw or crop.shape[0] > ph):
            s = min(pw / crop.shape[1], ph / crop.shape[0])
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                              interpolation=cv2.INTER_AREA)
        self.preview.setPixmap(QPixmap.fromImage(bgr_to_qimage(crop)))

    def _preview_blank(self):
        self.preview.setPixmap(QPixmap())
        self.preview.setText("非高光片段")

    def _update_mark_btn(self):
        if self.open_start is None:
            self.btn_mark.setText("标记高光(起点)")
        else:
            self.btn_mark.setText(f"标记终点 (起点@帧{self.open_start})")

    def _refresh_timeline(self):
        seg_ranges = [(s.start, s.end) for s in self.segments]
        kf_frames = [k.frame for s in self.segments for k in s.keyframes]
        self.timeline.set_data(self.reader.total, self.reader.fps, seg_ranges,
                               kf_frames, self.playhead, self.selected)

    # ---------------- 播放 / 跳转 ----------------
    def on_seek(self, frame):
        self.playhead = frame
        self.refresh()

    def on_segment_clicked(self, idx):
        self.selected = idx
        self._refresh_timeline()

    def step(self, d):
        if self.reader:
            self.playhead = max(0, min(self.reader.total - 1, self.playhead + d))
            self.refresh()

    def toggle_play(self):
        if not self.reader:
            return
        if self._playing:
            self._timer.stop()
            self._playing = False
        else:
            interval = max(12, int(round(1000.0 / self.reader.fps)))
            self._timer.start(interval)
            self._playing = True
        self._update_play_btn()

    def _update_play_btn(self):
        if self._playing:
            self.btn_play.setText("■ 暂停")
        else:
            self.btn_play.setText("▶ 播放")

    def _tick(self):
        if self.playhead >= self.reader.total - 1:
            self._timer.stop()
            self._playing = False
            self._update_play_btn()
            return
        self.playhead += 1
        self.refresh()

    # ---------------- 打标 ----------------
    def toggle_mark(self):
        if not self.reader:
            return
        if self.open_start is None:
            self.open_start = self.playhead
        else:
            if self.playhead <= self.open_start:
                QMessageBox.warning(self, "提示", "终点必须在起点之后。")
                return
            seg = Segment(self.open_start, self.playhead,
                          [default_keyframe(self.open_start, self.reader.w,
                                            self.reader.h, self.rw, self.rh)])
            self.segments.append(seg)
            self.segments.sort(key=lambda s: s.start)
            self.open_start = None
            self.selected = self.segments.index(seg)
        self._update_mark_btn()
        self._refresh_timeline()
        self.refresh()

    def undo_mark(self):
        if self.open_start is not None:
            self.open_start = None
        elif self.segments:
            self.segments.pop()
            self.selected = -1
        self._update_mark_btn()
        self._refresh_timeline()
        self.refresh()

    def on_box_changed(self, x, y, w):
        seg = self._segment_at(self.playhead)
        if seg is None:
            return
        updated = False
        for k in seg.keyframes:
            if k.frame == self.playhead:
                k.x, k.y, k.w = int(x), int(y), int(w)
                updated = True
                break
        if not updated:
            seg.keyframes.append(Keyframe(self.playhead, int(x), int(y), int(w)))
            seg.keyframes.sort(key=lambda k: k.frame)
        self._refresh_timeline()

    def add_keyframe(self):
        seg = self._segment_at(self.playhead)
        if seg is None:
            QMessageBox.warning(self, "提示", "播放头不在高光段内，无法添加关键帧。")
            return
        box = interp_box(seg.sorted_kf(), self.playhead)
        if box is None:
            return
        self.on_box_changed(box[0], box[1], box[2])
        self.refresh()

    def del_keyframe(self):
        seg = self._segment_at(self.playhead)
        if seg is None:
            QMessageBox.warning(self, "提示", "播放头不在高光段内。")
            return
        if len(seg.keyframes) <= 1:
            QMessageBox.warning(self, "提示", "每段至少保留一个关键帧。")
            return
        kfs = seg.sorted_kf()
        nearest = min(kfs, key=lambda k: abs(k.frame - self.playhead))
        if abs(nearest.frame - self.playhead) > 3:
            QMessageBox.warning(self, "提示", "当前播放头附近没有关键帧可删。")
            return
        seg.keyframes.remove(nearest)
        self._refresh_timeline()
        self.refresh()

    # ---------------- 保存 ----------------
    def save_and_next(self):
        if not self.reader:
            return
        if not self.segments and self.open_start is None:
            ret = QMessageBox.question(self, "确认", "当前没有标注任何高光，仍要保存空结果吗？")
            if ret != QMessageBox.StandardButton.Yes:
                return
        elif self.open_start is not None:
            QMessageBox.warning(self, "提示", "还有未闭合的高光起点，请先标记终点或撤销。")
            return

        predictions = segments_to_predictions(self.segments, self.rw, self.rh)
        out_dir = self.cfg.out_dir
        os.makedirs(out_dir, exist_ok=True)
        save_video_json(self.vid, self.rw, self.rh, predictions, out_dir)
        append_jsonl(self.vid, self.rw, self.rh, predictions, out_dir)

        out_mp4 = os.path.join(out_dir, f"{self.vid}.mp4")
        video_path = self.video_list[self.index][2]

        self._busy = True
        self.btn_save.setEnabled(False)
        self._prog = QProgressDialog("正在生成对比视频…", None, 0, self.reader.total, self)
        self._prog.setWindowModality(Qt.WindowModal)
        self._prog.setCancelButton(None)
        self._prog.show()

        t = threading.Thread(target=self._render_thread,
                             args=(video_path, self.segments, self.rw, self.rh, out_mp4),
                             daemon=True)
        t.start()

    def _render_thread(self, video_path, segments, rw, rh, out_mp4):
        try:
            render_comparison(video_path, segments, rw, rh, out_mp4,
                              progress_cb=lambda a, b: self._progress.emit(a, b))
            self._done.emit(out_mp4)
        except Exception as e:
            self._error.emit(str(e))

    def _on_progress(self, a, b):
        if self._prog is not None:
            self._prog.setMaximum(b)
            self._prog.setValue(a)

    def _on_done(self, out_mp4):
        self._prog.close()
        self._prog = None
        self.btn_save.setEnabled(True)
        if self.index + 1 < len(self.video_list):
            self.load_video(self.index + 1)
        else:
            QMessageBox.information(self, "完成", "全部视频打标完成！")
            self.close()

    def _on_error(self, msg):
        self._prog.close()
        self._prog = None
        self.btn_save.setEnabled(True)
        QMessageBox.critical(self, "导出失败", f"对比视频生成失败：{msg}\n（json 已保存）")

    def reload(self, cfg, video_list, prefill):
        self.cfg = cfg
        self.video_list = video_list
        self.prefill = prefill
        start = self._first_unannotated()
        self.load_video(start)
