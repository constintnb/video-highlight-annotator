# -*- coding: utf-8 -*-
"""自定义控件：VideoView（原视频 + 比例锁定裁剪框）、TimelineView（可拖动时间轴）。"""
import cv2

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QWidget


def bgr_to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return qimg.copy()


class VideoView(QWidget):
    """显示原视频并在其上叠加可拖动、比例锁定的裁剪框。"""

    boxChanged = Signal(int, int, int)  # x, y, w（源像素坐标，用户改框后发出）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None      # QPixmap（原分辨率）
        self._src_w = 0
        self._src_h = 0
        self._box = None         # (x, y, w, h) 源坐标
        self.rw, self.rh = 16, 9
        self.interactive = False  # 仅高光帧允许拖动

        self._mode = None         # None / 'move' / 'resize'
        self._corner = None
        self._start_box = None
        self._press_src = None
        self._fixed = None        # 缩放时固定对角 (fx, fy) 源坐标
        self._handle = 14

    # ---- 数据设置 ----
    def set_frame(self, bgr):
        if bgr is None:
            self._pixmap = None
            self._src_w = self._src_h = 0
        else:
            self._src_h, self._src_w = bgr.shape[:2]
            self._pixmap = QPixmap.fromImage(bgr_to_qimage(bgr))
        self.update()

    def set_box(self, box):
        self._box = box
        self.update()

    def set_ratio(self, rw, rh):
        self.rw, self.rh = rw, rh
        self.update()

    # ---- 坐标换算 ----
    def _scale_offsets(self):
        w = self.width(); h = self.height()
        if not self._src_w or not self._src_h:
            return 1.0, 0, 0
        s = min(w / self._src_w, h / self._src_h)
        ox = (w - self._src_w * s) / 2
        oy = (h - self._src_h * s) / 2
        return s, ox, oy

    def _to_src(self, px, py):
        s, ox, oy = self._scale_offsets()
        return (px - ox) / s, (py - oy) / s

    # ---- 绘制 ----
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(15, 16, 18))
        if self._pixmap is None:
            p.setPen(QColor(150, 154, 158))
            p.drawText(self.rect(), Qt.AlignCenter, "无视频")
            return

        s, ox, oy = self._scale_offsets()
        sw = self._src_w * s
        sh = self._src_h * s
        p.drawPixmap(QRectF(ox, oy, sw, sh), self._pixmap, QRectF(0, 0, self._src_w, self._src_h))

        if self._box is None:
            return
        x, y, w, h = self._box
        wx, wy, ww, wh = ox + x * s, oy + y * s, w * s, h * s

        # 框外压暗
        dim = QColor(0, 0, 0, 150)
        p.fillRect(QRectF(ox, oy, sw, wy - oy), dim)                              # top
        p.fillRect(QRectF(ox, wy + wh, sw, (oy + sh) - (wy + wh)), dim)           # bottom
        p.fillRect(QRectF(ox, wy, wx - ox, wh), dim)                              # left
        p.fillRect(QRectF(wx + ww, wy, (ox + sw) - (wx + ww), wh), dim)           # right

        # 绿色框
        pen = QPen(QColor(0, 210, 90), 2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(wx, wy, ww, wh))

        # 四个角 + 右下角手柄
        p.setBrush(QColor(0, 210, 90))
        r = 5
        for cx, cy in ((wx, wy), (wx + ww, wy), (wx, wy + wh), (wx + ww, wy + wh)):
            p.drawRect(QRectF(cx - r / 2, cy - r / 2, r, r))
        p.drawRect(QRectF(wx + ww - self._handle / 2, wy + wh - self._handle / 2,
                          self._handle, self._handle))

    # ---- 交互 ----
    def mousePressEvent(self, e):
        if self._box is None or not self.interactive or e.button() != Qt.LeftButton:
            return
        x, y, w, h = self._box
        s, ox, oy = self._scale_offsets()
        wx, wy, ww, wh = ox + x * s, oy + y * s, w * s, h * s
        px, py = e.position().x(), e.position().y()

        corners = {"tl": (wx, wy), "tr": (wx + ww, wy),
                   "bl": (wx, wy + wh), "br": (wx + ww, wy + wh)}
        for name, (cx, cy) in corners.items():
            if abs(px - cx) <= 14 and abs(py - cy) <= 14:
                self._mode = "resize"
                self._corner = name
                self._fixed = self._fixed_corner(x, y, w, h, name)
                self._start_box = (x, y, w, h)
                return
        if wx <= px <= wx + ww and wy <= py <= wy + wh:
            self._mode = "move"
            self._start_box = (x, y, w, h)
            self._press_src = self._to_src(px, py)
            return
        self._mode = None

    def _fixed_corner(self, x, y, w, h, name):
        if name == "br":
            return (x, y)
        if name == "bl":
            return (x + w, y)
        if name == "tr":
            return (x, y + h)
        return (x + w, y + h)  # tl

    def mouseMoveEvent(self, e):
        if self._mode is None or self._box is None:
            return
        x, y, w, h = self._start_box
        mx, my = self._to_src(e.position().x(), e.position().y())
        if self._mode == "move":
            dx = mx - self._press_src[0]
            dy = my - self._press_src[1]
            nx = x + round(dx)
            ny = y + round(dy)
            nx, ny, w, h = self._clamp(nx, ny, w, h)
            self._box = (nx, ny, w, h)
        elif self._mode == "resize":
            self._box = self._resize_box(mx, my)
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._mode is not None and self._box is not None:
            x, y, w, h = self._box
            self._mode = None
            self.boxChanged.emit(int(x), int(y), int(w))
        self._mode = None

    def _resize_box(self, mx, my):
        fx, fy = self._fixed
        r = self.rw / self.rh
        dx = mx - fx
        dy = my - fy
        w = max(abs(dx), abs(dy) * r)
        w = max(1.0, w)
        w = min(w, self._src_w, self._src_h * r)
        w = max(1.0, w)
        h = w / r
        x = fx if dx >= 0 else fx - w
        y = fy if dy >= 0 else fy - h
        return self._clamp(x, y, w, h)

    def _clamp(self, x, y, w, h):
        w = max(1.0, min(float(w), self._src_w))
        h = max(1.0, min(float(h), self._src_h))
        x = max(0.0, min(float(x), self._src_w - w))
        y = max(0.0, min(float(y), self._src_h - h))
        return x, y, w, h


class TimelineView(QWidget):
    """可拖动时间轴：高光段、裁剪关键帧、播放头。"""

    seekRequested = Signal(int)
    segmentClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.total = 0
        self.fps = 30.0
        self.segments = []       # [(start, end), ...]
        self.keyframes = []      # [frame, ...] 关键帧帧号
        self.playhead = 0
        self.selected = -1
        self.setMinimumHeight(92)

    def set_data(self, total, fps, segments, keyframes, playhead, selected):
        self.total = max(1, total)
        self.fps = fps if fps else 30.0
        self.segments = segments
        self.keyframes = keyframes
        self.playhead = playhead
        self.selected = selected
        self.update()

    def _bar(self):
        pad = 14
        return pad, self.width() - 2 * pad, 40, self.height() - 40 - 26

    def _x(self, frame):
        bx, bw, by, bh = self._bar()
        return bx + (frame / self.total) * bw

    def _frame_at(self, px):
        bx, bw, by, bh = self._bar()
        f = (px - bx) / bw * self.total
        return int(round(max(0, min(self.total - 1, f))))

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(28, 30, 34))
        bx, bw, by, bh = self._bar()

        # 底条
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(60, 62, 66))
        p.drawRect(bx, by, bw, bh)

        # 高光段
        for i, (s, e) in enumerate(self.segments):
            col = QColor(255, 170, 40) if i == self.selected else QColor(255, 138, 0)
            p.setBrush(col)
            sx = self._x(s); ex = max(sx + 1, self._x(min(e, self.total - 1)))
            p.drawRect(int(sx), by, int(ex - sx), bh)

        # 关键帧标记（菱形）
        p.setPen(QPen(QColor(80, 200, 255), 2))
        for f in self.keyframes:
            cx = int(self._x(f))
            p.drawLine(cx - 5, by, cx, by - 6)
            p.drawLine(cx, by - 6, cx + 5, by)
            p.drawLine(cx + 5, by, cx, by + 6)
            p.drawLine(cx, by + 6, cx - 5, by)

        # 播放头
        px = self._x(min(self.playhead, self.total - 1))
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(int(px), by - 6, int(px), by + bh + 6)
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(QRectF(px - 4, by - 10, 8, 8))

        # 时间刻度
        p.setPen(QColor(170, 174, 178))
        dur = self.total / self.fps
        t = 0.0
        while t <= dur + 1e-6:
            xx = self._x(int(t * self.fps))
            p.drawLine(int(xx), by + bh, int(xx), by + bh + 4)
            mm, ss = int(t // 60), int(t % 60)
            p.drawText(int(xx - 12), by + bh + 18, f"{mm}:{ss:02d}")
            t += 10.0

        # 当前时间
        cur = self.playhead / self.fps
        mm, ss = int(cur // 60), int(cur % 60)
        p.drawText(bx, by - 12, f"t = {mm}:{ss:02d}  帧 {self.playhead}/{self.total - 1}")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._seek(e.position().x(), e.position().y())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._seek(e.position().x(), e.position().y())

    def _seek(self, px, py):
        f = self._frame_at(px)
        # 若点在高光段内则选中该段
        sel = -1
        for i, (s, e) in enumerate(self.segments):
            if s <= f <= e:
                sel = i
                break
        if sel != self.selected:
            self.selected = sel
            self.segmentClicked.emit(sel)
        self.seekRequested.emit(f)
