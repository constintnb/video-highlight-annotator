# -*- coding: utf-8 -*-
"""逐帧视频读取：cv2 顺序读 + 随机跳转 + 小缓存。"""
import cv2


class VideoReader:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cache = {}
        self._pos = -1

    def read(self, n):
        n = max(0, min(n, self.total - 1))
        if n in self._cache:
            return self._cache[n]
        if n != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = self.cap.read()
        if not ok:
            # 重开再跳转兜底
            self.cap.release()
            self.cap = cv2.VideoCapture(self.path)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = self.cap.read()
        if not ok:
            return None
        self._pos = n
        self._cache[n] = frame
        if len(self._cache) > 16:
            self._cache.pop(next(iter(self._cache)))
        return frame

    def release(self):
        self.cap.release()
