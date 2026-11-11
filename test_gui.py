# -*- coding: utf-8 -*-
"""无显示集成测试：读视频、恢复已有标注、修改和清空后重新导出。"""
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from config import Config, load_saved_annotation, scan_videos
from exporter import build_line, save_video_json
from main_window import MainWindow
from model import Keyframe, Segment, segments_to_predictions


def save_and_wait(win):
    with patch.object(QMessageBox, "critical") as errors:
        win.btn_save.click()
        deadline = monotonic() + 15
        while win._prog is not None and monotonic() < deadline:
            QApplication.processEvents()
            sleep(0.02)
        assert win._prog is None, "导出超时"
        assert not errors.called, errors.call_args


def check_edit_saved_annotations(tmp):
    source = tmp / "video"
    output = tmp / "output"
    ratios = tmp / "jsonl"
    source.mkdir()
    output.mkdir()
    ratios.mkdir()
    vid_path = source / "0.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 90))
    assert writer.isOpened()
    for i in range(60):
        frame = np.full((90, 160, 3), (i * 4 % 255, 90, 90), np.uint8)
        writer.write(frame)
    writer.release()
    for vid in (1, 2, 3):
        shutil.copyfile(vid_path, source / f"{vid}.mp4")

    segments = [
        Segment(10, 40, [Keyframe(10, 0, 20, 40), Keyframe(15, 38, 24, 44),
                         Keyframe(40, 40, 20, 40)]),
        Segment(42, 45, [Keyframe(42, 10, 20, 40)]),
    ]
    original = segments_to_predictions(segments, 1, 1)
    save_video_json(0, 1, 1, original, output)
    save_video_json(1, 16, 9, [], output)
    (output / "3.json").write_text("{", encoding="utf-8")
    (ratios / "0.jsonl").write_text('{"targetRatioWH":[16,9]}\n', encoding="utf-8")
    summary = output / "annotations.jsonl"
    old_row = build_line(0, 1, 1, original)
    other_row = build_line(1, 16, 9, [])
    summary.write_text("".join(json.dumps(row) + "\n" for row in (old_row, other_row, old_row)),
                       encoding="utf-8")
    prefill = {vid: build_line(vid, 4, 3, [{"frame": 7, "bboxes": [0, 0, 40]}])
               for vid in (0, 1, 2)}

    cfg = Config(source_dir=str(source), ratio_dir=str(ratios), out_dir=str(output))
    win = MainWindow(cfg, scan_videos(source), prefill)
    try:
        assert win.vid == 2 and win.timeline.segments == [(7, 7)]
        assert (win.rw, win.rh) == (4, 3)
        assert win.reader.read(30).shape == (90, 160, 3)
        win.sidebar.itemClicked.emit(win.sidebar.item(1))
        assert win.vid == 1 and not win.segments
        with patch.object(QMessageBox, "critical") as error:
            win.sidebar.itemClicked.emit(win.sidebar.item(3))
            assert error.called and win.vid == 1 and win.sidebar.currentRow() == 1

        win.sidebar.itemClicked.emit(win.sidebar.item(0))
        assert win.vid == 0 and win.playhead == 10
        assert (win.rw, win.rh) == (1, 1)
        assert win.timeline.segments == [(10, 40), (42, 45)]
        assert segments_to_predictions(win.segments, win.rw, win.rh) == original
        win.video_view.boxChanged.emit(4, 20, 40)
        win.on_seek(43)
        win.btn_undo.click()
        edited = segments_to_predictions(win.segments, win.rw, win.rh)
        assert edited[0]["bboxes"] == [4, 20, 40]
        assert len(edited) == 31

        save_and_wait(win)
        assert win.vid == 1
        saved = load_saved_annotation(output, 0)
        assert saved["predictions"] == edited
        assert [json.loads(line) for line in summary.read_text(encoding="utf-8").splitlines()] == [
            saved, other_row,
        ]
        assert (output / "0.mp4").stat().st_size > 0

        win.sidebar.itemClicked.emit(win.sidebar.item(0))
        assert segments_to_predictions(win.segments, win.rw, win.rh) == edited
        win.btn_mark.click()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            win.btn_clear.click()
        assert win.open_start == 10 and win.segments
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            win.btn_clear.click()
        assert win.open_start is None and win.playhead == 0 and win.selected == -1
        assert not win.timeline.segments and not win.timeline.keyframes
        assert not win.video_view.interactive and not win.btn_clear.isEnabled()
        assert load_saved_annotation(output, 0) == saved

        win.on_seek(3)
        win.btn_mark.click()
        win.on_seek(6)
        win.btn_mark.click()
        save_and_wait(win)
        saved = load_saved_annotation(output, 0)
        assert [p["frame"] for p in saved["predictions"]] == [3, 4, 5, 6]
        assert [json.loads(line) for line in summary.read_text(encoding="utf-8").splitlines()] == [
            saved, other_row,
        ]
        print("LOAD_EDIT_CLEAR_RESAVE_OK")
    finally:
        if win.reader:
            win.reader.release()
        win.close()


if __name__ == "__main__":
    app = QApplication([])
    tmp_root = Path(__file__).resolve().parent / "_tmp"
    tmp_root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=tmp_root) as tmp:
        check_edit_saved_annotations(Path(tmp))
