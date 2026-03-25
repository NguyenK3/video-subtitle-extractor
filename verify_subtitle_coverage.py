#!/usr/bin/env python3
"""Realtime subtitle coverage QA verifier.

Implements frame-level subtitle presence detection and compares detected
subtitle segments against realtime SRT coverage.
"""

from __future__ import annotations

import argparse
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SRTEntry:
    start_s: float
    end_s: float
    text: str


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float


@dataclass
class DetectorConfig:
    edge_threshold: float
    min_components: int
    swt_threshold: float
    close_kernel: int
    min_segment_frames: int
    max_gap_frames: int
    use_ocr_check: bool
    ocr_conf_threshold: float


@dataclass
class DetectorState:
    ocr: Optional[object] = None
    ocr_cache: Dict[str, bool] = None

    def __post_init__(self) -> None:
        if self.ocr_cache is None:
            self.ocr_cache = {}


def parse_timestamp(ts: str) -> float:
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
    if not m:
        return 0.0
    h, mm, ss, ms = map(int, m.groups())
    return h * 3600 + mm * 60 + ss + ms / 1000.0


def parse_srt(path: str) -> List[SRTEntry]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    entries: List[SRTEntry] = []
    for block in blocks:
        lines = [line.rstrip("\n") for line in block.splitlines() if line.strip() != ""]
        if len(lines) < 3:
            continue
        time_line = lines[1]
        m = re.match(r"(.+?)\s*-->\s*(.+)", time_line)
        if not m:
            continue
        start_s = parse_timestamp(m.group(1))
        end_s = parse_timestamp(m.group(2))
        text = "\n".join(lines[2:]).strip()
        entries.append(SRTEntry(start_s=start_s, end_s=end_s, text=text))
    return entries


def _init_ocr(state: DetectorState) -> bool:
    if state.ocr is not None:
        return True
    try:
        from paddleocr import PaddleOCR

        state.ocr = PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=False, show_log=False)
        return True
    except Exception:
        return False


def _ocr_text_check(roi: np.ndarray, cfg: DetectorConfig, state: DetectorState) -> bool:
    if not cfg.use_ocr_check:
        return True
    if not _init_ocr(state):
        return True

    thumb = cv2.resize(roi, (96, 48), interpolation=cv2.INTER_AREA)
    key = hashlib.md5(thumb.tobytes()).hexdigest()
    if key in state.ocr_cache:
        return state.ocr_cache[key]

    test = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    try:
        result = state.ocr.ocr(test, cls=False)
        ok = False
        if result and result[0]:
            for line in result[0]:
                text = line[1][0].strip()
                conf = float(line[1][1])
                if conf < cfg.ocr_conf_threshold:
                    continue

                clean = re.sub(r"\s+", "", text)
                cjk_count = len(re.findall(r"[\u4e00-\u9fff]", clean))
                if cjk_count < 2:
                    continue

                box = np.array(line[0], dtype=np.float32)
                x_coords = box[:, 0]
                y_coords = box[:, 1]
                bw = max(x_coords) - min(x_coords)
                bh = max(y_coords) - min(y_coords)
                h, w = test.shape[:2]

                if bw < w * 0.08 or bw > w * 0.98:
                    continue
                if bh < h * 0.04 or bh > h * 0.45:
                    continue

                cy = (max(y_coords) + min(y_coords)) / 2.0
                if cy < h * 0.20 or cy > h * 0.95:
                    continue

                ok = True
                break
    except Exception:
        ok = False

    state.ocr_cache[key] = ok
    return ok


def is_text_present(roi: np.ndarray, cfg: DetectorConfig, state: DetectorState) -> bool:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.close_kernel, cfg.close_kernel))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    h, w = gray.shape
    valid = 0
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < 12 or area > 5000:
            continue
        if hh < 6 or hh > h * 0.8:
            continue
        if ww < 2 or ww > w * 0.9:
            continue
        ar = ww / max(hh, 1)
        if ar < 0.1 or ar > 15:
            continue
        valid += 1

    dist = cv2.distanceTransform(th, cv2.DIST_L2, 3)
    swt_proxy = float(np.mean((dist > 0.8) & (dist < 4.5)))

    coarse_ok = edge_density >= cfg.edge_threshold and (valid >= cfg.min_components or swt_proxy >= cfg.swt_threshold)
    if not coarse_ok:
        return False

    return _ocr_text_check(roi, cfg, state)


def smooth_flags(flags: List[bool], max_gap_frames: int, min_segment_frames: int) -> List[bool]:
    arr = flags[:]
    n = len(arr)

    i = 0
    while i < n:
        if arr[i]:
            i += 1
            continue
        j = i
        while j < n and not arr[j]:
            j += 1
        gap_len = j - i
        left = i - 1 >= 0 and arr[i - 1]
        right = j < n and arr[j]
        if left and right and gap_len <= max_gap_frames:
            for k in range(i, j):
                arr[k] = True
        i = j

    i = 0
    while i < n:
        if not arr[i]:
            i += 1
            continue
        j = i
        while j < n and arr[j]:
            j += 1
        seg_len = j - i
        if seg_len < min_segment_frames:
            for k in range(i, j):
                arr[k] = False
        i = j

    return arr


def build_segments(flags: List[bool], fps: float) -> List[Segment]:
    segments: List[Segment] = []
    n = len(flags)
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        start_frame = i
        end_frame = j - 1
        segments.append(
            Segment(
                start_frame=start_frame,
                end_frame=end_frame,
                start_s=start_frame / fps,
                end_s=(end_frame + 1) / fps,
            )
        )
        i = j
    return segments


def overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def evaluate_segments(segments: List[Segment], srt_entries: List[SRTEntry]) -> Tuple[List[Segment], List[Segment], int]:
    missed: List[Segment] = []
    partial: List[Segment] = []
    covered = 0

    for seg in segments:
        seg_dur = max(seg.end_s - seg.start_s, 1e-9)
        total_overlap = 0.0
        max_single_overlap = 0.0
        for srt in srt_entries:
            ov = overlap_duration(seg.start_s, seg.end_s, srt.start_s, srt.end_s)
            if ov <= 0:
                continue
            total_overlap += ov
            if ov > max_single_overlap:
                max_single_overlap = ov

        overlap_ratio = max_single_overlap / seg_dur
        coverage_ratio = min(total_overlap / seg_dur, 1.0)

        if overlap_ratio < 0.40:
            missed.append(seg)
            continue

        covered += 1
        if coverage_ratio < 0.90:
            partial.append(seg)

    return missed, partial, covered


def frame_level_coverage(flags: List[bool], fps: float, srt_entries: List[SRTEntry]) -> float:
    text_frames = [i for i, f in enumerate(flags) if f]
    if not text_frames:
        return 100.0

    covered = 0
    for fi in text_frames:
        t = fi / fps
        if any(s.start_s <= t <= s.end_s for s in srt_entries):
            covered += 1
    return covered / len(text_frames) * 100.0


def save_debug_frames(
    video_path: str,
    missed: List[Segment],
    partial: List[Segment],
    out_dir: str,
    limit_each: int = 6,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    def save_from_segment(seg: Segment, path: str) -> None:
        mid = (seg.start_frame + seg.end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        if not ok:
            return
        cv2.imwrite(path, frame)

    for i, seg in enumerate(missed[:limit_each], start=1):
        path = os.path.join(out_dir, f"missed_segment_{i:02d}_frame_{(seg.start_frame + seg.end_frame)//2:04d}.png")
        save_from_segment(seg, path)

    for i, seg in enumerate(partial[:limit_each], start=1):
        path = os.path.join(out_dir, f"partial_segment_{i:02d}_frame_{(seg.start_frame + seg.end_frame)//2:04d}.png")
        save_from_segment(seg, path)

    cap.release()


def scan_video(video_path: str, cfg: DetectorConfig, state: DetectorState) -> Tuple[List[bool], float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    flags: List[bool] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, _ = frame.shape[:2]
        y1 = int(h * 0.70)
        y2 = int(h * 0.95)
        x1 = int(frame.shape[1] * 0.08)
        x2 = int(frame.shape[1] * 0.92)
        roi = frame[y1:y2, x1:x2]
        flags.append(is_text_present(roi, cfg, state))

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"Scan progress: {frame_idx}/{total_frames}")

    cap.release()
    return flags, fps, total_frames


def run_once(video_path: str, srt_path: str, cfg: DetectorConfig) -> Dict[str, object]:
    srt_entries = parse_srt(srt_path)
    state = DetectorState()
    flags_raw, fps, total_frames = scan_video(video_path, cfg, state)
    flags = smooth_flags(flags_raw, cfg.max_gap_frames, cfg.min_segment_frames)
    segments = build_segments(flags, fps)
    missed, partial, covered = evaluate_segments(segments, srt_entries)
    coverage_segments = (covered / len(segments) * 100.0) if segments else 100.0
    coverage_frames = frame_level_coverage(flags, fps, srt_entries)

    return {
        "fps": fps,
        "total_frames": total_frames,
        "srt_entries": srt_entries,
        "flags": flags,
        "segments": segments,
        "missed": missed,
        "partial": partial,
        "covered": covered,
        "coverage_segments": coverage_segments,
        "coverage_frames": coverage_frames,
        "status": "PASS" if coverage_frames >= 98.0 and len(missed) == 0 else "FAIL",
    }


def print_report(result: Dict[str, object], attempt: int) -> None:
    segments = result["segments"]
    missed = result["missed"]
    partial = result["partial"]
    covered = result["covered"]
    coverage_frames = result["coverage_frames"]
    coverage_segments = result["coverage_segments"]

    print("\n---")
    print(f"VIDEO SUBTITLE COVERAGE REPORT (attempt {attempt})")
    print(f"Total detected subtitle segments: {len(segments)}")
    print(f"Segments covered by realtime SRT: {covered}")
    print(f"Missed segments: {len(missed)}")
    print(f"Partial segments: {len(partial)}")
    print()
    print(f"Coverage percentage: {coverage_segments:.2f} %")
    print("---")
    print(f"Frame-level subtitle coverage: {coverage_frames:.2f} %")
    print(f"STATUS = {result['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Subtitle coverage QA verifier")
    parser.add_argument("--video", default="test/test_cn.mp4", help="Path to video")
    parser.add_argument("--srt", default="test/test_cn.realtime.srt", help="Path to realtime SRT")
    parser.add_argument("--debug-dir", default="debug", help="Debug output directory")
    parser.add_argument("--max-attempts", type=int, default=4, help="Detector tuning attempts")
    args = parser.parse_args()

    attempts = [
        DetectorConfig(0.010, 28, 0.014, 3, 3, 2, True, 0.45),
        DetectorConfig(0.009, 22, 0.012, 3, 3, 2, True, 0.40),
        DetectorConfig(0.008, 18, 0.010, 3, 2, 3, True, 0.35),
        DetectorConfig(0.007, 14, 0.008, 5, 2, 4, True, 0.30),
    ]
    attempts = attempts[: max(1, min(args.max_attempts, len(attempts)))]

    best_result = None
    for idx, cfg in enumerate(attempts, start=1):
        print(f"\nRunning attempt {idx} with config: {cfg}")
        result = run_once(args.video, args.srt, cfg)
        print_report(result, idx)

        if best_result is None:
            best_result = result
        else:
            better = (
                result["coverage_frames"],
                -len(result["missed"]),
                -len(result["partial"]),
            ) > (
                best_result["coverage_frames"],
                -len(best_result["missed"]),
                -len(best_result["partial"]),
            )
            if better:
                best_result = result

        if result["status"] == "PASS":
            save_debug_frames(args.video, result["missed"], result["partial"], args.debug_dir)
            return

    if best_result is not None:
        print("\nNo attempt reached PASS. Saving debug frames for best attempt.")
        save_debug_frames(args.video, best_result["missed"], best_result["partial"], args.debug_dir)
        print("STATUS = FAIL")


if __name__ == "__main__":
    main()
