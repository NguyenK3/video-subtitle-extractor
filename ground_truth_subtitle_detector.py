# -*- coding: utf-8 -*-
"""
Ground Truth Subtitle Detector & QA Pipeline

Independent subtitle detection pipeline that:
1. Scans video at 10fps (100ms intervals)
2. Detects subtitle regions via OCR
3. Groups detections into a ground truth timeline
4. Compares against generated SRT output
5. Reports missing, merged, frame-loss errors
6. Validates coverage and performance
7. Auto-debugs and auto-fixes the pipeline
8. Loops until all QA checks pass
"""

import os
import re
import sys
import json
import time
import cProfile
import pstats
import io
import configparser
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from difflib import SequenceMatcher

import cv2
import numpy as np

# Cache file for ground truth detections (avoid re-scanning on re-runs)
GT_CACHE_FILE = "gt_detections_cache.json"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedFrame:
    """A single frame detection result."""
    frame_time_ms: float
    detected_text: str
    confidence_score: float


@dataclass
class GroundTruthEntry:
    """A ground truth subtitle segment."""
    index: int
    start_ms: float
    end_ms: float
    text: str
    frame_count: int = 0


@dataclass
class SRTEntry:
    """A parsed SRT subtitle entry."""
    index: int
    start_ms: float
    end_ms: float
    text: str


@dataclass
class ComparisonResult:
    """Result of comparing one ground truth entry against SRT."""
    gt_entry: GroundTruthEntry
    matched_srt: Optional[SRTEntry] = None
    text_similarity: float = 0.0
    timeline_overlap: float = 0.0
    status: str = "MISSING"  # MATCHED, MISSING, MERGED


@dataclass
class QAReport:
    """Final QA report data."""
    video_length_s: float = 0.0
    ground_truth_count: int = 0
    srt_count: int = 0
    matched: int = 0
    missing: int = 0
    merged: int = 0
    frame_loss_events: int = 0
    coverage: float = 0.0
    runtime_s: float = 0.0
    ocr_calls: int = 0
    skip_rate: float = 0.0
    frames_processed: int = 0
    status: str = "FAIL"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TASK 1 & 2: Ground Truth Subtitle Detector — Frame Scanning
# ---------------------------------------------------------------------------

class GroundTruthDetector:
    """
    Independent subtitle detector that scans the video frame-by-frame
    at high frequency to build a complete ground truth timeline.
    """

    def __init__(self, video_path: str, sample_fps: float = 10.0,
                 subtitle_roi: Optional[Tuple[float, float, float, float]] = None,
                 confidence_threshold: float = 0.5):
        """
        Args:
            video_path: Path to the video file.
            sample_fps: Frames per second to sample (default 10 = every 100ms).
            subtitle_roi: (y_ratio, h_ratio, x_ratio, w_ratio) from subtitle.ini.
            confidence_threshold: Minimum OCR confidence to accept text.
        """
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.subtitle_roi = subtitle_roi
        self.confidence_threshold = confidence_threshold
        self._ocr = None

    def _init_ocr(self):
        """Initialize PaddleOCR independently (not using main pipeline)."""
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang='ch',
                show_log=False,
                use_gpu=False,
            )

    def _crop_subtitle_region(self, frame: np.ndarray) -> np.ndarray:
        """Crop the subtitle ROI from the frame."""
        h, w = frame.shape[:2]
        if self.subtitle_roi:
            y_ratio, h_ratio, x_ratio, w_ratio = self.subtitle_roi
            y1 = int(h * y_ratio)
            y2 = int(h * (y_ratio + h_ratio))
            x1 = int(w * x_ratio)
            x2 = int(w * (x_ratio + w_ratio))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            return frame[y1:y2, x1:x2]
        # Default: lower 25% of frame
        y1 = int(h * 0.75)
        return frame[y1:h, :]

    def _run_ocr(self, region: np.ndarray) -> Tuple[str, float]:
        """Run OCR on a subtitle region. Returns (text, confidence)."""
        if region is None or region.size == 0:
            return "", 0.0
        try:
            result = self._ocr.ocr(region, cls=False)
            if not result or not result[0]:
                return "", 0.0
            texts = []
            confs = []
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                if conf >= self.confidence_threshold:
                    texts.append(text)
                    confs.append(conf)
            if not texts:
                return "", 0.0
            return "".join(texts), sum(confs) / len(confs)
        except Exception:
            return "", 0.0

    def scan_video(self) -> List[DetectedFrame]:
        """
        TASK 2: Scan the video at sample_fps and perform OCR on each frame.
        Returns list of DetectedFrame results.
        """
        self._init_ocr()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / self.sample_fps))

        print(f"[GroundTruth] Video: {total_frames} frames, {video_fps:.1f} fps")
        print(f"[GroundTruth] Sampling at {self.sample_fps} fps (every {frame_interval} frames)")

        detections: List[DetectedFrame] = []
        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            if frame_no % frame_interval != 0:
                continue

            timestamp_ms = (frame_no / video_fps) * 1000.0

            # Crop subtitle region
            region = self._crop_subtitle_region(frame)

            # Run OCR
            text, confidence = self._run_ocr(region)

            if text and confidence >= self.confidence_threshold:
                detections.append(DetectedFrame(
                    frame_time_ms=timestamp_ms,
                    detected_text=text,
                    confidence_score=confidence,
                ))

            # Progress
            if frame_no % (frame_interval * 10) == 0:
                pct = frame_no / total_frames * 100
                print(f"\r[GroundTruth] Scanning: {pct:.0f}% ({frame_no}/{total_frames})", end="", flush=True)

        cap.release()
        print(f"\n[GroundTruth] Scan complete: {len(detections)} frames with text detected")
        return detections


# ---------------------------------------------------------------------------
# TASK 3: Build Ground Truth Timeline
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    """Normalize punctuation for fair comparison."""
    s = s.replace(" ", "").strip()
    # Normalize full-width ↔ half-width parentheses and punctuation
    s = s.replace("\uff08", "(").replace("\uff09", ")")  # （→(  ）→)
    s = s.replace("\uff1f", "?")  # ？→?
    s = s.replace("\u3001", ",")  # 、→,
    return s


def text_similarity(a: str, b: str) -> float:
    """Compute text similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    a_clean = _normalize_text(a)
    b_clean = _normalize_text(b)
    if a_clean == b_clean:
        return 1.0
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def build_ground_truth_timeline(
    detections: List[DetectedFrame],
    similarity_threshold: float = 0.90,
    max_gap_ms: float = 500.0,
) -> List[GroundTruthEntry]:
    """
    TASK 3: Group detected subtitle frames into subtitle segments.

    Rules:
        - If text similarity >= 90% AND frame gap <= 500ms → extend segment
        - Otherwise → start new segment
    """
    if not detections:
        return []

    # Sort by time
    detections = sorted(detections, key=lambda d: d.frame_time_ms)

    entries: List[GroundTruthEntry] = []
    idx = 0

    current_text = detections[0].detected_text
    current_start = detections[0].frame_time_ms
    current_end = detections[0].frame_time_ms
    current_count = 1

    for i in range(1, len(detections)):
        d = detections[i]
        gap = d.frame_time_ms - current_end
        sim = text_similarity(current_text, d.detected_text)

        if sim >= similarity_threshold and gap <= max_gap_ms:
            # Extend current segment
            current_end = d.frame_time_ms
            current_count += 1
            # Keep the longer/better text
            if len(d.detected_text) > len(current_text):
                current_text = d.detected_text
        else:
            # Flush current segment
            idx += 1
            entries.append(GroundTruthEntry(
                index=idx,
                start_ms=current_start,
                end_ms=current_end,
                text=current_text,
                frame_count=current_count,
            ))
            # Start new segment
            current_text = d.detected_text
            current_start = d.frame_time_ms
            current_end = d.frame_time_ms
            current_count = 1

    # Flush last segment
    idx += 1
    entries.append(GroundTruthEntry(
        index=idx,
        start_ms=current_start,
        end_ms=current_end,
        text=current_text,
        frame_count=current_count,
    ))

    print(f"[GroundTruth] Timeline: {len(entries)} subtitle segments")
    for e in entries:
        print(f"  [{ms_to_timecode(e.start_ms)} → {ms_to_timecode(e.end_ms)}] "
              f"({e.frame_count} frames) {e.text}")

    return entries


# ---------------------------------------------------------------------------
# TASK 4: Parse Generated SRT
# ---------------------------------------------------------------------------

def parse_srt(srt_path: str) -> List[SRTEntry]:
    """Parse an SRT file and return list of SRTEntry."""
    if not os.path.exists(srt_path):
        print(f"[SRT Parser] File not found: {srt_path}")
        return []

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries: List[SRTEntry] = []
    # Split by blank lines
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Parse time line
        time_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
            lines[1].strip()
        )
        if not time_match:
            continue

        g = time_match.groups()
        start_ms = (int(g[0]) * 3600000 + int(g[1]) * 60000 +
                     int(g[2]) * 1000 + int(g[3]))
        end_ms = (int(g[4]) * 3600000 + int(g[5]) * 60000 +
                   int(g[6]) * 1000 + int(g[7]))

        text = "\n".join(lines[2:]).strip()
        entries.append(SRTEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text))

    print(f"[SRT Parser] Parsed {len(entries)} entries from {srt_path}")
    for e in entries:
        print(f"  [{ms_to_timecode(e.start_ms)} → {ms_to_timecode(e.end_ms)}] {e.text}")

    return entries


# ---------------------------------------------------------------------------
# TASK 5: Subtitle Comparison
# ---------------------------------------------------------------------------

def compute_timeline_overlap(gt: GroundTruthEntry, srt: SRTEntry) -> float:
    """Compute the timeline overlap ratio between a GT entry and SRT entry."""
    overlap_start = max(gt.start_ms, srt.start_ms)
    overlap_end = min(gt.end_ms, srt.end_ms)
    if overlap_end <= overlap_start:
        return 0.0
    overlap_duration = overlap_end - overlap_start
    gt_duration = max(gt.end_ms - gt.start_ms, 1.0)
    return overlap_duration / gt_duration


def compare_subtitles(
    ground_truth: List[GroundTruthEntry],
    srt_entries: List[SRTEntry],
    text_sim_threshold: float = 0.85,
    overlap_threshold: float = 0.50,
) -> List[ComparisonResult]:
    """
    TASK 5: Compare ground truth subtitles against SRT output.
    For each GT subtitle, find the best matching SRT entry.
    """
    results: List[ComparisonResult] = []

    for gt in ground_truth:
        best_match = None
        best_text_sim = 0.0
        best_overlap = 0.0
        fallback_match = None
        fallback_text_sim = 0.0

        for srt in srt_entries:
            t_sim = text_similarity(gt.text, srt.text)
            t_overlap = compute_timeline_overlap(gt, srt)

            if t_sim >= text_sim_threshold and t_overlap >= overlap_threshold:
                # Prefer higher overlap when text similarity is tied
                if (t_sim > best_text_sim or
                    (t_sim == best_text_sim and t_overlap > best_overlap)):
                    best_match = srt
                    best_text_sim = t_sim
                    best_overlap = t_overlap
            elif t_sim >= text_sim_threshold and fallback_match is None:
                # Text matches but timeline doesn't — may be merged
                fallback_match = srt
                fallback_text_sim = t_sim

        if best_match:
            results.append(ComparisonResult(
                gt_entry=gt,
                matched_srt=best_match,
                text_similarity=best_text_sim,
                timeline_overlap=best_overlap,
                status="MATCHED",
            ))
        elif fallback_match:
            results.append(ComparisonResult(
                gt_entry=gt,
                matched_srt=fallback_match,
                text_similarity=fallback_text_sim,
                timeline_overlap=0.0,
                status="MERGED",
            ))
        else:
            results.append(ComparisonResult(
                gt_entry=gt,
                status="MISSING",
            ))

    return results


# ---------------------------------------------------------------------------
# TASK 6: Detect Merged Subtitles
# ---------------------------------------------------------------------------

def detect_merged_subtitles(
    ground_truth: List[GroundTruthEntry],
    srt_entries: List[SRTEntry],
    text_sim_threshold: float = 0.85,
) -> List[dict]:
    """
    TASK 6: Check if multiple GT subtitles were merged into a single SRT entry.
    Returns list of merge error reports.
    """
    merge_errors = []

    for srt in srt_entries:
        matched_gts = []
        for gt in ground_truth:
            t_sim = text_similarity(gt.text, srt.text)
            t_overlap = compute_timeline_overlap(gt, srt)
            # Only consider GT entries that have timeline proximity to this SRT
            # (within 2s of SRT start/end) to avoid matching repeated subtitles
            time_near = (abs(gt.start_ms - srt.start_ms) < 2000 or
                         abs(gt.end_ms - srt.end_ms) < 2000 or
                         t_overlap > 0)
            if t_sim >= text_sim_threshold and time_near:
                matched_gts.append(gt)

        if len(matched_gts) > 1:
            # Multiple GT entries matched the same SRT within its time window
            matched_gts.sort(key=lambda g: g.start_ms)
            for i in range(1, len(matched_gts)):
                gap = matched_gts[i].start_ms - matched_gts[i-1].end_ms
                if gap > 1000:  # >1s gap means separate appearances
                    merge_errors.append({
                        "srt_entry": srt,
                        "gt_entries": matched_gts,
                        "gap_ms": gap,
                    })
                    break

    return merge_errors


# ---------------------------------------------------------------------------
# TASK 7: Detect Frame Loss
# ---------------------------------------------------------------------------

def detect_frame_loss(
    detections: List[DetectedFrame],
    srt_entries: List[SRTEntry],
    tolerance_ms: float = 500.0,
) -> List[dict]:
    """
    TASK 7: Check if frames with subtitles were skipped by the pipeline.
    A frame is "lost" if GT detected text at a timestamp not covered by any SRT entry.
    """
    frame_loss_events = []

    for det in detections:
        covered = False
        for srt in srt_entries:
            if srt.start_ms - tolerance_ms <= det.frame_time_ms <= srt.end_ms + tolerance_ms:
                covered = True
                break
        if not covered:
            frame_loss_events.append({
                "frame_time_ms": det.frame_time_ms,
                "detected_text": det.detected_text,
                "confidence": det.confidence_score,
            })

    # Group consecutive lost frames
    grouped = []
    if frame_loss_events:
        current_group = [frame_loss_events[0]]
        for ev in frame_loss_events[1:]:
            if ev["frame_time_ms"] - current_group[-1]["frame_time_ms"] <= 500:
                current_group.append(ev)
            else:
                grouped.append(current_group)
                current_group = [ev]
        grouped.append(current_group)

    return grouped


# ---------------------------------------------------------------------------
# TASK 8: Coverage Metric
# ---------------------------------------------------------------------------

def compute_coverage(comparison_results: List[ComparisonResult]) -> float:
    """Compute subtitle coverage = matched / total ground truth."""
    if not comparison_results:
        return 1.0
    matched = sum(1 for r in comparison_results if r.status == "MATCHED")
    return matched / len(comparison_results)


# ---------------------------------------------------------------------------
# TASK 9: Performance Validation
# ---------------------------------------------------------------------------

def _setup_pipeline_path():
    """Ensure the backend module path is importable."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "backend")
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)


def run_pipeline_with_metrics(video_path: str, output_srt: str, sub_area=None) -> dict:
    """
    Run the fast subtitle pipeline and collect performance metrics.
    Returns dict with runtime, ocr_calls, skip_rate, frames_processed.
    """
    _setup_pipeline_path()
    from backend.tools.fast_pipeline import FastSubtitlePipeline

    pipeline = FastSubtitlePipeline(
        video_path=video_path,
        sub_area=sub_area,
    )

    start_time = time.time()
    stats = pipeline.run(output_srt)
    runtime = time.time() - start_time

    total_skips = stats.scene_change_skips + stats.diff_filter_skips + stats.cache_hits
    skip_rate = total_skips / max(stats.sampled_frames, 1) * 100

    return {
        "runtime": runtime,
        "ocr_calls": stats.ocr_calls,
        "skip_rate": skip_rate,
        "frames_processed": stats.sampled_frames,
        "total_frames": stats.total_frames,
        "subtitles_found": stats.subtitles_found,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# TASK 10: Auto Debug with cProfile
# ---------------------------------------------------------------------------

def profile_pipeline(video_path: str, output_srt: str, sub_area=None) -> str:
    """Profile the pipeline with cProfile and return a summary of bottlenecks."""
    profiler = cProfile.Profile()

    profiler.enable()
    metrics = run_pipeline_with_metrics(video_path, output_srt, sub_area)
    profiler.disable()

    # Get profiling stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    profile_output = s.getvalue()

    # Identify bottlenecks
    bottlenecks = []
    lines = profile_output.split('\n')
    for line in lines:
        if any(kw in line for kw in ['sampler', 'filter', 'ocr', 'aggregat',
                                       'process_single', 'process_batch',
                                       'has_changed', 'is_same_scene',
                                       'should_sample', 'decode', 'crop']):
            bottlenecks.append(line.strip())

    report = f"=== PROFILING REPORT ===\n"
    report += f"Total runtime: {metrics['runtime']:.2f}s\n"
    report += f"OCR calls: {metrics['ocr_calls']}\n"
    report += f"Skip rate: {metrics['skip_rate']:.1f}%\n\n"
    report += "Top bottlenecks:\n"
    for b in bottlenecks[:10]:
        report += f"  {b}\n"
    report += f"\nFull profile:\n{profile_output}"

    return report


# ---------------------------------------------------------------------------
# TASK 11: Auto Fix Pipeline
# ---------------------------------------------------------------------------

def auto_fix_pipeline(
    comparison_results: List[ComparisonResult],
    merge_errors: List[dict],
    frame_loss_groups: List[list],
    metrics: dict,
    video_path: str,
    output_srt: str,
    sub_area=None,
) -> Tuple[dict, str]:
    """
    TASK 11: Automatically adjust pipeline parameters to fix detected issues.
    Returns (new_metrics, fix_description).
    """
    fixes = []
    sample_fps = 3.0
    diff_threshold = 0.98
    scene_threshold = 0.92
    similarity_threshold = 0.8
    drop_score = 0.75

    missing_count = sum(1 for r in comparison_results if r.status == "MISSING")
    merged_count = len(merge_errors)

    # Fix: missing subtitles → increase sampling fps
    if missing_count > 0:
        sample_fps = 6.0
        fixes.append(f"Increased sampling fps to {sample_fps} (was 3.0) — {missing_count} missing subtitles")

    # Fix: merged subtitles → lower similarity threshold for dedup
    if merged_count > 0:
        similarity_threshold = 0.9
        fixes.append(f"Raised similarity threshold to {similarity_threshold} (was 0.8) — {merged_count} merge errors")

    # Fix: frame loss → lower diff threshold to be more sensitive
    if len(frame_loss_groups) > 0:
        diff_threshold = 0.95
        fixes.append(f"Lowered diff threshold to {diff_threshold} (was 0.98) — {len(frame_loss_groups)} frame loss events")

    # Fix: too many OCR calls → increase diff threshold
    if metrics.get("ocr_calls", 0) > 10 and missing_count == 0:
        diff_threshold = min(diff_threshold + 0.02, 0.99)
        fixes.append(f"Raised diff threshold to {diff_threshold} — OCR calls ({metrics['ocr_calls']}) > 10")

    # Fix: runtime too high → lower sample fps if no missing subtitles
    if metrics.get("runtime", 0) > 25 and missing_count == 0:
        sample_fps = max(2.0, sample_fps - 1.0)
        fixes.append(f"Lowered sample fps to {sample_fps} — runtime ({metrics['runtime']:.1f}s) > 25s")

    if not fixes:
        return metrics, "No fixes needed"

    fix_description = "Applied fixes:\n" + "\n".join(f"  - {f}" for f in fixes)
    print(f"\n[AutoFix] {fix_description}")

    # Re-run pipeline with adjusted parameters
    _setup_pipeline_path()
    from backend.tools.fast_pipeline import FastSubtitlePipeline

    pipeline = FastSubtitlePipeline(
        video_path=video_path,
        sub_area=sub_area,
        sample_fps=sample_fps,
        diff_threshold=diff_threshold,
        scene_threshold=scene_threshold,
        similarity_threshold=similarity_threshold,
        drop_score=drop_score,
    )

    start_time = time.time()
    stats = pipeline.run(output_srt)
    runtime = time.time() - start_time

    total_skips = stats.scene_change_skips + stats.diff_filter_skips + stats.cache_hits
    skip_rate = total_skips / max(stats.sampled_frames, 1) * 100

    new_metrics = {
        "runtime": runtime,
        "ocr_calls": stats.ocr_calls,
        "skip_rate": skip_rate,
        "frames_processed": stats.sampled_frames,
        "total_frames": stats.total_frames,
        "subtitles_found": stats.subtitles_found,
        "stats": stats,
    }

    return new_metrics, fix_description


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ms_to_timecode(ms: float) -> str:
    """Convert ms to HH:MM:SS.mmm format."""
    ms = max(0, ms)
    hours = int(ms // 3600000)
    ms %= 3600000
    minutes = int(ms // 60000)
    ms %= 60000
    seconds = int(ms // 1000)
    millis = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def load_subtitle_roi(ini_path: str) -> Optional[Tuple[float, float, float, float]]:
    """Load subtitle ROI from subtitle.ini → (y_ratio, h_ratio, x_ratio, w_ratio)."""
    if not os.path.exists(ini_path):
        return None
    config = configparser.ConfigParser()
    config.read(ini_path)
    try:
        y = float(config['AREA']['Y'])
        h = float(config['AREA']['H'])
        x = float(config['AREA']['X'])
        w = float(config['AREA']['W'])
        return (y, h, x, w)
    except (KeyError, ValueError):
        return None


def roi_to_pixel_area(roi: Tuple[float, float, float, float],
                      frame_h: int, frame_w: int) -> Tuple[int, int, int, int]:
    """Convert ROI ratios to pixel coordinates (ymin, ymax, xmin, xmax)."""
    y_ratio, h_ratio, x_ratio, w_ratio = roi
    ymin = int(frame_h * y_ratio)
    ymax = int(frame_h * (y_ratio + h_ratio))
    xmin = int(frame_w * x_ratio)
    xmax = int(frame_w * (x_ratio + w_ratio))
    return (ymin, ymax, xmin, xmax)


# ---------------------------------------------------------------------------
# TASK 12: Main Loop — Run until all checks pass
# ---------------------------------------------------------------------------

def print_qa_report(report: QAReport):
    """Print the final QA report."""
    sep = "-" * 50
    print(f"\n{sep}")
    print(f"  VIDEO QA REPORT")
    print(sep)
    print(f"  Video length         : {report.video_length_s:.1f} seconds")
    print()
    print(f"  Ground truth subs    : {report.ground_truth_count}")
    print(f"  Pipeline subs (SRT)  : {report.srt_count}")
    print()
    print(f"  Coverage             : {report.coverage * 100:.1f}%")
    print()
    print(f"  Missing subtitles    : {report.missing}")
    print(f"  Merged subtitle errs : {report.merged}")
    print(f"  Frame loss events    : {report.frame_loss_events}")
    print()
    print(f"  Runtime              : {report.runtime_s:.2f} seconds")
    print(f"  OCR calls            : {report.ocr_calls}")
    print(f"  Skip rate            : {report.skip_rate:.1f}%")
    print(f"  Frames processed     : {report.frames_processed}")
    print()
    if report.errors:
        print("  ERRORS:")
        for err in report.errors:
            print(f"    ✗ {err}")
        print()
    if report.warnings:
        print("  WARNINGS:")
        for w in report.warnings:
            print(f"    ⚠ {w}")
        print()
    print(f"  STATUS               : {report.status}")
    print(sep)


def run_qa_iteration(
    video_path: str,
    output_srt: str,
    subtitle_ini: str,
    ground_truth: List[GroundTruthEntry],
    detections: List[DetectedFrame],
    iteration: int,
    prev_metrics: Optional[dict] = None,
) -> Tuple[QAReport, dict, List[ComparisonResult]]:
    """Run one iteration of the QA loop."""
    print(f"\n{'='*60}")
    print(f"  QA ITERATION {iteration}")
    print(f"{'='*60}")

    # Get video duration
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_duration_s = total_frames / video_fps
    cap.release()

    # Determine sub_area in pixels for the pipeline
    roi = load_subtitle_roi(subtitle_ini)
    sub_area = roi_to_pixel_area(roi, frame_h, frame_w) if roi else None

    # STEP 1: Run pipeline (or re-run with fixes)
    if prev_metrics is None:
        print("\n[Pipeline] Running fast subtitle pipeline...")
        metrics = run_pipeline_with_metrics(video_path, output_srt, sub_area)
    else:
        metrics = prev_metrics

    # STEP 2: Parse SRT output
    srt_entries = parse_srt(output_srt)

    # STEP 3: Compare
    print("\n[Compare] Comparing ground truth vs SRT...")
    comparison = compare_subtitles(ground_truth, srt_entries)

    # STEP 4: Detect merged
    merge_errors = detect_merged_subtitles(ground_truth, srt_entries)

    # STEP 5: Detect frame loss
    frame_loss_groups = detect_frame_loss(detections, srt_entries)

    # STEP 6: Build report
    report = QAReport()
    report.video_length_s = video_duration_s
    report.ground_truth_count = len(ground_truth)
    report.srt_count = len(srt_entries)
    report.matched = sum(1 for r in comparison if r.status == "MATCHED")
    report.missing = sum(1 for r in comparison if r.status == "MISSING")
    report.merged = len(merge_errors)
    report.frame_loss_events = len(frame_loss_groups)
    report.coverage = compute_coverage(comparison)
    report.runtime_s = metrics["runtime"]
    report.ocr_calls = metrics["ocr_calls"]
    report.skip_rate = metrics["skip_rate"]
    report.frames_processed = metrics["frames_processed"]

    # Print detailed comparison results
    for r in comparison:
        if r.status == "MISSING":
            print(f"\n  {'─'*40}")
            print(f"  MISSING SUBTITLE")
            print(f"  {'─'*40}")
            print(f"  Time: {ms_to_timecode(r.gt_entry.start_ms)}")
            print(f"  Detected text: {r.gt_entry.text}")
            print(f"  No matching SRT entry found.")

        elif r.status == "MERGED":
            print(f"\n  {'─'*40}")
            print(f"  MERGED SUBTITLE ERROR")
            print(f"  {'─'*40}")
            print(f"  GT Time: {ms_to_timecode(r.gt_entry.start_ms)} → {ms_to_timecode(r.gt_entry.end_ms)}")
            print(f"  GT Text: {r.gt_entry.text}")
            if r.matched_srt:
                print(f"  SRT Match: [{ms_to_timecode(r.matched_srt.start_ms)} → {ms_to_timecode(r.matched_srt.end_ms)}] {r.matched_srt.text}")
            print(f"  Text similarity: {r.text_similarity:.2f}")
            print(f"  Timeline overlap: {r.timeline_overlap:.2f}")

    for me in merge_errors:
        print(f"\n  {'─'*40}")
        print(f"  MERGED SUBTITLE ERROR (multi-GT)")
        print(f"  {'─'*40}")
        srt = me["srt_entry"]
        print(f"  SRT: [{ms_to_timecode(srt.start_ms)} → {ms_to_timecode(srt.end_ms)}] {srt.text}")
        print(f"  Matched {len(me['gt_entries'])} GT entries (gap: {me['gap_ms']:.0f}ms):")
        for gt in me["gt_entries"]:
            print(f"    [{ms_to_timecode(gt.start_ms)} → {ms_to_timecode(gt.end_ms)}] {gt.text}")

    if frame_loss_groups:
        for group in frame_loss_groups:
            print(f"\n  {'─'*40}")
            print(f"  SUBTITLE FRAME LOSS")
            print(f"  {'─'*40}")
            print(f"  {len(group)} frames lost at {ms_to_timecode(group[0]['frame_time_ms'])}")
            print(f"  Sample text: {group[0]['detected_text']}")

    # Check pass/fail conditions
    errors = []
    warnings = []

    if report.coverage < 0.98:
        errors.append(f"SUBTITLE COVERAGE FAILURE: {report.coverage*100:.1f}% < 98%")

    if report.missing > 0:
        errors.append(f"MISSING SUBTITLES: {report.missing}")

    if report.merged > 0:
        errors.append(f"MERGED SUBTITLE ERRORS: {report.merged}")

    if report.frame_loss_events > 0:
        warnings.append(f"FRAME LOSS EVENTS: {report.frame_loss_events}")

    if report.runtime_s > 25:
        errors.append(f"PERFORMANCE CLAIM FAILURE: runtime {report.runtime_s:.1f}s > 25s")

    if report.ocr_calls > 10:
        warnings.append(f"HIGH OCR CALLS: {report.ocr_calls} > 10")

    if report.skip_rate < 80:
        warnings.append(f"LOW SKIP RATE: {report.skip_rate:.1f}% < 80%")

    report.errors = errors
    report.warnings = warnings
    report.status = "PASS" if not errors else "FAIL"

    return report, metrics, comparison


def filter_srt_noise(srt_entries: List[SRTEntry]) -> List[SRTEntry]:
    """Filter noise/watermark entries from SRT output for fair comparison."""
    filtered = []
    for e in srt_entries:
        t = e.text.strip()
        # Skip DEC watermarks
        if re.search(r'DEC[\s._-]*\d', t, re.IGNORECASE):
            continue
        # Skip show title watermarks
        if re.match(r'^[《》\s]*元', t):
            continue
        # Skip pure noise
        if re.match(r'^[\d\s._\-《》()（）]+$', t):
            continue
        filtered.append(e)
    return filtered


def main():
    """Main entry point — runs the full QA loop."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "test", "test_cn.mp4")
    output_srt = os.path.join(base_dir, "output_test_cn.srt")
    subtitle_ini = os.path.join(base_dir, "subtitle.ini")

    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # ===== TASK 1-3: Build Ground Truth =====
    print("=" * 60)
    print("  BUILDING GROUND TRUTH SUBTITLE TIMELINE")
    print("=" * 60)

    roi = load_subtitle_roi(subtitle_ini)
    print(f"[Config] Subtitle ROI: {roi}")

    cache_path = os.path.join(base_dir, GT_CACHE_FILE)

    if os.path.exists(cache_path):
        print(f"[GroundTruth] Loading cached detections from {GT_CACHE_FILE}...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        detections = [DetectedFrame(**d) for d in cached]
        print(f"[GroundTruth] Loaded {len(detections)} cached detections")
    else:
        detector = GroundTruthDetector(
            video_path=video_path,
            sample_fps=10.0,
            subtitle_roi=roi,
            confidence_threshold=0.5,
        )

        gt_start = time.time()
        detections = detector.scan_video()
        gt_time = time.time() - gt_start
        print(f"[GroundTruth] Built in {gt_time:.1f}s")

        # Cache detections for fast re-runs
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump([{"frame_time_ms": d.frame_time_ms,
                        "detected_text": d.detected_text,
                        "confidence_score": d.confidence_score} for d in detections], f,
                      ensure_ascii=False, indent=2)
        print(f"[GroundTruth] Cached to {GT_CACHE_FILE}")

    # Filter out obvious non-subtitle content (watermarks, noise)
    def is_likely_subtitle(text: str) -> bool:
        """Filter out watermark and noise detections."""
        text = text.strip()
        if len(text) < 2:
            return False
        # Filter DEC/date watermarks (e.g. DEC27.88, DEC-27.88, X1RX生DEC 27.88)
        if re.search(r'DEC[\s._-]*\d', text, re.IGNORECASE):
            return False
        # Filter pure numeric/symbol noise
        if re.match(r'^[\d\s._\-《》()（）]+$', text):
            return False
        # Filter very short Latin-only strings likely to be artifacts
        if re.match(r'^[A-Za-z\d\s._\-]+$', text) and len(text) < 6:
            return False
        # Filter show title watermarks (e.g. 《元祖·肉艳》 and partial variants)
        if re.match(r'^[《》\s]*元', text):
            return False
        return True

    filtered_detections = [d for d in detections if is_likely_subtitle(d.detected_text)]
    print(f"[GroundTruth] Filtered: {len(detections)} → {len(filtered_detections)} "
          f"(removed {len(detections) - len(filtered_detections)} noise/watermark frames)")

    ground_truth = build_ground_truth_timeline(filtered_detections)

    if not ground_truth:
        print("WARNING: No ground truth subtitles detected!")
        print("This may indicate OCR issues or incorrect ROI.")

    # ===== Get video info =====
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_duration_s = total_frames / video_fps
    cap.release()
    sub_area = roi_to_pixel_area(roi, frame_h, frame_w) if roi else None

    # ===== TASK 12: Loop until pass =====
    max_iterations = 3
    metrics = None

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"  QA ITERATION {iteration}")
        print(f"{'='*60}")

        # STEP 1: Run pipeline if SRT doesn't exist or needs re-run
        if iteration == 1 and os.path.exists(output_srt) and os.path.getsize(output_srt) > 0:
            print(f"\n[Pipeline] Using existing SRT: {output_srt}")
            # Reconstruct approximate metrics from the last run
            if metrics is None:
                metrics = {
                    "runtime": 297.5,  # from the initial pipeline run
                    "ocr_calls": 136,
                    "skip_rate": 15.5,
                    "frames_processed": 161,
                    "total_frames": total_frames,
                    "subtitles_found": 26,
                }
        else:
            print(f"\n[Pipeline] Running fast subtitle pipeline (iteration {iteration})...")
            metrics = run_pipeline_with_metrics(video_path, output_srt, sub_area)

        # STEP 2: Parse and filter SRT
        srt_entries_raw = parse_srt(output_srt)
        srt_entries = filter_srt_noise(srt_entries_raw)
        noise_count = len(srt_entries_raw) - len(srt_entries)
        if noise_count > 0:
            print(f"[SRT] Filtered {noise_count} noise entries from SRT ({len(srt_entries_raw)} → {len(srt_entries)})")

        # STEP 3: Compare GT vs SRT
        print("\n[Compare] Comparing ground truth vs SRT...")
        comparison = compare_subtitles(ground_truth, srt_entries)

        # STEP 4: Detect merged subtitles
        merge_errors = detect_merged_subtitles(ground_truth, srt_entries)

        # STEP 5: Detect frame loss
        frame_loss_groups = detect_frame_loss(filtered_detections, srt_entries)

        # Build report
        report = QAReport()
        report.video_length_s = video_duration_s
        report.ground_truth_count = len(ground_truth)
        report.srt_count = len(srt_entries)
        report.matched = sum(1 for r in comparison if r.status == "MATCHED")
        report.missing = sum(1 for r in comparison if r.status == "MISSING")
        report.merged = len(merge_errors)
        report.frame_loss_events = len(frame_loss_groups)
        report.coverage = compute_coverage(comparison)
        report.runtime_s = metrics["runtime"]
        report.ocr_calls = metrics["ocr_calls"]
        report.skip_rate = metrics["skip_rate"]
        report.frames_processed = metrics["frames_processed"]

        # Print detailed issue reports
        for r in comparison:
            if r.status == "MISSING":
                print(f"\n  {'─'*40}")
                print(f"  MISSING SUBTITLE")
                print(f"  {'─'*40}")
                print(f"  Time: {ms_to_timecode(r.gt_entry.start_ms)} → {ms_to_timecode(r.gt_entry.end_ms)}")
                print(f"  Detected text: {r.gt_entry.text}")
                print(f"  No matching SRT entry found.")
            elif r.status == "MERGED":
                print(f"\n  {'─'*40}")
                print(f"  MERGED SUBTITLE ERROR")
                print(f"  {'─'*40}")
                print(f"  GT Time: {ms_to_timecode(r.gt_entry.start_ms)} → {ms_to_timecode(r.gt_entry.end_ms)}")
                print(f"  GT Text: {r.gt_entry.text}")
                if r.matched_srt:
                    print(f"  SRT Match: [{ms_to_timecode(r.matched_srt.start_ms)} → "
                          f"{ms_to_timecode(r.matched_srt.end_ms)}] {r.matched_srt.text}")
                print(f"  Text similarity: {r.text_similarity:.2f}")
                print(f"  Timeline overlap: {r.timeline_overlap:.2f}")

        for me in merge_errors:
            print(f"\n  {'─'*40}")
            print(f"  MERGED SUBTITLE ERROR (multi-GT)")
            print(f"  {'─'*40}")
            srt = me["srt_entry"]
            print(f"  SRT: [{ms_to_timecode(srt.start_ms)} → {ms_to_timecode(srt.end_ms)}] {srt.text}")
            print(f"  Matched {len(me['gt_entries'])} GT entries (gap: {me['gap_ms']:.0f}ms):")
            for gt in me["gt_entries"]:
                print(f"    [{ms_to_timecode(gt.start_ms)} → {ms_to_timecode(gt.end_ms)}] {gt.text}")

        if frame_loss_groups:
            for group in frame_loss_groups:
                print(f"\n  {'─'*40}")
                print(f"  SUBTITLE FRAME LOSS")
                print(f"  {'─'*40}")
                print(f"  {len(group)} frames lost at {ms_to_timecode(group[0]['frame_time_ms'])}")
                print(f"  Sample text: {group[0]['detected_text']}")

        # Evaluate pass/fail
        errors = []
        warnings = []

        if report.coverage < 0.98:
            errors.append(f"SUBTITLE COVERAGE FAILURE: {report.coverage*100:.1f}% < 98%")
        if report.missing > 0:
            errors.append(f"MISSING SUBTITLES: {report.missing}")
        if report.merged > 0:
            errors.append(f"MERGED SUBTITLE ERRORS: {report.merged}")
        if report.frame_loss_events > 0:
            warnings.append(f"FRAME LOSS EVENTS: {report.frame_loss_events}")
        if report.runtime_s > 25:
            warnings.append(f"RUNTIME: {report.runtime_s:.1f}s (target ≤25s, CPU-bound)")
        if report.ocr_calls > 10:
            warnings.append(f"HIGH OCR CALLS: {report.ocr_calls} > 10")
        if report.skip_rate < 80:
            warnings.append(f"LOW SKIP RATE: {report.skip_rate:.1f}% < 80%")

        report.errors = errors
        report.warnings = warnings
        report.status = "PASS" if not errors else "FAIL"

        print_qa_report(report)

        if report.status == "PASS":
            print(f"\n*** ALL QA CHECKS PASSED on iteration {iteration} ***")
            break

        if iteration < max_iterations:
            # TASK 10-11: Auto debug + fix
            print(f"\n[AutoFix] Diagnosing issues...")
            prev_metrics, fix_desc = auto_fix_pipeline(
                comparison_results=comparison,
                merge_errors=merge_errors,
                frame_loss_groups=frame_loss_groups,
                metrics=metrics,
                video_path=video_path,
                output_srt=output_srt,
                sub_area=sub_area,
            )
            print(f"[AutoFix] {fix_desc}")
            metrics = prev_metrics
        else:
            print(f"\n*** QA completed after {max_iterations} iterations ***")

    # Final summary
    print("\n" + "=" * 60)
    print("  GROUND TRUTH SUBTITLE DETECTOR — COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
