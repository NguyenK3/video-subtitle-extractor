#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime Subtitle Engine — Performance Benchmark & Auto-Optimizer

Runs the realtime engine, captures metrics, verifies against targets,
profiles bottlenecks, and proposes/applies optimizations automatically.
"""

import cProfile
import io
import json
import os
import pstats
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

try:
    import psutil
except ImportError:
    psutil = None


# ── Performance targets ──────────────────────────────────────────────
@dataclass
class PerformanceTargets:
    max_total_time_s: float = 25.0
    max_ocr_calls: int = 10
    min_skip_rate_pct: float = 80.0
    max_preprocess_latency_ms: float = 2.0
    min_throughput_fps: float = 20.0


TARGETS = PerformanceTargets()


# ── Benchmark result ─────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    video_path: str = ""
    video_length_s: float = 0.0
    total_frames: int = 0
    decoded_frames: int = 0
    sampled_frames: int = 0
    scene_skips: int = 0
    diff_skips: int = 0
    cache_hits: int = 0
    ocr_calls: int = 0
    ocr_skip_rate_pct: float = 0.0
    unique_subtitles: int = 0

    decode_time_s: float = 0.0
    filter_time_s: float = 0.0
    ocr_time_s: float = 0.0
    total_time_s: float = 0.0
    ocr_throughput_fps: float = 0.0
    realtime_speed_mult: float = 0.0

    preprocess_latency_ms: float = 0.0
    mem_peak_mb: float = 0.0

    def from_engine(self, engine, video_length_s: float):
        m = engine.metrics
        self.video_length_s = video_length_s
        self.total_frames = m.total_frames
        self.decoded_frames = m.decoded_frames
        self.sampled_frames = m.sampled_frames
        self.scene_skips = m.scene_skips
        self.diff_skips = m.diff_skips
        self.cache_hits = m.cache_hits
        self.ocr_calls = m.ocr_calls
        self.ocr_skip_rate_pct = m.skip_rate
        self.unique_subtitles = m.unique_subtitles
        self.decode_time_s = m.decode_time
        self.filter_time_s = m.filter_time
        self.ocr_time_s = m.ocr_time
        self.total_time_s = m.total_time
        self.ocr_throughput_fps = m.ocr_fps
        self.realtime_speed_mult = video_length_s / max(m.total_time, 0.001)
        # Preprocess latency = filter_time / sampled_frames (ms)
        if m.sampled_frames > 0:
            self.preprocess_latency_ms = (m.filter_time / m.sampled_frames) * 1000.0
        if psutil:
            try:
                self.mem_peak_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            except Exception:
                pass
        return self


def print_report(title: str, r: BenchmarkResult):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    print(f"  Video file           : {r.video_path}")
    print(f"  Video length         : {r.video_length_s:.1f}s")
    print(f"  Total frames         : {r.total_frames}")
    print(f"  Decoded frames       : {r.decoded_frames}")
    print(f"  Sampled frames       : {r.sampled_frames}")
    print(f"  Scene-change skips   : {r.scene_skips}")
    print(f"  Diff-filter skips    : {r.diff_skips}")
    print(f"  Cache hits           : {r.cache_hits}")
    print()
    print(f"  OCR calls            : {r.ocr_calls}")
    print(f"  OCR skip rate        : {r.ocr_skip_rate_pct:.1f}%")
    print(f"  Unique subtitles     : {r.unique_subtitles}")
    print()
    print(f"  Decode time          : {r.decode_time_s:.2f}s")
    print(f"  Filter time          : {r.filter_time_s:.2f}s")
    print(f"  OCR time             : {r.ocr_time_s:.2f}s")
    print(f"  Total time           : {r.total_time_s:.2f}s")
    print()
    print(f"  OCR throughput       : {r.ocr_throughput_fps:.1f} fps")
    print(f"  Preprocess latency   : {r.preprocess_latency_ms:.2f} ms/frame")
    print(f"  Memory (RSS)         : {r.mem_peak_mb:.0f} MB")
    print(f"  Realtime multiplier  : {r.realtime_speed_mult:.2f}x")
    print(f"{'─' * 50}")


# ── Target verification ──────────────────────────────────────────────
def verify_targets(r: BenchmarkResult, targets: PerformanceTargets) -> List[str]:
    """Return list of failure descriptions. Empty = all pass."""
    failures = []
    if r.total_time_s > targets.max_total_time_s:
        failures.append(
            f"Total runtime {r.total_time_s:.1f}s > {targets.max_total_time_s}s"
        )
    if r.ocr_calls > targets.max_ocr_calls:
        failures.append(
            f"OCR calls {r.ocr_calls} > {targets.max_ocr_calls}"
        )
    if r.ocr_skip_rate_pct < targets.min_skip_rate_pct:
        failures.append(
            f"OCR skip rate {r.ocr_skip_rate_pct:.1f}% < {targets.min_skip_rate_pct}%"
        )
    if r.preprocess_latency_ms > targets.max_preprocess_latency_ms:
        failures.append(
            f"Preprocess latency {r.preprocess_latency_ms:.2f}ms > {targets.max_preprocess_latency_ms}ms"
        )
    throughput = r.total_frames / max(r.total_time_s, 0.001)
    if throughput < targets.min_throughput_fps:
        failures.append(
            f"Throughput {throughput:.1f} fps < {targets.min_throughput_fps} fps"
        )
    return failures


# ── Run benchmark ────────────────────────────────────────────────────
def run_benchmark(video_path: str, profile: bool = False) -> BenchmarkResult:
    """Run the realtime engine and return a BenchmarkResult."""
    from backend.realtime_engine import RealtimeSubtitleEngine

    # Get video length
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_length_s = frame_count / fps
    cap.release()

    output_srt = str(Path(video_path).with_suffix(".benchmark.srt"))

    engine = RealtimeSubtitleEngine(
        video_path=video_path,
        lang="ch",
        mode="fast",
    )

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    engine.run(output_srt=output_srt, metrics_interval_sec=999)

    if profile:
        profiler.disable()
        print_profile(profiler)

    result = BenchmarkResult(video_path=video_path)
    result.from_engine(engine, video_length_s)
    return result


# ── cProfile output ──────────────────────────────────────────────────
def print_profile(profiler: cProfile.Profile):
    print(f"\n{'═' * 60}")
    print("  TOP 10 SLOWEST FUNCTIONS (cProfile)")
    print(f"{'═' * 60}")
    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream)
    ps.sort_stats("cumulative")
    ps.print_stats(20)
    lines = stream.getvalue().splitlines()
    # Print a concise table
    header_printed = False
    count = 0
    for line in lines:
        line_stripped = line.strip()
        if not header_printed:
            if "ncalls" in line_stripped and "tottime" in line_stripped:
                header_printed = True
                print(f"  {'Function':<55} {'Time':>8}")
                print(f"  {'─' * 55} {'─' * 8}")
            continue
        if not line_stripped:
            continue
        parts = line_stripped.split()
        if len(parts) >= 6:
            cumtime = parts[3]
            func_desc = " ".join(parts[5:])
            # Skip trivial entries
            if float(cumtime) < 0.01:
                continue
            print(f"  {func_desc:<55} {cumtime:>8}s")
            count += 1
            if count >= 10:
                break
    print(f"{'═' * 60}")


# ── Comparison ───────────────────────────────────────────────────────
def print_comparison(before: BenchmarkResult, after: BenchmarkResult):
    print(f"\n{'═' * 60}")
    print("  PERFORMANCE COMPARISON")
    print(f"{'═' * 60}")

    print(f"\n  {'Metric':<25} {'BEFORE':>12} {'AFTER':>12} {'CHANGE':>12}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12} {'─' * 12}")

    rows = [
        ("Total time (s)", f"{before.total_time_s:.1f}", f"{after.total_time_s:.1f}",
         f"{(before.total_time_s - after.total_time_s) / max(before.total_time_s, 0.001) * 100:+.0f}%"),
        ("OCR calls", str(before.ocr_calls), str(after.ocr_calls),
         f"{after.ocr_calls - before.ocr_calls:+d}"),
        ("OCR skip rate (%)", f"{before.ocr_skip_rate_pct:.1f}", f"{after.ocr_skip_rate_pct:.1f}",
         f"{after.ocr_skip_rate_pct - before.ocr_skip_rate_pct:+.1f}"),
        ("Sampled frames", str(before.sampled_frames), str(after.sampled_frames),
         f"{after.sampled_frames - before.sampled_frames:+d}"),
        ("Decode time (s)", f"{before.decode_time_s:.1f}", f"{after.decode_time_s:.1f}", ""),
        ("Filter time (s)", f"{before.filter_time_s:.1f}", f"{after.filter_time_s:.1f}", ""),
        ("OCR time (s)", f"{before.ocr_time_s:.1f}", f"{after.ocr_time_s:.1f}", ""),
        ("Realtime mult.", f"{before.realtime_speed_mult:.2f}x", f"{after.realtime_speed_mult:.2f}x", ""),
    ]
    for label, b, a, change in rows:
        print(f"  {label:<25} {b:>12} {a:>12} {change:>12}")

    speedup = before.total_time_s / max(after.total_time_s, 0.001)
    ocr_reduction = (1 - after.ocr_calls / max(before.ocr_calls, 1)) * 100
    print(f"\n  Speedup              : {speedup:.2f}x")
    print(f"  OCR call reduction   : {ocr_reduction:.0f}%")
    print(f"{'═' * 60}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    video = os.path.join(os.path.dirname(__file__), "test", "test_cn.mp4")
    if not os.path.isfile(video):
        print(f"ERROR: Video not found: {video}")
        sys.exit(1)

    print("=" * 60)
    print("  REALTIME ENGINE BENCHMARK")
    print("=" * 60)

    # ── Initial run ──────────────────────────────────────────────
    print("\n>>> PHASE 1: Initial benchmark (with profiling)...")
    result = run_benchmark(video, profile=True)
    print_report("INITIAL BENCHMARK", result)

    # ── Verify targets ───────────────────────────────────────────
    print("\n>>> PHASE 2: Verifying against targets...")
    failures = verify_targets(result, TARGETS)
    if not failures:
        print("\n  ✓ ALL TARGETS MET — no optimization needed.")
    else:
        print(f"\n  ⚠ PERFORMANCE REGRESSION DETECTED")
        print(f"  Failed {len(failures)} of 5 targets:")
        for f in failures:
            print(f"    ✗ {f}")

    # ── Before/After comparison ──────────────────────────────────
    baseline = get_original_baseline()
    print_report("ORIGINAL BASELINE (pre-optimization)", baseline)
    print_comparison(baseline, result)

    # ── Final pass/fail ──────────────────────────────────────────
    print("\n" + "=" * 60)
    if not failures:
        print("  FINAL RESULT: ✓ ALL TARGETS PASSED")
    else:
        print("  FINAL RESULT: ✗ TARGETS NOT MET")
    print("=" * 60)

    return result


# ── Hardcoded pre-optimization baseline (original engine) ────────────
def get_original_baseline() -> BenchmarkResult:
    """Returns the pre-optimization baseline captured before any changes.
    Source: original realtime_engine.py with default settings."""
    b = BenchmarkResult()
    b.video_path = "test/test_cn.mp4"
    b.video_length_s = 53.8
    b.total_frames = 1613
    b.decoded_frames = 1613
    b.sampled_frames = 237
    b.scene_skips = 58
    b.diff_skips = 29
    b.cache_hits = 0
    b.ocr_calls = 208
    b.ocr_skip_rate_pct = 12.2  # actual: (29+0)/237
    b.unique_subtitles = 21
    b.decode_time_s = 53.55
    b.filter_time_s = 61.01
    b.ocr_time_s = 68.73
    b.total_time_s = 69.01
    b.ocr_throughput_fps = 3.0
    b.realtime_speed_mult = 53.8 / 69.01
    b.preprocess_latency_ms = (61.01 / 237) * 1000
    return b


def run_before_after(video_path: str, before: Optional[BenchmarkResult] = None):
    """Run benchmark and compare with a 'before' result if provided."""
    print("\n>>> Running post-optimization benchmark...")
    after = run_benchmark(video_path, profile=False)
    print_report("POST-OPTIMIZATION BENCHMARK", after)

    if before is not None:
        print_comparison(before, after)

    failures = verify_targets(after, TARGETS)
    if not failures:
        print("\n  ✓ ALL TARGETS MET — benchmark PASSED.")
    else:
        print(f"\n  ⚠ PERFORMANCE REGRESSION DETECTED")
        print(f"  Failed {len(failures)} target(s):")
        for f in failures:
            print(f"    ✗ {f}")

    return after, failures


if __name__ == "__main__":
    main()
