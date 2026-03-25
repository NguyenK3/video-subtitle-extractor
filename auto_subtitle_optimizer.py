#!/usr/bin/env python3
"""Genetic evolution optimizer for realtime subtitle extraction."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from backend.realtime_engine import RealtimeSubtitleEngine


@dataclass
class SubtitleEvent:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "frame_sampling_base_fps": (1.0, 6.0),
    "burst_sampling_fps": (4.0, 12.0),
    "burst_duration_sec": (0.3, 2.0),
    "frame_diff_threshold": (5.0, 40.0),
    "edge_density_trigger": (0.01, 0.15),
    "OCR_confidence_threshold": (0.4, 0.9),
    "temporal_merge_similarity": (60.0, 95.0),
    "subtitle_min_duration_ms": (150.0, 800.0),
    "OCR_cache_size": (32.0, 2048.0),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def as_runtime_config(cfg: Dict[str, float]) -> Dict[str, float]:
    out = dict(cfg)
    out["temporal_merge_similarity"] = out["temporal_merge_similarity"] / 100.0
    out["OCR_cache_size"] = int(round(out["OCR_cache_size"]))
    out["subtitle_min_duration_ms"] = int(round(out["subtitle_min_duration_ms"]))
    return out


def random_config(rng: random.Random) -> Dict[str, float]:
    cfg: Dict[str, float] = {}
    for key, (lo, hi) in PARAM_RANGES.items():
        cfg[key] = rng.uniform(lo, hi)
    return cfg


def mutate_config(base: Dict[str, float], rng: random.Random, mutation_rate: float = 0.25) -> Dict[str, float]:
    cfg = dict(base)
    for key, (lo, hi) in PARAM_RANGES.items():
        if rng.random() < mutation_rate:
            sigma = (hi - lo) * 0.12
            cfg[key] = clamp(cfg[key] + rng.gauss(0.0, sigma), lo, hi)
    return cfg


def crossover_config(p1: Dict[str, float], p2: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    child: Dict[str, float] = {}
    for key in PARAM_RANGES.keys():
        child[key] = p1[key] if rng.random() < 0.5 else p2[key]
    return child


def parse_time_ms(ts: str) -> int:
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
    if not m:
        return 0
    hh, mm, ss, ms = map(int, m.groups())
    return hh * 3600000 + mm * 60000 + ss * 1000 + ms


def parse_srt(path: str) -> List[SubtitleEvent]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    blocks = re.split(r"\n\s*\n", content)
    out: List[SubtitleEvent] = []
    for block in blocks:
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0])
        except ValueError:
            continue
        m = re.match(r"(.+?)\s*-->\s*(.+)", lines[1])
        if not m:
            continue
        out.append(
            SubtitleEvent(
                index=idx,
                start_ms=parse_time_ms(m.group(1)),
                end_ms=parse_time_ms(m.group(2)),
                text=" ".join(lines[2:]).strip(),
            )
        )
    return out


def norm_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("？", "?")
    return s


def text_similarity(a: str, b: str) -> float:
    aa = norm_text(a)
    bb = norm_text(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    return SequenceMatcher(None, aa, bb).ratio()


def overlap_ratio(a: SubtitleEvent, b: SubtitleEvent) -> float:
    st = max(a.start_ms, b.start_ms)
    ed = min(a.end_ms, b.end_ms)
    overlap = max(0, ed - st)
    return overlap / max(a.duration_ms, 1)


def evaluate_prediction(gt_events: List[SubtitleEvent], pred_events: List[SubtitleEvent]) -> Dict[str, float]:
    used_pred = set()
    matched = 0
    missed = 0

    for gt in gt_events:
        best_idx = -1
        best_score = -1.0
        for i, pd in enumerate(pred_events):
            if i in used_pred:
                continue
            sim = text_similarity(gt.text, pd.text)
            ov = overlap_ratio(gt, pd)
            if sim < 0.80 or ov < 0.35:
                continue
            score = sim * 0.7 + ov * 0.3
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            used_pred.add(best_idx)
            matched += 1
        else:
            missed += 1

    false_subtitles = max(0, len(pred_events) - len(used_pred))
    coverage = matched / max(len(gt_events), 1) * 100.0
    return {
        "coverage": coverage,
        "matched": matched,
        "missed": missed,
        "false_subtitles": false_subtitles,
    }


def likely_subtitle_text(s: str) -> bool:
    t = norm_text(s)
    if len(t) < 2:
        return False
    if re.fullmatch(r"[\W_]+", t):
        return False
    return True


def write_srt(events: List[SubtitleEvent], path: str) -> None:
    def to_srt_time(ms: int) -> str:
        ms = int(max(ms, 0))
        hh = ms // 3600000
        ms -= hh * 3600000
        mm = ms // 60000
        ms -= mm * 60000
        ss = ms // 1000
        ms -= ss * 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    with open(path, "w", encoding="utf-8") as f:
        for i, ev in enumerate(events, start=1):
            f.write(f"{i}\n")
            f.write(f"{to_srt_time(ev.start_ms)} --> {to_srt_time(ev.end_ms)}\n")
            f.write(f"{ev.text}\n\n")


def build_ai_ground_truth(video_path: str, output_srt: str) -> Dict[str, float]:
    from paddleocr import PaddleOCR

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, show_log=False)

    print("[STEP 1] BUILD AI GROUND TRUTH")
    print(f"[GT] scanning full video: {total_frames} frames")

    per_frame: List[Tuple[int, str]] = []
    frame_no = 0
    ocr_calls = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        h, w = frame.shape[:2]
        y1 = int(h * 0.70)
        y2 = int(h * 0.95)
        x1 = int(w * 0.06)
        x2 = int(w * 0.94)
        roi = frame[y1:y2, x1:x2]

        ocr_calls += 1
        result = ocr.ocr(roi, cls=True)
        texts = []
        for line in (result[0] if result and result[0] else []):
            txt = line[1][0].strip()
            conf = float(line[1][1])
            if conf >= 0.55 and txt:
                texts.append(txt)
        if texts:
            text = "".join(texts)
            if likely_subtitle_text(text):
                per_frame.append((frame_no, text))

        if frame_no % 240 == 0:
            print(f"[GT] progress: {frame_no}/{total_frames}")

    cap.release()

    events: List[SubtitleEvent] = []
    cur_text = ""
    start_ms = 0.0
    end_ms = 0.0
    last_frame = -10**9
    idx = 0

    for frame_id, text in per_frame:
        ts_ms = frame_id / fps * 1000.0
        if not cur_text:
            cur_text = text
            start_ms = ts_ms
            end_ms = ts_ms
            last_frame = frame_id
            continue

        sim = text_similarity(cur_text, text)
        gap_frames = frame_id - last_frame
        if sim >= 0.82 and gap_frames <= max(1, int(fps * 0.45)):
            end_ms = ts_ms
            if len(norm_text(text)) > len(norm_text(cur_text)):
                cur_text = text
            last_frame = frame_id
            continue

        idx += 1
        if end_ms - start_ms < 150:
            end_ms = start_ms + 150
        events.append(SubtitleEvent(idx, int(start_ms), int(end_ms), cur_text))

        cur_text = text
        start_ms = ts_ms
        end_ms = ts_ms
        last_frame = frame_id

    if cur_text:
        idx += 1
        if end_ms - start_ms < 150:
            end_ms = start_ms + 150
        events.append(SubtitleEvent(idx, int(start_ms), int(end_ms), cur_text))

    write_srt(events, output_srt)
    print(f"[GT] saved {output_srt} with {len(events)} events")
    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration_sec": total_frames / fps,
        "ocr_calls": ocr_calls,
        "events": len(events),
    }


def resolve_video_path(video_arg: str) -> str:
    candidates = [video_arg, os.path.join("test", video_arg), "test/test_cn.mp4", "test_cn.mp4"]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Video not found. Tried: {candidates}")


def compute_fitness(coverage: float, runtime_sec: float, ocr_calls: int, missed: int, false_subtitles: int) -> float:
    # Higher is better: reward coverage, penalize cost/errors.
    return (
        coverage * 5.0
        - runtime_sec * 0.2
        - ocr_calls * 0.05
        - missed * 3.0
        - false_subtitles * 2.0
    )


def run_generation(
    generation_idx: int,
    population: List[Dict[str, float]],
    video_path: str,
    gt_events: List[SubtitleEvent],
    workspace_out_dir: str,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for i, cfg in enumerate(population, start=1):
        runtime_cfg = as_runtime_config(cfg)
        out_srt = os.path.join(workspace_out_dir, f"gen{generation_idx:02d}_cfg{i:02d}.srt")
        t0 = time.time()
        engine = RealtimeSubtitleEngine(
            video_path=video_path,
            lang="ch",
            mode="fast",
            tuning=runtime_cfg,
        )
        engine.run(output_srt=out_srt, metrics_interval_sec=999)
        runtime_sec = time.time() - t0

        pred_events = parse_srt(out_srt)
        cmp = evaluate_prediction(gt_events, pred_events)
        fitness = compute_fitness(
            coverage=cmp["coverage"],
            runtime_sec=runtime_sec,
            ocr_calls=engine.metrics.ocr_calls,
            missed=cmp["missed"],
            false_subtitles=cmp["false_subtitles"],
        )

        row = {
            "generation": generation_idx,
            "index": i,
            "config": cfg,
            "output_srt": out_srt,
            "runtime_sec": runtime_sec,
            "ocr_calls": int(engine.metrics.ocr_calls),
            "coverage": float(cmp["coverage"]),
            "missed_subtitles": int(cmp["missed"]),
            "false_subtitles": int(cmp["false_subtitles"]),
            "fitness": float(fitness),
        }
        results.append(row)

        print(
            f"[GEN {generation_idx:02d} | CFG {i:02d}] "
            f"coverage={row['coverage']:.2f}% runtime={row['runtime_sec']:.2f}s "
            f"ocr={row['ocr_calls']} missed={row['missed_subtitles']} "
            f"false={row['false_subtitles']} fitness={row['fitness']:.2f}"
        )
    return results


def evolve_population(
    ranked: List[Dict[str, object]],
    rng: random.Random,
    population_size: int,
    elite_size: int,
) -> List[Dict[str, float]]:
    elites = [dict(row["config"]) for row in ranked[:elite_size]]
    new_population = elites[:]
    while len(new_population) < population_size:
        p1 = rng.choice(elites)
        p2 = rng.choice(elites)
        child = crossover_config(p1, p2, rng)
        child = mutate_config(child, rng)
        new_population.append(child)
    return new_population


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous realtime subtitle genetic optimizer")
    parser.add_argument("--video", default="test_cn.mp4")
    parser.add_argument("--groundtruth", default="groundtruth.srt")
    parser.add_argument("--output", default="test_cn.realtime.srt")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--elite", type=int, default=5)
    parser.add_argument("--realtime-target", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    video_path = resolve_video_path(args.video)
    gt_meta = build_ai_ground_truth(video_path, args.groundtruth)
    gt_events = parse_srt(args.groundtruth)
    if not gt_events:
        raise RuntimeError("No ground truth events detected. Cannot evolve.")

    realtime_target = args.realtime_target
    if realtime_target is None:
        realtime_target = float(gt_meta["duration_sec"])

    os.makedirs("reports", exist_ok=True)
    workspace_out_dir = os.path.join("reports", "evolution_runs")
    os.makedirs(workspace_out_dir, exist_ok=True)

    print("\n[STEP 2] DEFINE MUTATION PARAMETERS")
    print(json.dumps(PARAM_RANGES, indent=2))

    population = [random_config(rng) for _ in range(args.population)]
    best_overall: Dict[str, object] | None = None

    for gen in range(1, args.generations + 1):
        print(f"\n[STEP 3] RUN POPULATION BENCHMARK - GENERATION {gen}")
        results = run_generation(gen, population, video_path, gt_events, workspace_out_dir)
        ranked = sorted(results, key=lambda x: x["fitness"], reverse=True)

        if best_overall is None or ranked[0]["fitness"] > best_overall["fitness"]:
            best_overall = ranked[0]

        print("[STEP 5] TOP CONFIGS")
        for i, row in enumerate(ranked[: args.elite], start=1):
            print(
                f"  TOP {i}: fitness={row['fitness']:.2f}, coverage={row['coverage']:.2f}%, "
                f"runtime={row['runtime_sec']:.2f}s, ocr={row['ocr_calls']}"
            )

        if (
            ranked[0]["coverage"] >= 98.0
            and ranked[0]["runtime_sec"] <= realtime_target
            and ranked[0]["missed_subtitles"] == 0
        ):
            print("[STEP 7] Early stop condition reached.")
            best_overall = ranked[0]
            break

        print("[STEP 6] CROSSOVER + MUTATION")
        population = evolve_population(ranked, rng, args.population, args.elite)

    if best_overall is None:
        raise RuntimeError("No evaluated configuration found.")

    best_runtime_cfg = as_runtime_config(best_overall["config"])
    print("\n[FINAL] Re-running best configuration for final output file")
    final_engine = RealtimeSubtitleEngine(
        video_path=video_path,
        lang="ch",
        mode="fast",
        tuning=best_runtime_cfg,
    )
    t0 = time.time()
    final_engine.run(output_srt=args.output, metrics_interval_sec=999)
    final_runtime = time.time() - t0
    final_eval = evaluate_prediction(gt_events, parse_srt(args.output))

    final_fitness = compute_fitness(
        coverage=final_eval["coverage"],
        runtime_sec=final_runtime,
        ocr_calls=final_engine.metrics.ocr_calls,
        missed=final_eval["missed"],
        false_subtitles=final_eval["false_subtitles"],
    )

    report = {
        "video": video_path,
        "groundtruth": args.groundtruth,
        "output": args.output,
        "best_config": best_overall["config"],
        "runtime_config": best_runtime_cfg,
        "coverage_percent": final_eval["coverage"],
        "runtime_sec": final_runtime,
        "ocr_calls": int(final_engine.metrics.ocr_calls),
        "missed_subtitles": int(final_eval["missed"]),
        "false_subtitles": int(final_eval["false_subtitles"]),
        "fitness": final_fitness,
    }

    with open("best_pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n---")
    print("GENETIC OPTIMIZER RESULT")
    print("\nBest configuration:\n")
    for k, v in best_overall["config"].items():
        if isinstance(v, float):
            print(f"{k} = {v:.4f}")
        else:
            print(f"{k} = {v}")
    print()
    print(f"Coverage = {final_eval['coverage']:.2f} %")
    print(f"Runtime = {final_runtime:.2f} sec")
    print(f"OCR calls = {int(final_engine.metrics.ocr_calls)}")
    print("---")


if __name__ == "__main__":
    main()
