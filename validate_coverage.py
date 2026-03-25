#!/usr/bin/env python3
"""Independent strict subtitle coverage validation.

Compares GT and predicted SRT with explicit matching constraints:
 - text similarity >= 0.8
 - time overlap >= 50%
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


@dataclass
class SubEvent:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


def parse_time_ms(ts: str) -> int:
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
    if not m:
        return 0
    hh, mm, ss, ms = map(int, m.groups())
    return hh * 3600000 + mm * 60000 + ss * 1000 + ms


def ms_to_time(ms: int) -> str:
    ms = max(0, int(ms))
    hh = ms // 3600000
    ms -= hh * 3600000
    mm = ms // 60000
    ms -= mm * 60000
    ss = ms // 1000
    ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def parse_srt(path: str) -> List[SubEvent]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    out: List[SubEvent] = []

    for block in blocks:
        lines = [x for x in block.splitlines() if x.strip() != ""]
        if len(lines) < 3:
            continue

        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue

        tm = re.match(r"(.+?)\s*-->\s*(.+)", lines[1].strip())
        if not tm:
            continue

        start_ms = parse_time_ms(tm.group(1))
        end_ms = parse_time_ms(tm.group(2))
        text = "\n".join(lines[2:]).strip()

        out.append(SubEvent(index=idx, start_ms=start_ms, end_ms=end_ms, text=text))

    return out


def norm_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("？", "?")
    s = s.replace("（", "(").replace("）", ")")
    return s


def text_similarity(a: str, b: str) -> float:
    aa = norm_text(a)
    bb = norm_text(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0
    return SequenceMatcher(None, aa, bb).ratio()


def overlap_ratio(gt: SubEvent, pred: SubEvent) -> float:
    st = max(gt.start_ms, pred.start_ms)
    ed = min(gt.end_ms, pred.end_ms)
    inter = max(0, ed - st)
    base = max(gt.duration_ms, 1)
    return inter / base


def find_best_match(gt: SubEvent, preds: List[SubEvent], used_pred: set) -> Tuple[Optional[int], float, float]:
    best_idx = None
    best_sim = 0.0
    best_ov = 0.0
    best_score = -1.0

    for i, p in enumerate(preds):
        if i in used_pred:
            continue
        sim = text_similarity(gt.text, p.text)
        ov = overlap_ratio(gt, p)
        if sim < 0.8 or ov < 0.5:
            continue
        score = sim * 0.7 + ov * 0.3
        if score > best_score:
            best_score = score
            best_idx = i
            best_sim = sim
            best_ov = ov

    return best_idx, best_sim, best_ov


def validate(gt_path: str, pred_path: str) -> Dict[str, object]:
    gt_events = parse_srt(gt_path)
    pred_events = parse_srt(pred_path)

    used_pred = set()
    rows = []
    matched_pairs = []

    for gt in gt_events:
        pi, sim, ov = find_best_match(gt, pred_events, used_pred)
        if pi is None:
            rows.append({
                "gt": gt,
                "pred": None,
                "status": "MISS",
                "sim": 0.0,
                "ov": 0.0,
                "offset_ms": None,
                "dur_err_ms": None,
            })
            continue

        pred = pred_events[pi]
        used_pred.add(pi)
        offset_ms = abs(pred.start_ms - gt.start_ms)
        dur_err_ms = abs(pred.duration_ms - gt.duration_ms)

        rows.append({
            "gt": gt,
            "pred": pred,
            "status": "MATCH",
            "sim": sim,
            "ov": ov,
            "offset_ms": offset_ms,
            "dur_err_ms": dur_err_ms,
        })
        matched_pairs.append((gt, pred, sim, ov, offset_ms, dur_err_ms))

    false_positive_indices = [i for i in range(len(pred_events)) if i not in used_pred]
    false_positives = [pred_events[i] for i in false_positive_indices]

    matched_count = sum(1 for r in rows if r["status"] == "MATCH")
    missed_count = len(gt_events) - matched_count
    total_gt = len(gt_events)
    coverage = matched_count / max(total_gt, 1)

    if matched_pairs:
        avg_offset = sum(x[4] for x in matched_pairs) / len(matched_pairs)
        avg_dur_err = sum(x[5] for x in matched_pairs) / len(matched_pairs)
    else:
        avg_offset = 0.0
        avg_dur_err = 0.0

    return {
        "gt_events": gt_events,
        "pred_events": pred_events,
        "rows": rows,
        "false_positives": false_positives,
        "total_gt_events": total_gt,
        "matched_events": matched_count,
        "missed_events": missed_count,
        "false_positive_events": len(false_positives),
        "avg_time_offset_ms": avg_offset,
        "avg_duration_error_ms": avg_dur_err,
        "coverage": coverage,
    }


def special_check(rows: List[Dict[str, object]], target_text: str) -> Dict[str, object]:
    target = norm_text(target_text)
    best = None
    for r in rows:
        gt: SubEvent = r["gt"]
        if target in norm_text(gt.text):
            best = r
            break
    if best is None:
        return {
            "target": target_text,
            "found_in_gt": False,
        }

    gt = best["gt"]
    pred = best["pred"]
    ov = best["ov"]
    return {
        "target": target_text,
        "found_in_gt": True,
        "gt_start": gt.start_ms,
        "gt_end": gt.end_ms,
        "pred_start": pred.start_ms if pred else None,
        "pred_end": pred.end_ms if pred else None,
        "overlap_pct": ov * 100.0,
        "status": best["status"],
    }


def write_report(path: str, result: Dict[str, object], check_a: Dict[str, object], check_b: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines: List[str] = []
    lines.append("# Coverage Validation Report")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append(f"- total_gt_events: {result['total_gt_events']}")
    lines.append(f"- matched_events: {result['matched_events']}")
    lines.append(f"- missed_events: {result['missed_events']}")
    lines.append(f"- false_positive_events: {result['false_positive_events']}")
    lines.append(f"- avg_time_offset_ms: {result['avg_time_offset_ms']:.2f}")
    lines.append(f"- avg_duration_error_ms: {result['avg_duration_error_ms']:.2f}")
    lines.append(f"- coverage: {result['coverage']*100:.2f}%")
    lines.append("")

    lines.append("## Mandatory Checks")
    lines.append("")
    for ck in [check_a, check_b]:
        lines.append(f"### {ck['target']}")
        if not ck.get("found_in_gt", False):
            lines.append("- GT event not found")
            lines.append("")
            continue
        lines.append(f"- status: {ck['status']}")
        lines.append(f"- GT start/end: {ms_to_time(ck['gt_start'])} --> {ms_to_time(ck['gt_end'])}")
        if ck["pred_start"] is None:
            lines.append("- Predicted start/end: N/A")
        else:
            lines.append(
                f"- Predicted start/end: {ms_to_time(ck['pred_start'])} --> {ms_to_time(ck['pred_end'])}"
            )
        lines.append(f"- overlap: {ck['overlap_pct']:.2f}%")
        lines.append("")

    lines.append("## Event-by-Event Diff")
    lines.append("")
    lines.append("| GT idx | GT time | GT text | Status | Pred idx | Pred time | Pred text | Sim | Overlap % | Start offset ms | Duration error ms |")
    lines.append("|---:|---|---|---|---:|---|---|---:|---:|---:|---:|")
    for r in result["rows"]:
        gt: SubEvent = r["gt"]
        pred: Optional[SubEvent] = r["pred"]
        gt_time = f"{ms_to_time(gt.start_ms)} -> {ms_to_time(gt.end_ms)}"
        if pred is None:
            lines.append(
                f"| {gt.index} | {gt_time} | {gt.text.replace('|', '\\|')} | MISS | - | - | - | 0.00 | 0.00 | - | - |"
            )
            continue

        pred_time = f"{ms_to_time(pred.start_ms)} -> {ms_to_time(pred.end_ms)}"
        lines.append(
            f"| {gt.index} | {gt_time} | {gt.text.replace('|', '\\|')} | MATCH | {pred.index} | {pred_time} | {pred.text.replace('|', '\\|')} | {r['sim']:.2f} | {r['ov']*100:.2f} | {int(r['offset_ms'])} | {int(r['dur_err_ms'])} |"
        )

    lines.append("")
    lines.append("## False Positives")
    lines.append("")
    if not result["false_positives"]:
        lines.append("None")
    else:
        lines.append("| Pred idx | Pred time | Pred text |")
        lines.append("|---:|---|---|")
        for ev in result["false_positives"]:
            t = f"{ms_to_time(ev.start_ms)} -> {ms_to_time(ev.end_ms)}"
            lines.append(f"| {ev.index} | {t} | {ev.text.replace('|', '\\|')} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict SRT coverage validator")
    ap.add_argument("--gt", default="test/test_cn.srt", help="Ground-truth SRT")
    ap.add_argument("--pred", default="test/test_cn.optimized.realtime.srt", help="Predicted SRT")
    ap.add_argument("--report", default="reports/coverage_validation.md", help="Markdown report output")
    args = ap.parse_args()

    result = validate(args.gt, args.pred)
    ck1 = special_check(result["rows"], "无所谓")
    ck2 = special_check(result["rows"], "（学小猫咪)")
    write_report(args.report, result, ck1, ck2)

    print("COVERAGE_VALIDATION")
    print(f"total_gt_events={result['total_gt_events']}")
    print(f"matched_events={result['matched_events']}")
    print(f"missed_events={result['missed_events']}")
    print(f"false_positive_events={result['false_positive_events']}")
    print(f"avg_time_offset_ms={result['avg_time_offset_ms']:.2f}")
    print(f"avg_duration_error_ms={result['avg_duration_error_ms']:.2f}")
    print(f"coverage={result['coverage']*100:.2f}%")
    print(
        "mandatory_无所谓="
        f"{ck1.get('status', 'N/A')} "
        f"GT[{ms_to_time(ck1.get('gt_start', 0))}->{ms_to_time(ck1.get('gt_end', 0))}] "
        f"PRED[{ms_to_time(ck1.get('pred_start', 0)) if ck1.get('pred_start') is not None else 'N/A'}"
        f"->{ms_to_time(ck1.get('pred_end', 0)) if ck1.get('pred_end') is not None else 'N/A'}] "
        f"overlap={ck1.get('overlap_pct', 0.0):.2f}%"
    )
    print(
        "mandatory_（学小猫咪)="
        f"{ck2.get('status', 'N/A')} "
        f"GT[{ms_to_time(ck2.get('gt_start', 0))}->{ms_to_time(ck2.get('gt_end', 0))}] "
        f"PRED[{ms_to_time(ck2.get('pred_start', 0)) if ck2.get('pred_start') is not None else 'N/A'}"
        f"->{ms_to_time(ck2.get('pred_end', 0)) if ck2.get('pred_end') is not None else 'N/A'}] "
        f"overlap={ck2.get('overlap_pct', 0.0):.2f}%"
    )
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
