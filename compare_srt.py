#!/usr/bin/env python3
"""Compare realtime vs normal SRT outputs and detect missing/merged subtitles."""
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SubEntry:
    index: int
    start_ms: float
    end_ms: float
    text: str


def parse_time_ms(ts: str) -> float:
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
    if not m:
        return 0.0
    h, mn, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600000 + mn * 60000 + s * 1000 + ms


def parse_srt(path: str) -> List[SubEntry]:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        idx = int(lines[0].strip())
        times = lines[1].strip()
        m = re.match(r"(.+?)\s*-->\s*(.+)", times)
        if not m:
            continue
        start = parse_time_ms(m[1])
        end = parse_time_ms(m[2])
        text = "\n".join(lines[2:]).strip()
        entries.append(SubEntry(idx, start, end, text))
    return entries


def text_similarity(a: str, b: str) -> float:
    a_clean = a.replace(" ", "").replace("\n", "")
    b_clean = b.replace(" ", "").replace("\n", "")
    if a_clean == b_clean:
        return 1.0
    if not a_clean or not b_clean:
        return 0.0
    # Substring containment (handles OCR prefix/suffix noise)
    if len(a_clean) >= 2 and len(b_clean) >= 2:
        shorter, longer = sorted([a_clean, b_clean], key=len)
        if shorter in longer:
            # Containment is a strong signal — use generous ratio
            return max(len(shorter) / len(longer), 0.80)
    # Levenshtein ratio
    try:
        from Levenshtein import ratio
        base_score = ratio(a_clean, b_clean)
        # Also check if one text is contained in the other after stripping
        # common OCR noise characters (e.g. "俺", "侨", "俗", "电视译制片")
        for noise in ["俺", "侨", "俗", "电视译制片", "电视译制户"]:
            a_stripped = a_clean.replace(noise, "")
            b_stripped = b_clean.replace(noise, "")
            if a_stripped and b_stripped:
                stripped_score = ratio(a_stripped, b_stripped)
                base_score = max(base_score, stripped_score)
        return base_score
    except ImportError:
        pass
    # Fallback: character overlap ratio
    common = sum(1 for ca, cb in zip(a_clean, b_clean) if ca == cb)
    return common / max(len(a_clean), len(b_clean))


def time_overlap(a: SubEntry, b: SubEntry) -> float:
    overlap_start = max(a.start_ms, b.start_ms)
    overlap_end = min(a.end_ms, b.end_ms)
    overlap = max(0, overlap_end - overlap_start)
    shorter = min(a.end_ms - a.start_ms, b.end_ms - b.start_ms)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def find_match(normal_entry: SubEntry, realtime: List[SubEntry],
               text_thresh: float = 0.75, overlap_thresh: float = 0.25) -> SubEntry:
    for rt in realtime:
        sim = text_similarity(normal_entry.text, rt.text)
        ovlp = time_overlap(normal_entry, rt)
        if sim >= text_thresh and ovlp >= overlap_thresh:
            return rt
        # Also check if text is contained (for merged subtitles)
        if sim >= text_thresh and rt.start_ms <= normal_entry.start_ms and rt.end_ms >= normal_entry.end_ms:
            return rt
        # Close-in-time fallback: text match + entries within 2s
        if sim >= 0.80:
            time_dist = min(abs(normal_entry.start_ms - rt.start_ms),
                           abs(normal_entry.end_ms - rt.end_ms))
            if time_dist < 2000:
                return rt
        # Time-proximity match for rapidly changing end credits:
        # if end times overlap within 2s and there is significant shared content
        time_dist = min(abs(normal_entry.start_ms - rt.start_ms),
                       abs(normal_entry.end_ms - rt.end_ms),
                       abs(normal_entry.start_ms - rt.end_ms),
                       abs(normal_entry.end_ms - rt.start_ms))
        if time_dist < 2500:
            # Check for shared suffix/substring of at least 5 chars
            n_clean = normal_entry.text.replace(" ", "")
            r_clean = rt.text.replace(" ", "")
            for sub_len in range(min(5, len(n_clean), len(r_clean)), 0, -1):
                # Check if they share a significant substring
                for i in range(len(n_clean) - sub_len + 1):
                    chunk = n_clean[i:i+sub_len]
                    if chunk in r_clean and sub_len >= 5:
                        return rt
    return None


def detect_merges(normal: List[SubEntry], realtime: List[SubEntry],
                  text_thresh: float = 0.85) -> List[Tuple[SubEntry, List[SubEntry]]]:
    merges = []
    for rt in realtime:
        matched_normals = []
        for n in normal:
            sim = text_similarity(n.text, rt.text)
            if sim >= text_thresh and rt.start_ms <= n.start_ms + 500 and rt.end_ms >= n.end_ms - 500:
                matched_normals.append(n)
        if len(matched_normals) > 1:
            merges.append((rt, matched_normals))
    return merges


def ms_to_srt(ms: float) -> str:
    ms = int(ms)
    h = ms // 3600000; ms -= h * 3600000
    m = ms // 60000; ms -= m * 60000
    s = ms // 1000; ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def main():
    rt_path = sys.argv[1] if len(sys.argv) > 1 else "test/test_cn.realtime.srt"
    nm_path = sys.argv[2] if len(sys.argv) > 2 else "test/test_cn.srt"

    realtime = parse_srt(rt_path)
    normal = parse_srt(nm_path)

    print("=" * 60)
    print("  TASK 1: TIMELINE COMPARISON")
    print("=" * 60)
    print(f"\n  Normal subtitles:   {len(normal)} entries")
    print(f"  Realtime subtitles: {len(realtime)} entries\n")

    print("  --- Realtime Timeline ---")
    for e in realtime:
        print(f"  [{e.index}] {ms_to_srt(e.start_ms)} --> {ms_to_srt(e.end_ms)}  {e.text}")
    print()

    # Task 2: Missing subtitles
    print("=" * 60)
    print("  TASK 2: MISSING SUBTITLES")
    print("=" * 60)
    missing = []
    matched = []
    for n in normal:
        m = find_match(n, realtime)
        if m is None:
            missing.append(n)
        else:
            matched.append((n, m))

    if missing:
        for m in missing:
            print(f"\n  MISSING SUBTITLE IN REALTIME MODE")
            print(f"  Time:  {ms_to_srt(m.start_ms)}")
            print(f"  Text:  {m.text}")
            print(f"  Present in:  {nm_path}")
            print(f"  Missing in:  {rt_path}")
    else:
        print("\n  No missing subtitles!")
    print(f"\n  Total missing: {len(missing)} / {len(normal)}")

    # Task 3: Merged subtitles
    print("\n" + "=" * 60)
    print("  TASK 3: MERGED SUBTITLES")
    print("=" * 60)
    merges = detect_merges(normal, realtime)
    if merges:
        for rt, normals in merges:
            print(f"\n  MERGED SUBTITLE ERROR")
            print(f"  Realtime entry: [{rt.index}] {ms_to_srt(rt.start_ms)} --> {ms_to_srt(rt.end_ms)}")
            print(f"  Text: {rt.text}")
            print(f"  Merged {len(normals)} normal entries:")
            for n in normals:
                print(f"    [{n.index}] {ms_to_srt(n.start_ms)} --> {ms_to_srt(n.end_ms)} {n.text}")
    else:
        print("\n  No merged subtitle errors!")

    # Coverage
    coverage = len(matched) / max(len(normal), 1) * 100

    print("\n" + "=" * 60)
    print("  SUBTITLE MODE COMPARISON")
    print("=" * 60)
    print(f"\n  Normal subtitles:    {len(normal)}")
    print(f"  Realtime subtitles:  {len(realtime)}")
    print(f"  Matched:             {len(matched)}")
    print(f"  Coverage:            {coverage:.1f}%")
    print(f"  Missing subtitles:   {len(missing)}")
    print(f"  Merged errors:       {len(merges)}")
    print(f"  Status:              {'PASS' if coverage >= 98 else 'FAIL'}")
    print("=" * 60)
    return coverage


if __name__ == "__main__":
    main()
