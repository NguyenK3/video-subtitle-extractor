import argparse
import importlib
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from backend import config
from backend.tools.ocr import OcrRecogniser

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None

try:
    import psutil
except Exception:
    psutil = None


@dataclass
class FramePacket:
    frame_no: int
    time_ms: float
    frame: np.ndarray


@dataclass
class OCRItem:
    frame_no: int
    time_ms: float
    crop: np.ndarray


@dataclass
class OCRResult:
    frame_no: int
    time_ms: float
    text: str


class PerformanceMonitor:
    def __init__(self):
        self.t0 = time.time()
        self.decoded = 0
        self.analyzed = 0
        self.ocr_count = 0
        self.lock = threading.Lock()

    def inc(self, attr, v=1):
        with self.lock:
            setattr(self, attr, getattr(self, attr) + v)

    def snapshot(self):
        with self.lock:
            elapsed = max(time.time() - self.t0, 1e-6)
            mem_mb = 0.0
            cpu_pct = 0.0
            if psutil is not None:
                p = psutil.Process(os.getpid())
                mem_mb = p.memory_info().rss / 1024 / 1024
                cpu_pct = p.cpu_percent(interval=0.0)
            return {
                "elapsed_sec": round(elapsed, 2),
                "decoded": self.decoded,
                "analyzed": self.analyzed,
                "ocr_count": self.ocr_count,
                "analysis_fps": round(self.analyzed / elapsed, 2),
                "ocr_tps": round(self.ocr_count / elapsed, 2),
                "cpu_pct": round(cpu_pct, 2),
                "mem_mb": round(mem_mb, 2),
            }


class RealtimeSubtitleEngine:
    def __init__(
        self,
        video_path,
        lang="ch",
        mode="fast",
        default_interval_sec=0.8,
        min_interval_sec=0.2,
        max_interval_sec=1.2,
        batch_size=8,
        hash_threshold=6,
        diff_threshold=7.0,
        ssim_threshold=0.93,
    ):
        self.video_path = video_path
        self.lang = lang
        self.mode = mode
        self.default_interval_sec = default_interval_sec
        self.min_interval_sec = min_interval_sec
        self.max_interval_sec = max_interval_sec
        self.current_interval_sec = default_interval_sec
        self.batch_size = batch_size
        self.hash_threshold = hash_threshold
        self.diff_threshold = diff_threshold
        self.ssim_threshold = ssim_threshold

        self.monitor = PerformanceMonitor()
        self.stop_event = threading.Event()

        self.decode_q = queue.Queue(maxsize=64)
        self.ocr_q = queue.Queue(maxsize=64)
        self.result_q = queue.Queue(maxsize=128)

        self.subtitle_bbox = None
        self.last_hash = None
        self.last_small = None

        self.timeline = []
        self.last_event = None

        self.use_cuda_preprocess = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

    def _update_runtime_settings(self):
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.ini")
        with open(settings_path, "w", encoding="utf-8") as f:
            f.write("[DEFAULT]\n")
            f.write("Interface = 简体中文\n")
            f.write(f"Language = {self.lang}\n")
            f.write(f"Mode = {self.mode}\n")
        importlib.reload(config)

    @staticmethod
    def _to_srt_time(ms):
        ms = int(max(0, ms))
        h = ms // 3600000
        ms -= h * 3600000
        m = ms // 60000
        ms -= m * 60000
        s = ms // 1000
        ms -= s * 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _dhash(gray):
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return diff.astype(np.uint8).flatten()

    @staticmethod
    def _hamming(a, b):
        if a is None or b is None:
            return 999
        return int(np.count_nonzero(a != b))

    def _detect_subtitle_region(self, frame):
        h, w = frame.shape[:2]
        lower = frame[int(h * 0.45):, :]
        gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 80, 180)
        row_energy = edges.mean(axis=1)
        if row_energy.max() <= 0:
            return int(h * 0.70), int(h * 0.98), int(w * 0.03), int(w * 0.97)
        thr = max(8.0, float(np.percentile(row_energy, 80)))
        rows = np.where(row_energy >= thr)[0]
        if len(rows) == 0:
            return int(h * 0.70), int(h * 0.98), int(w * 0.03), int(w * 0.97)
        y1 = int(h * 0.45) + max(0, int(rows.min()) - 20)
        y2 = int(h * 0.45) + min(lower.shape[0] - 1, int(rows.max()) + 35)
        x1 = int(w * 0.03)
        x2 = int(w * 0.97)
        y1 = max(0, min(y1, h - 2))
        y2 = max(y1 + 1, min(y2, h - 1))
        return y1, y2, x1, x2

    def _preprocess_crop(self, crop):
        if self.use_cuda_preprocess:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(crop)
            gpu_gray = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)
            gpu_resize = cv2.cuda.resize(gpu_gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = gpu_resize.download()
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        denoise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        binary = cv2.adaptiveThreshold(
            denoise,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), binary

    def _is_changed(self, small_gray):
        curr_hash = self._dhash(small_gray)
        ham = self._hamming(curr_hash, self.last_hash)

        mad = 255.0
        ssim_score = 0.0
        if self.last_small is not None and self.last_small.shape == small_gray.shape:
            mad = float(np.mean(np.abs(small_gray.astype(np.int16) - self.last_small.astype(np.int16))))
            if ssim is not None:
                ssim_score = float(ssim(self.last_small, small_gray, data_range=255))

        changed = (ham >= self.hash_threshold) or (mad >= self.diff_threshold)
        if ssim is not None:
            changed = changed or (ssim_score <= self.ssim_threshold)

        self.last_hash = curr_hash
        self.last_small = small_gray.copy()
        return changed

    def _decoder_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_no = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_no += 1
                time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                pkt = FramePacket(frame_no=frame_no, time_ms=time_ms, frame=frame)
                self.decode_q.put(pkt)
                self.monitor.inc("decoded")

                step = max(1, int(self.current_interval_sec * fps) - 1)
                for _ in range(step):
                    if not cap.grab():
                        break
                    frame_no += 1
            self.decode_q.put(None)
        finally:
            cap.release()

    def _analyzer_loop(self):
        while not self.stop_event.is_set():
            pkt = self.decode_q.get()
            if pkt is None:
                self.ocr_q.put(None)
                return
            try:
                frame = pkt.frame
                if self.subtitle_bbox is None:
                    self.subtitle_bbox = self._detect_subtitle_region(frame)

                y1, y2, x1, x2 = self.subtitle_bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                proc_bgr, proc_gray = self._preprocess_crop(crop)
                tiny = cv2.resize(proc_gray, (160, 48), interpolation=cv2.INTER_AREA)
                changed = self._is_changed(tiny)

                if changed:
                    self.current_interval_sec = max(self.min_interval_sec, self.current_interval_sec * 0.7)
                    self.ocr_q.put(OCRItem(pkt.frame_no, pkt.time_ms, proc_bgr))
                else:
                    self.current_interval_sec = min(self.max_interval_sec, self.current_interval_sec * 1.12)

                self.monitor.inc("analyzed")
            except Exception as e:
                print(f"[Analyzer] skip frame {pkt.frame_no}: {e}")

    def _ocr_loop(self):
        self._update_runtime_settings()
        recogniser = OcrRecogniser()
        batch = []
        last_flush = time.time()

        def flush_batch():
            if not batch:
                return
            for item in batch:
                try:
                    dt_box, rec_res = recogniser.predict(item.crop)
                    texts = [x[0] for x in rec_res] if rec_res else []
                    text = " ".join(t.strip() for t in texts if t and t.strip())
                    self.result_q.put(OCRResult(item.frame_no, item.time_ms, text))
                    self.monitor.inc("ocr_count")
                except Exception as oe:
                    print(f"[OCR] frame {item.frame_no} failed: {oe}")
            batch.clear()

        while not self.stop_event.is_set():
            item = self.ocr_q.get()
            if item is None:
                flush_batch()
                self.result_q.put(None)
                return
            batch.append(item)
            if len(batch) >= self.batch_size or (time.time() - last_flush) >= 0.25:
                flush_batch()
                last_flush = time.time()

    def _aggregator_loop(self):
        while not self.stop_event.is_set():
            result = self.result_q.get()
            if result is None:
                if self.last_event is not None:
                    self.timeline.append(self.last_event)
                    self.last_event = None
                return
            txt = (result.text or "").strip()
            if not txt:
                continue
            if self.last_event is None:
                self.last_event = {
                    "start_ms": result.time_ms,
                    "end_ms": result.time_ms,
                    "text": txt,
                }
                continue

            if txt == self.last_event["text"]:
                self.last_event["end_ms"] = result.time_ms
            else:
                self.timeline.append(self.last_event)
                self.last_event = {
                    "start_ms": result.time_ms,
                    "end_ms": result.time_ms,
                    "text": txt,
                }

    def run(self, output_srt=None, metrics_interval_sec=5.0):
        if output_srt is None:
            output_srt = str(Path(self.video_path).with_suffix(".realtime.srt"))

        t_decoder = threading.Thread(target=self._decoder_loop, daemon=True)
        t_analyzer = threading.Thread(target=self._analyzer_loop, daemon=True)
        t_ocr = threading.Thread(target=self._ocr_loop, daemon=True)
        t_agg = threading.Thread(target=self._aggregator_loop, daemon=True)

        t_decoder.start()
        t_analyzer.start()
        t_ocr.start()
        t_agg.start()

        last_report = 0.0
        while t_agg.is_alive():
            time.sleep(0.5)
            now = time.time()
            if now - last_report >= metrics_interval_sec:
                m = self.monitor.snapshot()
                print(
                    f"[Metrics] elapsed={m['elapsed_sec']}s analyzed={m['analyzed']} "
                    f"analysis_fps={m['analysis_fps']} ocr={m['ocr_count']} ocr_tps={m['ocr_tps']} "
                    f"cpu={m['cpu_pct']}% mem={m['mem_mb']}MB"
                )
                last_report = now

        self.stop_event.set()
        t_decoder.join(timeout=1)
        t_analyzer.join(timeout=1)
        t_ocr.join(timeout=1)
        t_agg.join(timeout=1)

        self._write_srt(output_srt)
        print(f"[Realtime] subtitle saved: {output_srt}")
        return output_srt

    def _write_srt(self, output_srt):
        os.makedirs(os.path.dirname(output_srt) or ".", exist_ok=True)
        with open(output_srt, "w", encoding="utf-8") as f:
            for i, it in enumerate(self.timeline, start=1):
                start_ms = it["start_ms"]
                end_ms = max(it["end_ms"], start_ms + 800)
                f.write(f"{i}\n")
                f.write(f"{self._to_srt_time(start_ms)} --> {self._to_srt_time(end_ms)}\n")
                f.write(f"{it['text']}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Realtime subtitle OCR engine")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--lang", default="ch", help="OCR language, e.g. ch/en/japan/korean")
    parser.add_argument("--mode", default="fast", choices=["fast", "auto", "accurate"])
    parser.add_argument("--out", default=None, help="Output srt path")
    parser.add_argument("--sample", type=float, default=0.8, help="Default sample interval seconds")
    parser.add_argument("--min-sample", type=float, default=0.2)
    parser.add_argument("--max-sample", type=float, default=1.2)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    engine = RealtimeSubtitleEngine(
        video_path=args.video,
        lang=args.lang,
        mode=args.mode,
        default_interval_sec=args.sample,
        min_interval_sec=args.min_sample,
        max_interval_sec=args.max_sample,
        batch_size=args.batch_size,
    )
    engine.run(output_srt=args.out)


if __name__ == "__main__":
    main()
