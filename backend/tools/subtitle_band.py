# -*- coding: utf-8 -*-
"""
Subtitle Band Detector — ROI filtering for subtitle-only OCR.

Enforces a strict subtitle band to prevent OCR contamination from
non-subtitle text (shirt prints, logos, watermarks, background signs).

Pipeline:
    1. Static subtitle band: 75%-95% of frame height
    2. Red background detection (for styled subtitles)
    3. Contour-based subtitle box detection
    4. Strict OCR boundary with padding
    5. Hard rule: ignore anything above 70% of frame height
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np


class SubtitleBandDetector:
    """Detects and crops the subtitle region within a strict vertical band."""

    # Static band boundaries (fraction of frame height)
    # FIX 3: Strict subtitle region 70%-92%
    BAND_TOP = 0.70
    BAND_BOTTOM = 0.92

    # Hard ceiling: anything above this is never subtitle
    HARD_CEILING = 0.70

    # FIX 4: 92%-100% is excluded (lower HUD area with timestamps/overlays)

    # Subtitle box constraints (pixels)
    MIN_BOX_WIDTH_RATIO = 0.25   # min 25% of frame width
    MIN_BOX_HEIGHT = 40
    MAX_BOX_HEIGHT = 120

    # OCR padding around detected subtitle box
    OCR_PADDING = 5

    # Debug output directory
    DEBUG_DIR = "debug"

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._frame_count = 0
        if self.debug:
            os.makedirs(self.DEBUG_DIR, exist_ok=True)

    def get_subtitle_band(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Return the static subtitle band as (y1, y2, x1, x2)."""
        h, w = frame.shape[:2]
        y1 = int(h * self.BAND_TOP)
        y2 = int(h * self.BAND_BOTTOM)
        x1 = 0
        x2 = w
        return y1, y2, x1, x2

    def crop_subtitle_band(self, frame: np.ndarray) -> np.ndarray:
        """Crop the static subtitle band from a full frame."""
        y1, y2, x1, x2 = self.get_subtitle_band(frame)
        return frame[y1:y2, x1:x2].copy()

    def detect_red_subtitle_box(self, band: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect red background subtitle region within the band.

        Returns (y1, y2, x1, x2) relative to band, or None if not found.
        """
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        h_band, w_band = band.shape[:2]

        # Red hue wraps around 0/180 in HSV
        # Lower red range: H=0-10, S=70-255, V=50-255
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        # Upper red range: H=160-180
        mask2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find the largest red contour that matches subtitle dimensions
        best = None
        best_area = 0
        min_w = int(w_band * self.MIN_BOX_WIDTH_RATIO)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= min_w and self.MIN_BOX_HEIGHT <= h <= self.MAX_BOX_HEIGHT:
                area = w * h
                if area > best_area:
                    best_area = area
                    best = (y, y + h, x, x + w)

        return best

    def detect_subtitle_contour(self, band: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect subtitle box via contour analysis within the band.

        Returns (y1, y2, x1, x2) relative to band, or None if not found.
        """
        h_band, w_band = band.shape[:2]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if len(band.shape) == 3 else band

        # Adaptive threshold to find text/box regions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

        # Horizontal morphological close to merge text into lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_w = int(w_band * self.MIN_BOX_WIDTH_RATIO)
        candidates = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= min_w and self.MIN_BOX_HEIGHT <= h <= self.MAX_BOX_HEIGHT:
                # Prefer boxes near horizontal center
                center_x = x + w // 2
                dist_from_center = abs(center_x - w_band // 2)
                candidates.append((dist_from_center, x, y, w, h))

        if not candidates:
            return None

        # Merge nearby candidates into one bounding box
        all_x1 = min(c[1] for c in candidates)
        all_y1 = min(c[2] for c in candidates)
        all_x2 = max(c[1] + c[3] for c in candidates)
        all_y2 = max(c[2] + c[4] for c in candidates)

        return all_y1, all_y2, all_x1, all_x2

    def detect_and_crop(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Full pipeline: detect subtitle region and return tight crop.

        Returns:
            crop: The cropped subtitle region for OCR
            bbox: (y1, y2, x1, x2) absolute coordinates in original frame
        """
        h, w = frame.shape[:2]
        self._frame_count += 1

        # Step 1: Static subtitle band (75%-95%)
        band_y1, band_y2, band_x1, band_x2 = self.get_subtitle_band(frame)
        band = frame[band_y1:band_y2, band_x1:band_x2]

        if band.size == 0:
            return band, (band_y1, band_y2, band_x1, band_x2)

        # Step 2: Try red background detection
        red_box = self.detect_red_subtitle_box(band)

        # Step 3: Try contour detection
        contour_box = self.detect_subtitle_contour(band)

        # Choose the tightest valid box
        box = red_box or contour_box

        if box is not None:
            by1, by2, bx1, bx2 = box

            # Apply padding
            by1 = max(0, by1 - self.OCR_PADDING)
            by2 = min(band.shape[0], by2 + self.OCR_PADDING)
            bx1 = max(0, bx1 - self.OCR_PADDING)
            bx2 = min(band.shape[1], bx2 + self.OCR_PADDING)

            crop = band[by1:by2, bx1:bx2].copy()

            # Convert to absolute frame coordinates
            abs_y1 = band_y1 + by1
            abs_y2 = band_y1 + by2
            abs_x1 = band_x1 + bx1
            abs_x2 = band_x1 + bx2

            # Hard ceiling check: reject if box extends above 70%
            if abs_y1 < int(h * self.HARD_CEILING):
                # Box is too high — fall back to full band
                crop = band.copy()
                abs_y1, abs_y2, abs_x1, abs_x2 = band_y1, band_y2, band_x1, band_x2
        else:
            # No specific box found — use full subtitle band
            crop = band.copy()
            abs_y1, abs_y2 = band_y1, band_y2
            abs_x1, abs_x2 = band_x1, band_x2

        # Debug visualization
        if self.debug:
            self._save_debug_frame(frame, (abs_y1, abs_y2, abs_x1, abs_x2))

        return crop, (abs_y1, abs_y2, abs_x1, abs_x2)

    def filter_ocr_results(self, detections: list, frame_height: int) -> list:
        """Filter OCR detections: remove any text above the hard ceiling.

        Args:
            detections: list of (box, (text, conf)) from OCR
            frame_height: original frame height

        Returns:
            Filtered detections with non-subtitle text removed
        """
        ceiling_px = int(frame_height * self.HARD_CEILING)
        filtered = []
        for box, result in detections:
            # box is typically [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            if hasattr(box, '__len__') and len(box) >= 4:
                min_y = min(pt[1] for pt in box)
                if min_y < ceiling_px:
                    continue  # Above ceiling — skip
            filtered.append((box, result))
        return filtered

    def _save_debug_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Save a debug frame with the detected subtitle box drawn."""
        y1, y2, x1, x2 = bbox
        debug_frame = frame.copy()

        # Draw subtitle band boundaries (blue dashed)
        h = frame.shape[0]
        band_top = int(h * self.BAND_TOP)
        band_bot = int(h * self.BAND_BOTTOM)
        cv2.line(debug_frame, (0, band_top), (frame.shape[1], band_top), (255, 0, 0), 1)
        cv2.line(debug_frame, (0, band_bot), (frame.shape[1], band_bot), (255, 0, 0), 1)

        # Draw hard ceiling (red)
        ceiling = int(h * self.HARD_CEILING)
        cv2.line(debug_frame, (0, ceiling), (frame.shape[1], ceiling), (0, 0, 255), 1)

        # Draw detected subtitle box (green)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Labels
        cv2.putText(debug_frame, "HARD CEILING (70%)", (10, ceiling - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_frame, "BAND TOP (75%)", (10, band_top - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_frame, "BAND BOT (95%)", (10, band_bot + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_frame, "OCR REGION", (x1, y1 - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = os.path.join(self.DEBUG_DIR, f"frame_{self._frame_count:04d}_subtitle_box.png")
        cv2.imwrite(out_path, debug_frame)
