"""
annotate.py
-----------
Reusable drawing utilities wrapping the `supervision` annotator API.

All public functions accept a BGR numpy frame and return an annotated copy,
so they can be chained without mutating the original.

Exported helpers
----------------
- build_annotators()          -> dict of supervision annotators
- annotate_frame()            -> draw boxes + IDs + trails on one frame
- draw_count_overlay()        -> top-left HUD showing live subject count
- draw_fps_overlay()          -> bottom-right FPS counter
- draw_frame_number()         -> bottom-left frame index
- color_for_id()              -> deterministic BGR color per tracker ID
- create_id_color_palette()   -> generate a large palette of distinct colors
"""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

import cv2
import numpy as np
import supervision as sv

if TYPE_CHECKING:
    pass   # avoid circular imports in type hints only


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def create_id_color_palette(n: int = 100) -> list[tuple[int, int, int]]:
    """
    Generate `n` visually distinct BGR colors using the golden-ratio hue step.

    Returns a list of (B, G, R) tuples.
    """
    golden = 0.618033988749895
    palette: list[tuple[int, int, int]] = []
    h = 0.0
    for _ in range(n):
        h = (h + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        palette.append((int(b * 255), int(g * 255), int(r * 255)))
    return palette


_PALETTE = create_id_color_palette(200)


def color_for_id(tracker_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR color for a given tracker ID."""
    return _PALETTE[tracker_id % len(_PALETTE)]


def sv_color_for_id(tracker_id: int) -> sv.Color:
    """Return a supervision Color object for a given tracker ID."""
    b, g, r = color_for_id(tracker_id)
    return sv.Color(r=r, g=g, b=b)


# ---------------------------------------------------------------------------
# Annotator factory
# ---------------------------------------------------------------------------

def build_annotators(
    trace_length: int = 40,
    thickness: int = 2,
    text_scale: float = 0.55,
    text_thickness: int = 1,
) -> dict:
    """
    Build and return a dict of supervision annotator objects.

    Keys
    ----
    box         : sv.BoxAnnotator
    label       : sv.LabelAnnotator
    trace       : sv.TraceAnnotator
    round_box   : sv.RoundBoxAnnotator   (alternative style)
    ellipse     : sv.EllipseAnnotator    (feet-dot style, common in sports)
    """
    color_lookup = sv.ColorLookup.TRACK  # color tied to tracker ID

    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex([
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#A9DFBF",
        ]),
        thickness=thickness,
        color_lookup=color_lookup,
    )

    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex([
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#A9DFBF",
        ]),
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=4,
        color_lookup=color_lookup,
    )

    trace_annotator = sv.TraceAnnotator(
        color=sv.ColorPalette.from_hex([
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#A9DFBF",
        ]),
        position=sv.Position.BOTTOM_CENTER,
        trace_length=trace_length,
        thickness=max(1, thickness - 1),
        color_lookup=color_lookup,
    )

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex([
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        ]),
        thickness=thickness,
        color_lookup=color_lookup,
    )

    return {
        "box": box_annotator,
        "label": label_annotator,
        "trace": trace_annotator,
        "ellipse": ellipse_annotator,
    }


# ---------------------------------------------------------------------------
# Per-frame annotation
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    annotators: dict,
    show_trace: bool = True,
    show_ellipse: bool = False,
    confidence: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes, ID labels, and (optionally) trajectory traces.

    Parameters
    ----------
    frame        : BGR image array
    detections   : sv.Detections with tracker_id set
    annotators   : dict returned by build_annotators()
    show_trace   : draw trajectory trails
    show_ellipse : draw ellipse at feet (useful for top-down views)
    confidence   : include confidence score in label

    Returns
    -------
    Annotated BGR frame (copy).
    """
    annotated = frame.copy()

    if detections.tracker_id is None or len(detections) == 0:
        return annotated

    # Build label strings  e.g.  "ID 7  0.82"
    labels = []
    for tid, conf in zip(
        detections.tracker_id,
        detections.confidence if detections.confidence is not None
        else [None] * len(detections),
    ):
        if confidence and conf is not None:
            labels.append(f"#{tid}  {conf:.2f}")
        else:
            labels.append(f"#{tid}")

    # Draw trail first (underneath boxes)
    if show_trace:
        annotated = annotators["trace"].annotate(annotated, detections)

    # Ellipse at feet (sports style)
    if show_ellipse:
        annotated = annotators["ellipse"].annotate(annotated, detections)

    # Bounding boxes
    annotated = annotators["box"].annotate(annotated, detections)

    # Labels
    annotated = annotators["label"].annotate(annotated, detections, labels=labels)

    return annotated


# ---------------------------------------------------------------------------
# HUD overlays
# ---------------------------------------------------------------------------

def _semi_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int] = (20, 20, 20),
    alpha: float = 0.55,
) -> np.ndarray:
    """Draw a semi-transparent filled rectangle in-place (no copy)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_count_overlay(
    frame: np.ndarray,
    count: int,
    label: str = "Tracked",
    position: tuple[int, int] = (12, 12),
) -> np.ndarray:
    """
    Draw a top-left HUD showing the live subject count.

    Returns the modified frame (mutates in place for performance).
    """
    x, y = position
    box_w, box_h = 165, 52
    _semi_rect(frame, x, y, box_w, box_h)

    cv2.putText(
        frame, label.upper(),
        (x + 10, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (180, 180, 180), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, str(count),
        (x + 10, y + 44),
        cv2.FONT_HERSHEY_SIMPLEX, 1.05,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


def draw_fps_overlay(
    frame: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Draw FPS counter at bottom-right."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = w - tw - 14, h - 10
    _semi_rect(frame, x - 6, y - th - 6, tw + 12, th + 12)
    cv2.putText(
        frame, text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (200, 200, 200), 1, cv2.LINE_AA,
    )
    return frame


def draw_frame_number(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int | None = None,
) -> np.ndarray:
    """Draw frame counter at bottom-left."""
    h = frame.shape[0]
    if total_frames:
        text = f"Frame {frame_idx}/{total_frames}"
    else:
        text = f"Frame {frame_idx}"
    cv2.putText(
        frame, text,
        (12, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
        (140, 140, 140), 1, cv2.LINE_AA,
    )
    return frame


def draw_title_banner(
    frame: np.ndarray,
    title: str = "Sports Tracker — YOLOv8 + ByteTrack",
) -> np.ndarray:
    """Draw a subtle title banner at the very top of the frame."""
    w = frame.shape[1]
    _semi_rect(frame, 0, 0, w, 24, color=(10, 10, 10), alpha=0.7)
    cv2.putText(
        frame, title,
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44,
        (200, 200, 200), 1, cv2.LINE_AA,
    )
    return frame