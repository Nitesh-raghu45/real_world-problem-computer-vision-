"""
detect_and_track.py
--------------------
Main tracking pipeline:
  1. Open source video
  2. Run YOLOv8 detection (person class) on each frame
  3. Pass detections through ByteTrack for persistent ID assignment
  4. Annotate frame with bounding boxes, IDs, and trajectory trails
  5. Write annotated frames to output MP4
  6. Save per-frame counts for downstream charting (count_over_time.png)

Usage
-----
    # Basic (uses all defaults)
    python detect_and_track.py --video outputs/source_video.mp4

    # Full options
    python detect_and_track.py \
        --video  outputs/source_video.mp4 \
        --output outputs/tracked_output.mp4 \
        --model  yolov8s.pt \
        --conf   0.35 \
        --skip   2 \
        --tracker bytetrack \
        --device  cpu

    # Side-by-side model comparison (ByteTrack vs BoT-SORT)
    python detect_and_track.py --video outputs/source_video.mp4 --compare
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from annotate import (
    annotate_frame,
    build_annotators,
    draw_count_overlay,
    draw_fps_overlay,
    draw_frame_number,
    draw_title_banner,
)


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL      = "yolov8s.pt"
DEFAULT_CONF       = 0.35
DEFAULT_IOU        = 0.45
PERSON_CLASS_ID    = 0        # COCO class 0 = person
DEFAULT_SKIP       = 1        # process every N-th frame (1 = every frame)
DEFAULT_TRACKER    = "bytetrack"
STATS_FILE         = "outputs/tracking_stats.json"


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def open_video(path: str | Path) -> tuple[cv2.VideoCapture, dict]:
    """
    Open a video file and return (cap, meta) where meta contains:
        width, height, fps, total_frames, fourcc_str
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    meta = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, meta


def make_writer(path: str | Path, meta: dict, skip: int = 1) -> cv2.VideoWriter:
    """Create a VideoWriter that outputs at the source FPS (adjusted for skip)."""
    out_fps = meta["fps"] / max(1, skip)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(
        str(path), fourcc, out_fps,
        (meta["width"], meta["height"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer at: {path}")
    return writer


# ---------------------------------------------------------------------------
# Tracker factory
# ---------------------------------------------------------------------------

def build_tracker(name: str = "bytetrack") -> sv.ByteTrack:
    """
    Return a supervision tracker instance.

    Supported names: 'bytetrack'  (sv.ByteTrack)
                     'botsort'    (sv.ByteTrack with different params —
                                   true BoT-SORT requires separate install;
                                   here we approximate with tighter IoU)
    """
    name = name.lower()
    if name == "bytetrack":
        return sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
    elif name == "botsort":
        # Approximation — tighter association for comparison experiments
        return sv.ByteTrack(
            track_activation_threshold=0.30,
            lost_track_buffer=60,     # longer buffer
            minimum_matching_threshold=0.85,
            frame_rate=30,
        )
    else:
        raise ValueError(f"Unknown tracker: '{name}'. Choose 'bytetrack' or 'botsort'.")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: str | Path,
    output_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    skip: int = DEFAULT_SKIP,
    tracker_name: str = DEFAULT_TRACKER,
    device: str = "cpu",
    show_trace: bool = True,
    show_ellipse: bool = False,
    banner_title: str | None = None,
    stats_out: str | Path = STATS_FILE,
) -> dict:
    """
    Full detection + tracking + annotation pipeline.

    Returns a stats dict with per-frame counts and summary metrics.
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Video   : {video_path}")
    print(f"  Output  : {output_path}")
    print(f"  Model   : {model_name}")
    print(f"  Tracker : {tracker_name}")
    print(f"  Conf    : {conf}  |  IoU: {iou}  |  Skip: {skip}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    model     = YOLO(model_name)
    tracker   = build_tracker(tracker_name)
    annotators = build_annotators(trace_length=45, thickness=2)

    cap, meta = open_video(video_path)
    writer    = make_writer(output_path, meta, skip)

    total = meta["total_frames"]
    fps_video = meta["fps"]
    print(f"[INFO] Video: {meta['width']}x{meta['height']} @ {fps_video:.1f} fps  |  {total} frames")

    # ------------------------------------------------------------------
    # Per-frame stats storage
    # ------------------------------------------------------------------
    stats: dict = {
        "tracker":        tracker_name,
        "model":          model_name,
        "conf":           conf,
        "source":         str(video_path),
        "per_frame":      [],   # list of {"frame": N, "count": K, "ids": [...]}
        "all_ids_seen":   set(),
        "id_switches":    0,
        "processing_fps": [],
    }
    prev_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    frame_idx = 0
    pbar = tqdm(total=total, unit="frame", desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        if frame_idx % skip != 0:
            continue

        t0 = time.perf_counter()

        # --- Detection ---
        results = model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # --- Tracking ---
        detections = tracker.update_with_detections(detections)

        # --- Stats ---
        elapsed = time.perf_counter() - t0
        proc_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        current_ids: set[int] = set(
            detections.tracker_id.tolist()
            if detections.tracker_id is not None else []
        )
        stats["all_ids_seen"].update(current_ids)

        # Count ID switches: IDs present now that weren't in the previous
        # frame (and vice-versa) beyond expected arrivals/departures.
        # Simple heuristic: switches += max(0, |new_ids| - |appeared|)
        appeared  = current_ids - prev_ids
        departed  = prev_ids - current_ids
        # If count stays same but some IDs changed → potential switch
        if len(current_ids) == len(prev_ids) and appeared:
            stats["id_switches"] += len(appeared)

        prev_ids = current_ids

        stats["per_frame"].append({
            "frame": frame_idx,
            "count": len(current_ids),
            "ids":   sorted(current_ids),
        })
        stats["processing_fps"].append(round(proc_fps, 2))

        # --- Annotation ---
        annotated = annotate_frame(
            frame, detections, annotators,
            show_trace=show_trace,
            show_ellipse=show_ellipse,
        )
        draw_count_overlay(annotated, len(current_ids), label="Tracked")
        draw_fps_overlay(annotated, proc_fps)
        draw_frame_number(annotated, frame_idx, total)
        if banner_title:
            draw_title_banner(annotated, banner_title)

        writer.write(annotated)

    pbar.close()
    cap.release()
    writer.release()

    # ------------------------------------------------------------------
    # Finalize stats
    # ------------------------------------------------------------------
    stats["all_ids_seen"]  = sorted(stats["all_ids_seen"])
    stats["unique_ids"]    = len(stats["all_ids_seen"])
    stats["max_concurrent"] = max(
        (f["count"] for f in stats["per_frame"]), default=0
    )
    stats["avg_concurrent"] = round(
        sum(f["count"] for f in stats["per_frame"]) / max(1, len(stats["per_frame"])), 2
    )
    stats["avg_processing_fps"] = round(
        sum(stats["processing_fps"]) / max(1, len(stats["processing_fps"])), 2
    )

    # Save JSON stats for other scripts (heatmap.py, count chart)
    Path(stats_out).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_out, "w") as f:
        json.dump(stats, f, indent=2)

    _print_summary(stats, output_path)
    return stats


def _print_summary(stats: dict, output_path: str | Path) -> None:
    print(f"\n{'='*60}")
    print("  TRACKING SUMMARY")
    print(f"{'='*60}")
    print(f"  Unique IDs assigned  : {stats['unique_ids']}")
    print(f"  Max concurrent       : {stats['max_concurrent']}")
    print(f"  Avg concurrent       : {stats['avg_concurrent']}")
    print(f"  ID switches (approx) : {stats['id_switches']}")
    print(f"  Avg processing FPS   : {stats['avg_processing_fps']}")
    print(f"  Output video         : {output_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Comparison mode (ByteTrack vs BoT-SORT)
# ---------------------------------------------------------------------------

def run_comparison(
    video_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    conf: float = DEFAULT_CONF,
    skip: int = DEFAULT_SKIP,
    device: str = "cpu",
) -> None:
    """
    Run both ByteTrack and BoT-SORT on the same video and print a
    side-by-side metric comparison.  Outputs two separate annotated videos.
    """
    results_all: dict[str, dict] = {}

    for tracker_name, out_name in [
        ("bytetrack", "outputs/tracked_bytetrack.mp4"),
        ("botsort",   "outputs/tracked_botsort.mp4"),
    ]:
        print(f"\n>>> Running: {tracker_name.upper()}")
        stats = run_pipeline(
            video_path=video_path,
            output_path=out_name,
            model_name=model_name,
            conf=conf,
            skip=skip,
            tracker_name=tracker_name,
            device=device,
            banner_title=f"YOLOv8 + {tracker_name.upper()}",
            stats_out=f"outputs/stats_{tracker_name}.json",
        )
        results_all[tracker_name] = stats

    # --- Comparison table ---
    print("\n" + "="*60)
    print("  TRACKER COMPARISON")
    print("="*60)
    header = f"{'Metric':<28} {'ByteTrack':>12} {'BoT-SORT':>12}"
    print(header)
    print("-"*60)
    metrics = [
        ("Unique IDs assigned",  "unique_ids"),
        ("Max concurrent",       "max_concurrent"),
        ("Avg concurrent",       "avg_concurrent"),
        ("ID switches (approx)", "id_switches"),
        ("Avg processing FPS",   "avg_processing_fps"),
    ]
    for label, key in metrics:
        bt = results_all["bytetrack"].get(key, "—")
        bs = results_all["botsort"].get(key, "—")
        print(f"  {label:<26} {str(bt):>12} {str(bs):>12}")
    print("="*60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 + ByteTrack sports video tracking pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video",   required=True, help="Path to input video file")
    parser.add_argument("--output",  default="outputs/tracked_output.mp4", help="Output video path")
    parser.add_argument("--model",   default=DEFAULT_MODEL, help="YOLOv8 model weights")
    parser.add_argument("--conf",    type=float, default=DEFAULT_CONF, help="Detection confidence threshold")
    parser.add_argument("--iou",     type=float, default=DEFAULT_IOU,  help="NMS IoU threshold")
    parser.add_argument("--skip",    type=int,   default=DEFAULT_SKIP, help="Process every N-th frame")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, choices=["bytetrack", "botsort"],
                        help="Tracking algorithm")
    parser.add_argument("--device",  default="cpu", help="Inference device: 'cpu', '0', 'cuda', 'mps'")
    parser.add_argument("--no-trace",   action="store_true", help="Disable trajectory trails")
    parser.add_argument("--ellipse",    action="store_true", help="Show ellipse at feet (top-down footage)")
    parser.add_argument("--compare",    action="store_true", help="Run ByteTrack AND BoT-SORT for comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.compare:
        run_comparison(
            video_path=args.video,
            model_name=args.model,
            conf=args.conf,
            skip=args.skip,
            device=args.device,
        )
    else:
        run_pipeline(
            video_path=args.video,
            output_path=args.output,
            model_name=args.model,
            conf=args.conf,
            iou=args.iou,
            skip=args.skip,
            tracker_name=args.tracker,
            device=args.device,
            show_trace=not args.no_trace,
            show_ellipse=args.ellipse,
            banner_title=f"YOLOv8 + {args.tracker.upper()}",
        )

        # Auto-generate count chart after pipeline completes
        print("[INFO] Generating count-over-time chart…")
        try:
            from heatmap import plot_count_over_time
            plot_count_over_time(
                stats_file="outputs/tracking_stats.json",
                output_path="outputs/count_over_time.png",
            )
        except Exception as exc:
            print(f"[WARN] Could not generate count chart: {exc}")


if __name__ == "__main__":
    main()