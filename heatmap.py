"""
heatmap.py
----------
Post-processing visualizations generated from the tracking pipeline output.

Produces
--------
1. Player position heatmap           (outputs/heatmap.png)
2. Tracked subject count over time   (outputs/count_over_time.png)
3. Per-ID trajectory plot            (outputs/trajectories.png)
4. Top-down bird's-eye projection    (outputs/topdown.png)   [optional]

All functions can be run individually or via the CLI:

    # Generate all visuals from a processed video
    python heatmap.py --video outputs/tracked_output.mp4 --stats outputs/tracking_stats.json

    # Count chart only (from stats JSON, no video re-processing)
    python heatmap.py --stats outputs/tracking_stats.json --count-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "#0f0f0f",
    "axes.facecolor":    "#1a1a1a",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#cccccc",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "text.color":        "#cccccc",
    "grid.color":        "#333333",
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
})

COLORMAP_HEATMAP = "inferno"   # high contrast for sports pitch
ID_COLORS = plt.cm.tab20.colors + plt.cm.Set3.colors   # 32 distinct colors


# ---------------------------------------------------------------------------
# 1. Position heatmap
# ---------------------------------------------------------------------------

def extract_positions_from_video(
    video_path: str | Path,
    max_frames: int = 5000,
    skip: int = 2,
    conf: float = 0.35,
    device: str = "cpu",
) -> tuple[np.ndarray, tuple[int, int], dict[int, list[tuple[float, float]]]]:
    """
    Re-run detection on the *original* (unannotated) video to collect centroid
    positions, OR read from an already-annotated video for approximate positions.

    Returns
    -------
    positions   : (N, 2) float array of (x_norm, y_norm) in [0,1]
    frame_size  : (width, height) of the video
    id_tracks   : {tracker_id: [(x_norm, y_norm), ...]}  per-ID trajectory
    """
    from ultralytics import YOLO
    import supervision as sv

    model   = YOLO("yolov8s.pt")
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_xy: list[tuple[float, float]] = []
    id_tracks: dict[int, list[tuple[float, float]]] = {}

    frame_idx = 0
    pbar = tqdm(total=min(total, max_frames * skip), unit="frame", desc="Extracting positions")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        if frame_idx > max_frames * skip:
            break
        if frame_idx % skip != 0:
            continue

        results    = model(frame, classes=[0], conf=conf, verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is None or len(detections) == 0:
            continue

        xyxy = detections.xyxy
        for i, tid in enumerate(detections.tracker_id):
            cx = ((xyxy[i, 0] + xyxy[i, 2]) / 2) / W
            cy = ((xyxy[i, 1] + xyxy[i, 3]) / 2) / H
            all_xy.append((cx, cy))
            id_tracks.setdefault(int(tid), []).append((cx, cy))

    pbar.close()
    cap.release()

    positions = np.array(all_xy) if all_xy else np.empty((0, 2))
    return positions, (W, H), id_tracks


def plot_heatmap(
    positions: np.ndarray,
    frame_size: tuple[int, int] | None = None,
    output_path: str | Path = "outputs/heatmap.png",
    title: str = "Player position heatmap",
    sigma: float = 18.0,
    resolution: int = 200,
    pitch_overlay: bool = True,
) -> None:
    """
    Render a 2D Gaussian-smoothed heatmap from normalised (x, y) positions.

    Parameters
    ----------
    positions    : (N, 2) array of (x_norm, y_norm)
    frame_size   : original (width, height) — used to set aspect ratio
    output_path  : where to save the PNG
    sigma        : Gaussian blur radius (higher = smoother)
    resolution   : grid resolution (pixels per axis)
    pitch_overlay: draw a simple football/cricket pitch outline
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if len(positions) == 0:
        print("[WARN] No positions to plot — heatmap skipped.")
        return

    # --- Build 2D histogram ---
    xvals = np.clip(positions[:, 0], 0.0, 1.0)
    yvals = np.clip(positions[:, 1], 0.0, 1.0)

    grid, _, _ = np.histogram2d(
        yvals, xvals,
        bins=resolution,
        range=[[0, 1], [0, 1]],
    )
    grid = gaussian_filter(grid, sigma=sigma)
    grid = grid / (grid.max() + 1e-8)  # normalise to [0, 1]

    # --- Plot ---
    aspect = (frame_size[0] / frame_size[1]) if frame_size else 16 / 9
    fig_w  = 10
    fig_h  = fig_w / aspect

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(
        grid,
        cmap=COLORMAP_HEATMAP,
        interpolation="bilinear",
        origin="upper",
        extent=[0, 1, 1, 0],
        aspect="auto",
        vmin=0, vmax=1,
    )

    # Simple pitch markings (center circle + halftime line)
    if pitch_overlay:
        _draw_pitch_overlay(ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=COLORMAP_HEATMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relative density", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Pitch width →", fontsize=9)
    ax.set_ylabel("Pitch length →", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Heatmap saved: {output_path}")


def _draw_pitch_overlay(ax: plt.Axes) -> None:
    """Overlay a minimal football/cricket pitch diagram (normalised coords)."""
    lc = "#ffffff"
    lw = 0.8
    alpha = 0.35

    # Outline
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=lw,
                          edgecolor=lc, facecolor="none", alpha=alpha)
    ax.add_patch(rect)

    # Centre line
    ax.plot([0.02, 0.98], [0.5, 0.5], color=lc, linewidth=lw, alpha=alpha)

    # Centre circle (approximate)
    circle = plt.Circle((0.5, 0.5), 0.1, linewidth=lw,
                         edgecolor=lc, facecolor="none", alpha=alpha)
    ax.add_patch(circle)

    # Penalty areas (top & bottom)
    for y_start in [0.02, 0.75]:
        box = plt.Rectangle((0.25, y_start), 0.5, 0.23, linewidth=lw,
                             edgecolor=lc, facecolor="none", alpha=alpha)
        ax.add_patch(box)


# ---------------------------------------------------------------------------
# 2. Count-over-time chart
# ---------------------------------------------------------------------------

def plot_count_over_time(
    stats_file: str | Path = "outputs/tracking_stats.json",
    output_path: str | Path = "outputs/count_over_time.png",
    smooth_window: int = 15,
) -> None:
    """
    Plot number of tracked subjects per processed frame.

    Reads from the JSON produced by detect_and_track.py.
    Also draws a smoothed trendline and annotates peak count.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(stats_file) as f:
        stats = json.load(f)

    per_frame = stats["per_frame"]
    frames    = [d["frame"] for d in per_frame]
    counts    = [d["count"] for d in per_frame]

    # Moving average
    kernel = np.ones(smooth_window) / smooth_window
    smooth = np.convolve(counts, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(frames, counts, alpha=0.25, color="#4ECDC4")
    ax.plot(frames, counts,  color="#4ECDC4", linewidth=0.8, alpha=0.6, label="Raw count")
    ax.plot(frames, smooth,  color="#FF6B6B", linewidth=1.8, label=f"Smoothed (w={smooth_window})")

    # Peak annotation
    peak_idx  = int(np.argmax(counts))
    peak_val  = counts[peak_idx]
    peak_frame = frames[peak_idx]
    ax.annotate(
        f"Peak: {peak_val}",
        xy=(peak_frame, peak_val),
        xytext=(peak_frame, peak_val + 0.8),
        fontsize=8,
        color="#FFD93D",
        arrowprops=dict(arrowstyle="->", color="#FFD93D", lw=0.8),
    )

    # Summary stats as text box
    unique   = stats.get("unique_ids", "—")
    max_conc = stats.get("max_concurrent", "—")
    avg_conc = stats.get("avg_concurrent", "—")
    tracker  = stats.get("tracker", "—")
    info = (
        f"Tracker: {tracker}  |  "
        f"Unique IDs: {unique}  |  "
        f"Max concurrent: {max_conc}  |  "
        f"Avg concurrent: {avg_conc}"
    )
    ax.set_title(f"Tracked subjects over time\n{info}", fontsize=11, pad=10)
    ax.set_xlabel("Frame index", fontsize=9)
    ax.set_ylabel("Subject count", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Count chart saved: {output_path}")


# ---------------------------------------------------------------------------
# 3. Per-ID trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectories(
    id_tracks: dict[int, list[tuple[float, float]]],
    frame_size: tuple[int, int] | None = None,
    output_path: str | Path = "outputs/trajectories.png",
    max_ids: int = 20,
    min_points: int = 5,
) -> None:
    """
    Draw per-ID trajectory paths on a dark background.

    Parameters
    ----------
    id_tracks  : {tracker_id: [(x_norm, y_norm), ...]}
    max_ids    : cap for legibility
    min_points : skip IDs seen in fewer than this many frames
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    aspect = (frame_size[0] / frame_size[1]) if frame_size else 16 / 9
    fig, ax = plt.subplots(figsize=(10, 10 / aspect))

    # Filter and sort by track length (longest first)
    tracks = {
        tid: pts for tid, pts in id_tracks.items()
        if len(pts) >= min_points
    }
    tracks = dict(
        sorted(tracks.items(), key=lambda kv: -len(kv[1]))[:max_ids]
    )

    legend_patches = []
    for i, (tid, pts) in enumerate(tracks.items()):
        color = ID_COLORS[i % len(ID_COLORS)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8)
        ax.scatter(xs[0],  ys[0],  color=color, s=30, zorder=5, marker="o")
        ax.scatter(xs[-1], ys[-1], color=color, s=30, zorder=5, marker="x")
        legend_patches.append(
            mpatches.Patch(color=color, label=f"ID {tid} ({len(pts)} pts)")
        )

    _draw_pitch_overlay(ax)

    ax.set_title("Per-ID player trajectories  (○ start  × end)", fontsize=12, pad=10)
    ax.set_xlabel("Normalised x")
    ax.set_ylabel("Normalised y")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.legend(
        handles=legend_patches,
        fontsize=7,
        loc="lower right",
        ncol=2,
        framealpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Trajectories saved: {output_path}")


# ---------------------------------------------------------------------------
# 4. Speed estimation (pixels/frame → relative units)
# ---------------------------------------------------------------------------

def estimate_speeds(
    id_tracks: dict[int, list[tuple[float, float]]],
    fps: float = 30.0,
    pitch_length_m: float = 105.0,
) -> dict[int, float]:
    """
    Rough speed estimate per ID in metres/second (assuming pitch length = 105m).

    Returns {tracker_id: avg_speed_m_s}
    """
    speeds: dict[int, float] = {}
    for tid, pts in id_tracks.items():
        if len(pts) < 2:
            continue
        dists = [
            np.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
            for i in range(1, len(pts))
        ]
        avg_norm_dist = float(np.mean(dists))
        # 1 unit of normalised x ≈ pitch_length_m (rough assumption)
        avg_m_per_frame = avg_norm_dist * pitch_length_m
        avg_speed = avg_m_per_frame * fps
        speeds[tid] = round(avg_speed, 2)
    return speeds


def plot_speed_chart(
    speeds: dict[int, float],
    output_path: str | Path = "outputs/speed_chart.png",
) -> None:
    """Bar chart of estimated per-ID speeds."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not speeds:
        return

    ids    = [str(tid) for tid in speeds]
    vals   = list(speeds.values())
    colors = [ID_COLORS[i % len(ID_COLORS)] for i in range(len(ids))]

    fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.5), 4))
    bars = ax.bar(ids, vals, color=colors, width=0.6, edgecolor="none")

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=7, color="#cccccc",
        )

    ax.set_title("Estimated avg speed per player (m/s)  [approximate]", fontsize=11)
    ax.set_xlabel("Tracker ID")
    ax.set_ylabel("Speed (m/s)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Speed chart saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate heatmaps and analytics from tracked video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video",      default=None, help="Source (unannotated) video path for position extraction")
    parser.add_argument("--stats",      default="outputs/tracking_stats.json", help="Stats JSON from detect_and_track.py")
    parser.add_argument("--out-dir",    default="outputs", help="Directory to save all outputs")
    parser.add_argument("--conf",       type=float, default=0.35)
    parser.add_argument("--skip",       type=int,   default=3, help="Frame skip during re-processing")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--count-only", action="store_true", help="Only generate count-over-time chart (no video re-processing)")
    parser.add_argument("--fps",        type=float, default=30.0, help="Video FPS for speed estimation")
    return parser.parse_args()


def main() -> None:
    args  = parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Always generate count chart from stats JSON
    if Path(args.stats).exists():
        plot_count_over_time(
            stats_file=args.stats,
            output_path=outdir / "count_over_time.png",
        )
    else:
        print(f"[WARN] Stats file not found: {args.stats}  — run detect_and_track.py first.")

    if args.count_only:
        return

    if not args.video:
        print("[INFO] No --video provided. Skipping heatmap and trajectory plots.")
        print("       Run with --video <path> to generate all visuals.")
        return

    # Extract positions (re-runs detector on source video)
    print("[INFO] Extracting positions from video (this may take a while)…")
    positions, frame_size, id_tracks = extract_positions_from_video(
        video_path=args.video,
        skip=args.skip,
        conf=args.conf,
        device=args.device,
    )

    print(f"[INFO] Collected {len(positions)} position samples from {len(id_tracks)} unique IDs")

    plot_heatmap(
        positions=positions,
        frame_size=frame_size,
        output_path=outdir / "heatmap.png",
    )

    plot_trajectories(
        id_tracks=id_tracks,
        frame_size=frame_size,
        output_path=outdir / "trajectories.png",
    )

    speeds = estimate_speeds(id_tracks, fps=args.fps)
    plot_speed_chart(speeds, output_path=outdir / "speed_chart.png")

    print(f"\n[DONE] All visuals saved to: {outdir}/")


if __name__ == "__main__":
    main()