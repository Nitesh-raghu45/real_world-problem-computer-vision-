# Sports Tracker — Multi-Object Detection & Persistent ID Tracking

A computer vision pipeline for detecting and tracking multiple moving subjects (players, athletes) in public sports footage, using **YOLOv8** for detection and **ByteTrack** for persistent ID assignment.

---

## Demo

**Source video:** [IPL Cricket Highlights — YouTube](https://www.youtube.com/watch?v=REPLACE_WITH_ACTUAL_URL)
*(Replace with your chosen public video URL)*

**Output:** Annotated video with bounding boxes, persistent IDs, and trajectory trails.

---

## Project Structure

```
sports_tracker/
├── README.md               ← You are here
├── requirements.txt        ← All dependencies
├── download_video.py       ← yt-dlp wrapper to fetch public videos
├── detect_and_track.py     ← Main pipeline (detection + tracking + annotation)
├── annotate.py             ← Drawing utilities (boxes, labels, overlays)
├── heatmap.py              ← Post-processing visualizations
├── report.md               ← Technical report
└── outputs/
    ├── source_video.mp4        ← Downloaded source
    ├── tracked_output.mp4      ← Annotated output video
    ├── tracking_stats.json     ← Per-frame stats (auto-generated)
    ├── heatmap.png             ← Player position heatmap
    ├── count_over_time.png     ← Tracked subject count per frame
    ├── trajectories.png        ← Per-ID trajectory paths
    └── speed_chart.png         ← Estimated player speeds
```

---

## Installation

### 1. Clone / download the project

```bash
git clone https://github.com/YOUR_USERNAME/sports-tracker.git
cd sports-tracker
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is installed as a CPU build by default. For GPU acceleration, install the appropriate CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) **before** running `pip install -r requirements.txt`.

### 4. Install FFmpeg (for video encoding)

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

### 5. Verify installation

```bash
python -c "import ultralytics, supervision, cv2; print('All dependencies OK')"
```

---

## Usage

### Step 1 — Download a public video

```bash
python download_video.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

Options:
```
--url          Public video URL (required)
--output       Output path  [default: outputs/source_video.mp4]
--start        Start time in seconds  [default: 0]
--duration     Duration to download in seconds, 0 = full video  [default: 120]
--max-height   Max resolution height  [default: 720]
```

Example — download 90 seconds starting from 0:30:
```bash
python download_video.py \
    --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
    --start 30 \
    --duration 90
```

---

### Step 2 — Run the tracking pipeline

```bash
python detect_and_track.py --video outputs/source_video.mp4
```

Full options:
```
--video        Path to input video (required)
--output       Output video path  [default: outputs/tracked_output.mp4]
--model        YOLOv8 model weights  [default: yolov8s.pt]
--conf         Detection confidence threshold  [default: 0.35]
--iou          NMS IoU threshold  [default: 0.45]
--skip         Process every N-th frame (1=every frame)  [default: 1]
--tracker      Tracking algorithm: bytetrack | botsort  [default: bytetrack]
--device       Inference device: cpu | 0 | cuda | mps  [default: cpu]
--no-trace     Disable trajectory trails
--ellipse      Draw ellipse at feet (useful for drone/top-down footage)
--compare      Run BOTH ByteTrack and BoT-SORT side-by-side for comparison
```

Example — GPU, higher confidence, with ellipse marker:
```bash
python detect_and_track.py \
    --video outputs/source_video.mp4 \
    --model yolov8m.pt \
    --conf 0.40 \
    --device 0 \
    --ellipse
```

Example — run tracker comparison:
```bash
python detect_and_track.py --video outputs/source_video.mp4 --compare
```

---

### Step 3 — Generate heatmaps and analytics

```bash
# Count chart only (fast — reads from stats JSON, no re-processing)
python heatmap.py --stats outputs/tracking_stats.json --count-only

# Full analytics (re-runs detection on source video)
python heatmap.py --video outputs/source_video.mp4
```

Full options:
```
--video        Source video for position extraction (optional)
--stats        Stats JSON from detect_and_track.py
--out-dir      Directory to save visuals  [default: outputs]
--skip         Frame skip during re-processing  [default: 3]
--fps          Video FPS for speed estimation  [default: 30]
--count-only   Only generate count chart (no video re-processing)
```

---

## Model Choices

| Model | Size | Speed (CPU) | Accuracy | Recommended for |
|-------|------|------------|----------|-----------------|
| `yolov8n.pt` | 6 MB  | Fastest | Lower  | Quick prototyping |
| `yolov8s.pt` | 22 MB | Fast    | Good   | **Default — best balance** |
| `yolov8m.pt` | 52 MB | Moderate | Better | Final submission output |
| `yolov8l.pt` | 87 MB | Slow    | Best   | GPU only |

Models are downloaded automatically on first run from Ultralytics.

---

## Assumptions

- **Detection class:** Only COCO class 0 (person) is tracked. Balls, goalposts, and other objects are ignored.
- **Camera:** Works for both fixed and slowly panning cameras. Fast camera shake can cause temporary ID switches.
- **Subject size:** Players must occupy at least ~20×50 pixels to be reliably detected. Very distant players in aerial shots may be missed.
- **Frame rate:** The pipeline is tested on 25–60 fps source videos. Very low fps (< 10) may hurt tracking continuity.
- **Speed estimation** is a rough approximation assuming a standard 105m football pitch length. Results are relative, not calibrated.

---

## Limitations

- **ID switches** occur when two players cross or occlude each other — ByteTrack uses IoU matching which can confuse similarly-positioned players.
- **Fast motion blur** reduces detection confidence, potentially dropping a player for several frames.
- **Similar appearances** (same-colored jerseys) are not distinguished by appearance alone without a ReID model (OSNet / Fast-ReID).
- **Top-down projection** requires manual calibration of homography points per video — not included in this pipeline.
- CPU inference is slow (~3–8 fps). Use `--device 0` with a CUDA GPU for real-time processing.

---

## Optional Enhancements (included)

| Feature | Command |
|---------|---------|
| Trajectory trails | Enabled by default in output video |
| Position heatmap | `python heatmap.py --video ...` |
| Count over time chart | Auto-generated after `detect_and_track.py` |
| Per-ID trajectory plot | `python heatmap.py --video ...` |
| Speed estimation | `python heatmap.py --video ...` |
| Tracker comparison | `python detect_and_track.py --compare` |

---

## Requirements

- Python 3.9+
- torch >= 2.0
- ultralytics >= 8.2
- supervision >= 0.21
- opencv-python-headless >= 4.9
- numpy, matplotlib, scipy, seaborn, tqdm
- yt-dlp (for video download)
- FFmpeg (system install)

See `requirements.txt` for pinned versions.

---

## Troubleshooting

**`yt-dlp: command not found`**
```bash
pip install yt-dlp
```

**`No module named 'supervision'`**
```bash
pip install supervision
```

**Video writes but won't play**
```bash
# Re-encode with FFmpeg
ffmpeg -i outputs/tracked_output.mp4 -vcodec libx264 outputs/tracked_final.mp4
```

**Out of memory on GPU**
```bash
# Use smaller model or process at lower resolution
python detect_and_track.py --video ... --model yolov8n.pt --skip 2
```

---

## License

This project is released for educational and research purposes.  
Video content must be from publicly accessible sources only.