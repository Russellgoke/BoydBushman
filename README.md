# Boyd Bushman Magnet Drop Experiment Analysis

A Python toolkit for tracking and analyzing the trajectories of falling magnets from video recordings. This project was created to investigate claims about anomalous acceleration behavior in magnetized objects.

## Overview

This project provides tools to:
- **Track falling objects** in video using background subtraction and contour detection
- **Extract trajectory data** with frame-accurate position measurements
- **Analyze acceleration** using both numerical differentiation and parabola fitting
- **Visualize results** with interactive plots and statistical summaries

## Project Structure

```
├── trackmagnet.py          # Main object tracking tool
├── plotTrajectories.py     # Trajectory visualization and analysis
├── ROItool.py              # Region of Interest selection utility
├── trackBottle.py          # [WIP] Bottle tracking variant
├── trackmeterstickmotion.py # Camera stability verification (not for data collection)
├── temp.py                 # Scratch file
├── Data/
│   ├── attractv0.csv       # Attracting magnet trajectories
│   ├── attractv1.csv
│   ├── attractv2.csv
│   ├── repelv1.csv         # Repelling magnet trajectories
│   └── repelv2.csv
└── Videos/
    └── *.MP4               # Source video recordings
```

## Video Sources

The source video recordings used in this analysis are available on Google Drive:

- [MVI_1635.MP4 (Attracting configuration)](https://drive.google.com/file/d/1i7MLJ87Q46n6Nftmq-XlbmNFEyhirVyn/view?usp=sharing)
- [MVI_1636.MP4 (Repelling configuration)](https://drive.google.com/file/d/1h-OUpiATpLf5LY-_PWcl8N1MK6_FQxuj/view?usp=sharing)

Download these files and place them in the `Videos/` folder to run the tracking tools.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pandas
- Matplotlib
- mplcursors (for interactive plots)

Install dependencies:
```bash
pip install opencv-python numpy pandas matplotlib mplcursors
```

## Usage

### 1. Tracking Objects (`trackmagnet.py`)

Tracks falling objects in video and exports trajectory data to CSV.

```bash
python trackmagnet.py
```

**Configuration** (edit at top of file):
- `VIDEO_PATH` - Path to source video
- `LANE_ROI` - Region of interest for tracking (x, y, width, height)
- `START_FRAME` - Frame to begin tracking
- `MIN_WIDTH` / `MIN_HEIGHT` - Size filters for valid detections
- `MIN_FRAMES_TO_VALIDATE` - Minimum trajectory length to save

**Controls:**
| Key | Action |
|-----|--------|
| `Space` | Pause/Play |
| `d` | Step forward 1 frame |
| `f` | Step forward 20 frames |
| `s` | Step backward 1 frame |
| `a` | Step backward 20 frames |
| `r` | Reset background model |
| `q` | Quit and save |

**Output:** Timestamped CSV file with columns:
- `trajectory_id` - Unique identifier for each drop
- `point_index` - Sequential point number within trajectory
- `frame_num` - Video frame number
- `x`, `y` - Centroid position in pixels

### 2. Analyzing Trajectories (`plotTrajectories.py`)

Visualizes trajectory data and calculates acceleration statistics.

```bash
python plotTrajectories.py Data/attractv2.csv Data/repelv2.csv
python plotTrajectories.py Data/              # Process all CSVs in folder
```

**Configuration** (edit at top of file):
- `PLOT_Y` - Plot Y position vs time
- `PLOT_ACCEL_VS_TIME` - Overlay acceleration curves
- `PLOT_ACCEL_HIST` - Histogram of accelerations from parabola fits
- `PRINT_STATS` - Print mean/std dev statistics
- `SMOOTH_WINDOW` - Moving average window (0 = disabled)

**Outlier Handling:** Add trajectory IDs to exclude in the CSV header:
```csv
#OUTLIERS: 52, 67, 103
```

### 3. ROI Selection (`ROItool.py`)

Helper utility for selecting regions of interest and navigating to specific frames.

```bash
python ROItool.py
```

Edit `video_path` in the file to point to your video. Use this to find the exact ROI coordinates and frame numbers needed for `trackmagnet.py`.

**Commands:**
- Enter a frame number to jump to that frame
- Enter `mm:ss` timestamp to jump to that time
- Press `s` to select a 4-point ROI
- Press `b` to select a bounding box ROI
- Press `q` to quit

## Data Format

Trajectory CSV files contain position data for each tracked drop:

```csv
#META: VIDEO=Videos\MVI_1636.MP4, ROI=(609, 0, 143, 749), TIMESTAMP=...
#OUTLIERS:
trajectory_id,point_index,frame_num,x,y
0,0,13,709,14
0,1,14,706,26
...
```

## Analysis Output

The `plotTrajectories.py` script provides:

- **Position plots** - Y position vs frames since drop start
- **Acceleration histograms** - Distribution of calculated accelerations
- **Statistical summary** - Per-drop and per-frame acceleration statistics
  - Mean acceleration (px/fr²)
  - Standard deviation
  - Standard error

## Notes

- **How `trackmagnet.py` works:** Uses OpenCV's MOG2 background subtractor to isolate moving objects, then applies morphological operations (opening + dilation) to clean up the foreground mask. Contours are detected and filtered by minimum width/height. For each valid detection, the centroid is calculated using image moments. A candidate tracking system associates detections across frames using a distance gate (`MAX_VELOCITY`), with kinematic constraints requiring downward motion and limited horizontal drift (`MAX_X_DRIFT`). Candidates that persist for `MIN_FRAMES_TO_VALIDATE` frames become valid trajectories; those that disappear too quickly are discarded. A grace period (`MAX_MISSED_FRAMES`) handles brief detection dropouts.
- **`trackBottle.py`** is a work in progress and not yet functional
- **`trackmeterstickmotion.py`** was used to verify camera stability and is not part of the data collection workflow
- Acceleration units are in pixels per frame squared (px/fr²)

## License

This project is for research and educational purposes.
