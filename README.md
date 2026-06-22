# LiftDetect — Weightlifting Analysis Dashboard

Real-time barbell biomechanics analysis.  
Upload a gym video → the backend runs RTMPose-L + YOLOv8 + DeepSort → the dashboard streams live metrics and renders charts.

---

## Project structure

```
liftdetect/
├── backend/
│   ├── server.py            ← FastAPI server (upload, SSE stream, download)
│   └── analysis_engine.py   ← Refactored weight_analysis.py (callable, headless)
├── frontend/
│   └── index.html           ← Single-file dashboard (no build step)
├── uploads/                 ← Created automatically, gitignored
├── outputs/                 ← Created automatically, gitignored
├── models/                  ← Put your .pth and .torchscript here (gitignored)
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Tool | Why needed |
|---|---|
| Python 3.10+ | Backend runtime |
| ffmpeg | **Required** — re-encodes output video to H.264 so the browser can play it. `winget install ffmpeg` (Win) / `brew install ffmpeg` (Mac) / `sudo apt install ffmpeg` (Linux) |
| NVIDIA GPU + CUDA | Real inference only — not needed for demo mode |

---

## Quick start (two terminals)

> **This project uses the `vb` conda environment** which already has MMPose, MMDet,
> YOLO, and all ML dependencies installed.
> Environment path: `C:\Users\S5\miniconda3\envs\vb`

### Terminal 1 — start the backend

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/liftdetect.git
cd liftdetect

# 2. Activate the vb conda environment
conda activate vb

# 3. Install the 3 web server packages (one-time, if not already installed)
pip install fastapi "uvicorn[standard]" python-multipart

# 4. Start the server from the project root
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

> **Important:** always run `uvicorn` from the **project root** (the folder containing
> `backend/`, `frontend/`, `models/`). This ensures `./models/` paths in
> `analysis_engine.py` resolve correctly.

### Terminal 2 — open the dashboard

Just open the frontend in your browser. Two options:

**Option A — served by FastAPI (recommended)**
```
http://localhost:8000
```

**Option B — open the HTML file directly**
```
liftdetect/frontend/index.html   (double-click or drag into browser)
```

---

## Adding your model weights

Place these files in the `models/` folder (create it if needed):

| File | What |
|---|---|
| `rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth` | RTMPose-L weights |
| `best.torchscript` | Your YOLO barbell detector |

To change these paths, edit the top of `backend/analysis_engine.py`:
```python
POSE_WEIGHTS = "./models/rtmpose-l_simcc-...pth"
YOLO_MODEL   = "./models/best.torchscript"
DEVICE       = "cuda:0"   # change to "cpu" if no GPU
```

---

## Running without a GPU (demo / review mode)

If you are a **reviewer without the `vb` environment**, the engine automatically falls back
to demo mode when MMPose/YOLO are not installed. Create a plain Python environment instead:

```bash
# One-time setup for reviewers
conda create -n liftdetect-review python=3.10 -y
conda activate liftdetect-review
pip install fastapi "uvicorn[standard]" python-multipart numpy opencv-python

# Then run the server the same way
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

The dashboard will show a **"DEMO MODE — No GPU"** watermark on the output video.
All charts, exports, and the video player work identically in demo mode.

---

## MMPose / MMDet / YOLO

These are **already installed in the `vb` conda environment** (`C:\Users\S5\miniconda3\envs\vb`).
No additional ML package installation is needed if you are running on the development machine.

If you need to recreate the environment from scratch, refer to the
[MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html)
and install in this order: PyTorch+CUDA → mmcv → mmdet → mmpose → ultralytics → deep-sort-realtime.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a video file, returns `job_id` |
| `POST` | `/api/start/{job_id}` | Start analysis for an uploaded job |
| `GET` | `/api/stream/{job_id}` | SSE stream of frame-level metrics |
| `GET` | `/api/status/{job_id}` | Poll job status and progress |
| `GET` | `/api/download/{job_id}` | Download the annotated output video |
| `GET` | `/api/session/{job_id}` | Get the full session JSON after completion |

---

## What gets streamed

Every 3 frames the server pushes a `frame` SSE event with this payload:

```json
{
  "frame": 450,
  "t": 15.0,
  "progress": 42.3,
  "round": 2,
  "squat_count": 5,
  "speed": 1.24,
  "max_speed": 1.84,
  "avg_speed": 0.93,
  "l_hip": 82.4,
  "r_hip": 80.1,
  "com_dist": 0.112,
  "barbell_x": 640,
  "barbell_y": 280,
  "is_stopped": false
}
```

When complete, a single `done` event carries the full session JSON (rounds, trajectories, summary).

---

## Dashboard charts

| Chart | Data source |
|---|---|
| Barbell trajectory | `rounds[n].trajectory` — barbell XY positions per round |
| Speed per rep | `rounds[n].peak_speed` |
| Hip angle over time | `frames[n].l_hip` / `frames[n].r_hip` |
| CoM–bar distance | `frames[n].com_dist` |
| Hip symmetry | `rounds[n].avg_l_hip` vs `rounds[n].avg_r_hip` |
| Round breakdown table | `rounds` array |

---

## Export

From the dashboard Export section or the **Export all** button in the top bar:

- **Session JSON** — complete `session_data` object
- **Rep CSV** — per-round stats table
- **Trajectory PNG** — barbell path canvas export
- **Annotated video** — the `output_*.mp4` from the server

---

## Troubleshooting

**"Upload failed — is the server running?"**  
Make sure `uvicorn server:app ...` is running in Terminal 1 before clicking Analyse.

**CORS error in browser console**  
The server allows all origins in dev mode (`allow_origins=["*"]`). This is fine for local use.

**Video plays but no metrics stream**  
Check Terminal 1 for Python errors. If models aren't installed, confirm you see `[WARN] ML models not installed — running in DEMO mode`.

**`cv2.error` or shape mismatch**  
Your video codec may need re-encoding. Try: `ffmpeg -i input.mp4 -vcodec libx264 fixed.mp4`

**Port already in use**  
Change the port: `uvicorn server:app --port 8001` and update `const API = 'http://localhost:8001'` in `frontend/index.html`.

---

## Hardware used for development

- GPU: NVIDIA RTX 4090
- Python: 3.10
- PyTorch: 2.1 + CUDA 11.8
- OS: Windows 11