"""
LiftDetect — FastAPI Backend
Receives video uploads, runs weight_analysis_engine.py in a background thread,
streams frame-level metrics to the browser via Server-Sent Events (SSE),
and serves the annotated output video for download.
"""

import sys
import os

# Ensure analysis_engine.py is always found regardless of where uvicorn launches from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uuid
import json
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil

from analysis_engine import run_analysis   # ← the refactored pipeline

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="LiftDetect API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # local dev — lock this down for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from /frontend
app.mount(
    "/app",
    StaticFiles(directory=str(BASE_DIR / "frontend"), html=True),
    name="frontend",
)

# ── In-memory job store ───────────────────────────────────────────────────
# { job_id: {"status": "queued|running|done|error",
#             "progress": 0-100,
#             "stage": "...",
#             "queue": asyncio.Queue (SSE events),
#             "output_path": Path,
#             "session_data": dict } }
# JOBS: dict[str, dict] = {}
JOBS: Dict[str, Dict[str, Any]] = {}


# ═════════════════════════════════════════════════════════════════════════
#  UPLOAD — POST /api/upload
# ═════════════════════════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Receive a video file, save it, return a job_id."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")

    job_id    = str(uuid.uuid4())[:8]
    suffix    = Path(file.filename).suffix or ".mp4"
    in_path   = UPLOAD_DIR / f"{job_id}_input{suffix}"
    out_path  = OUTPUT_DIR / f"{job_id}_output.mp4"

    # Write upload to disk
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Register job
    JOBS[job_id] = {
        "status":       "queued",
        "progress":     0,
        "stage":        "Waiting to start…",
        "queue":        asyncio.Queue(),
        "input_path":   in_path,
        "output_path":  out_path,
        "filename":     file.filename,
        "session_data": None,
    }

    return {"job_id": job_id, "filename": file.filename}


# ═════════════════════════════════════════════════════════════════════════
#  START — POST /api/start/{job_id}
# ═════════════════════════════════════════════════════════════════════════
@app.post("/api/start/{job_id}")
async def start_job(job_id: str):
    """Kick off analysis for an uploaded job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "running":
        raise HTTPException(400, "Already running")

    job["status"] = "running"
    loop = asyncio.get_event_loop()

    def _run():
        """Runs in a background thread so it doesn't block the event loop."""
        try:
            run_analysis(
                input_path  = str(job["input_path"]),
                output_path = str(job["output_path"]),
                on_frame    = lambda data: _emit(loop, job["queue"], data),
                on_done     = lambda session: _finish(loop, job, session),
                on_error    = lambda err:     _error(loop, job, err),
            )
        except Exception as e:
            _error(loop, job, str(e))

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


def _emit(loop, queue: asyncio.Queue, data: dict):
    """Thread-safe: push a frame event onto the async queue."""
    asyncio.run_coroutine_threadsafe(queue.put(("frame", data)), loop)

def _finish(loop, job: dict, session_data: dict):
    job["status"]       = "done"
    job["progress"]     = 100
    job["session_data"] = session_data
    asyncio.run_coroutine_threadsafe(job["queue"].put(("done", session_data)), loop)

def _error(loop, job: dict, msg: str):
    job["status"] = "error"
    asyncio.run_coroutine_threadsafe(job["queue"].put(("error", {"message": msg})), loop)


# ═════════════════════════════════════════════════════════════════════════
#  SSE STREAM — GET /api/stream/{job_id}
# ═════════════════════════════════════════════════════════════════════════
@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    """Server-Sent Events: pushes frame data to the browser in real time."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        queue = job["queue"]
        while True:
            try:
                event_type, payload = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                yield "event: ping\ndata: {}\n\n"
                continue

            yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

            if event_type in ("done", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ═════════════════════════════════════════════════════════════════════════
#  STATUS — GET /api/status/{job_id}
# ═════════════════════════════════════════════════════════════════════════
@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status":   job["status"],
        "progress": job["progress"],
        "stage":    job["stage"],
    }


# ═════════════════════════════════════════════════════════════════════════
#  STREAM VIDEO — GET /api/download/{job_id}
#  Supports HTTP Range requests so the browser <video> element can seek.
# ═════════════════════════════════════════════════════════════════════════
@app.get("/api/download/{job_id}")
async def download_video(job_id: str, request: Request):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    out = job["output_path"]
    if not out.exists():
        raise HTTPException(404, "Output video not ready yet")

    file_size = out.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end"
        range_val = range_header.strip().replace("bytes=", "")
        parts     = range_val.split("-")
        start     = int(parts[0]) if parts[0] else 0
        end       = int(parts[1]) if parts[1] else file_size - 1
        end       = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iterfile():
            with open(out, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range":              f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges":              "bytes",
            "Content-Length":             str(chunk_size),
            "Content-Disposition":        f'inline; filename="liftdetect_{job_id}_output.mp4"',
            "Access-Control-Allow-Origin": "*",
        }
        return StreamingResponse(iterfile(), status_code=206,
                                 media_type="video/mp4", headers=headers)

    # Full file (no Range header)
    def iterfile_full():
        with open(out, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    headers = {
        "Accept-Ranges":               "bytes",
        "Content-Length":              str(file_size),
        "Content-Disposition":         f'attachment; filename="liftdetect_{job_id}_output.mp4"',
        "Access-Control-Allow-Origin": "*",
    }
    return StreamingResponse(iterfile_full(), media_type="video/mp4", headers=headers)


# ═════════════════════════════════════════════════════════════════════════
#  SESSION DATA — GET /api/session/{job_id}
# ═════════════════════════════════════════════════════════════════════════
@app.get("/api/session/{job_id}")
async def session_data(job_id: str):
    job = JOBS.get(job_id)
    if not job or job["session_data"] is None:
        raise HTTPException(404, "Session data not available yet")
    return job["session_data"]


# ── Root redirect ─────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)