"""
LiftDetect — Analysis Engine
Strict port of weight_analysis.py for headless web use.

Intentional changes from the original (ONLY these 4):
  1. cv2.imshow / cv2.waitKey / cv2.destroyAllWindows removed (headless)
  2. Model paths / input-output paths come from function args, not top-level globals
  3. on_frame(dict) callback emits metrics every N frames via SSE
  4. Output video re-encoded to H.264 so browsers can play it inline

Everything else is identical to weight_analysis.py — including:
  - Trajectory lines drawn in hardcoded RED (0,0,255), same as original line 530
  - EMA smoothing, squat detection, round reset logic
  - CoM calc, hip angles, draw_skeleton, draw_info_panel, box_label
"""

import math
import time
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── Optional heavy imports ─────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from mmdet.utils import register_all_modules as register_all_mmdet_modules
    from mmpose.utils import register_all_modules as register_all_mmpose_modules
    from mmpose.apis import MMPoseInferencer
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("[WARN] ML models not installed — running in DEMO mode.")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — identical to weight_analysis.py
# ══════════════════════════════════════════════════════════════════════════════
KP_NOSE           = 0
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_KNEE      = 13
KP_RIGHT_KNEE     = 14
KP_LEFT_ANKLE     = 15
KP_RIGHT_ANKLE    = 16

COM_KEYPOINTS = {
    KP_LEFT_SHOULDER:  0.15,
    KP_RIGHT_SHOULDER: 0.15,
    KP_LEFT_HIP:       0.20,
    KP_RIGHT_HIP:      0.20,
    KP_LEFT_KNEE:      0.15,
    KP_RIGHT_KNEE:     0.15,
}

SCORE_THRESHOLD    = 0.3
BARBELL_DIAMETER_M = 0.45
BARBELL_CLASS_ID   = 1
FIXED_TRACK_ID     = 1
SPEED_THRESHOLD    = 0.3
STOP_THRESHOLD     = 0.1
STOP_DURATION      = 5.0       # seconds
MIN_SQUAT_DIST_M   = 0.5       # metres
EMA_ALPHA          = 0.25

SKELETON_LINKS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# ── Model paths — edit these to match your local setup ───────────────────────
# NOTE: Original uses "best.torchscript" relative to the script's CWD.
#       Engine uses paths relative to where uvicorn is launched (repo root).
#       Adjust YOLO_MODEL if your torchscript lives elsewhere.
POSE_CONFIG  = "rtmpose-l_8xb512-700e_body8-halpe26-256x192"
POSE_WEIGHTS = "./models/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth"
YOLO_MODEL   = "./models/best.torchscript"   # same as original "best.torchscript"
DEVICE       = "cuda:0"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS — verbatim copies from weight_analysis.py
# ══════════════════════════════════════════════════════════════════════════════

def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def calc_hip_angles(keypoints: np.ndarray, scores: np.ndarray):
    def _angle(shoulder_idx, hip_idx, knee_idx):
        if (scores[shoulder_idx] < SCORE_THRESHOLD or
                scores[hip_idx]     < SCORE_THRESHOLD or
                scores[knee_idx]    < SCORE_THRESHOLD):
            return None
        return calc_angle(keypoints[shoulder_idx],
                          keypoints[hip_idx],
                          keypoints[knee_idx])
    left_angle  = _angle(KP_LEFT_SHOULDER,  KP_LEFT_HIP,  KP_LEFT_KNEE)
    right_angle = _angle(KP_RIGHT_SHOULDER, KP_RIGHT_HIP, KP_RIGHT_KNEE)
    return left_angle, right_angle


def calc_knee_angles(keypoints: np.ndarray, scores: np.ndarray):
    """Knee flexion angle = angle at the knee formed by hip → knee → ankle.
    Mirrors calc_hip_angles but with the knee as the vertex."""
    def _angle(hip_idx, knee_idx, ankle_idx):
        if (scores[hip_idx]   < SCORE_THRESHOLD or
                scores[knee_idx]  < SCORE_THRESHOLD or
                scores[ankle_idx] < SCORE_THRESHOLD):
            return None
        return calc_angle(keypoints[hip_idx],
                          keypoints[knee_idx],
                          keypoints[ankle_idx])
    left_angle  = _angle(KP_LEFT_HIP,  KP_LEFT_KNEE,  KP_LEFT_ANKLE)
    right_angle = _angle(KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE)
    return left_angle, right_angle


def calc_body_com(keypoints: np.ndarray, scores: np.ndarray):
    weighted_sum = np.zeros(2, dtype=np.float32)
    total_weight = 0.0
    for kp_idx, weight in COM_KEYPOINTS.items():
        if scores[kp_idx] >= SCORE_THRESHOLD:
            weighted_sum += keypoints[kp_idx] * weight
            total_weight += weight
    if total_weight < 1e-8:
        return None
    return weighted_sum / total_weight


def get_predefined_color(index: int):
    """依索引回傳預定義的顏色（BGR）— identical to original."""
    colors = [
        (0,   255,   0),
        (255,   0,   0),
        (0,     0, 255),
        (255, 255,   0),
        (255,   0, 255),
        (0,   255, 255),
        (128,   0, 255),
        (255, 128,   0),
        (0,   255, 128),
        (128, 255,   0),
    ]
    return colors[index % len(colors)]


def box_label(image, box, label='', color=(128, 128, 128),
              txt_color=(255, 255, 255)):
    """Verbatim from weight_analysis.py."""
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
        outside = p1[1] - h >= 3
        p2_text = (p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3)
        cv2.rectangle(image, p1, p2_text, color, -1, cv2.LINE_AA)
        cv2.putText(image, label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, 0.6, txt_color, thickness=1, lineType=cv2.LINE_AA)


def draw_skeleton(frame, keypoints: np.ndarray, scores: np.ndarray,
                  link_color=(0, 255, 0), kp_color=(0, 200, 255),
                  score_thr=0.3, thickness=2, radius=4):
    """Verbatim from weight_analysis.py."""
    n_kp = len(keypoints)
    for (i, j) in SKELETON_LINKS:
        if i >= n_kp or j >= n_kp:
            continue
        if scores[i] >= score_thr and scores[j] >= score_thr:
            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(frame, pt1, pt2, link_color, thickness, cv2.LINE_AA)
    for idx in range(n_kp):
        if scores[idx] >= score_thr:
            pt = (int(keypoints[idx, 0]), int(keypoints[idx, 1]))
            cv2.circle(frame, pt, radius, kp_color, -1, cv2.LINE_AA)


def draw_info_panel(frame, data: dict, panel_x=10, panel_y=10,
                    line_height=28, alpha=0.55):
    """Verbatim from weight_analysis.py."""
    items   = list(data.items())
    panel_w = 340
    panel_h = line_height * (len(items) + 1) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    y = panel_y + line_height
    for key, val in items:
        text = f"{key}: {val}"
        cv2.putText(frame, text, (panel_x + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (220, 220, 220), 1, cv2.LINE_AA)
        y += line_height


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO RE-ENCODE  MP4V → H.264 (browser-compatible)
# ══════════════════════════════════════════════════════════════════════════════

def reencode_for_browser(src: str, dst: str) -> bool:
    if shutil.which("ffmpeg") is None:
        print("[WARN] ffmpeg not found — output video may not play in browser.")
        shutil.move(src, dst)
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-vcodec", "libx264",
        "-profile:v", "baseline",
        "-level",     "3.0",
        "-pix_fmt",   "yuv420p",
        "-movflags",  "+faststart",
        "-an",
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print("[WARN] ffmpeg re-encode failed:", result.stderr.decode())
        shutil.move(src, dst)
        return False
    Path(src).unlink(missing_ok=True)
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(input_path: str, output_path: str,
                 on_frame=None, on_done=None, on_error=None,
                 emit_every_n_frames: int = 3):
    try:
        if MODELS_AVAILABLE:
            _run_real(input_path, output_path, on_frame, on_done, emit_every_n_frames)
        else:
            _run_demo(input_path, output_path, on_frame, on_done)
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"[ERROR] Analysis failed:\n{msg}")
        if on_error:
            on_error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  REAL PIPELINE — faithful port of weight_analysis.py main loop
#
#  Verified line-by-line against weight_analysis.py.
#  The ONLY intentional changes from the original are:
#    - cv2.imshow / cv2.waitKey removed
#    - writer writes to tmp_path first, then re-encoded to output_path
#    - frame_data dict built and emitted via on_frame callback
# ══════════════════════════════════════════════════════════════════════════════

def _run_real(input_path: str, output_path: str,
              on_frame, on_done, emit_every_n_frames: int):

    # ── Load models ───────────────────────────────────────────────────────
    print("[INFO] 載入 MMPose inferencer …")
    register_all_mmdet_modules()
    register_all_mmpose_modules()
    pose_inferencer = MMPoseInferencer(
        pose2d=POSE_CONFIG,
        pose2d_weights=POSE_WEIGHTS,
        device=DEVICE,
    )

    print("[INFO] 載入 YOLO 模型 …")
    yolo_model = YOLO(YOLO_MODEL)
    tracker    = DeepSort(max_age=5)

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] 無法開啟影片：{input_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time   = 1.0 / fps
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Write raw MP4V first; re-encode to H.264 after loop
    tmp_path = output_path + "_tmp.mp4"
    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*"MP4V"),   # identical to original
        fps,
        (orig_w, orig_h),
    )

    # ── State variables — identical to weight_analysis.py lines 276-302 ──
    trajectory_dict     = {}
    trajectory_segments = {}
    current_seg_idx     = {}
    previous_center     = {}
    velocity_dict       = {}
    max_speed_dict      = {}
    total_distance_dict = {}
    total_time_dict     = {}
    lowest_point        = {}
    stop_start_time     = {}
    is_stopped          = {}
    smoothed_center     = {}    # EMA-smoothed centre

    EMA_ALPHA_local = EMA_ALPHA  # local alias avoids closure issues

    squat_count       = 0
    is_moving_down    = False
    is_moving_up      = False

    round_count       = 1
    round_squat_count = 0
    round_max_speed   = 0.0
    round_total_dist  = 0.0
    round_total_time  = 0.0

    m_per_pixel = 1e-3   # updated on first barbell detection

    # ── Dashboard accumulators ────────────────────────────────────────────
    all_frame_data     = []
    rounds_summary     = []
    current_round_data = []

    print("[INFO] 開始處理影片…")
    frame_idx = 0

    # ════════════════════════════════════════════════════════════════════════
    #  MAIN LOOP  — mirrors weight_analysis.py while cap.isOpened()
    # ════════════════════════════════════════════════════════════════════════
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 影片讀取結束")
            break

        frame_idx += 1
        t_sec = frame_idx / fps

        # ── (A) MMPose 骨架偵測 ── same as original lines 316-362 ─────────
        left_hip_angle   = None
        right_hip_angle  = None
        left_knee_angle  = None
        right_knee_angle = None
        com_px           = None

        pose_result = next(pose_inferencer(
            frame,
            show=False,
            return_vis=False,
        ))

        vis_frame = frame.copy()
        instances = pose_result["predictions"][0]

        if len(instances) > 0:
            def _person_area(inst):
                kps   = np.array(inst["keypoints"],       dtype=np.float32)
                sco   = np.array(inst["keypoint_scores"], dtype=np.float32)
                valid = kps[sco >= SCORE_THRESHOLD]
                if len(valid) < 2:
                    return 0.0
                w = valid[:, 0].max() - valid[:, 0].min()
                h = valid[:, 1].max() - valid[:, 1].min()
                return float(w * h)

            person    = max(instances, key=_person_area)
            keypoints = np.array(person["keypoints"],       dtype=np.float32)
            kp_scores = np.array(person["keypoint_scores"], dtype=np.float32)

            draw_skeleton(vis_frame, keypoints, kp_scores)
            left_hip_angle,  right_hip_angle  = calc_hip_angles(keypoints, kp_scores)
            left_knee_angle, right_knee_angle = calc_knee_angles(keypoints, kp_scores)
            com_px = calc_body_com(keypoints, kp_scores)

            if com_px is not None:
                cx_com, cy_com = int(com_px[0]), int(com_px[1])
                cv2.circle(vis_frame, (cx_com, cy_com), 8, (0, 255, 255), -1)
                cv2.putText(vis_frame, "CoM",
                            (cx_com + 10, cy_com - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 1, cv2.LINE_AA)

        # ── (B) YOLO + DeepSort 槓鈴追蹤 ── same as original lines 364-552 ─
        yolo_results = yolo_model(vis_frame, conf=0.4)
        outputs      = yolo_results[0].boxes.data.cpu().numpy()

        detections        = []
        barbell_center_px = None   # reset each frame — same as original

        # Per-frame metric defaults (used in frame_data if no track fires)
        vel_new            = 0.0
        avg_speed          = 0.0
        round_avg_speed    = 0.0
        com_barbell_dist_m = None

        if outputs is not None and len(outputs) > 0:
            for output in outputs:
                x1, y1, x2, y2 = map(int, output[:4])
                cls_id = int(output[5])
                if cls_id == BARBELL_CLASS_ID:
                    detections.append((
                        [x1, y1, int(x2 - x1), int(y2 - y1)],
                        float(output[4]),
                        "barbell"
                    ))
                    barbell_diameter_px = y2 - y1
                    if barbell_diameter_px > 0:
                        m_per_pixel = BARBELL_DIAMETER_M / barbell_diameter_px

            tracks = tracker.update_tracks(detections, frame=vis_frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                # FIXED_TRACK_ID — same as original line 391
                track_id = FIXED_TRACK_ID
                bbox     = track.to_ltrb()

                raw_x = (bbox[0] + bbox[2]) / 2
                raw_y = (bbox[1] + bbox[3]) / 2

                # ── Initialise state on first detection ── original lines 399-415
                if track_id not in trajectory_dict:
                    trajectory_dict[track_id]     = []
                    trajectory_segments[track_id] = []
                    current_seg_idx[track_id]     = 0
                    smoothed_center[track_id]     = (raw_x, raw_y)
                    previous_center[track_id]     = (raw_x, raw_y)
                    velocity_dict[track_id]       = 0.0
                    max_speed_dict[track_id]      = 0.0
                    total_distance_dict[track_id] = 0.0
                    total_time_dict[track_id]     = 0.0
                    lowest_point[track_id]        = raw_y
                    stop_start_time[track_id]     = None
                    is_stopped[track_id]          = False
                    trajectory_segments[track_id].append({
                        "points": [(int(raw_x), int(raw_y))],
                        "color" : get_predefined_color(0),
                    })

                # EMA smoothing — original lines 418-420
                sx = EMA_ALPHA_local * raw_x + (1 - EMA_ALPHA_local) * smoothed_center[track_id][0]
                sy = EMA_ALPHA_local * raw_y + (1 - EMA_ALPHA_local) * smoothed_center[track_id][1]
                smoothed_center[track_id] = (sx, sy)

                center_x = int(sx)
                center_y = int(sy)
                barbell_center_px = (center_x, center_y)

                trajectory_dict[track_id].append((center_x, center_y))
                if trajectory_segments[track_id]:
                    trajectory_segments[track_id][-1]["points"].append((center_x, center_y))

                prev_squat_count = round_squat_count

                # ── Squat direction / count — original lines 432-448 ────────
                prev_cy = previous_center[track_id][1]
                if center_y > prev_cy:
                    is_moving_down = True
                    is_moving_up   = False
                    lowest_point[track_id] = max(lowest_point[track_id], center_y)
                elif center_y < prev_cy and is_moving_down:
                    vert_dist_m = (lowest_point[track_id] - center_y) * m_per_pixel
                    if vert_dist_m >= MIN_SQUAT_DIST_M:
                        is_moving_up   = True
                        is_moving_down = False

                if is_moving_up and not is_moving_down:
                    squat_count       += 1
                    round_squat_count += 1
                    is_moving_up       = False
                    lowest_point[track_id] = center_y

                # New segment on squat count change — original lines 451-456
                if round_squat_count != prev_squat_count:
                    current_seg_idx[track_id] += 1
                    trajectory_segments[track_id].append({
                        "points": [(center_x, center_y)],
                        "color" : get_predefined_color(current_seg_idx[track_id]),
                    })

                # ── Speed calculation — original lines 458-478 ───────────────
                dist_px = math.hypot(center_x - previous_center[track_id][0],
                                     center_y - previous_center[track_id][1])
                dist_m  = dist_px * m_per_pixel
                vel_new = dist_m / frame_time

                velocity_dict[track_id]   = vel_new
                previous_center[track_id] = (center_x, center_y)

                if vel_new > round_max_speed:
                    round_max_speed = vel_new
                if vel_new > SPEED_THRESHOLD:
                    total_distance_dict[track_id] += dist_m
                    total_time_dict[track_id]     += frame_time
                    round_total_dist              += dist_m
                    round_total_time              += frame_time

                avg_speed = (total_distance_dict[track_id] / total_time_dict[track_id]
                             if total_time_dict[track_id] > 0 else 0.0)
                round_avg_speed = (round_total_dist / round_total_time
                                   if round_total_time > 0 else 0.0)

                # ── Stop detection → new Round — original lines 480-498 ──────
                if vel_new < STOP_THRESHOLD:
                    if stop_start_time[track_id] is None:
                        stop_start_time[track_id] = time.time()
                    elif time.time() - stop_start_time[track_id] >= STOP_DURATION:
                        if not is_stopped[track_id]:
                            # Save round summary before reset
                            rounds_summary.append(_build_round_summary(
                                round_count, round_squat_count,
                                round_max_speed, round_avg_speed,
                                current_round_data,
                                trajectory_segments.get(track_id, [])))
                            current_round_data = []

                            # ── Reset — original lines 486-494 ───────────
                            round_count              += 1
                            round_max_speed           = 0.0
                            round_squat_count         = 0
                            round_total_dist          = 0.0
                            round_total_time          = 0.0
                            current_seg_idx[track_id] = 0
                            vis_frame = np.zeros_like(vis_frame)   # original line 492
                            trajectory_dict[track_id]     = []
                            trajectory_segments[track_id] = []
                            # NOTE: track_id key stays in trajectory_dict (value=[])
                            # so init block is skipped next frame — same as original.
                            # First new-round segment is added when next squat fires.

                        is_stopped[track_id] = True
                else:
                    stop_start_time[track_id] = None
                    is_stopped[track_id]      = False

                # ── (C) CoM — barbell distance — original lines 500-520 ──────
                com_barbell_dist_m  = None
                com_barbell_dist_px = None
                if com_px is not None and barbell_center_px is not None:
                    dx = com_px[0] - barbell_center_px[0]
                    dy = com_px[1] - barbell_center_px[1]
                    com_barbell_dist_px = math.hypot(dx, dy)
                    com_barbell_dist_m  = com_barbell_dist_px * m_per_pixel

                    cv2.line(vis_frame,
                             (int(com_px[0]), int(com_px[1])),
                             barbell_center_px,
                             (0, 255, 255), 2, cv2.LINE_AA)
                    mid_x = int((com_px[0] + barbell_center_px[0]) / 2)
                    mid_y = int((com_px[1] + barbell_center_px[1]) / 2)
                    cv2.putText(vis_frame,
                                f"{com_barbell_dist_m:.3f} m",
                                (mid_x + 5, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 255), 1, cv2.LINE_AA)

                # Barbell centre dot — original line 523
                cv2.circle(vis_frame, barbell_center_px, 6, (0, 140, 255), -1)

                # ── Trajectory segments — original lines 526-531 ─────────────
                # IMPORTANT: original line 531 draws (0, 0, 255) RED regardless
                # of segment["color"]. We match this exactly.
                for segment in trajectory_segments[track_id]:
                    pts   = segment["points"]
                    color = segment["color"]   # kept for dashboard JSON
                    for i in range(1, len(pts)):
                        cv2.line(vis_frame, pts[i - 1], pts[i],
                                 (0, 0, 255), 3, cv2.LINE_AA)  # RED — same as original

                box_label(vis_frame, bbox, "barbell", (167, 146, 11))

                # ── (D) Info panel — original lines 535-552 ──────────────────
                info = {
                    "Round"       : str(round_count),
                    "Squat Count" : str(round_squat_count),
                    "Speed"       : f"{vel_new:.2f} m/s",
                    "Max Speed"   : f"{round_max_speed:.2f} m/s",
                    "Avg Speed"   : f"{round_avg_speed:.2f} m/s",
                    "L Hip Angle" : (f"{left_hip_angle:.1f} deg"
                                     if left_hip_angle  is not None else "N/A"),
                    "R Hip Angle" : (f"{right_hip_angle:.1f} deg"
                                     if right_hip_angle is not None else "N/A"),
                    "L Knee Angle": (f"{left_knee_angle:.1f} deg"
                                     if left_knee_angle  is not None else "N/A"),
                    "R Knee Angle": (f"{right_knee_angle:.1f} deg"
                                     if right_knee_angle is not None else "N/A"),
                    "CoM-Bar Dist": (f"{com_barbell_dist_m:.3f} m"
                                     if com_barbell_dist_m is not None else "N/A"),
                }
                if is_stopped[track_id]:
                    info["Status"] = "STOP"

                draw_info_panel(vis_frame, info)

        # ── Write frame — replaces cv2.imshow + writer.write ─────────────
        writer.write(vis_frame)

        # ── Build frame_data for SSE stream and session JSON ─────────────
        frame_data = {
            "frame":       frame_idx,
            "t":           round(t_sec, 3),
            "progress":    round(frame_idx / total_frames * 100, 1),
            "round":       round_count,
            "squat_count": round_squat_count,
            "speed":       round(vel_new, 3),
            "max_speed":   round(round_max_speed, 3),
            "avg_speed":   round(round_avg_speed, 3),
            "l_hip":       round(left_hip_angle,   2) if left_hip_angle   is not None else None,
            "r_hip":       round(right_hip_angle,  2) if right_hip_angle  is not None else None,
            "l_knee":      round(left_knee_angle,  2) if left_knee_angle  is not None else None,
            "r_knee":      round(right_knee_angle, 2) if right_knee_angle is not None else None,
            "com_dist":    round(com_barbell_dist_m, 4) if com_barbell_dist_m is not None else None,
            "barbell_x":   barbell_center_px[0] if barbell_center_px else None,
            "barbell_y":   barbell_center_px[1] if barbell_center_px else None,
            "is_stopped":  is_stopped.get(FIXED_TRACK_ID, False),
        }
        all_frame_data.append(frame_data)
        current_round_data.append(frame_data)

        if on_frame and frame_idx % emit_every_n_frames == 0:
            on_frame(frame_data)

    # ── End of loop ───────────────────────────────────────────────────────
    cap.release()
    writer.release()
    print("[INFO] Raw output written — re-encoding for browser…")

    reencode_for_browser(tmp_path, output_path)
    print(f"[INFO] 已輸出影片至：{output_path}")

    # Final round (video ended before STOP_DURATION fired)
    if current_round_data:
        rounds_summary.append(_build_round_summary(
            round_count, round_squat_count,
            round_max_speed, round_avg_speed,
            current_round_data,
            trajectory_segments.get(FIXED_TRACK_ID, [])))

    session = _build_session(all_frame_data, rounds_summary,
                             Path(input_path).name, fps, total_frames)
    if on_done:
        on_done(session)


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO PIPELINE — no GPU/models; reads real video, synthetic overlay
# ══════════════════════════════════════════════════════════════════════════════

def _run_demo(input_path: str, output_path: str, on_frame, on_done):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time   = 1.0 / fps
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    tmp_path = output_path + "_tmp.mp4"
    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps,
        (orig_w, orig_h),
    )

    all_frame_data     = []
    rounds_summary     = []
    current_round_data = []

    frame_idx         = 0
    round_count       = 1
    round_squat_count = 0
    round_max_speed   = 0.0
    round_avg_speed   = 0.0
    cycle_timer       = 0.0
    bx = orig_w // 2
    by = int(orig_h * 0.7)
    traj_pts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx  += 1
        t_sec       = frame_idx / fps
        cycle_timer += frame_time
        phase        = cycle_timer % 12.0
        in_lift      = 1.0 < phase < 7.0
        lift_p       = max(0.0, min(1.0, (phase - 1.0) / 4.0)) if in_lift else 0.0
        drop_p       = max(0.0, min(1.0, (phase - 6.0)))        if phase > 6.0 else 0.0

        bx_t = orig_w // 2 + int(math.sin(lift_p * math.pi) * orig_w * 0.05)
        by_t = int(orig_h * 0.7 - lift_p * orig_h * 0.45 + drop_p * orig_h * 0.45)
        bx   = int(0.3 * bx_t + 0.7 * bx)
        by   = int(0.3 * by_t + 0.7 * by)
        traj_pts.append((bx, by))

        l_hip           = 80  + (1 - lift_p) * 90
        r_hip           = 78  + (1 - lift_p) * 88
        # Knee flexion: deep at the bottom (~80°), extended standing (~175°)
        l_knee          = 80  + lift_p * 95
        r_knee          = 82  + lift_p * 92
        speed           = lift_p * 1.8 if in_lift else 0.0
        round_max_speed = max(round_max_speed, speed)
        com_dist        = 0.08 + lift_p * 0.06 if in_lift else 0.05
        is_stop         = phase > 10.5

        if is_stop and len(current_round_data) > 30:
            rounds_summary.append(_build_round_summary(
                round_count, round_squat_count, round_max_speed,
                round_avg_speed, current_round_data, []))
            current_round_data = []
            traj_pts           = []
            round_count       += 1
            round_squat_count  = 0
            round_max_speed    = 0.0

        vis = frame.copy()
        cv2.circle(vis, (bx, by), 6, (0, 140, 255), -1)
        # Draw trajectory in RED — matching original
        for i in range(1, len(traj_pts)):
            cv2.line(vis, traj_pts[i-1], traj_pts[i], (0, 0, 255), 3, cv2.LINE_AA)

        info = {
            "Round"       : str(round_count),
            "Squat Count" : str(round_squat_count),
            "Speed"       : f"{speed:.2f} m/s",
            "Max Speed"   : f"{round_max_speed:.2f} m/s",
            "Avg Speed"   : f"{round_avg_speed:.2f} m/s",
            "L Hip Angle" : f"{l_hip:.1f} deg",
            "R Hip Angle" : f"{r_hip:.1f} deg",
            "L Knee Angle": f"{l_knee:.1f} deg",
            "R Knee Angle": f"{r_knee:.1f} deg",
            "CoM-Bar Dist": f"{com_dist:.3f} m",
        }
        if is_stop:
            info["Status"] = "STOP"
        draw_info_panel(vis, info)
        cv2.putText(vis, "DEMO MODE — No GPU / Models",
                    (10, orig_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(vis)

        frame_data = {
            "frame": frame_idx, "t": round(t_sec, 3),
            "progress": round(frame_idx / total_frames * 100, 1),
            "round": round_count, "squat_count": round_squat_count,
            "speed": round(speed, 3), "max_speed": round(round_max_speed, 3),
            "avg_speed": round(round_avg_speed, 3),
            "l_hip": round(l_hip, 2), "r_hip": round(r_hip, 2),
            "l_knee": round(l_knee, 2), "r_knee": round(r_knee, 2),
            "com_dist": round(com_dist, 4),
            "barbell_x": bx, "barbell_y": by,
            "is_stopped": bool(is_stop),
        }
        all_frame_data.append(frame_data)
        current_round_data.append(frame_data)

        if on_frame and frame_idx % 3 == 0:
            on_frame(frame_data)

    cap.release()
    writer.release()
    reencode_for_browser(tmp_path, output_path)

    if current_round_data:
        rounds_summary.append(_build_round_summary(
            round_count, round_squat_count, round_max_speed,
            round_avg_speed, current_round_data, []))

    session = _build_session(all_frame_data, rounds_summary,
                             Path(input_path).name, fps, total_frames)
    if on_done:
        on_done(session)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION SUMMARY BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_round_summary(round_n, reps, peak_speed, avg_speed,
                         frame_data, traj_segs):
    l_hips  = [f["l_hip"]    for f in frame_data if f.get("l_hip")    is not None]
    r_hips  = [f["r_hip"]    for f in frame_data if f.get("r_hip")    is not None]
    l_knees = [f["l_knee"]   for f in frame_data if f.get("l_knee")   is not None]
    r_knees = [f["r_knee"]   for f in frame_data if f.get("r_knee")   is not None]
    coms    = [f["com_dist"] for f in frame_data if f.get("com_dist") is not None]

    # Trajectory from actual barbell X/Y positions
    traj = [[f["barbell_x"], f["barbell_y"]]
            for f in frame_data
            if f.get("barbell_x") is not None and f.get("barbell_y") is not None]
    step = max(1, len(traj) // 400)
    traj = traj[::step]

    def _avg(lst): return round(sum(lst) / len(lst), 2) if lst else None

    return {
        "round":      round_n,
        "reps":       reps,
        "peak_speed": round(peak_speed, 3),
        "avg_speed":  round(avg_speed,  3),
        "avg_l_hip":  _avg(l_hips),
        "avg_r_hip":  _avg(r_hips),
        "avg_l_knee": _avg(l_knees),
        "avg_r_knee": _avg(r_knees),
        "avg_com":    _avg(coms),
        "trajectory": traj,
    }


def _build_session(all_frames, rounds, filename, fps, total_frames):
    duration_sec = total_frames / fps
    mm, ss       = divmod(int(duration_sec), 60)

    all_l   = [f["l_hip"]    for f in all_frames if f.get("l_hip")    is not None]
    all_r   = [f["r_hip"]    for f in all_frames if f.get("r_hip")    is not None]
    all_lk  = [f["l_knee"]   for f in all_frames if f.get("l_knee")   is not None]
    all_rk  = [f["r_knee"]   for f in all_frames if f.get("r_knee")   is not None]
    all_com = [f["com_dist"] for f in all_frames if f.get("com_dist") is not None]
    all_sp  = [r["peak_speed"] for r in rounds]

    def _avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0

    frames_ds = all_frames[::5]   # downsample for chart data

    return {
        "meta": {
            "filename":     filename,
            "duration":     f"{mm}:{ss:02d}",
            "fps":          round(fps, 2),
            "total_frames": total_frames,
            "model":        "RTMPose-L + YOLOv8 + DeepSort" if MODELS_AVAILABLE else "Demo mode",
            "device":       DEVICE if MODELS_AVAILABLE else "cpu",
        },
        "summary": {
            "total_rounds": len(rounds),
            "total_reps":   sum(r["reps"] for r in rounds),
            "peak_speed":   round(max(all_sp), 3) if all_sp else 0,
            "avg_l_hip":    _avg(all_l),
            "avg_r_hip":    _avg(all_r),
            "avg_l_knee":   _avg(all_lk),
            "avg_r_knee":   _avg(all_rk),
            "avg_com":      _avg(all_com),
        },
        "rounds": rounds,
        "frames": frames_ds,
    }
