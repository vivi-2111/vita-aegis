
import cv2
import csv
import math
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# 1. FILES AND VIDEO SETTINGS
# --------------------------------------------------

VIDEO = "accident_clip.mp4"
OUTPUT = "fall_sequence_analysis.mp4"
CSV_OUTPUT = "fall_sequence_metrics.csv"

# Approximate rider region in the original video.
CROP_X1, CROP_Y1 = 450, 180
CROP_X2, CROP_Y2 = 850, 520

# Approximate rider center in crop coordinates.
ANCHOR = np.array([230.0, 130.0], dtype=np.float32)

MAX_CENTER_DISTANCE = 60.0

START_TIME = 3.5
END_TIME = 7.0

CONFIDENCE_THRESHOLD = 0.15

# IMPORTANT:
# Learn normal posture only before the suspected fall.
BASELINE_END_TIME = 4.0
BASELINE_FRAMES = 20

# A detection missing for several frames resets tracking.
MAX_LOST_FRAMES = 5

# --------------------------------------------------
# 2. LOAD MODEL AND VIDEO
# --------------------------------------------------

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)

if not math.isfinite(fps) or fps <= 0:
    cap.release()
    raise SystemExit("Invalid video FPS.")

writer = None
frame_no = 0

previous_center = None
previous_motion_center = None
lost_frames = 0

rows = []
baseline_angles = []


def midpoint(a, b):
    return (a + b) / 2.0


def valid_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        time_sec = frame_no / fps

        if START_TIME <= time_sec <= END_TIME:
            h, w = frame.shape[:2]

            xa = max(0, CROP_X1)
            ya = max(0, CROP_Y1)
            xb = min(w, CROP_X2)
            yb = min(h, CROP_Y2)

            if xb <= xa or yb <= ya:
                raise SystemExit(
                    "Invalid crop coordinates. "
                    "Check CROP_X1/Y1/X2/Y2."
                )

            crop = frame[ya:yb, xa:xb].copy()

            result = model.predict(
                crop,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=640,
                verbose=False
            )[0]

            chosen = None
            center = None
            motion = None

            # --------------------------------------------------
            # 3. DETECT AND TRACK THE RIDER
            # --------------------------------------------------

            if (
                result.boxes is not None
                and result.keypoints is not None
                and len(result.boxes) > 0
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                keypoints = result.keypoints.xy.cpu().numpy()

                if result.keypoints.conf is not None:
                    kp_conf = (
                        result.keypoints.conf.cpu().numpy()
                    )
                else:
                    kp_conf = np.ones(keypoints.shape[:2])

                centers = np.column_stack((
                    (boxes[:, 0] + boxes[:, 2]) / 2.0,
                    (boxes[:, 1] + boxes[:, 3]) / 2.0
                )).astype(np.float32)

                reference = (
                    previous_center
                    if previous_center is not None
                    else ANCHOR
                )

                distances = np.linalg.norm(
                    centers - reference,
                    axis=1
                )

                best = int(np.argmin(distances))

                # Accept only detections near the tracked position.
                if distances[best] <= MAX_CENTER_DISTANCE:
                    chosen = best
                    center = centers[best].copy()
                    lost_frames = 0
                else:
                    lost_frames += 1

            else:
                lost_frames += 1

            # Reset stale tracking after repeated missing detections.
            if lost_frames > MAX_LOST_FRAMES:
                previous_center = None
                previous_motion_center = None

            # --------------------------------------------------
            # 4. ANALYZE BODY POSTURE
            # --------------------------------------------------

            if chosen is not None:
                box = boxes[chosen]
                kp = keypoints[chosen]
                confs = kp_conf[chosen]

                # COCO keypoints:
                # 5, 6 = shoulders
                # 11, 12 = hips
                needed = [5, 6, 11, 12]

                valid = all(
                    np.isfinite(kp[i]).all()
                    and valid_number(confs[i])
                    and confs[i] >= CONFIDENCE_THRESHOLD
                    for i in needed
                )

                x1, y1, x2, y2 = box

                box_width = float(x2 - x1)
                box_height = float(y2 - y1)

                aspect_ratio = (
                    box_width / box_height
                    if box_height > 0
                    else 0.0
                )

                torso_angle = None
                shoulder_mid = None
                hip_mid = None

                if valid:
                    shoulder_mid = midpoint(kp[5], kp[6])
                    hip_mid = midpoint(kp[11], kp[12])

                    dx = float(
                        shoulder_mid[0] - hip_mid[0]
                    )
                    dy = float(
                        shoulder_mid[1] - hip_mid[1]
                    )

                    # 0 degrees = vertical torso.
                    # 90 degrees = horizontal torso.
                    torso_angle = math.degrees(
                        math.atan2(
                            abs(dx),
                            abs(dy) + 1e-6
                        )
                    )

                    # IMPORTANT FIX:
                    # Build the baseline only before 4.0 seconds.
                    # Do not let the suspected fall change it.
                    if (
                        time_sec < BASELINE_END_TIME
                        and len(baseline_angles) < BASELINE_FRAMES
                    ):
                        baseline_angles.append(torso_angle)

                baseline = (
                    float(np.median(baseline_angles))
                    if baseline_angles
                    else None
                )

                angle_change = (
                    torso_angle - baseline
                    if (
                        torso_angle is not None
                        and baseline is not None
                    )
                    else None
                )

                # --------------------------------------------------
                # 5. CALCULATE MOVEMENT BEFORE UPDATING POSITION
                # --------------------------------------------------

                if (
                    previous_motion_center is not None
                    and np.isfinite(center).all()
                ):
                    motion = math.hypot(
                        float(center[0])
                        - float(previous_motion_center[0]),
                        float(center[1])
                        - float(previous_motion_center[1])
                    )

                # Update positions only after measuring movement.
                previous_center = center.copy()
                previous_motion_center = center.copy()

                # --------------------------------------------------
                # 6. PRELIMINARY POSTURE-CHANGE MARKER
                # --------------------------------------------------

                posture_change = (
                    torso_angle is not None
                    and angle_change is not None
                    and torso_angle >= 45
                    and angle_change >= 25
                )

                # --------------------------------------------------
                # 7. DRAW DIAGNOSTIC ANNOTATIONS
                # --------------------------------------------------

                x1i, y1i, x2i, y2i = map(int, box)

                cv2.rectangle(
                    crop,
                    (x1i, y1i),
                    (x2i, y2i),
                    (0, 255, 255),
                    2
                )

                for idx in needed:
                    px, py = map(int, kp[idx])

                    cv2.circle(
                        crop,
                        (px, py),
                        4,
                        (0, 255, 0),
                        -1
                    )

                if valid:
                    cv2.line(
                        crop,
                        tuple(map(int, shoulder_mid)),
                        tuple(map(int, hip_mid)),
                        (255, 0, 255),
                        3
                    )

                angle_text = (
                    f"{torso_angle:.1f} deg"
                    if torso_angle is not None
                    else "angle unavailable"
                )

                label = (
                    f"Posture: {angle_text}"
                    + (" | CHECK" if posture_change else "")
                )

                cv2.putText(
                    crop,
                    label,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                rows.append({
                    "time_sec": round(time_sec, 3),
                    "frame": frame_no,
                    "detected": 1,
                    "center_x": round(float(center[0]), 2),
                    "center_y": round(float(center[1]), 2),
                    "box_width": round(box_width, 2),
                    "box_height": round(box_height, 2),
                    "aspect_ratio": round(aspect_ratio, 3),
                    "torso_angle_deg": (
                        round(torso_angle, 2)
                        if torso_angle is not None
                        else ""
                    ),
                    "baseline_angle_deg": (
                        round(baseline, 2)
                        if baseline is not None
                        else ""
                    ),
                    "angle_change_deg": (
                        round(angle_change, 2)
                        if angle_change is not None
                        else ""
                    ),
                    "center_displacement_px": (
                        round(motion, 2)
                        if motion is not None
                        else ""
                    ),
                    "posture_change_marker": int(posture_change),
                    "keypoints_valid": int(valid)
                })

            else:
                # Record missing detections without inventing values.
                rows.append({
                    "time_sec": round(time_sec, 3),
                    "frame": frame_no,
                    "detected": 0,
                    "center_x": "",
                    "center_y": "",
                    "box_width": "",
                    "box_height": "",
                    "aspect_ratio": "",
                    "torso_angle_deg": "",
                    "baseline_angle_deg": (
                        round(float(np.median(baseline_angles)), 2)
                        if baseline_angles
                        else ""
                    ),
                    "angle_change_deg": "",
                    "center_displacement_px": "",
                    "posture_change_marker": 0,
                    "keypoints_valid": 0
                })

            # --------------------------------------------------
            # 8. CREATE OUTPUT VIDEO
            # --------------------------------------------------

            if writer is None:
                ch, cw = crop.shape[:2]

                writer = cv2.VideoWriter(
                    OUTPUT,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (cw, ch)
                )

                if not writer.isOpened():
                    raise SystemExit(
                        f"Cannot create output video: {OUTPUT}"
                    )

            cv2.putText(
                crop,
                f"Clip time: {time_sec:.2f}s",
                (10, crop.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            writer.write(crop)

        frame_no += 1

finally:
    cap.release()

    if writer is not None:
        writer.release()

# --------------------------------------------------
# 9. SAVE CSV METRICS
# --------------------------------------------------

columns = [
    "time_sec",
    "frame",
    "detected",
    "center_x",
    "center_y",
    "box_width",
    "box_height",
    "aspect_ratio",
    "torso_angle_deg",
    "baseline_angle_deg",
    "angle_change_deg",
    "center_displacement_px",
    "posture_change_marker",
    "keypoints_valid"
]

with open(
    CSV_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    output = csv.DictWriter(f, fieldnames=columns)
    output.writeheader()
    output.writerows(rows)

# --------------------------------------------------
# 10. DISPLAY SUMMARY
# --------------------------------------------------

print("\nTemporal analysis complete.")
print("Video:", OUTPUT)
print("Metrics:", CSV_OUTPUT)
print("Frames analyzed:", len(rows))

print(
    "Frames with selected-person detections:",
    sum(r["detected"] for r in rows)
)

print(
    "Valid keypoint frames:",
    sum(r["keypoints_valid"] for r in rows)
)

print(
    "Baseline samples collected:",
    len(baseline_angles)
)

print(
    "Fixed baseline end time:",
    BASELINE_END_TIME,
    "seconds"
)

print(
    "Preliminary posture-change markers:",
    sum(r["posture_change_marker"] for r in rows)
)

print(
    "Reminder: posture markers are not confirmed accident events."
)
