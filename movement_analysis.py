
import cv2
import csv
import math
import os
from collections import deque
from datetime import datetime

from ultralytics import YOLO

# ============================================================
# VITA AEGIS
# CCTV ACCIDENT CANDIDATE DETECTION
# YOLO + ByteTrack + Optical Flow + Motion + Interaction
# ============================================================

VIDEO_PATH = "Road (4).mp4"
MODEL_PATH = "yolov8n.pt"

OUTPUT_PATH = "vita_aegis_output.mp4"
CSV_PATH = "incident_alerts.csv"

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

CONFIDENCE = 0.20
IOU_THRESHOLD = 0.45

# Track history
HISTORY_SECONDS = 2.0
MOTION_GAP_SECONDS = 0.35
MIN_HISTORY_POINTS = 5

# Motion thresholds (initial values; calibrate to footage)
MIN_PREVIOUS_SPEED = 0.25
BRAKE_RATIO = 0.55
MIN_SPEED_DROP = 0.25
DIRECTION_COSINE = -0.10

# Vehicle interactions
INTERACTION_IOU = 0.03
INTERACTION_DISTANCE = 1.5

# Optical flow
FLOW_POINTS = 30
FLOW_MIN_POINTS = 5
FLOW_MIN_DISPLACEMENT = 2.0

# Alert handling
EVIDENCE_WINDOW = 2.0
ALERT_COOLDOWN = 8.0
MIN_ALERT_SCORE = 2

DISPLAY_VIDEO = True
SAVE_VIDEO = True

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_diagonal(box):
    x1, y1, x2, y2 = box
    return max(1.0, math.hypot(x2 - x1, y2 - y1))


def vector_cosine(a, b):
    ma = math.hypot(a[0], a[1])
    mb = math.hypot(b[0], b[1])

    if ma < 1e-6 or mb < 1e-6:
        return 1.0

    return (a[0] * b[0] + a[1] * b[1]) / (ma * mb)


def get_motion_signals(history):
    """
    History entries:
        (video_time, center_x, center_y, box)

    Speeds are approximate, normalized by vehicle size.
    """

    if len(history) < MIN_HISTORY_POINTS:
        return {
            "speed": 0.0,
            "previous_speed": 0.0,
            "brake": False,
            "direction": False,
            "stationary": False,
        }

    points = list(history)
    latest = points[-1]

    target_time = latest[0] - MOTION_GAP_SECONDS

    past = min(
        points[:-1],
        key=lambda p: abs(p[0] - target_time)
    )

    older_target = past[0] - MOTION_GAP_SECONDS

    older_candidates = [
        p for p in points
        if p[0] < past[0]
    ]

    if not older_candidates:
        return {
            "speed": 0.0,
            "previous_speed": 0.0,
            "brake": False,
            "direction": False,
            "stationary": False,
        }

    older = min(
        older_candidates,
        key=lambda p: abs(p[0] - older_target)
    )

    dt_recent = latest[0] - past[0]
    dt_old = past[0] - older[0]

    if dt_recent <= 0 or dt_old <= 0:
        return {
            "speed": 0.0,
            "previous_speed": 0.0,
            "brake": False,
            "direction": False,
            "stationary": False,
        }

    recent_vector = (
        latest[1] - past[1],
        latest[2] - past[2]
    )

    old_vector = (
        past[1] - older[1],
        past[2] - older[2]
    )

    scale = box_diagonal(latest[3])

    speed = math.hypot(*recent_vector) / scale / dt_recent
    previous_speed = math.hypot(*old_vector) / scale / dt_old

    brake = (
        previous_speed >= MIN_PREVIOUS_SPEED
        and previous_speed - speed >= MIN_SPEED_DROP
        and speed <= previous_speed * BRAKE_RATIO
    )

    direction = (
        previous_speed >= MIN_PREVIOUS_SPEED
        and speed >= 0.20
        and vector_cosine(old_vector, recent_vector)
        <= DIRECTION_COSINE
    )

    recent_history = [
        p for p in points
        if latest[0] - p[0] <= 1.0
    ]

    stationary = False

    if len(recent_history) >= 3:
        first = recent_history[0]
        displacement = math.hypot(
            latest[1] - first[1],
            latest[2] - first[2]
        )

        stationary = (
            displacement / scale < 0.08
            and latest[0] - first[0] >= 0.8
            and previous_speed >= MIN_PREVIOUS_SPEED
        )

    return {
        "speed": speed,
        "previous_speed": previous_speed,
        "brake": brake,
        "direction": direction,
        "stationary": stationary,
    }


def optical_flow_motion(previous_gray, current_gray, box):
    """
    Estimate local image motion inside a vehicle bounding box.

    Optical flow is supporting evidence only. It is not a
    collision classifier and can be affected by camera motion.
    """

    if previous_gray is None or current_gray is None:
        return 0.0, 0

    height, width = current_gray.shape

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0, 0

    roi = previous_gray[y1:y2, x1:x2]

    points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=FLOW_POINTS,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=5
    )

    if points is None or len(points) < FLOW_MIN_POINTS:
        return 0.0, 0

    points[:, 0, 0] += x1
    points[:, 0, 1] += y1

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None
    )

    if next_points is None or status is None:
        return 0.0, 0

    valid_old = points[status.flatten() == 1]
    valid_new = next_points[status.flatten() == 1]

    if len(valid_old) < FLOW_MIN_POINTS:
        return 0.0, len(valid_old)

    movement = valid_new - valid_old

    magnitudes = [
        math.hypot(float(v[0]), float(v[1]))
        for v in movement.reshape(-1, 2)
    ]

    # Median resists a few noisy feature points.
    magnitudes.sort()
    median_movement = magnitudes[len(magnitudes) // 2]

    return median_movement, len(magnitudes)


# ============================================================
# INITIALIZATION
# ============================================================

if not os.path.isfile(VIDEO_PATH):
    raise SystemExit(f"ERROR: Video not found: {VIDEO_PATH}")

if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f"ERROR: Model not found: {MODEL_PATH}")

print("Loading Vita Aegis...")
model = YOLO(MODEL_PATH)

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise SystemExit("ERROR: Could not open input video.")

fps = video.get(cv2.CAP_PROP_FPS)

if not math.isfinite(fps) or fps <= 0 or fps > 120:
    fps = 25.0

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0 or height <= 0:
    video.release()
    raise SystemExit("ERROR: Invalid video dimensions.")

writer = None

if SAVE_VIDEO:
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        video.release()
        raise SystemExit("ERROR: Could not create output video.")

# Per-vehicle history
vehicle_data = {}

# Candidate evidence timestamps
event_evidence = {}

# Last alert timestamp per incident key
last_alert_time = {}

previous_gray = None
previous_video_time = None

frame_number = 0
total_alerts = 0

print(f"FPS: {fps:.2f}")
print(f"Resolution: {width}x{height}")
print("Monitoring started. Press Q to quit.")

# ============================================================
# MAIN LOOP
# ============================================================

try:
    with open(
        CSV_PATH, "w", newline="", encoding="utf-8"
    ) as csv_file:

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([
            "Alert Time",
            "Video Time (seconds)",
            "Vehicle IDs",
            "Evidence",
            "Review Status"
        ])

        while True:
            success, frame = video.read()

            if not success:
                break

            frame_number += 1

            time_ms = video.get(cv2.CAP_PROP_POS_MSEC)

            video_time = (
                time_ms / 1000.0
                if time_ms > 0
                else frame_number / fps
            )

            current_gray = cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY
            )

            # Persist tracking between frames.
            result = model.track(
                frame,
                persist=True,
                classes=VEHICLE_CLASSES,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False
            )[0]

            tracks = {}

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, vehicle_id, class_id in zip(
                    boxes, ids, classes
                ):
                    x1, y1, x2, y2 = map(int, box)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    vehicle_box = (x1, y1, x2, y2)

                    tracks[vehicle_id] = {
                        "box": vehicle_box,
                        "center": (center_x, center_y),
                        "class_id": class_id,
                    }

                    if vehicle_id not in vehicle_data:
                        vehicle_data[vehicle_id] = {
                            "history": deque(),
                            "last_seen": video_time,
                            "previous_flow": None,
                        }

                    data = vehicle_data[vehicle_id]

                    data["history"].append((
                        video_time,
                        center_x,
                        center_y,
                        vehicle_box
                    ))

                    data["last_seen"] = video_time

                    while (
                        data["history"]
                        and video_time - data["history"][0][0]
                        > HISTORY_SECONDS
                    ):
                        data["history"].popleft()

            # ------------------------------------------------
            # A. MOTION AND OPTICAL FLOW SIGNALS
            # ------------------------------------------------

            motion_signals = {}

            for vehicle_id, track in tracks.items():
                data = vehicle_data[vehicle_id]

                motion = get_motion_signals(data["history"])

                flow_pixels, flow_points = optical_flow_motion(
                    previous_gray,
                    current_gray,
                    track["box"]
                )

                flow_anomaly = (
                    flow_points >= FLOW_MIN_POINTS
                    and flow_pixels >= FLOW_MIN_DISPLACEMENT
                )

                motion["flow_pixels"] = flow_pixels
                motion["flow_points"] = flow_points
                motion["flow_anomaly"] = flow_anomaly

                motion_signals[vehicle_id] = motion

            # ------------------------------------------------
            # B. NEARBY VEHICLES / BOX OVERLAP
            # ------------------------------------------------

            interactions = set()
            ids = list(tracks.keys())

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id_a, id_b = ids[i], ids[j]

                    a = tracks[id_a]
                    b = tracks[id_b]

                    overlap = box_iou(a["box"], b["box"])

                    distance = math.dist(
                        a["center"], b["center"]
                    )

                    scale = (
                        box_diagonal(a["box"])
                        + box_diagonal(b["box"])
                    ) / 2.0

                    near = distance / max(scale, 1.0)

                    if (
                        overlap >= INTERACTION_IOU
                        or near <= INTERACTION_DISTANCE
                    ):
                        interactions.add(
                            tuple(sorted((id_a, id_b)))
                        )

            # ------------------------------------------------
            # C. BUILD TEMPORAL EVIDENCE
            # ------------------------------------------------

            # Each distinct evidence type gets its own history.
            for vehicle_id, motion in motion_signals.items():
                if motion["brake"]:
                    key = ("vehicle", vehicle_id, "brake")
                    event_evidence.setdefault(key, deque()).append(
                        video_time
                    )

                if motion["direction"]:
                    key = ("vehicle", vehicle_id, "direction")
                    event_evidence.setdefault(key, deque()).append(
                        video_time
                    )

                if motion["stationary"]:
                    key = ("vehicle", vehicle_id, "stop")
                    event_evidence.setdefault(key, deque()).append(
                        video_time
                    )

            for id_a, id_b in interactions:
                motion_a = motion_signals[id_a]
                motion_b = motion_signals[id_b]

                abnormal = (
                    motion_a["brake"]
                    or motion_a["direction"]
                    or motion_b["brake"]
                    or motion_b["direction"]
                )

                if abnormal:
                    key = ("pair", id_a, id_b, "interaction")
                    event_evidence.setdefault(key, deque()).append(
                        video_time
                    )

            # Prune old evidence.
            for key in list(event_evidence.keys()):
                timestamps = event_evidence[key]

                while (
                    timestamps
                    and video_time - timestamps[0] > EVIDENCE_WINDOW
                ):
                    timestamps.popleft()

                if not timestamps:
                    del event_evidence[key]

            # ------------------------------------------------
            # D. CANDIDATE DETECTION
            # ------------------------------------------------

            candidates = {}

            # A motion anomaly is enough to flag a review
            # candidate. Interaction adds supporting evidence.
            for vehicle_id, motion in motion_signals.items():
                evidence = set()

                if motion["brake"]:
                    evidence.add("sudden_speed_drop")

                if motion["direction"]:
                    evidence.add("abrupt_direction_change")

                # Optical flow alone is not enough to declare
                # an accident. It is displayed for diagnostics.
                if evidence:
                    candidates[(vehicle_id,)] = evidence

            # Promote interacting vehicles when motion changes.
            for id_a, id_b in interactions:
                motion_a = motion_signals[id_a]
                motion_b = motion_signals[id_b]

                evidence = {"vehicle_interaction"}

                if motion_a["brake"]:
                    evidence.add("vehicle_A_speed_drop")

                if motion_a["direction"]:
                    evidence.add("vehicle_A_direction_change")

                if motion_b["brake"]:
                    evidence.add("vehicle_B_speed_drop")

                if motion_b["direction"]:
                    evidence.add("vehicle_B_direction_change")

                if len(evidence) > 1:
                    candidates[(id_a, id_b)] = evidence

            # ------------------------------------------------
            # E. ALERTS AND CSV LOGGING
            # ------------------------------------------------

            alert_ids = set()

            for incident_ids, evidence in candidates.items():

                # Two distinct clues are preferred for a pair.
                # Single-vehicle anomalies can be reviewed too.
                if len(incident_ids) > 1:
                    if len(evidence) < MIN_ALERT_SCORE:
                        continue

                alert_key = tuple(sorted(incident_ids))

                last_time = last_alert_time.get(alert_key, -1e9)

                if video_time - last_time < ALERT_COOLDOWN:
                    continue

                last_alert_time[alert_key] = video_time
                total_alerts += 1
                alert_ids.update(incident_ids)

                alert_time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                evidence_text = ", ".join(sorted(evidence))
                vehicle_text = ", ".join(map(str, incident_ids))

                print(
                    f"[POSSIBLE ACCIDENT] "
                    f"t={video_time:.2f}s "
                    f"vehicles={vehicle_text} "
                    f"evidence={evidence_text}"
                )

                csv_writer.writerow([
                    alert_time,
                    round(video_time, 2),
                    vehicle_text,
                    evidence_text,
                    "Requires human review"
                ])

                csv_file.flush()

                try:
                    import winsound
                    winsound.MessageBeep(
                        winsound.MB_ICONEXCLAMATION
                    )
                except (ImportError, RuntimeError):
                    pass

            # ------------------------------------------------
            # F. VISUALIZATION
            # ------------------------------------------------

            output = frame.copy()

            for vehicle_id, track in tracks.items():
                x1, y1, x2, y2 = track["box"]
                motion = motion_signals[vehicle_id]

                if vehicle_id in alert_ids:
                    color = (0, 0, 255)
                    label = f"REVIEW ID:{vehicle_id}"
                else:
                    color = (0, 200, 0)
                    label = f"ID:{vehicle_id}"

                cv2.rectangle(
                    output, (x1, y1), (x2, y2), color, 2
                )

                cv2.putText(
                    output,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                diagnostic = (
                    f"v:{motion['speed']:.2f} "
                    f"old:{motion['previous_speed']:.2f} "
                    f"flow:{motion['flow_pixels']:.1f}"
                )

                cv2.putText(
                    output,
                    diagnostic,
                    (x1, min(height - 5, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 0),
                    1
                )

            status = (
                "POSSIBLE INCIDENT - REVIEW REQUIRED"
                if alert_ids
                else "MONITORING ROAD TRAFFIC"
            )

            cv2.rectangle(
                output, (0, 0), (width, 72), (30, 30, 30), -1
            )

            cv2.putText(
                output,
                "VITA AEGIS | AI ROAD MONITOR",
                (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                output,
                status,
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if alert_ids else (0, 200, 0),
                2
            )

            cv2.putText(
                output,
                f"Alerts: {total_alerts} | t={video_time:.1f}s",
                (max(10, width - 280), height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            if DISPLAY_VIDEO:
                cv2.imshow("Vita Aegis", output)

            if writer is not None:
                writer.write(output)

            if DISPLAY_VIDEO and cv2.waitKey(1) & 0xFF == ord("q"):
                break

            previous_gray = current_gray
            previous_video_time = video_time

finally:
    video.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

print("\nVita Aegis processing completed.")
print(f"Total candidate alerts: {total_alerts}")
print(f"CSV log: {CSV_PATH}")

if SAVE_VIDEO:
    print(f"Annotated video: {OUTPUT_PATH}")
