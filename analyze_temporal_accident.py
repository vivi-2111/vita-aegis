
import cv2
import numpy as np
import pandas as pd
import math
from ultralytics import YOLO
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

VIDEO_PATH = "accident_clip.mp4"
MODEL_PATH = "yolov8n-pose.pt"

OUTPUT_VIDEO = "temporal_accident_analysis.mp4"
OUTPUT_METRICS = "temporal_accident_metrics.csv"
OUTPUT_CANDIDATES = "temporal_accident_candidates.csv"

# Accident area selected earlier
ROI_X1 = 609
ROI_Y1 = 275
ROI_X2 = 872
ROI_Y2 = 426

# Expand the area to avoid missing partially visible people
ROI_PADDING = 100

CONFIDENCE = 0.15
IMAGE_SIZE = 960
KEYPOINT_CONFIDENCE = 0.25

# COCO keypoint names
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def midpoint(keypoints, confidence, index1, index2):
    """Calculate midpoint when both keypoints are reliable."""

    if confidence[index1] < KEYPOINT_CONFIDENCE:
        return None

    if confidence[index2] < KEYPOINT_CONFIDENCE:
        return None

    x = (keypoints[index1][0] + keypoints[index2][0]) / 2
    y = (keypoints[index1][1] + keypoints[index2][1]) / 2

    return float(x), float(y)


def calculate_torso_angle(shoulder, hip):
    """
    0 degrees   = vertical torso in image
    90 degrees  = horizontal torso in image
    """

    if shoulder is None or hip is None:
        return np.nan

    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]

    return math.degrees(math.atan2(abs(dx), abs(dy)))


def intersects_roi(box):
    """Check whether a detection overlaps the expanded accident area."""

    x1, y1, x2, y2 = box

    rx1 = ROI_X1 - ROI_PADDING
    ry1 = ROI_Y1 - ROI_PADDING
    rx2 = ROI_X2 + ROI_PADDING
    ry2 = ROI_Y2 + ROI_PADDING

    return (
        x2 >= rx1 and x1 <= rx2 and
        y2 >= ry1 and y1 <= ry2
    )


# ==========================================
# 3. LOAD VIDEO AND MODEL
# ==========================================

if not Path(VIDEO_PATH).exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

print("Loading YOLOv8 Pose model...")

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nVideo information")
print("-----------------")
print("FPS:", fps)
print("Total frames:", total_frames)
print("Duration:", round(total_frames / fps, 2), "seconds")
print("Resolution:", width, "x", height)


# ==========================================
# 4. CREATE OUTPUT VIDEO
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

all_records = []

frame_number = 0

print("\nStarting full-video pose tracking...")


# ==========================================
# 5. PROCESS EVERY FRAME
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    time_sec = frame_number - 1
    time_sec = time_sec / fps

    # Persistent ByteTrack tracking across frames
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    result = results[0]

    # Draw the expanded accident ROI
    rx1 = max(0, ROI_X1 - ROI_PADDING)
    ry1 = max(0, ROI_Y1 - ROI_PADDING)
    rx2 = min(width, ROI_X2 + ROI_PADDING)
    ry2 = min(height, ROI_Y2 + ROI_PADDING)

    cv2.rectangle(
        frame,
        (rx1, ry1),
        (rx2, ry2),
        (255, 200, 0),
        2
    )

    cv2.putText(
        frame,
        "ACCIDENT ANALYSIS AREA",
        (rx1, max(20, ry1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 0),
        2
    )

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        box_conf = result.boxes.conf.cpu().numpy()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes)

        keypoints = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            keypoint_conf = result.keypoints.conf.cpu().numpy()
        else:
            keypoint_conf = np.zeros(
                (len(boxes), 17),
                dtype=float
            )

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box.tolist()

            if not intersects_roi(box):
                continue

            track_id = track_ids[i]
            confidence = float(box_conf[i])

            person_keypoints = keypoints[i]
            person_confidence = keypoint_conf[i]

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            box_width = x2 - x1
            box_height = y2 - y1

            aspect_ratio = (
                box_width / box_height
                if box_height > 0 else np.nan
            )

            # Calculate body joint midpoints
            shoulder = midpoint(
                person_keypoints,
                person_confidence,
                5, 6
            )

            hip = midpoint(
                person_keypoints,
                person_confidence,
                11, 12
            )

            knee = midpoint(
                person_keypoints,
                person_confidence,
                13, 14
            )

            ankle = midpoint(
                person_keypoints,
                person_confidence,
                15, 16
            )

            torso_angle = calculate_torso_angle(
                shoulder,
                hip
            )

            torso_length = np.nan

            if shoulder is not None and hip is not None:

                torso_length = math.dist(
                    shoulder,
                    hip
                )

            record = {
                "time_sec": round(time_sec, 3),
                "frame": frame_number - 1,
                "track_id": track_id,
                "confidence": confidence,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,

                "center_x": center_x,
                "center_y": center_y,

                "box_width": box_width,
                "box_height": box_height,
                "aspect_ratio": aspect_ratio,

                "shoulder_mid_x": (
                    shoulder[0] if shoulder else np.nan
                ),
                "shoulder_mid_y": (
                    shoulder[1] if shoulder else np.nan
                ),

                "hip_mid_x": (
                    hip[0] if hip else np.nan
                ),
                "hip_mid_y": (
                    hip[1] if hip else np.nan
                ),

                "knee_mid_x": (
                    knee[0] if knee else np.nan
                ),
                "knee_mid_y": (
                    knee[1] if knee else np.nan
                ),

                "ankle_mid_x": (
                    ankle[0] if ankle else np.nan
                ),
                "ankle_mid_y": (
                    ankle[1] if ankle else np.nan
                ),

                "torso_angle_deg": torso_angle,
                "torso_length_px": torso_length
            }

            # Save all 17 keypoints
            for j, name in enumerate(KEYPOINT_NAMES):

                record[name + "_x"] = float(
                    person_keypoints[j][0]
                )

                record[name + "_y"] = float(
                    person_keypoints[j][1]
                )

                record[name + "_conf"] = float(
                    person_confidence[j]
                )

            all_records.append(record)

            # Draw detection box
            color = (
                (0, 255, 0)
                if track_id != -1
                else (0, 165, 255)
            )

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )

            label = f"ID: {track_id}"

            cv2.putText(
                frame,
                label,
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Draw important pose points
            for joint in [5, 6, 11, 12, 13, 14, 15, 16]:

                if person_confidence[joint] >= KEYPOINT_CONFIDENCE:

                    px, py = person_keypoints[joint]

                    cv2.circle(
                        frame,
                        (int(px), int(py)),
                        4,
                        (0, 0, 255),
                        -1
                    )

            # Draw shoulder-to-hip line
            if shoulder is not None and hip is not None:

                cv2.line(
                    frame,
                    (int(shoulder[0]), int(shoulder[1])),
                    (int(hip[0]), int(hip[1])),
                    (0, 255, 255),
                    3
                )

                cv2.putText(
                    frame,
                    f"Angle: {torso_angle:.1f}",
                    (int(x1), min(height - 10, int(y2) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

    # Display timestamp
    cv2.putText(
        frame,
        f"Time: {time_sec:.2f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(frame)

    if frame_number % 30 == 0:

        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


# ==========================================
# 6. RELEASE VIDEO RESOURCES
# ==========================================

cap.release()
writer.release()

print("\nVideo processing completed.")


# ==========================================
# 7. CREATE TEMPORAL METRICS
# ==========================================

df = pd.DataFrame(all_records)

if df.empty:

    print("No people detected in the expanded accident area.")

    df.to_csv(OUTPUT_METRICS, index=False)

    pd.DataFrame().to_csv(
        OUTPUT_CANDIDATES,
        index=False
    )

    raise SystemExit


# Track IDs -1 are untracked detections.
# Do not combine them into a single person track.

df["center_displacement_px"] = np.nan
df["angle_change_deg"] = np.nan
df["normalized_displacement"] = np.nan
df["preliminary_fall_cue"] = False

for track_id in df["track_id"].dropna().unique():

    if track_id == -1:
        continue

    indices = df.index[df["track_id"] == track_id]

    track = df.loc[indices].sort_values("frame")

    previous = None

    for index, current in track.iterrows():

        if previous is not None:

            frame_gap = (
                current["frame"] - previous["frame"]
            )

            time_gap = frame_gap / fps

            if 0 < time_gap <= 0.20:

                displacement = math.dist(
                    (current["center_x"], current["center_y"]),
                    (previous["center_x"], previous["center_y"])
                )

                angle_change = np.nan

                if (
                    pd.notna(current["torso_angle_deg"])
                    and pd.notna(previous["torso_angle_deg"])
                ):

                    angle_change = abs(
                        current["torso_angle_deg"]
                        - previous["torso_angle_deg"]
                    )

                normalized = (
                    displacement / current["box_height"]
                    if current["box_height"] > 0
                    else np.nan
                )

                df.loc[index, "center_displacement_px"] = displacement
                df.loc[index, "angle_change_deg"] = angle_change
                df.loc[index, "normalized_displacement"] = normalized

                # Preliminary cue only, not a confirmed accident
                if (
                    pd.notna(current["torso_angle_deg"])
                    and current["torso_angle_deg"] >= 35
                    and pd.notna(angle_change)
                    and angle_change >= 20
                    and pd.notna(normalized)
                    and normalized >= 0.10
                ):

                    df.loc[index, "preliminary_fall_cue"] = True

        previous = current


# ==========================================
# 8. SAVE METRICS AND CANDIDATES
# ==========================================

df.to_csv(OUTPUT_METRICS, index=False)

candidates = df[
    df["preliminary_fall_cue"] == True
].copy()

candidates.to_csv(
    OUTPUT_CANDIDATES,
    index=False
)


# ==========================================
# 9. PRINT SUMMARY
# ==========================================

print("\n========== ANALYSIS SUMMARY ==========")

print("Total frames:", total_frames)
print("Processed frames:", frame_number)

print("Detection records:", len(df))

valid_ids = sorted(
    int(value)
    for value in df["track_id"].dropna().unique()
    if value != -1
)

print("Tracked person IDs:", valid_ids)

print("\nRecords per track:")

print(
    df[df["track_id"] != -1]
    .groupby("track_id")
    .size()
)

print("\nPreliminary fall-cue records:", len(candidates))

print("\nOutput files:")
print(OUTPUT_VIDEO)
print(OUTPUT_METRICS)
print(OUTPUT_CANDIDATES)

print("\nTemporal analysis completed successfully.")
