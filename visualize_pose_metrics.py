
import cv2
import pandas as pd
import numpy as np

# ==========================================
# FILE CONFIGURATION
# ==========================================

VIDEO_FILE = "occupants_pose_960.mp4"
POSE_FILE = "occupants_pose_960.csv"
METRICS_FILE = "pose_timeline_metrics.csv"

OUTPUT_FILE = "pose_metrics_visualization.mp4"

CONF_THRESHOLD = 0.25
VIDEO_START_TIME = 3.0

# ==========================================
# LOAD CSV FILES
# ==========================================

print("Loading pose and metrics data...")

pose_df = pd.read_csv(POSE_FILE)
metrics_df = pd.read_csv(METRICS_FILE)

print("Pose records:", len(pose_df))
print("Metrics records:", len(metrics_df))

# Merge original keypoints with calculated angles
df = pd.merge(
    pose_df,
    metrics_df[["frame", "track_id", "torso_angle_deg"]],
    on=["frame", "track_id"],
    how="left"
)

print("Merged records:", len(df))

# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Unable to open video: {VIDEO_FILE}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    raise ValueError("Invalid video FPS.")

print("\nVideo information:")
print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Total frames:", total_frames)

# ==========================================
# CREATE OUTPUT VIDEO
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_FILE,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError("Unable to create output video.")

# ==========================================
# CALCULATE JOINT MIDPOINT
# ==========================================

def get_midpoint(row, joint):

    points = []

    for side in ["left", "right"]:

        x = row[f"{side}_{joint}_x"]
        y = row[f"{side}_{joint}_y"]
        conf = row[f"{side}_{joint}_conf"]

        if (
            pd.notna(x)
            and pd.notna(y)
            and pd.notna(conf)
            and x > 0
            and y > 0
            and conf >= CONF_THRESHOLD
        ):

            points.append((int(x), int(y)))

    if not points:
        return None

    midpoint_x = int(
        np.mean([point[0] for point in points])
    )

    midpoint_y = int(
        np.mean([point[1] for point in points])
    )

    return midpoint_x, midpoint_y


# ==========================================
# GROUP DATA BY FRAME
# ==========================================

grouped = {
    int(frame_number): group
    for frame_number, group in df.groupby("frame")
}

# ==========================================
# PROCESS VIDEO FRAME BY FRAME
# ==========================================

print("\nStarting visualization...")

frame_index = 0
valid_detections = 0
skipped_detections = 0

START_FRAME = int(round(VIDEO_START_TIME * fps))

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Map video frame to original source frame
    original_frame = START_FRAME + frame_index

    frame_data = grouped.get(original_frame)

    if frame_data is not None:

        for _, row in frame_data.iterrows():

            # Skip missing tracking IDs
            if pd.isna(row["track_id"]):
                skipped_detections += 1
                continue

            track_id = int(row["track_id"])

            # Only visualize our two verified occupants
            if track_id not in [16, 19]:
                continue

            shoulder = get_midpoint(row, "shoulder")
            hip = get_midpoint(row, "hip")

            angle = row["torso_angle_deg"]

            if shoulder is None or hip is None:
                continue

            if pd.isna(angle):
                continue

            valid_detections += 1

            # Draw shoulder point - GREEN
            cv2.circle(
                frame,
                shoulder,
                7,
                (0, 255, 0),
                -1
            )

            # Draw hip point - RED
            cv2.circle(
                frame,
                hip,
                7,
                (0, 0, 255),
                -1
            )

            # Connect shoulder and hip - YELLOW
            cv2.line(
                frame,
                shoulder,
                hip,
                (0, 255, 255),
                3
            )

            # Display track ID and torso angle
            label = (
                f"ID {track_id} | "
                f"Angle: {angle:.1f} deg"
            )

            cv2.putText(
                frame,
                label,
                (
                    shoulder[0],
                    max(25, shoulder[1] - 15)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # Display timestamp
    timestamp = (
        VIDEO_START_TIME + frame_index / fps
    )

    cv2.putText(
        frame,
        f"Time: {timestamp:.2f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    out.write(frame)

    frame_index += 1

    if frame_index % 30 == 0:
        print(
            f"Processed {frame_index}/{total_frames} frames"
        )

# ==========================================
# RELEASE RESOURCES
# ==========================================

cap.release()
out.release()

cv2.destroyAllWindows()

# ==========================================
# FINAL REPORT
# ==========================================

print("\n================================")
print("VISUALIZATION COMPLETED")
print("================================")

print("Output video:", OUTPUT_FILE)
print("Frames processed:", frame_index)
print("Valid occupant measurements:", valid_detections)
print("Untracked detections skipped:", skipped_detections)

print("\nTracked occupants:")
print("ID 16")
print("ID 19")

print("\nColor reference:")
print("GREEN  = Shoulder midpoint")
print("RED    = Hip midpoint")
print("YELLOW = Shoulder-to-hip line")

print("\n================================")
