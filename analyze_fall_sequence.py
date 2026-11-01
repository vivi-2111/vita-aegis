
import cv2
import csv
import math
import numpy as np
from ultralytics import YOLO


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

VIDEO = "accident_clip.mp4"

OUTPUT = "fall_sequence_analysis.mp4"
CSV_OUTPUT = "fall_sequence_metrics.csv"

START_TIME = 3.5
END_TIME = 7.0

CONFIDENCE_THRESHOLD = 0.15

BASELINE_END_TIME = 4.0
BASELINE_FRAMES = 20

MAX_LOST_FRAMES = 5

# Original video crop coordinates
CROP_X1, CROP_Y1 = 450, 180
CROP_X2, CROP_Y2 = 850, 520

# Rider's approximate position inside the crop
ANCHOR = np.array([230, 130], dtype=float)

MAX_CENTER_DISTANCE = 60


# YOLO COCO-17 body keypoint indices
KEYPOINT_NAMES = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}


# ==========================================================
# 2. HELPER FUNCTIONS
# ==========================================================

def valid_number(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def calculate_torso_angle(shoulder_mid, hip_mid):
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    angle = math.degrees(
        math.atan2(abs(dx), abs(dy))
    )

    return angle


def create_empty_keypoint_data():
    data = {}

    for name in KEYPOINT_NAMES:
        data[f"{name}_x"] = ""
        data[f"{name}_y"] = ""
        data[f"{name}_conf"] = ""

    return data


def extract_keypoint_data(kp, confs):
    data = create_empty_keypoint_data()

    for name, idx in KEYPOINT_NAMES.items():

        if idx >= len(kp) or idx >= len(confs):
            continue

        x, y = kp[idx]
        confidence = confs[idx]

        if (
            valid_number(x)
            and valid_number(y)
            and valid_number(confidence)
            and confidence >= CONFIDENCE_THRESHOLD
        ):
            data[f"{name}_x"] = round(float(x), 2)
            data[f"{name}_y"] = round(float(y), 2)
            data[f"{name}_conf"] = round(float(confidence), 3)

    return data


# ==========================================================
# 3. LOAD YOLO POSE MODEL
# ==========================================================

print("Loading YOLOv8 Pose model...")

model = YOLO("yolov8n-pose.pt")


# ==========================================================
# 4. OPEN VIDEO
# ==========================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open input video: {VIDEO}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if not valid_number(fps) or fps <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS.")

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(f"Video FPS: {fps}")
print(f"Total frames: {total_frames}")


# ==========================================================
# 5. CREATE OUTPUT VIDEO
# ==========================================================

crop_width = CROP_X2 - CROP_X1
crop_height = CROP_Y2 - CROP_Y1

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    fps,
    (crop_width, crop_height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError(
        "Could not create output video."
    )


# ==========================================================
# 6. INITIALIZE ANALYSIS VARIABLES
# ==========================================================

rows = []

baseline_angles = []

previous_center = None

last_center_for_movement = None

lost_frames = 0

frames_processed = 0

valid_pose_frames = 0


# ==========================================================
# 7. PROCESS VIDEO
# ==========================================================

print("\nStarting fall-sequence analysis...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number = int(
        cap.get(cv2.CAP_PROP_POS_FRAMES)
    ) - 1

    time_sec = frame_number / fps

    # Process only the required time window
    if time_sec < START_TIME:
        continue

    if time_sec > END_TIME:
        break

    # ------------------------------------------------------
    # Extract rider's region
    # ------------------------------------------------------

    crop = frame[
        CROP_Y1:CROP_Y2,
        CROP_X1:CROP_X2
    ].copy()

    if crop.shape[:2] != (crop_height, crop_width):
        print("Warning: Invalid crop dimensions.")
        continue

    frames_processed += 1

    # ------------------------------------------------------
    # Initialize frame measurements
    # ------------------------------------------------------

    detected = 0

    center_x = ""
    center_y = ""

    box_width = ""
    box_height = ""

    aspect_ratio = ""

    torso_angle = ""
    baseline_angle = ""
    angle_change = ""

    center_displacement = ""

    posture_marker = 0
    keypoints_valid = 0

    keypoint_data = create_empty_keypoint_data()

    # ------------------------------------------------------
    # Run YOLOv8 Pose
    # ------------------------------------------------------

    results = model.predict(
        crop,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    if (
        result.boxes is not None
        and result.keypoints is not None
        and len(result.boxes) > 0
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        keypoints = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            keypoint_confidences = (
                result.keypoints.conf.cpu().numpy()
            )
        else:
            keypoint_confidences = np.ones(
                (len(keypoints), len(keypoints[0]))
            )

        # --------------------------------------------------
        # Select the detection closest to the rider
        # --------------------------------------------------

        candidates = []

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = box

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            center = np.array(
                [cx, cy],
                dtype=float
            )

            if previous_center is not None:

                distance = np.linalg.norm(
                    center - previous_center
                )

            else:

                distance = np.linalg.norm(
                    center - ANCHOR
                )

            candidates.append(
                (distance, i, center)
            )

        candidates.sort(
            key=lambda item: item[0]
        )

        best_distance, best_index, best_center = candidates[0]

        # --------------------------------------------------
        # Accept detection only within tracking distance
        # --------------------------------------------------

        if best_distance <= MAX_CENTER_DISTANCE:

            detected = 1

            lost_frames = 0

            previous_center = best_center.copy()

            x1, y1, x2, y2 = boxes[best_index]

            kp = keypoints[best_index]

            confs = keypoint_confidences[best_index]

            # Bounding box measurements
            center_x = float(best_center[0])
            center_y = float(best_center[1])

            box_width = float(x2 - x1)
            box_height = float(y2 - y1)

            if box_height > 0:
                aspect_ratio = box_width / box_height

            # --------------------------------------------------
            # Extract all 8 body keypoints
            # --------------------------------------------------

            keypoint_data = extract_keypoint_data(
                kp,
                confs
            )

            # --------------------------------------------------
            # Validate torso keypoints
            # --------------------------------------------------

            needed = [5, 6, 11, 12]

            valid = True

            for idx in needed:

                if idx >= len(kp) or idx >= len(confs):
                    valid = False
                    break

                if not (
                    np.isfinite(kp[idx]).all()
                    and valid_number(confs[idx])
                    and confs[idx] >= CONFIDENCE_THRESHOLD
                ):
                    valid = False
                    break

            # --------------------------------------------------
            # Calculate torso posture
            # --------------------------------------------------

            if valid:

                keypoints_valid = 1

                valid_pose_frames += 1

                shoulder_mid = (
                    kp[5] + kp[6]
                ) / 2

                hip_mid = (
                    kp[11] + kp[12]
                ) / 2

                torso_angle = calculate_torso_angle(
                    shoulder_mid,
                    hip_mid
                )

                # Collect baseline posture samples
                if (
                    time_sec < BASELINE_END_TIME
                    and len(baseline_angles) < BASELINE_FRAMES
                ):
                    baseline_angles.append(
                        torso_angle
                    )

                # Calculate angle change
                if len(baseline_angles) > 0:

                    baseline_angle = float(
                        np.median(baseline_angles)
                    )

                    angle_change = abs(
                        torso_angle - baseline_angle
                    )

                    # Existing posture signal
                    if (
                        torso_angle >= 45
                        and angle_change >= 25
                    ):
                        posture_marker = 1

                # Draw torso line
                shoulder_point = tuple(
                    np.array(shoulder_mid, dtype=int)
                )

                hip_point = tuple(
                    np.array(hip_mid, dtype=int)
                )

                cv2.line(
                    crop,
                    shoulder_point,
                    hip_point,
                    (0, 255, 255),
                    2
                )

            # --------------------------------------------------
            # Calculate center displacement
            # --------------------------------------------------

            current_center = np.array(
                [center_x, center_y],
                dtype=float
            )

            if last_center_for_movement is not None:

                center_displacement = float(
                    np.linalg.norm(
                        current_center - last_center_for_movement
                    )
                )

            last_center_for_movement = current_center.copy()

            # --------------------------------------------------
            # Draw bounding box
            # --------------------------------------------------

            cv2.rectangle(
                crop,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # --------------------------------------------------
            # Draw body keypoints
            # --------------------------------------------------

            for idx in KEYPOINT_NAMES.values():

                if idx >= len(kp) or idx >= len(confs):
                    continue

                if (
                    valid_number(confs[idx])
                    and confs[idx] >= CONFIDENCE_THRESHOLD
                    and np.isfinite(kp[idx]).all()
                ):

                    point = tuple(
                        np.array(kp[idx], dtype=int)
                    )

                    cv2.circle(
                        crop,
                        point,
                        4,
                        (0, 0, 255),
                        -1
                    )

        else:

            lost_frames += 1

    else:

        lost_frames += 1

    # ------------------------------------------------------
    # Reset tracking after too many missing frames
    # ------------------------------------------------------

    if lost_frames > MAX_LOST_FRAMES:

        previous_center = None

        last_center_for_movement = None

    # ------------------------------------------------------
    # Display information on video
    # ------------------------------------------------------

    cv2.putText(
        crop,
        f"Time: {time_sec:.2f}s",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    if torso_angle != "":

        cv2.putText(
            crop,
            f"Torso Angle: {torso_angle:.1f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    if posture_marker == 1:

        cv2.putText(
            crop,
            "POSTURE CHANGE",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # ------------------------------------------------------
    # Save measurements for this frame
    # ------------------------------------------------------

    row = {
        "time_sec": round(time_sec, 3),
        "frame": frame_number,
        "detected": detected,
        "center_x": center_x,
        "center_y": center_y,
        "box_width": box_width,
        "box_height": box_height,
        "aspect_ratio": aspect_ratio,
        "torso_angle_deg": torso_angle,
        "baseline_angle_deg": baseline_angle,
        "angle_change_deg": angle_change,
        "center_displacement_px": center_displacement,
        "posture_change_marker": posture_marker,
        "keypoints_valid": keypoints_valid,
        **keypoint_data
    }

    rows.append(row)

    # Write annotated frame
    out.write(crop)


# ==========================================================
# 8. RELEASE VIDEO RESOURCES
# ==========================================================

cap.release()
out.release()


# ==========================================================
# 9. SAVE CSV FILE
# ==========================================================

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

for name in KEYPOINT_NAMES:

    columns.extend([
        f"{name}_x",
        f"{name}_y",
        f"{name}_conf"
    ])


with open(
    CSV_OUTPUT,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=columns
    )

    writer.writeheader()

    writer.writerows(rows)


# ==========================================================
# 10. FINAL REPORT
# ==========================================================

print("\n" + "=" * 50)

print("FALL-SEQUENCE ANALYSIS COMPLETED")

print("=" * 50)

print(f"Input video: {VIDEO}")

print(f"Annotated video: {OUTPUT}")

print(f"Metrics CSV: {CSV_OUTPUT}")

print(f"Frames processed: {frames_processed}")

print(f"Frames with rider detection: {sum(row['detected'] for row in rows)}")

print(f"Frames with valid torso keypoints: {valid_pose_frames}")

print(f"Baseline samples collected: {len(baseline_angles)}")

print("=" * 50)

print("\nAll 8 body keypoints have been saved.")
