
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "filtered_detection_inspection/filtered_results.csv"
OUTPUT_FILE = "nms_detection_results.csv"

IOU_THRESHOLD = 0.50

# --------------------------------------------------
# 1. Load detections
# --------------------------------------------------

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(f"Cannot find {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required = {
    "time_sec", "frame", "confidence",
    "x1", "y1", "x2", "y2"
}

if not required.issubset(df.columns):
    raise ValueError("Required bounding-box columns are missing.")

print("Loaded:", len(df), "detections")

# --------------------------------------------------
# 2. Calculate IoU
# --------------------------------------------------

def calculate_iou(a, b):

    left = max(a["x1"], b["x1"])
    top = max(a["y1"], b["y1"])

    right = min(a["x2"], b["x2"])
    bottom = min(a["y2"], b["y2"])

    width = max(0, right - left)
    height = max(0, bottom - top)

    intersection = width * height

    area_a = max(0, a["x2"] - a["x1"]) * max(
        0, a["y2"] - a["y1"]
    )

    area_b = max(0, b["x2"] - b["x1"]) * max(
        0, b["y2"] - b["y1"]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


# --------------------------------------------------
# 3. Apply Non-Maximum Suppression
# --------------------------------------------------

def apply_nms(frame_data, threshold):

    candidates = frame_data.sort_values(
        "confidence", ascending=False
    ).to_dict("records")

    kept = []

    while candidates:

        best = candidates.pop(0)
        kept.append(best)

        remaining = []

        for candidate in candidates:

            iou = calculate_iou(best, candidate)

            if iou <= threshold:
                remaining.append(candidate)

        candidates = remaining

    return kept


# --------------------------------------------------
# 4. Process each frame
# --------------------------------------------------

final_records = []

print("\n========== NMS RESULTS ==========")

for frame, frame_data in df.groupby("frame"):

    original_count = len(frame_data)

    kept = apply_nms(
        frame_data,
        IOU_THRESHOLD
    )

    final_count = len(kept)

    time_value = frame_data["time_sec"].iloc[0]

    print(
        f"{time_value:.3f}s | "
        f"Frame {frame} | "
        f"Before NMS: {original_count} | "
        f"After NMS: {final_count}"
    )

    for box_number, box in enumerate(kept, start=1):

        final_records.append({
            "time_sec": box["time_sec"],
            "frame": box["frame"],
            "detection_number": box_number,
            "confidence": box["confidence"],
            "x1": box["x1"],
            "y1": box["y1"],
            "x2": box["x2"],
            "y2": box["y2"],
            "width": box["x2"] - box["x1"],
            "height": box["y2"] - box["y1"]
        })

# --------------------------------------------------
# 5. Save output
# --------------------------------------------------

result = pd.DataFrame(final_records)

result.to_csv(OUTPUT_FILE, index=False)

print("\nNMS processing completed.")
print("Original detections:", len(df))
print("Retained detections:", len(result))
print("Saved:", OUTPUT_FILE)

print("\nFinal detection summary:")
print(
    result.groupby("time_sec").size().to_string()
)

