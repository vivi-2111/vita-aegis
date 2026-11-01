
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Select a CSV containing individual box coordinates
# --------------------------------------------------

possible_files = [
    "filtered_results.csv",
    "temporal_accident_metrics.csv",
    "occupants_pose_960.csv"
]

required_columns = {"x1", "y1", "x2", "y2"}

data = None
source_file = None

for filename in possible_files:
    path = Path(filename)

    if not path.exists():
        continue

    temp = pd.read_csv(path)
    temp.columns = temp.columns.str.strip().str.lower()

    if required_columns.issubset(temp.columns):
        data = temp
        source_file = filename
        break

if data is None:
    print("ERROR: No CSV with x1, y1, x2, y2 columns found.")
    print("Please check the CSV filenames and column names.")
    raise SystemExit

print(f"\nUsing file: {source_file}")
print("Available columns:", list(data.columns))

# --------------------------------------------------
# 2. Identify timestamp and optional columns
# --------------------------------------------------

time_column = next(
    (c for c in ["time_sec", "timestamp", "time"] if c in data.columns),
    None
)

if time_column is None:
    print("ERROR: Timestamp column not found.")
    raise SystemExit

if "confidence" not in data.columns:
    if "conf" in data.columns:
        data["confidence"] = data["conf"]
    else:
        data["confidence"] = np.nan

if "track_id" not in data.columns:
    data["track_id"] = np.nan

# Convert required values to numeric
numeric_columns = [
    time_column, "x1", "y1", "x2", "y2", "confidence"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=[time_column, "x1", "y1", "x2", "y2"])

# --------------------------------------------------
# 3. Calculate box dimensions and area
# --------------------------------------------------

data["width"] = (data["x2"] - data["x1"]).clip(lower=0)
data["height"] = (data["y2"] - data["y1"]).clip(lower=0)
data["area"] = data["width"] * data["height"]

data["center_x"] = (data["x1"] + data["x2"]) / 2
data["center_y"] = (data["y1"] + data["y2"]) / 2

# --------------------------------------------------
# 4. Calculate overlap between two boxes
# --------------------------------------------------

def calculate_overlap(box_a, box_b):

    x_left = max(box_a["x1"], box_b["x1"])
    y_top = max(box_a["y1"], box_b["y1"])

    x_right = min(box_a["x2"], box_b["x2"])
    y_bottom = min(box_a["y2"], box_b["y2"])

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)

    intersection = intersection_width * intersection_height

    union = box_a["area"] + box_b["area"] - intersection

    iou = intersection / union if union > 0 else 0

    smaller_area = min(box_a["area"], box_b["area"])

    overlap_ratio = (
        intersection / smaller_area if smaller_area > 0 else 0
    )

    return iou, overlap_ratio


# --------------------------------------------------
# 5. Inspect important timestamps
# --------------------------------------------------

target_times = [4.8, 4.9, 5.0, 5.2, 5.4, 5.6, 5.633, 5.667]

output_records = []

print("\n========== BOX TRANSITION ANALYSIS ==========")

for target in target_times:

    frame_data = data[
        (data[time_column] - target).abs() <= 0.02
    ].copy()

    frame_data = frame_data.sort_values(
        by="confidence", ascending=False, na_position="last"
    )

    print(f"\n\nTIME: {target:.3f} seconds")
    print(f"Detections in CSV: {len(frame_data)}")

    boxes = frame_data.to_dict("records")

    for index, box in enumerate(boxes):

        print(f"\nBox {index + 1}")
        print(f"Track ID: {box.get('track_id')}")
        print(f"Confidence: {box.get('confidence'):.3f}"
              if pd.notna(box.get("confidence"))
              else "Confidence: unavailable")

        print(
            f"Coordinates: "
            f"({box['x1']:.1f}, {box['y1']:.1f}, "
            f"{box['x2']:.1f}, {box['y2']:.1f})"
        )

        print(
            f"Width: {box['width']:.1f} | "
            f"Height: {box['height']:.1f} | "
            f"Area: {box['area']:.1f}"
        )

        output_records.append({
            "target_time": target,
            "actual_time": box[time_column],
            "box_number": index + 1,
            "track_id": box.get("track_id"),
            "confidence": box.get("confidence"),
            "x1": box["x1"],
            "y1": box["y1"],
            "x2": box["x2"],
            "y2": box["y2"],
            "width": box["width"],
            "height": box["height"],
            "area": box["area"]
        })

    # Compare every pair of detections
    if len(boxes) >= 2:

        print("\nPairwise box comparisons:")

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):

                iou, overlap_ratio = calculate_overlap(
                    boxes[i], boxes[j]
                )

                center_distance = np.sqrt(
                    (boxes[i]["center_x"] - boxes[j]["center_x"]) ** 2
                    +
                    (boxes[i]["center_y"] - boxes[j]["center_y"]) ** 2
                )

                print(
                    f"Box {i + 1} vs Box {j + 1}: "
                    f"IoU={iou:.3f}, "
                    f"Overlap ratio={overlap_ratio:.3f}, "
                    f"Center distance={center_distance:.1f}px"
                )

# --------------------------------------------------
# 6. Save diagnostic report
# --------------------------------------------------

if output_records:

    result = pd.DataFrame(output_records)

    result.to_csv(
        "box_transition_details.csv",
        index=False
    )

    print("\nAnalysis completed successfully.")
    print("Saved: box_transition_details.csv")

else:
    print("\nNo matching detections found.")
    print("Check the timestamp column and selected CSV.")
