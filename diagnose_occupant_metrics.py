
import pandas as pd
import numpy as np

# Load tracking data
df = pd.read_csv("multi_person_tracking.csv")

# Calculate bounding box centre
df["center_x"] = (df["x1"] + df["x2"]) / 2
df["center_y"] = (df["y1"] + df["y2"]) / 2

# Calculate bounding box width
df["box_width"] = df["x2"] - df["x1"]

# Calculate average shoulder coordinates
shoulder_x = (
    df["left_shoulder_x"] + df["right_shoulder_x"]
) / 2

shoulder_y = (
    df["left_shoulder_y"] + df["right_shoulder_y"]
) / 2

# Calculate average hip coordinates
hip_x = (
    df["left_hip_x"] + df["right_hip_x"]
) / 2

hip_y = (
    df["left_hip_y"] + df["right_hip_y"]
) / 2

# Calculate shoulder-to-hip distance
df["torso_length"] = np.sqrt(
    (shoulder_x - hip_x) ** 2 +
    (shoulder_y - hip_y) ** 2
)

# Select our two occupants
occupants = df[df["track_id"].isin([12, 13])].copy()

# Select required columns
columns = [
    "time_sec",
    "track_id",
    "center_x",
    "center_y",
    "box_width",
    "box_height",
    "torso_length"
]

print("\n========== OCCUPANT DIAGNOSTICS ==========\n")

print(
    occupants[columns]
    .round(2)
    .to_string(index=False)
)

print("\n========== SUMMARY ==========\n")

print(
    occupants.groupby("track_id")[[
        "center_x",
        "center_y",
        "box_width",
        "box_height",
        "torso_length"
    ]]
    .agg(["min", "max", "mean"])
    .round(2)
)

print("\nDiagnostic analysis completed.")
