
import pandas as pd
import numpy as np

FILE = "temporal_accident_metrics.csv"

df = pd.read_csv(FILE)

if df.empty:
    print("No detection records found.")
    raise SystemExit

# Inspect the important accident timeline
start_time = 4.700
end_time = 6.000

window = df[
    (df["time_sec"] >= start_time) &
    (df["time_sec"] <= end_time)
].copy()

if window.empty:
    print("No detections found in the selected timeline.")
    raise SystemExit

print("\n========== DETECTIONS AROUND THE FALL ==========")

print(
    window[
        [
            "time_sec",
            "frame",
            "track_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "torso_angle_deg"
        ]
    ].sort_values(
        ["frame", "confidence"],
        ascending=[True, False]
    ).round(2).to_string(index=False)
)

print("\n========== DETECTION COUNT PER FRAME ==========")

for frame, group in window.groupby("frame"):

    ids = group["track_id"].tolist()

    print(
        f"Time: {group['time_sec'].iloc[0]:.3f}s | "
        f"Frame: {frame} | "
        f"Detections: {len(group)} | "
        f"Track IDs: {ids}"
    )

print("\n========== TRACK ID SUMMARY ==========")

for track_id, group in window.groupby("track_id", dropna=False):

    print(
        f"ID {track_id}: "
        f"{len(group)} records, "
        f"first={group['time_sec'].min():.3f}s, "
        f"last={group['time_sec'].max():.3f}s"
    )

print("\nInspection completed.")
