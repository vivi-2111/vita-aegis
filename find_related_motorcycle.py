
import pandas as pd
import numpy as np

df = pd.read_csv("crash_tracking.csv")

# Rider track identified during visual inspection.
rider = df[
    (df["object"] == "person") &
    (df["track_id"] == 55) &
    (df["time_sec"].between(4.0, 7.0))
].copy()

bikes = df[
    (df["object"] == "motorcycle") &
    (df["time_sec"].between(4.0, 7.0))
].copy()

if rider.empty or bikes.empty:
    print("Rider or motorcycle detections missing in this window.")
    raise SystemExit

# Compare detections only within the same video frame.
pairs = rider.merge(
    bikes,
    on="frame",
    suffixes=("_rider", "_bike")
)

if pairs.empty:
    print("No matching frames found.")
    raise SystemExit

pairs["distance"] = np.sqrt(
    (pairs["center_x_rider"] - pairs["center_x_bike"]) ** 2
    + (pairs["center_y_rider"] - pairs["center_y_bike"]) ** 2
)

# Keep the closest motorcycle candidate in each frame.
closest = pairs.loc[
    pairs.groupby("frame")["distance"].idxmin()
].copy()

summary = closest.groupby("track_id_bike").agg(
    frames=("frame", "nunique"),
    first_time=("time_sec_rider", "min"),
    last_time=("time_sec_rider", "max"),
    mean_distance=("distance", "mean"),
    min_distance=("distance", "min")
).sort_values(
    ["frames", "mean_distance"],
    ascending=[False, True]
)

print("\n=== MOTORCYCLE CANDIDATES NEAR RIDER 55 ===")
print(summary.head(10).round(2).to_string())

print("\nClosest candidate per frame:")
print(
    closest[
        ["time_sec_rider", "track_id_bike", "distance"]
    ].head(20).round(2).to_string(index=False)
)

print("\nNote: spatial proximity suggests candidates;")
print("it does not prove that a motorcycle belongs to the rider.")
