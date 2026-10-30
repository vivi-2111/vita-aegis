
import pandas as pd
import numpy as np

# Load existing tracking data
df = pd.read_csv("crash_tracking.csv")

# Sort objects by tracking ID and frame
df = df.sort_values(["track_id", "frame"]).copy()

# Calculate movement and bounding-box changes
df["speed_px_sec"] = 0.0
df["aspect_ratio"] = df["box_width"] / df["box_height"].replace(0, np.nan)
df["area"] = df["box_width"] * df["box_height"]

df["previous_speed"] = 0.0
df["aspect_change"] = 0.0
df["area_change"] = 0.0

for track_id, group in df.groupby("track_id"):
    idx = group.index

    dx = group["center_x"].diff()
    dy = group["center_y"].diff()
    dt = group["time_sec"].diff()

    valid_dt = dt.where(dt > 0)
    speed = np.sqrt(dx**2 + dy**2) / valid_dt

    df.loc[idx, "speed_px_sec"] = speed.fillna(0)
    df.loc[idx, "previous_speed"] = (
        speed.shift(1).fillna(0)
    )

    df.loc[idx, "aspect_change"] = (
        group["aspect_ratio"].pct_change().abs().fillna(0)
    )

    df.loc[idx, "area_change"] = (
        group["area"].pct_change().abs().fillna(0)
    )

# Identify unusual movement patterns for review
df["speed_change"] = (
    df["speed_px_sec"] - df["previous_speed"]
).abs()

# These are review thresholds, not trained accident classifiers
df["movement_flag"] = df["speed_change"] > 100
df["shape_flag"] = df["aspect_change"] > 0.30
df["size_flag"] = df["area_change"] > 0.40

df["candidate_score"] = (
    df["movement_flag"].astype(int)
    + df["shape_flag"].astype(int)
    + df["size_flag"].astype(int)
)

# Focus on the known accident window
window = df[
    df["time_sec"].between(15, 25)
].copy()

candidates = window[
    (window["candidate_score"] >= 1)
    & (window["confidence"] >= 0.25)
].copy()

columns = [
    "time_sec",
    "frame",
    "object",
    "track_id",
    "confidence",
    "center_x",
    "center_y",
    "speed_px_sec",
    "aspect_change",
    "area_change",
    "candidate_score",
]

candidates = candidates.sort_values(
    ["candidate_score", "time_sec"],
    ascending=[False, True],
)

candidates[columns].to_csv(
    "accident_candidates.csv", index=False
)

print("Movement analysis completed.")
print(f"Frames in target window: {window['frame'].nunique()}")
print(f"Candidate records: {len(candidates)}")
print("Saved report: accident_candidates.csv")

if not candidates.empty:
    print("\nTop candidate events:")
    print(candidates[columns].head(20).to_string(index=False))
else:
    print(
        "No candidates found with the current thresholds. "
        "This does not prove that no accident occurred."
    )
