
import pandas as pd

CSV_FILE = "multi_person_tracking.csv"

df = pd.read_csv(CSV_FILE)

print("\n========== TRACKING SUMMARY ==========\n")

summary = (
    df.groupby("track_id")
    .agg(
        records=("frame", "count"),
        first_time=("time_sec", "min"),
        last_time=("time_sec", "max"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        avg_box_height=("box_height", "mean"),
        valid_pose_frames=("keypoints_valid", "sum")
    )
    .reset_index()
)

summary["duration_sec"] = (
    summary["last_time"] - summary["first_time"]
)

print(summary.round(2).to_string(index=False))

print("\n========== TRACKING BY TIME ==========\n")

for timestamp in sorted(df["time_sec"].unique()):

    current = df[
        df["time_sec"] == timestamp
    ]

    ids = current["track_id"].tolist()

    print(
        f"Time: {timestamp:.2f}s | "
        f"Tracking IDs: {ids}"
    )

print("\n========== ID CONTINUITY ==========\n")

for track_id in sorted(df["track_id"].unique()):

    person = df[df["track_id"] == track_id]

    print(
        f"ID {track_id}: "
        f"{person['time_sec'].min():.2f}s - "
        f"{person['time_sec'].max():.2f}s | "
        f"{len(person)} records"
    )

print("\nAnalysis completed.")
