
import pandas as pd
import numpy as np

FILE = "temporal_accident_metrics.csv"

df = pd.read_csv(FILE)

if df.empty:
    print("No detection records found.")
    raise SystemExit

print("\n========== TRACK TIMELINE ==========")

for track_id, track in df.groupby("track_id"):

    if track_id == -1:
        continue

    track = track.sort_values("time_sec")

    print(f"\nTRACK ID: {int(track_id)}")
    print("First seen:", round(track["time_sec"].min(), 3))
    print("Last seen:", round(track["time_sec"].max(), 3))
    print("Records:", len(track))

    print(
        "Torso angle range:",
        round(track["torso_angle_deg"].min(), 2),
        "to",
        round(track["torso_angle_deg"].max(), 2)
    )

    print(
        "Maximum displacement:",
        round(track["center_displacement_px"].max(), 2)
    )

    print(
        "Maximum angle change:",
        round(track["angle_change_deg"].max(), 2)
    )

    print(
        "Maximum normalized movement:",
        round(track["normalized_displacement"].max(), 3)
    )


print("\n========== LARGEST POSTURE CHANGES ==========")

angle_changes = df.dropna(
    subset=["angle_change_deg"]
).sort_values(
    "angle_change_deg",
    ascending=False
)

print(
    angle_changes[
        [
            "time_sec",
            "track_id",
            "torso_angle_deg",
            "angle_change_deg",
            "center_displacement_px",
            "normalized_displacement"
        ]
    ].head(20).to_string(index=False)
)


print("\n========== LARGEST MOVEMENTS ==========")

movements = df.dropna(
    subset=["center_displacement_px"]
).sort_values(
    "center_displacement_px",
    ascending=False
)

print(
    movements[
        [
            "time_sec",
            "track_id",
            "center_displacement_px",
            "torso_angle_deg",
            "angle_change_deg"
        ]
    ].head(20).to_string(index=False)
)


print("\n========== FALL-CUE SUMMARY ==========")

print(
    df.groupby("track_id")["preliminary_fall_cue"]
    .sum()
)

print("\nDiagnosis completed.")
