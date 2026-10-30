
import pandas as pd
import numpy as np

INPUT_CSV = "fall_sequence_metrics.csv"
OUTPUT_CSV = "fall_event_candidates.csv"

# Tune these after comparing results with the actual video.
MIN_TORSO_ANGLE = 40.0
MIN_ANGLE_CHANGE = 25.0
MIN_POSTURE_FRAMES = 2

MIN_MOVEMENT_PX = 8.0
MAX_MOVEMENT_PX = 35.0

POSTURE_WINDOW_SECONDS = 0.20
MOVEMENT_WINDOW_SECONDS = 0.35

df = pd.read_csv(INPUT_CSV)
df = df.sort_values("time_sec").reset_index(drop=True)

required = [
    "time_sec",
    "detected",
    "torso_angle_deg",
    "angle_change_deg",
    "center_displacement_px",
    "keypoints_valid",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise SystemExit(f"Missing CSV columns: {missing}")

# Convert measurement columns safely to numeric values.
numeric_columns = [
    "time_sec",
    "detected",
    "torso_angle_deg",
    "angle_change_deg",
    "center_displacement_px",
    "keypoints_valid",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# A posture signal is valid only when the person and keypoints
# have actually been detected.
df["posture_signal"] = (
    (df["detected"] == 1)
    & (df["keypoints_valid"] == 1)
    & (df["torso_angle_deg"] >= MIN_TORSO_ANGLE)
    & (df["angle_change_deg"] >= MIN_ANGLE_CHANGE)
)

# Treat a very large single-frame jump as unreliable evidence.
df["plausible_movement"] = (
    df["center_displacement_px"].between(
        MIN_MOVEMENT_PX,
        MAX_MOVEMENT_PX,
        inclusive="both"
    )
)

# Find posture-change runs in chronological order.
candidate_events = []
i = 0

while i < len(df):
    if not bool(df.loc[i, "posture_signal"]):
        i += 1
        continue

    start_i = i
    start_time = float(df.loc[start_i, "time_sec"])
    end_time = start_time
    posture_indices = []

    # Count posture signals in a short time window.
    j = start_i

    while j < len(df):
        t = float(df.loc[j, "time_sec"])

        if t - start_time > POSTURE_WINDOW_SECONDS:
            break

        if bool(df.loc[j, "posture_signal"]):
            posture_indices.append(j)
            end_time = t

        j += 1

    if len(posture_indices) >= MIN_POSTURE_FRAMES:
        # Search for plausible movement after the posture change.
        movement_rows = df[
            (df["time_sec"] >= start_time)
            & (
                df["time_sec"]
                <= start_time + MOVEMENT_WINDOW_SECONDS
            )
            & df["plausible_movement"]
        ]

        if not movement_rows.empty:
            movement_time = float(
                movement_rows.iloc[0]["time_sec"]
            )

            candidate_events.append({
                "candidate_start_sec": round(start_time, 3),
                "posture_signal_end_sec": round(end_time, 3),
                "movement_cue_sec": round(movement_time, 3),
                "posture_frames": len(posture_indices),
                "candidate_status": "REVIEW_VIDEO",
            })

        # Avoid repeatedly reporting the same short posture run.
        i = max(j, start_i + 1)
    else:
        i += 1

# Save the results.
events_df = pd.DataFrame(
    candidate_events,
    columns=[
        "candidate_start_sec",
        "posture_signal_end_sec",
        "movement_cue_sec",
        "posture_frames",
        "candidate_status",
    ]
)

events_df.to_csv(OUTPUT_CSV, index=False)

print("\nFall-event candidate analysis complete.")
print("Input:", INPUT_CSV)
print("Output:", OUTPUT_CSV)
print("Candidate events:", len(events_df))

if events_df.empty:
    print(
        "No candidates matched these thresholds. "
        "This does not prove that no fall occurred."
    )
else:
    print("\nCandidate event windows:")
    print(events_df.to_string(index=False))

print(
    "\nImportant: these are candidates for video review, "
    "not confirmed falls or evidence of injury."
)
