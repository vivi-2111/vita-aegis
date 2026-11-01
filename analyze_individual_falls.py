
import pandas as pd
import numpy as np

# ==========================================
# VITA AEGIS - INDIVIDUAL FALL ANALYSIS
# ==========================================

INPUT_FILE = "multi_person_tracking.csv"
OUTPUT_FILE = "individual_fall_metrics.csv"

TARGET_IDS = [12, 13]

# Load tracking data
df = pd.read_csv(INPUT_FILE)

print("\nLoading multi-person tracking data...")

# ==========================================
# FUNCTION: CALCULATE TORSO ANGLE
# ==========================================

def calculate_torso_angle(row):

    # Average shoulder position
    shoulder_x = (
        row["left_shoulder_x"] +
        row["right_shoulder_x"]
    ) / 2

    shoulder_y = (
        row["left_shoulder_y"] +
        row["right_shoulder_y"]
    ) / 2

    # Average hip position
    hip_x = (
        row["left_hip_x"] +
        row["right_hip_x"]
    ) / 2

    hip_y = (
        row["left_hip_y"] +
        row["right_hip_y"]
    ) / 2

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y

    # 0 degrees = approximately vertical
    # 90 degrees = approximately horizontal
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))

    return angle


# ==========================================
# ANALYZE EACH OCCUPANT
# ==========================================

results = []

for person_id in TARGET_IDS:

    person = df[df["track_id"] == person_id].copy()

    person = person.sort_values("frame")

    print(f"\n========== OCCUPANT ID {person_id} ==========")

    if person.empty:
        print("No tracking records found.")
        continue

    # Calculate torso angle only for valid keypoints
    person["torso_angle_deg"] = np.nan

    valid = person["keypoints_valid"] == 1

    person.loc[valid, "torso_angle_deg"] = (
        person.loc[valid].apply(calculate_torso_angle, axis=1)
    )

    # Calculate vertical center of bounding box
    person["center_y"] = (
        person["y1"] + person["y2"]
    ) / 2

    # Calculate frame-to-frame movement
    person["vertical_movement_px"] = (
        person["center_y"].diff().abs()
    )

    # Baseline from the first 5 valid observations
    baseline_values = person.loc[
        person["torso_angle_deg"].notna(),
        "torso_angle_deg"
    ].head(5)

    if len(baseline_values) == 0:
        print("No valid pose measurements.")
        continue

    baseline_angle = baseline_values.median()

    person["angle_change_deg"] = (
        person["torso_angle_deg"] - baseline_angle
    ).abs()

    # Preliminary posture-change indicator.
    # This is a candidate, NOT a confirmed accident.
    person["posture_change_candidate"] = (
        (person["torso_angle_deg"] >= 45) &
        (person["angle_change_deg"] >= 25)
    )

    print(f"First detection: {person['time_sec'].min():.2f}s")
    print(f"Last detection:  {person['time_sec'].max():.2f}s")
    print(f"Total records:   {len(person)}")
    print(f"Baseline angle:  {baseline_angle:.2f} degrees")

    candidates = person[
        person["posture_change_candidate"] == True
    ]

    print(f"Posture-change candidate frames: {len(candidates)}")

    if not candidates.empty:
        print("\nCandidate timestamps:")
        print(
            candidates[
                ["time_sec", "torso_angle_deg",
                 "angle_change_deg", "vertical_movement_px"]
            ].round(2).to_string(index=False)
        )

    results.append(person)


# ==========================================
# SAVE COMBINED RESULTS
# ==========================================

if results:

    final_df = pd.concat(results, ignore_index=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n===================================")
    print("INDIVIDUAL FALL ANALYSIS COMPLETED")
    print("===================================")
    print(f"Saved report: {OUTPUT_FILE}")

else:
    print("No valid occupant data was available.")
