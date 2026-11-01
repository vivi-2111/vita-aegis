
import pandas as pd

CSV_FILE = "fall_sequence_metrics.csv"

df = pd.read_csv(CSV_FILE)

# Select the relevant accident window
df = df[
    (df["time_sec"] >= 3.5) &
    (df["time_sec"] <= 7.0)
].copy()

# Display every frame around the suspected event
columns = [
    "time_sec",
    "detected",
    "torso_angle_deg",
    "angle_change_deg",
    "center_displacement_px",
    "posture_change_marker",
    "keypoints_valid"
]

print("\n========== FALL EVENT TIMELINE ==========\n")

print(
    df[columns].round(2).to_string(index=False)
)

# Find strongest posture changes
print("\n========== STRONGEST POSTURE CHANGES ==========\n")

strongest = df.sort_values(
    "angle_change_deg",
    ascending=False
).head(10)

print(
    strongest[columns].round(2).to_string(index=False)
)

# Find largest center movements
print("\n========== LARGEST CENTER MOVEMENTS ==========\n")

movement = df.sort_values(
    "center_displacement_px",
    ascending=False
).head(10)

print(
    movement[columns].round(2).to_string(index=False)
)

print("\nTimeline inspection completed.")
