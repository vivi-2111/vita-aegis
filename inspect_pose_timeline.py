
import pandas as pd

CSV_FILE = "occupants_pose_960.csv"

df = pd.read_csv(CSV_FILE)

# Calculate bounding-box height from coordinates
df["box_height"] = df["y2"] - df["y1"]

print("\n========== CSV STRUCTURE ==========")
print("Total records:", len(df))
print("\nAvailable columns:")
for column in df.columns:
    print(column)

print("\n========== TRACK SUMMARY ==========")

for track_id in [16, 19]:

    person = df[df["track_id"] == track_id].copy()

    if person.empty:
        print(f"\nTrack ID {track_id}: No records found")
        continue

    person = person.sort_values("time_sec")

    print(f"\nTrack ID: {track_id}")
    print("First detected:", round(person["time_sec"].min(), 3))
    print("Last detected:", round(person["time_sec"].max(), 3))
    print("Total observations:", len(person))

    print("\nTimeline samples:")

    for _, row in person.iloc[::5].iterrows():

        print(
            f"Time: {row['time_sec']:.3f}s | "
            f"Frame: {row['frame']} | "
            f"Confidence: {row['confidence']:.3f} | "
            f"Box height: {row['box_height']:.2f}px"
        )

print("\n========== INSPECTION COMPLETED ==========")
