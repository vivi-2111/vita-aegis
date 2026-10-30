
import pandas as pd

df = pd.read_csv("crash_tracking.csv")

# Inspect the rider identified as track 55.
rider = df[
    (df["object"] == "person") &
    (df["track_id"] == 55) &
    (df["time_sec"] >= 3.0) &
    (df["time_sec"] <= 7.0)
].sort_values("time_sec")

if rider.empty:
    print("No detections for person track 55 in this window.")
    raise SystemExit

print("\n=== RIDER TRACK 55 ===")
print(
    rider[
        ["time_sec", "frame", "confidence",
         "center_x", "center_y",
         "box_width", "box_height"]
    ].to_string(index=False)
)

print("\n=== MOVEMENT SUMMARY ===")
print("Detections:", len(rider))
print("First time:", round(rider["time_sec"].min(), 2))
print("Last time:", round(rider["time_sec"].max(), 2))
print("Center Y range:",
      round(rider["center_y"].min(), 1),
      "to", round(rider["center_y"].max(), 1))
print("Box height range:",
      rider["box_height"].min(),
      "to", rider["box_height"].max())
print("Average confidence:",
      round(rider["confidence"].mean(), 3))
