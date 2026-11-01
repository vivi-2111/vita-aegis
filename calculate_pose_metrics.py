
import pandas as pd
import numpy as np
import math

INPUT_FILE = "occupants_pose_960.csv"
OUTPUT_FILE = "pose_timeline_metrics.csv"

CONF_THRESHOLD = 0.25

df = pd.read_csv(INPUT_FILE)


def get_joint_midpoint(row, joint_name):
    """Calculate a joint midpoint using reliable left/right keypoints."""

    points = []

    for side in ["left", "right"]:
        x = row[f"{side}_{joint_name}_x"]
        y = row[f"{side}_{joint_name}_y"]
        conf = row[f"{side}_{joint_name}_conf"]

        if (
            pd.notna(x) and pd.notna(y) and pd.notna(conf)
            and x > 0 and y > 0
            and conf >= CONF_THRESHOLD
        ):
            points.append((x, y))

    if not points:
        return None

    return (
        np.mean([p[0] for p in points]),
        np.mean([p[1] for p in points])
    )


results = []

for _, row in df.iterrows():

    shoulder = get_joint_midpoint(row, "shoulder")
    hip = get_joint_midpoint(row, "hip")

    angle = np.nan
    torso_length = np.nan

    if shoulder is not None and hip is not None:

        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]

        torso_length = math.sqrt(dx**2 + dy**2)

        if torso_length > 0:
            angle = math.degrees(
                math.atan2(abs(dx), abs(dy))
            )

    results.append({
        "time_sec": row["time_sec"],
        "frame": row["frame"],
        "track_id": row["track_id"],
        "confidence": row["confidence"],
        "box_height": row["y2"] - row["y1"],
        "torso_angle_deg": angle,
        "torso_length_px": torso_length
    })


metrics = pd.DataFrame(results)

metrics.to_csv(OUTPUT_FILE, index=False)

print("\n========== POSTURE METRICS ==========")

for track_id in [16, 19]:

    person = metrics[metrics["track_id"] == track_id].copy()
    valid = person.dropna(subset=["torso_angle_deg"])

    print(f"\nTRACK ID {track_id}")
    print("Total observations:", len(person))
    print("Valid posture measurements:", len(valid))

    if valid.empty:
        print("No reliable shoulder/hip measurements.")
        continue

    print(
        "Average torso angle:",
        round(valid["torso_angle_deg"].mean(), 2), "degrees"
    )

    print(
        "Minimum torso angle:",
        round(valid["torso_angle_deg"].min(), 2), "degrees"
    )

    print(
        "Maximum torso angle:",
        round(valid["torso_angle_deg"].max(), 2), "degrees"
    )

    print("\nTimeline:")

    for _, r in valid.iloc[::5].iterrows():
        print(
            f"Time: {r['time_sec']:.3f}s | "
            f"Angle: {r['torso_angle_deg']:.2f}° | "
            f"Torso length: {r['torso_length_px']:.2f}px"
        )

print("\nSaved:", OUTPUT_FILE)
print("POSTURE ANALYSIS COMPLETED")
