
import csv
import math
from collections import defaultdict

INPUT_CSV = "vehicle_tracks.csv"
OUTPUT_CSV = "motion_candidates.csv"

# Tune these values after inspecting your CCTV footage.
MIN_MOVING_SPEED = 20.0       # pixels per second
SUDDEN_DROP_RATIO = 0.50       # speed drops by at least 50%
MIN_DIRECTION_CHANGE = 70.0    # degrees
PAIR_DISTANCE_FACTOR = 0.80

tracks = defaultdict(list)

# Read tracking data
with open(INPUT_CSV, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    required = {
        "frame", "time_seconds", "track_id", "vehicle",
        "center_x", "center_y", "box_width", "box_height"
    }

    if not required.issubset(reader.fieldnames or []):
        raise ValueError(
            "CSV columns do not match vehicle_tracks.csv. "
            "Check the tracking script output."
        )

    for row in reader:
        track_id = row["track_id"]

        tracks[track_id].append({
            "frame": int(row["frame"]),
            "time": float(row["time_seconds"]),
            "vehicle": row["vehicle"],
            "x": float(row["center_x"]),
            "y": float(row["center_y"]),
            "w": float(row["box_width"]),
            "h": float(row["box_height"]),
        })

for observations in tracks.values():
    observations.sort(key=lambda item: item["frame"])

events = []
motion_by_frame = defaultdict(list)

# Calculate motion for each tracked vehicle
for track_id, observations in tracks.items():
    previous_speed = None
    previous_direction = None

    for i in range(1, len(observations)):
        prev = observations[i - 1]
        curr = observations[i]

        dt = curr["time"] - prev["time"]

        # Ignore invalid time differences or long tracking gaps
        if dt <= 0 or dt > 1.0:
            previous_speed = None
            previous_direction = None
            continue

        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]

        speed = math.hypot(dx, dy) / dt
        direction = math.degrees(math.atan2(dy, dx))

        record = {
            "track_id": track_id,
            "vehicle": curr["vehicle"],
            "frame": curr["frame"],
            "time": curr["time"],
            "x": curr["x"],
            "y": curr["y"],
            "w": curr["w"],
            "h": curr["h"],
            "speed": speed,
            "direction": direction,
        }

        motion_by_frame[curr["frame"]].append(record)

        if previous_speed is not None:
            if (
                previous_speed >= MIN_MOVING_SPEED
                and speed <= previous_speed * SUDDEN_DROP_RATIO
            ):
                events.append({
                    "time": curr["time"],
                    "frame": curr["frame"],
                    "track_id": track_id,
                    "vehicle": curr["vehicle"],
                    "event": "SUDDEN_SPEED_DROP",
                    "speed_before": round(previous_speed, 2),
                    "speed_after": round(speed, 2),
                    "other_vehicle": "",
                    "distance_px": "",
                })

        if previous_direction is not None and speed >= MIN_MOVING_SPEED:
            angle_diff = abs(direction - previous_direction)
            angle_diff = min(angle_diff, 360 - angle_diff)

            if angle_diff >= MIN_DIRECTION_CHANGE:
                events.append({
                    "time": curr["time"],
                    "frame": curr["frame"],
                    "track_id": track_id,
                    "vehicle": curr["vehicle"],
                    "event": "DIRECTION_CHANGE",
                    "speed_before": "",
                    "speed_after": round(speed, 2),
                    "other_vehicle": "",
                    "distance_px": "",
                })

        previous_speed = speed
        previous_direction = direction

# Look for nearby tracked vehicles in the same frame
for frame, vehicles in motion_by_frame.items():
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            a = vehicles[i]
            b = vehicles[j]

            if a["track_id"] == b["track_id"]:
                continue

            distance = math.hypot(a["x"] - b["x"], a["y"] - b["y"])

            # Approximate threshold based on detected box sizes
            size_a = math.hypot(a["w"], a["h"])
            size_b = math.hypot(b["w"], b["h"])
            threshold = max(35.0, PAIR_DISTANCE_FACTOR *
                            (size_a + size_b) / 2)

            if distance <= threshold:
                events.append({
                    "time": a["time"],
                    "frame": frame,
                    "track_id": a["track_id"],
                    "vehicle": a["vehicle"],
                    "event": "CLOSE_VEHICLE_INTERACTION",
                    "speed_before": "",
                    "speed_after": round(a["speed"], 2),
                    "other_vehicle": b["track_id"],
                    "distance_px": round(distance, 2),
                })

# Save all diagnostic candidates
events.sort(key=lambda event: (event["time"], event["frame"]))

columns = [
    "time", "frame", "track_id", "vehicle", "event",
    "speed_before", "speed_after", "other_vehicle", "distance_px"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(events)

print("\nMotion analysis completed.")
print(f"Tracked vehicles: {len(tracks)}")
print(f"Motion candidates: {len(events)}")
print(f"Saved report: {OUTPUT_CSV}")

if events:
    print("\nFirst 20 candidates:")
    for event in events[:20]:
        print(
            f"{event['time']:.2f}s | "
            f"Frame {event['frame']} | "
            f"Vehicle {event['track_id']} | "
            f"{event['event']}"
        )
else:
    print(
        "No candidates found with the current thresholds. "
        "This does not prove that no accident occurred."
    )
