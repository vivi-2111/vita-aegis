
import csv
from collections import Counter, defaultdict

INPUT_CSV = "motion_candidates.csv"

event_counts = Counter()
events_per_second = defaultdict(Counter)

with open(INPUT_CSV, "r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        event_type = row["event"]
        event_counts[event_type] += 1

        # Nearby vehicles alone are too noisy to rank as crashes.
        if event_type == "CLOSE_VEHICLE_INTERACTION":
            continue

        second = int(float(row["time"]))
        events_per_second[second][event_type] += 1

print("\n=== EVENT COUNTS ===")
for name, count in event_counts.most_common():
    print(f"{name}: {count}")

print("\n=== TOP SECONDS TO INSPECT ===")
ranked = sorted(
    events_per_second.items(),
    key=lambda item: sum(item[1].values()),
    reverse=True
)

if not ranked:
    print("No speed-drop or direction-change candidates found.")
else:
    for second, counts in ranked[:20]:
        details = ", ".join(
            f"{name}={count}" for name, count in counts.items()
        )
        print(f"{second}s: {details}")

print("\nThis ranks motion anomalies, not confirmed accidents.")
