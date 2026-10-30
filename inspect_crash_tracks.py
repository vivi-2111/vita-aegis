
import pandas as pd

CSV_FILE = "crash_tracking.csv"

df = pd.read_csv(CSV_FILE)

# Focus on the period around the observed fall.
window = df[
    (df["time_sec"] >= 3.0) &
    (df["time_sec"] <= 7.0)
].copy()

print("\n=== TRACK SUMMARY: 3 TO 7 SECONDS ===")

if window.empty:
    print("No detections found in this time window.")
    raise SystemExit

summaries = []

for (obj, track_id), group in window.groupby(
    ["object", "track_id"]
):
    group = group.sort_values("frame")

    if len(group) < 5:
        continue

    first = group.iloc[0]
    last = group.iloc[-1]

    summaries.append({
        "object": obj,
        "track_id": track_id,
        "detections": len(group),
        "start_time": round(first["time_sec"], 2),
        "end_time": round(last["time_sec"], 2),
        "start_cy": round(first["center_y"], 1),
        "end_cy": round(last["center_y"], 1),
        "start_h": round(first["box_height"], 1),
        "end_h": round(last["box_height"], 1),
        "start_w": round(first["box_width"], 1),
        "end_w": round(last["box_width"], 1),
        "confidence": round(group["confidence"].mean(), 3)
    })

if not summaries:
    print("No tracks have at least 5 detections.")
else:
    result = pd.DataFrame(summaries)
    result["vertical_change"] = (
        result["end_cy"] - result["start_cy"]
    )
    result["height_change"] = (
        result["end_h"] - result["start_h"]
    )

    print(
        result.sort_values(
            "vertical_change",
            key=lambda s: s.abs(),
            ascending=False
        ).to_string(index=False)
    )

print("\n=== DONE ===")
print("These measurements are clues, not proof of an accident.")
