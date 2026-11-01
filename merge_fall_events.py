
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. INPUT AND OUTPUT FILES
# --------------------------------------------------

CANDIDATES_CSV = "fall_event_candidates.csv"
METRICS_CSV = "fall_sequence_metrics.csv"
OUTPUT_CSV = "fall_event_summary.csv"

# Merge nearby signals when they likely belong to one event.
MERGE_GAP_SECONDS = 1.0

# Inspect measurements after the merged event.
POST_EVENT_WINDOW_SECONDS = 1.5


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

candidates = pd.read_csv(CANDIDATES_CSV)
metrics = pd.read_csv(METRICS_CSV)

if candidates.empty:
    print("No candidate events to merge.")
    raise SystemExit(0)

required_candidates = [
    "candidate_start_sec",
    "posture_signal_end_sec",
    "movement_cue_sec",
]

required_metrics = [
    "time_sec",
    "detected",
    "keypoints_valid",
    "torso_angle_deg",
    "box_height",
    "center_y",
]

for col in required_candidates:
    if col not in candidates.columns:
        raise SystemExit(f"Missing candidate column: {col}")

for col in required_metrics:
    if col not in metrics.columns:
        raise SystemExit(f"Missing metrics column: {col}")

for col in required_candidates:
    candidates[col] = pd.to_numeric(
        candidates[col], errors="coerce"
    )

for col in required_metrics:
    metrics[col] = pd.to_numeric(
        metrics[col], errors="coerce"
    )

candidates = candidates.dropna(
    subset=["candidate_start_sec", "posture_signal_end_sec"]
).sort_values("candidate_start_sec").reset_index(drop=True)

metrics = metrics.sort_values("time_sec").reset_index(drop=True)

if candidates.empty:
    print("No valid candidate timestamps found.")
    raise SystemExit(0)


# --------------------------------------------------
# 3. MERGE NEARBY CANDIDATES
# --------------------------------------------------

merged_events = []

current = None

for _, row in candidates.iterrows():
    start = float(row["candidate_start_sec"])
    end = float(row["posture_signal_end_sec"])
    movement_time = row["movement_cue_sec"]

    if current is None:
        current = {
            "start": start,
            "end": end,
            "candidate_count": 1,
            "movement_times": [],
        }

        if pd.notna(movement_time):
            current["movement_times"].append(
                float(movement_time)
            )

        continue

    # Merge if the next candidate starts close to the
    # end of the event already being tracked.
    if start <= current["end"] + MERGE_GAP_SECONDS:
        current["end"] = max(current["end"], end)
        current["candidate_count"] += 1

        if pd.notna(movement_time):
            current["movement_times"].append(
                float(movement_time)
            )
    else:
        merged_events.append(current)

        current = {
            "start": start,
            "end": end,
            "candidate_count": 1,
            "movement_times": [],
        }

        if pd.notna(movement_time):
            current["movement_times"].append(
                float(movement_time)
            )

if current is not None:
    merged_events.append(current)


# --------------------------------------------------
# 4. REVIEW THE POST-EVENT PERIOD
# --------------------------------------------------

summary_rows = []

for event_number, event in enumerate(merged_events, start=1):
    start = event["start"]
    end = event["end"]

    # Inspect the 1.5 seconds following the last candidate.
    post_start = end
    post_end = end + POST_EVENT_WINDOW_SECONDS

    post = metrics[
        (metrics["time_sec"] >= post_start)
        & (metrics["time_sec"] <= post_end)
    ].copy()

    total_post_frames = len(post)

    if total_post_frames:
        detected_rate = (
            post["detected"].fillna(0).eq(1).mean() * 100
        )

        valid_pose_rate = (
            post["keypoints_valid"].fillna(0).eq(1).mean()
            * 100
        )

        valid_angles = post.loc[
            post["keypoints_valid"].eq(1),
            "torso_angle_deg"
        ].dropna()

        median_angle = (
            float(valid_angles.median())
            if not valid_angles.empty
            else np.nan
        )

        median_height = (
            float(post["box_height"].median())
            if post["box_height"].notna().any()
            else np.nan
        )

        valid_centers = post["center_y"].dropna()

        if len(valid_centers) >= 2:
            center_y_change = (
                float(valid_centers.iloc[-1])
                - float(valid_centers.iloc[0])
            )
        else:
            center_y_change = np.nan

    else:
        detected_rate = np.nan
        valid_pose_rate = np.nan
        median_angle = np.nan
        median_height = np.nan
        center_y_change = np.nan

    movement_times = event["movement_times"]

    summary_rows.append({
        "event_id": event_number,
        "candidate_start_sec": round(start, 3),
        "candidate_end_sec": round(end, 3),
        "event_duration_sec": round(end - start, 3),
        "merged_candidate_count": event["candidate_count"],
        "movement_cue_times_sec": ";".join(
            f"{t:.3f}" for t in movement_times
        ),
        "post_event_window_start_sec": round(post_start, 3),
        "post_event_window_end_sec": round(post_end, 3),
        "post_event_frames": total_post_frames,
        "post_event_detection_rate_pct": (
            round(detected_rate, 1)
            if pd.notna(detected_rate) else ""
        ),
        "post_event_valid_pose_rate_pct": (
            round(valid_pose_rate, 1)
            if pd.notna(valid_pose_rate) else ""
        ),
        "post_event_median_torso_angle_deg": (
            round(median_angle, 2)
            if pd.notna(median_angle) else ""
        ),
        "post_event_median_box_height_px": (
            round(median_height, 2)
            if pd.notna(median_height) else ""
        ),
        "post_event_center_y_change_px": (
            round(center_y_change, 2)
            if pd.notna(center_y_change) else ""
        ),
        "event_status": "REVIEW_VIDEO",
        "recovery_status": "NOT_CONFIRMED",
        "injury_status": "UNKNOWN",
    })


# --------------------------------------------------
# 5. SAVE THE EVENT SUMMARY
# --------------------------------------------------

output = pd.DataFrame(summary_rows)
output.to_csv(OUTPUT_CSV, index=False)

print("\nFall-event grouping complete.")
print("Input candidates:", len(candidates))
print("Merged event groups:", len(output))
print("Saved report:", OUTPUT_CSV)

print("\nMerged event summary:")
print(output.to_string(index=False))

print(
    "\nThese are grouped fall candidates only. "
    "Recovery and injury status have not been confirmed."
)
