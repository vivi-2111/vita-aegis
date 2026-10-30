import pandas as pd

df = pd.read_csv("crash_tracking_botsort.csv")

people = df[
    (df["object"] == "person") &
    (df["time_sec"].between(3.5, 7.0))
].copy()

print("\nPERSON TRACKS AROUND THE FALL\n")

summary = people.groupby("track_id").agg(
    frames=("frame", "count"),
    first_time=("time_sec", "min"),
    last_time=("time_sec", "max"),
    min_x=("center_x", "min"),
    max_x=("center_x", "max"),
    min_y=("center_y", "min"),
    max_y=("center_y", "max"),
    min_height=("box_height", "min"),
    max_height=("box_height", "max")
)

summary["height_change"] = (
    summary["max_height"] - summary["min_height"]
)

print(
    summary.sort_values(
        ["height_change", "frames"], ascending=False
    ).head(20).round(2).to_string()
)
