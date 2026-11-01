import cv2
import pandas as pd

VIDEO_PATH = "accident_clip.mp4"
CSV_PATH = "multi_person_tracking.csv"

TARGET_IDS = [12, 13]

df = pd.read_csv(CSV_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Total frames:", total_frames)

# Inspect selected timestamps
timestamps = [
    3.8, 4.0, 4.2, 4.4, 4.6,
    4.8, 5.0, 5.2, 5.4, 5.6,
    5.8, 6.0, 6.2, 6.4, 6.6,
    6.8, 7.0
]

for timestamp in timestamps:

    frame_number = int(timestamp * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        continue

    current = df[
        (df["time_sec"].sub(timestamp).abs() < 0.025)
        & (df["track_id"].isin(TARGET_IDS))
    ]

    for _, row in current.iterrows():

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        track_id = int(row["track_id"])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Time: {timestamp:.2f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Occupant Timeline", frame)

    print(f"Inspecting timestamp: {timestamp:.2f}s")

    key = cv2.waitKey(0)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
