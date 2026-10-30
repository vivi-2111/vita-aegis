
import csv
import cv2
from ultralytics import YOLO

VIDEO_PATH = "Road (5).mp4"
MODEL_PATH = "yolov8n.pt"

OUTPUT_VIDEO = "tracked_vehicles.mp4"
OUTPUT_CSV = "vehicle_tracks.csv"

# COCO classes: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]

def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    print(f"Video: {width}x{height}, {fps:.2f} FPS")
    print(f"Frames: {total_frames}")

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Cannot create output video.")
        cap.release()
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([
            "frame", "time_seconds", "track_id", "vehicle",
            "center_x", "center_y", "box_width", "box_height"
        ])

        frame_number = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Track vehicles across frames; use CPU for compatibility.
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=VEHICLE_CLASSES,
                conf=0.20,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            result = results[0]
            annotated = result.plot()

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                ids = result.boxes.id

                if ids is not None:
                    track_ids = ids.cpu().numpy().astype(int)

                    for box, class_id, track_id in zip(
                        boxes, classes, track_ids
                    ):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        vehicle_name = model.names[class_id]
                        time_seconds = (frame_number - 1) / fps

                        csv_writer.writerow([
                            frame_number,
                            round(time_seconds, 3),
                            track_id,
                            vehicle_name,
                            round(float(center_x), 2),
                            round(float(center_y), 2),
                            round(float(x2 - x1), 2),
                            round(float(y2 - y1), 2)
                        ])

            writer.write(annotated)

            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames")

    cap.release()
    writer.release()

    print("\nTracking completed!")
    print(f"Annotated video: {OUTPUT_VIDEO}")
    print(f"Tracking data: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
