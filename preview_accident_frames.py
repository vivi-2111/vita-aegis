
import cv2

cap = cv2.VideoCapture("accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

for second in [3, 4, 5, 6, 7]:
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    success, frame = cap.read()

    if success:
        filename = f"accident_frame_{second}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)

cap.release()
