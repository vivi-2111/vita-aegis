import cv2

cap = cv2.VideoCapture("accident_clip.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

for timestamp in [3.5, 4.0, 4.5, 5.0, 6.0, 7.0]:

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    ret, frame = cap.read()

    if not ret:
        continue

    height, width = frame.shape[:2]

    print(
        f"\nTimestamp: {timestamp}s"
        f"\nFrame width: {width}"
        f"\nFrame height: {height}"
    )

    cv2.imshow("Accident Region Inspection", frame)

    print("Press any key for next frame. Press ESC to exit.")

    key = cv2.waitKey(0)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()