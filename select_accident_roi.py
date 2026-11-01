
import cv2

cap = cv2.VideoCapture("accident_clip.mp4")

cap.set(cv2.CAP_PROP_POS_MSEC, 3500)

ret, frame = cap.read()

cap.release()

if not ret:
    print("Could not read video frame.")
    exit()

print("Draw a rectangle around both accident occupants.")
print("Press ENTER or SPACE to confirm.")
print("Press C to cancel.")

roi = cv2.selectROI(
    "Select Accident Region",
    frame,
    fromCenter=False,
    showCrosshair=True
)

x, y, w, h = roi

print("\nSelected Accident Region:")
print("X:", x)
print("Y:", y)
print("Width:", w)
print("Height:", h)

cv2.destroyAllWindows()

if w > 0 and h > 0:
    with open("accident_roi.txt", "w") as file:
        file.write(f"{x},{y},{w},{h}")

    print("\nROI saved to accident_roi.txt")
else:
    print("No region selected.")
