
import cv2

for second in [3, 4, 5, 6, 7]:
    filename = f"accident_frame_{second}.jpg"
    image = cv2.imread(filename)

    if image is None:
        print("Could not open:", filename)
        continue

    cv2.imshow(f"Frame at {second} seconds", image)

    print(f"Showing {filename}. Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
