import cv2
import numpy as np

def nothing(x):
    pass

# Open camera
cap = cv2.VideoCapture(0)

# Create a window for trackbars
cv2.namedWindow("Trackbars")

# Create HSV trackbars
cv2.createTrackbar("LH", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("UH", "Trackbars", 25, 179, nothing)
cv2.createTrackbar("LS", "Trackbars", 120, 255, nothing)
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("LV", "Trackbars", 120, 255, nothing)
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of all trackbars
    lh = cv2.getTrackbarPos("LH", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    # Create lower and upper HSV bounds
    lower_orange = np.array([lh, ls, lv])
    upper_orange = np.array([uh, us, uv])

    print("Lower: ", lower_orange)
    print("Upper: ", upper_orange)

    


    # Apply the mask
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the original, mask, and result
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Exit with ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()