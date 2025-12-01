"""
# note that in this code the camera must be excactly normal to the ball
# otherwise prepare for false segmentation due to overlapping contours
# this is the limitation of OpenCV based computer vision
# The best computer vision uses machine learning models for detection.
# Things to test with this code:
# - Different hue values
# - Different heights of the camera
# - inputting different images to functions and modifying images  
"""

import cv2
import numpy as np
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

def nothing(x):
    pass

# Open camera
cap = cv2.VideoCapture(0)

# Create a window for trackbars
cv2.namedWindow("Trackbars")
# 'blue': ((90, 130, 114), (113, 255, 255))
# 'orange': ((0, 127, 168), (10, 255, 255)) <- here ((lh,ls,lv), (uh,us,uv))
# Create HSV trackbars - first value is initial value and second value is max of the entire bar
cv2.createTrackbar("LH", "Trackbars", 90, 179, nothing)
cv2.createTrackbar("UH", "Trackbars", 113, 179, nothing)
cv2.createTrackbar("LS", "Trackbars", 130, 255, nothing)
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("LV", "Trackbars", 114, 255, nothing)
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
    
    
    # =============================================================
    # === NEW SPECIAL SEGMENTATION FOR NORMAL-BIG BALLS BEGINS HERE ====
    # =============================================================
    
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Threshold
    _, thresh = cv2.threshold(result_gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    # Approximate each contour
    poly = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 3, True)
        poly.append(approx)
    
    # ---- FILLED CONTOURS ----
    for i in range(len(poly)):
        cv2.drawContours(thresh, poly, i, 255)
    
    
    # clean up noise
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.erode(thresh, np.ones((3,3), np.uint8), iterations=1)

    # empty canvas for small contours
    small_mask = np.zeros_like(mask)

    # find contours from the clean mask
    c_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ball_count = 0

    for cnt in c_contours:

        area = cv2.contourArea(cnt)

        if area < 100:
            if area < 1:
                continue
            # draw only this small contour onto small_mask
            cv2.drawContours(small_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            continue

        # big ball detected normally
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        ball_count += 1


    # =============================================================
    # === NEW SPECIAL SEGMENTATION FOR SMALL BALLS BEGINS HERE ====
    # =============================================================

    # polyDP for small mask
    s_contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    small_filled = np.zeros_like(mask)

    s_poly = []
    for cnt in s_contours:
        approx = cv2.approxPolyDP(cnt, 3, True)
        s_poly.append(approx)

    # fill them properly
    for i in range(len(s_poly)):
        cv2.drawContours(small_filled, s_poly, i, 255, cv2.FILLED)

    # --- distance transform segmentation ---
    dist = cv2.distanceTransform(small_filled, cv2.DIST_L2, 5)

    # find local maxima to split touching balls
    coords = peak_local_max(dist, min_distance=5, labels=small_filled)

    mask2 = np.zeros(dist.shape, dtype=bool)
    mask2[tuple(coords.T)] = True

    markers, _ = ndi.label(mask2)
    labels = watershed(-dist, markers, mask=small_filled)

    # draw each separated object
    for label in np.unique(labels):
        if label == 0:
            continue

        comp = np.uint8(labels == label)
        ccnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ccnts) == 0:
            continue

        cnt = ccnts[0]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        cv2.circle(result, center, radius, (0, 255, 255), 2)  # different color for small balls
        ball_count += 1

    # =============================================================
    # === END OF NEW SPECIAL SEGMENTATION =========================
    # =============================================================
    

    print("Total balls:", ball_count)

    # Show result
    cv2.imshow("small_mask", small_mask)
    
    
    
    
    
    
    # Show the original, mask, and result
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("thresh", thresh)
    cv2.imshow("Result", result)

    # Exit with ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()