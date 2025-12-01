import cv2
import numpy as np

# Load image
src = cv2.imread(r"C:\Users\leevi\Desktop\blue_floorballs_all_hard.png")        # Change name if needed
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Threshold
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
)



# Approximate each contour
poly = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 3, True)
    poly.append(approx)

# ---- FILLED CONTOURS here ----
for i in range(len(poly)):
    cv2.drawContours(thresh, poly, i, (255, 255, 255), thickness=cv2.FILLED)

# clear out noise for contour areas > 100
thresh = cv2.medianBlur(thresh, 3)
thresh = cv2.erode(thresh, (np.ones((3, 3), np.uint8)), iterations=1)

# empty canvas for small contours
small_mask = np.zeros_like(gray)  # same size as original
small_mask_filled = np.zeros_like(gray)

# find contours from the clean mask and draw the minimum enclosing circle
c_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
ball_count = 0
for cnt in c_contours:
    
    area = cv2.contourArea(cnt)
    
    # draw small contours (<100 area) to separate canvas
    if area < 100:
        if area < 1:
            continue
        # draw only this small contour into its own blank mask
        cv2.drawContours(small_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        continue
    
    # draw big contours directly to src image
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(src, center, radius, (0, 120, 0), 2)
    ball_count += 1
    #cv2.putText(src, f"id={ball_count}", (center[0] - 10, center[1] - radius - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 2)



# Show result
cv2.imshow("thresh", thresh)
cv2.imshow("small_mask", small_mask)
cv2.imshow("filled_contours", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
