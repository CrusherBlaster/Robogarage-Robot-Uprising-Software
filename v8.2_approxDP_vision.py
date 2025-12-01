import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

# Load image
src = cv2.imread(r"C:\Users\leevi\Desktop\blue_floorballs_all_hard.png")
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

# ---- FILLED CONTOURS ----
for i in range(len(poly)):
    cv2.drawContours(thresh, poly, i, 255, thickness=cv2.FILLED)

# clean up noise
thresh = cv2.medianBlur(thresh, 3)
thresh = cv2.erode(thresh, np.ones((3,3), np.uint8), iterations=1)

# empty canvas for small contours
small_mask = np.zeros_like(gray)

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
    cv2.circle(src, center, radius, (0, 255, 0), 2)
    ball_count += 1


# =============================================================
# === NEW SPECIAL SEGMENTATION FOR SMALL BALLS BEGINS HERE ====
# =============================================================

# polyDP for small mask
s_contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

small_filled = np.zeros_like(gray)

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

    cv2.circle(src, center, radius, (0, 255, 255), 2)  # different color for small balls
    ball_count += 1

# =============================================================
# === END OF NEW SPECIAL SEGMENTATION =========================
# =============================================================


print("Total balls:", ball_count)

# Show result
cv2.imshow("small_mask", small_mask)
cv2.imshow("segmented_small_balls", src)
cv2.waitKey(0)
cv2.destroyAllWindows()

