import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = r"C:\Users\leevi\Desktop\blue_floorballs_all.png"

# Utility for display
def show(img, title=None):
    plt.figure(figsize=(6,6))
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if title: plt.title(title)
    plt.axis("off")
    plt.show()


# 1) Load image
img = cv2.imread(IMG_PATH)
orig = img.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# 2) Threshold blue WITHOUT closing gaps between balls
lower_blue = np.array([95, 80, 40])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# remove only tiny noise, do NOT close gaps

# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
#mask = cv2.GaussianBlur(mask, (7,7), 0)
mask = cv2.medianBlur(mask, 5)
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
show(mask, "Blue mask (gaps preserved)")


# 3) Find contours and detect convexity-defect necks
mask_cut = mask.copy()
contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

MIN_DEPTH = 1       # lower because balls are small in your test image
CUT_THICKNESS = 3

for cnt in contours:
    if len(cnt) < 5:
        continue

    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        continue

    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        continue

    for d in defects:
        s,e,f,depth = d[0]
        depth = depth / 256.0

        if depth > MIN_DEPTH:
            start = tuple(cnt[s][0])
            end   = tuple(cnt[e][0])

            # This cut actually **removes pixels** between touching balls
            cv2.line(mask_cut, start, end, 0, thickness=CUT_THICKNESS)

show(mask_cut, "Mask after *real* convexity-defect cuts")


# 4) Watershed markers from distance transform
mask_bin = (mask_cut > 0).astype(np.uint8)

dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
show(dist, "Distance transform")

# Local maxima = seeds
local_max = (dist == cv2.dilate(dist, np.ones((7,7), np.uint8)))
seeds = np.zeros_like(dist, dtype=np.uint8)
seeds[(local_max) & (dist > 0.3 * dist.max())] = 255

show(seeds, "Seed peaks (ball centers)")


# Label markers
num_labels, markers = cv2.connectedComponents(seeds)
markers_ws = markers.copy().astype(np.int32)


# 5) Watershed
ws_in = img.copy()
cv2.watershed(ws_in, markers_ws)

# boundaries = -1
boundary_mask = markers_ws == -1

# Build segmentation mask (each ball has its own integer ID)
ball_segments = markers_ws.copy()
ball_segments[ball_segments < 1] = 0


# 6) Extract final balls: minEnclosingCircle per label
output = orig.copy()
count = 0

for lbl in range(1, ball_segments.max() + 1):
    ys, xs = np.where(ball_segments == lbl)
    if len(xs) < 10:
        continue

    pts = np.vstack((xs, ys)).T
    (cx, cy), r = cv2.minEnclosingCircle(pts)

    cx = int(cx)
    cy = int(cy)
    r  = int(r)

    # draw ball
    cv2.circle(output, (cx, cy), r, (0,255,0), 2)
    cv2.circle(output, (cx, cy), 2, (0,0,255), -1)

    count += 1

show(output, "Final detected balls (circles drawn)")
print("Total detected balls:", count)
