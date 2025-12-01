import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
IMG_PATH = r"C:\Users\leevi\Desktop\blue_floorballs_all.png"

MIN_DEFECT_DEPTH = 8
CUT_THICKNESS = 4
SEED_PEAK_THRESHOLD = 0.4     # fraction of dist.max()
MIN_SEED_SIZE = 30

# -------------------------------------------
def show(img, title=None):
    plt.figure(figsize=(6,6))
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img[...,::-1])
    if title: plt.title(title)
    plt.axis('off')
    plt.show()

# -------------------------------------------
# 1. LOAD IMAGE
# -------------------------------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("Image not found")

img_rgb = img[..., ::-1]

# -------------------------------------------
# 2. Threshold blue floorballs (HSV)
# -------------------------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([95, 90, 40])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# -------------------------------------------
# 3. Convexity defects → cut necks
# -------------------------------------------
contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_cut = mask.copy()

all_defects = []
for cnt in contours:
    if len(cnt) < 5: continue
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3: continue
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None: continue

    for d in defects:
        s,e,f,depth = d[0]
        depth /= 256.0
        if depth > MIN_DEFECT_DEPTH:
            start = tuple(cnt[s][0])
            end   = tuple(cnt[e][0])
            far   = tuple(cnt[f][0])
            all_defects.append((start,end,far,depth))
            cv2.line(mask_cut, start, end, 0, thickness=CUT_THICKNESS)

# -------------------------------------------
# 4. Distance transform → seed generation
# -------------------------------------------
bin_cut = (mask_cut > 0).astype(np.uint8)*255
dist = cv2.distanceTransform(bin_cut, cv2.DIST_L2, 5)

local_max = dist == cv2.dilate(dist, np.ones((9,9),np.uint8))
seed_mask = np.zeros_like(dist, np.uint8)
seed_mask[(local_max) & (dist > SEED_PEAK_THRESHOLD*dist.max())] = 255

seed_mask = morphology.remove_small_objects(seed_mask.astype(bool), min_size=MIN_SEED_SIZE)
seed_mask = (seed_mask.astype(np.uint8)*255)

num_labels, markers = cv2.connectedComponents(seed_mask)

# Fallback: if almost no seeds → use contour centroids
if num_labels <= 2:
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        if markers[cy,cx] == 0 and bin_cut[cy,cx] > 0:
            markers[cy,cx] = markers.max() + 1

# -------------------------------------------
# 5. Watershed
# -------------------------------------------
markers_ws = markers.astype(np.int32)
cv2.watershed(img, markers_ws)

# markers_ws:
#  -1 = boundaries
#  >0 = segment labels

# -------------------------------------------
# 6. Extract ball centroids + radii
# -------------------------------------------
ball_list = []
unique_labels = np.unique(markers_ws)
unique_labels = unique_labels[unique_labels > 0]

for lbl in unique_labels:
    mask_lbl = (markers_ws == lbl).astype(np.uint8)

    ys, xs = np.where(mask_lbl == 1)
    if len(xs) < 10: continue

    # Centroid
    cx = int(xs.mean())
    cy = int(ys.mean())

    # Fit minimum enclosing circle
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx2, cy2), radius = cv2.minEnclosingCircle(pts)

    ball_list.append((int(cx2), int(cy2), float(radius)))

print("\nDetected balls:", len(ball_list))

# -------------------------------------------
# 7. Draw detected balls on frame
# -------------------------------------------
out = img_rgb.copy()

for i,(cx,cy,r) in enumerate(ball_list, start=1):
    cv2.circle(out, (cx,cy), int(r), (0,255,0), 2)    # circle outline
    cv2.circle(out, (cx,cy), 3, (0,255,0), -1)        # center dot
    cv2.putText(out, str(i), (cx+5,cy-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

show(out, "Final detected balls with centers and IDs")

cv2.imwrite("detected_balls_output.png", out[..., ::-1])
print("Saved: detected_balls_output.png")
