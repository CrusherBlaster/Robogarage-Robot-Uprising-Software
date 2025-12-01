import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img, title=""):
    plt.figure(figsize=(6,6))
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

# ------------------------------------------------------------
# 1. Load image + extract blue mask (you can replace with your own mask)
# ------------------------------------------------------------
img = cv2.imread(r"C:\Users\leevi\Desktop\blue_floorballs_all.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([95, 80, 40])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# clean small noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)

show(mask, "Original ball mask")


# ------------------------------------------------------------
# 2. Find OUTER contours only
# ------------------------------------------------------------
contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

outline_mask = np.zeros_like(mask)

for cnt in contours:
    if len(cnt) < 20:      # ignore tiny noise
        continue

    # Smooth shape using convex hull (removes inside corners)
    hull = cv2.convexHull(cnt)

    # Draw hull as filled shape
    cv2.drawContours(outline_mask, [hull], -1, 255, thickness=3)

show(outline_mask, "Cleaned outer outlines only")


# ------------------------------------------------------------
# 3. Remove outlines from original mask → get ball centers
# ------------------------------------------------------------
centers_mask = cv2.subtract(mask, outline_mask)
centers_mask = cv2.morphologyEx(centers_mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)

show(centers_mask, "Inner ball centers (ready for detection)")


# ------------------------------------------------------------
# 4. Detect centers using connected components
# ------------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(centers_mask, connectivity=8)

# Draw detections
output = img.copy()
count = 0

for i in range(1, num_labels):   # skip background 0
    x = int(centroids[i][0])
    y = int(centroids[i][1])
    cv2.circle(output, (x,y), 3, (0,0,255), -1)
    count += 1

show(output, f"Detected ball centers: {count} balls")
print("Detected balls:", count)
