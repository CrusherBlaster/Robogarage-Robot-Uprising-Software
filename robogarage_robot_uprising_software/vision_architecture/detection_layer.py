

"""
This module detects:
- ArUco markers for arena corners and robots
- Balls of different colors and sizes close to each other, including small ones 
and stores them for the next module 2. Tracking layer
"""


# ============================================================
# 0. Initializing libraries, constants, and global variables
# ============================================================

# most important computer vision libraries for detection and image processing
import cv2
import numpy as np

# Used in small ball detection
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from dataclasses import dataclass


# ============================================================
# 1. Aruco detection functions
# ============================================================

class ArucoDetector():
    
    def __init__(self):
        
        # Aruco detection setup - adjust dictionary and parameters as needed for your markers and environment
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
    # Detect AruCo markers and store them - thread function to be run in the orchestrator
    def detect_aruco(self, frame):
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None:
            return [], []

        for marker_corners, marker_id in zip(corners_list, ids.flatten()):
            # marker_corners is shape (1,4,2), extract the 4 corners
            corners = marker_corners[0].astype(float)  # TL, TR, BR, BL

            detections.append({
                "id": int(marker_id),
                "corners": corners,
            })

        return detections

        
# ============================================================
# 2. Ball detection functions
# ============================================================

class BallDetector():

    # TODO: Use watershed everywhere: https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html https://docs.opencv.org/4.x/d7/d1c/tutorial_js_watershed.html
    # TODO: For circle recognition use HoughCircles instead: https://docs.opencv.org/4.x/d3/de5/tutorial_js_houghcircles.html & 
    # TODO: Check what happens with overlapping colors - previous tests show this doesn't affect detections but check again with more variability.
    # TODO: min_distance in peak_local_max, area thresholds, HSV ranges should ideally be configurable per resolution or environment.
    
    def __init__(self):
        # Add here all the colors of the balls you want to detect
        # TODO: In the future these will be taken from the game_config files but these really need variable/dynamic lighting possibility
        self.HSV_RANGES = {
            'blue': ((90, 130, 114), (113, 255, 255)),
            'orange': ((0, 127, 168), (10, 255, 255)),
        }
        
    def prepare_masks(self, hsv, lower, upper):
        # Threshold the hsv image with the help of lower and upper hue value ranges
        binary_mask = cv2.inRange(hsv, lower, upper)
        
        # If you want to see the filtered color frame for debugging, you can use this:
        # filtered_color_frame = cv2.bitwise_and(frame, frame, mask=mask)
        # but remember to add frame to prepare_masks parameters
        
        # Find contours with connected components - this returns contours that are connected together in two hierarchies: 1. external, 2. internal
        # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # https://docs.opencv.org/4.x/dc/dcf/tutorial_js_contour_features.html  the 4. contour approximation chapter is used here.
        # approximate contours to polygons - this separates the touching balls thanks to the concave parts of the contours.
        # When the polygon is approximated, the concave parts are cut and we can separate the touching balls by filling in 
        # the approximated polygons in the binary mask 
        poly = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]
        for i in range(len(poly)):
            # leave these filled polygons inside the binary_mask for another contour analysis later
            cv2.drawContours(binary_mask, poly, i, 255, thickness=cv2.FILLED)
        # clean up noise from the binary_mask and prepare for the big ball contour analysis
        # TODO: thresholding and medianBlur might sometimes remove very tiny balls or merge them. Consider tuning kernel size.
        clean_binary_mask = cv2.medianBlur(binary_mask, 3)
        clean_binary_mask = cv2.erode(clean_binary_mask, np.ones((3,3), np.uint8), iterations=1)
        # prepare empty canvas for adding small contour detections in the small ball contour analysis later on
        small_mask = np.zeros_like(clean_binary_mask)
        return clean_binary_mask, small_mask
    
    def detect_big_balls(self, clean_binary_mask, small_mask):
        detections = []
        # Find contours and filter by area to separate big balls from small ones and noise
        clean_contours, _ = cv2.findContours(clean_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in clean_contours:
            area = cv2.contourArea(cnt)
            # clean out noise - small meaningless pixels
            # TODO: This is arbitrary - camera position affects this too much
            if area < 100:
                if area < 1:
                    continue
                # add small contours to small mask for later processing
                cv2.drawContours(small_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                continue
            # prepare for drawing circles around detected balls
            (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
            area = float(cv2.contourArea(cnt))
            detections.append({
            "position": (float(center_x), float(center_y)),
            "radius": float(radius),
            "area": area,
            "size": "big"
            })    
        return detections, small_mask

    def detect_small_balls(self, small_mask):
        detections = []
        s_contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Small guard to avoid running computationally heavy operation on empty small mask
        if not s_contours:
            return detections
        # Make empty canvas
        small_filled = np.zeros_like(small_mask)
        # Find the polygon approximation of the contours to fill in the small mask 
        # and prepare for distance transform and watershed segmentation
        s_poly = [cv2.approxPolyDP(cnt, 3, True) for cnt in s_contours]
        # Fill the small contours to prepare for distance transform and watershed segmentation
        for poly in s_poly:
            cv2.drawContours(small_filled, [poly], -1, 255, cv2.FILLED)

        # distance transform and watershed segmentation to separate small balls that are close to each other, 
        # inspired by  https://www.youtube.com/watch?v=WQpXS9gBEu8 and ChatGPT ;)
        dist = cv2.distanceTransform(small_filled, cv2.DIST_L2, 5)
        # TODO: peak_local_max with min_distance=5 assumes a fixed pixel distance. In high-res frames, this might need scaling.
        coords = peak_local_max(dist, min_distance=5, labels=small_filled)
        mask2 = np.zeros(dist.shape, dtype=bool)
        mask2[tuple(coords.T)] = True
        markers, _ = ndi.label(mask2)
        labels = watershed(-dist, markers, mask=small_filled)

        # Add small ball detections
        for label in np.unique(labels):
            if label == 0:
                continue
            comp = np.uint8(labels == label)
            ccnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(ccnts) == 0:
                continue
            cnt = ccnts[0]
            # after watershed, contour is just a blob of pixels
            # minEnclose converts irregular contour → (x, y, r) meaningful geometry
            (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
            area = float(cv2.contourArea(cnt))
            detections.append({
            "position": (float(center_x), float(center_y)),
            "radius": float(radius),
            "area": area,
            "size": "small"
            })  
        return detections

    def detect_balls(self, frame):
        results = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Detecting different ball colors and sizes
        for color_name, (lower, upper) in self.HSV_RANGES.items():
            clean_binary_mask, small_mask = self.prepare_masks(hsv, lower, upper)
            big_detections, small_mask = self.detect_big_balls(clean_binary_mask, small_mask)
            small_detections = self.detect_small_balls(small_mask)
            for det in big_detections + small_detections:
                det["color"] = color_name
                results.append(det)
        return results


# ============================================================
# 3. Detector - the thread for all detections 
# ============================================================
# TODO: Draw all detected balls with color and size label. Helps to see watershed segmentation results.
class Detector():
    def __init__(self, frame_queue, detection_queue, stop_event):
        # Input queue
        self.frame_queue = frame_queue
        # Output queues
        self.detection_queue = detection_queue
        # Orchestrator control
        self.stop_event = stop_event
        # Initialize detectors we will use
        self.aruco_detector = ArucoDetector()
        self.ball_detector = BallDetector()
    
    # TODO: You might want to validate corners and ids before warping.
    def compute_warp_and_roi(self, frame, marker_corners, output_size=(900, 900)):

        if len(marker_corners) < 1:
            return None, None, None

        # 1️. Collect ALL marker corners
        all_pts = np.vstack(marker_corners).astype(np.float32)

        # 2️. Compute convex hull (orientation independent)
        hull = cv2.convexHull(all_pts)

        # 3️. Extract 4 extreme points for homography
        pts = hull.reshape(-1, 2)

        s = np.sum(pts, axis=1)
        diff = np.diff(pts, axis=1)

        src_quad = np.array([
            pts[np.argmin(s)],      # TL
            pts[np.argmin(diff)],   # TR
            pts[np.argmax(s)],      # BR
            pts[np.argmax(diff)]    # BL
        ], dtype=np.float32)

        # 4️. Destination rectangle
        W, H = output_size
        dst_quad = np.array([
            [0, 0],
            [W, 0],
            [W, H],
            [0, H]
        ], dtype=np.float32)

        # 5️. Compute homography
        H_matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)

        # 6. Warp full frame
        warped = cv2.warpPerspective(frame, H_matrix, (W, H))

        # 7. Warp hull into top-down space
        warped_hull = cv2.perspectiveTransform(hull.astype(np.float32), H_matrix)

        # 8️. Create ROI mask in warped space
        # empty canvas
        mask = np.zeros((H, W), dtype=np.uint8)
        # filling it with white polygon with the shape of warped_hull
        cv2.fillPoly(mask, [warped_hull.astype(np.int32)], 255)
        # cutting the warped frame with the mask
        warped_roi = cv2.bitwise_and(warped, warped, mask=mask)

        return warped_roi, H_matrix, warped_hull
    
    def warp_marker_detections(self):
        
    
    # This is the function that will run this entire layer and this function will be run in thread by the conductor in the orchestration_layer.    
    def detect_everything(self):
        # Run both detectors in the same thread by calling this function for simplicity and memory saving
        while not self.stop_event.is_set():
            frame = self.frame_queue.get(timeout=0.1)  # get the latest frame for detection
            if frame is None:
                continue # No frame available, skip iteration. (for clean exit)
            corners = self.aruco_detector.detect_aruco(frame)
            
            # Defining these two just in case nothing gets detected to prevent crash
            ball_results = {}
            
            if corners is not None:
                # Takes the first corner coordinates (top-left) of each marker found
                marker_corners = [c[0] for c in corners]

                warped_roi, H_matrix, warped_hull = self.compute_warp_and_roi(
                    frame,
                    marker_corners,
                    output_size=(900, 900)
                )

                if warped_roi is not None:
                    warped_corners = cv2.perspectiveTransform(warped_hull, H_matrix)
                    # Run ball detection in stabilized top-down view
                    ball_results = self.ball_detector.detect_balls(warped_roi)
            
            self.detection_queue.put((warped_corners))
            self.detection_queue.put(ball_results)
            
            