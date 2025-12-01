"""
Advanced multithreaded ball detection pipeline for holey floorball (salibandypallo)
Using: contours -> minEnclosingCircle + distanceTransform -> marker-based watershed

This file is written and documented for an advanced user: every nontrivial OpenCV call
and algorithmic choice is explained in docstring/comments. The code keeps your original
three-thread structure (capture, aruco, processing/display) and replaces the fragile
HoughCircles approach with a robust contour->segmentation pipeline.

The developer-provided test image path (useful for offline debugging) is included as:
TEST_IMAGE_PATH = '/mnt/data/2fcd5f99-f84b-430a-ac52-b75914bbfa95.png'

HOW TO USE:
 - If you want to run on a live camera, ensure cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
   and set USE_TEST_IMAGE = False.
 - If you want to debug on a single image, set USE_TEST_IMAGE = True.

Design goals and short summary of why this is superior to HoughCircles in your use case:
 - large morphological closing fills the hole-punctures of the floorball, producing
   a contiguous region for each ball.
 - minEnclosingCircle computes the smallest circumscribing circle of contour points;
   it is geometrically optimal for the given point set and robust to local edge gaps.
 - distanceTransform + marker-connectedComponents + watershed separate touching balls
   by discovering interior maxima (each ball center becomes a maxima in the distance map).
 - exponential smoothing stabilizes sporadic frame-to-frame detection failures.

This script purposely provides many tunable parameters near the top for easy experiment.

"""

import cv2
import numpy as np
import threading
import time
import math

# ---------------------------
# CONFIGURABLE PARAMETERS
# ---------------------------
USE_TEST_IMAGE = False
TEST_IMAGE_PATH = r"C:\Users\leevi\Desktop\blue_floorballs_all.png" # developer-provided file

CAMERA_INDEX = 0
FRAME_W = 960
FRAME_H = 960
FPS = 60

# HSV ranges - tune for your lighting. These are starting points from your existing code.
HSV_RANGES = {
    'blue': ((90, 130, 114), (113, 255, 255)),
    'orange': ((0, 127, 168), (10, 255, 255)),
}

# Morphology kernels: large close to fill holes, smaller open to remove small noise
MORPH_CLOSE_KERNEL = np.ones((0, 0), np.uint8)
MORPH_OPEN_KERNEL = np.ones((6, 6), np.uint8)
PEAKS_OPEN_KERNEL = np.ones((7, 7), np.uint8)

# Contour filters and expected sizes (in pixels) - tune according to camera distance
MIN_CONTOUR_AREA = 250
MAX_CONTOUR_AREA = 100000

# smoothing for detected positions (exponential moving average)
SMOOTHING_ALPHA = 0.25

# ------------------------------------------------------------------
# Thread-safe shared state and locks (kept similar to your original structure)
# ------------------------------------------------------------------
frame_lock = threading.Lock()
aruco_lock = threading.Lock()
balls_lock = threading.Lock()

latest_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
latest_corners = None
latest_ids = None
stop_requested = False

# balls_tracked: dictionary keyed by unique ids
# value: dict with keys: x, y, r, color, last_seen (timestamp), smoothed_x, smoothed_y, smoothed_r
balls_tracked = {}
next_local_ball_id = 0
ball_count = 0
debug_1 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
debug_2 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
debug_3 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
# ---------------------------
# Camera / capture thread
# ---------------------------
if not USE_TEST_IMAGE:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera. If you want, set USE_TEST_IMAGE=True and DEBUG on a saved frame.")
        # do not exit; allow user to debug with test image
    else:
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)


def get_frame():
    """Continuously capture frames from camera or feed the same test image repeatedly.

    We copy into latest_frame under frame_lock to provide consistent snapshots to the
    processing thread. This is the only place frames are read from the hardware.

    Note on threads: real-time camera I/O is I/O-bound; keeping capture in its own
    thread prevents blocking processing and allows stable frame rates.

    """
    global latest_frame, stop_requested
    if USE_TEST_IMAGE:
        # Read once and reuse to emulate camera
        img = cv2.imread(TEST_IMAGE_PATH)
        if img is None:
            print('Warning: test image not found at', TEST_IMAGE_PATH)
            stop_requested = True
            return
        img = cv2.resize(img, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        while not stop_requested:
            with frame_lock:
                latest_frame = img.copy()
            time.sleep(1.0 / max(1, FPS))
        return

    while not stop_requested:
        ret, frame = cap.read()
        if not ret:
            # camera read failed this cycle; wait a bit and continue
            time.sleep(0.01)
            continue
        # Ensure consistent size
        frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        with frame_lock:
            latest_frame = frame.copy()

# ---------------------------
# ArUco detection thread (left largely as-is)
# ---------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def get_aruco():
    """Detect ArUco markers on a grayscale snapshot of latest_frame and store results.

    cv2.aruco.ArucoDetector.detectMarkers expects a grayscale image. We store corners and ids
    under aruco_lock so the display thread can draw them.
    """
    global latest_corners, latest_ids, latest_frame, stop_requested
    while not stop_requested:
        with frame_lock:
            aruco_frame = latest_frame.copy()
        gray = cv2.cvtColor(aruco_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco_detector.detectMarkers(gray)
        with aruco_lock:
            latest_corners, latest_ids = corners, ids
        time.sleep(0.005)

# ---------------------------
# Helper geometric / image functions
# ---------------------------

def compute_circularity(contour):
    """Return circularity measure (4*pi*area / perimeter^2) for a contour.

    Circularity is 1.0 for a perfect circle and decreases for elongated or complex shapes.
    We use it to quickly accept clean single-ball contours.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1e-6:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


# ---------------------------
# Main processing thread (core of algorithm)
# ---------------------------

def process_and_display():
    """This function performs the following for each latest_frame snapshot:

    1) Convert to HSV and create color masks for each target color.
    2) Morphologically clean each mask: large closing to fill holes, small opening to
       remove small noise.
    3) Find contours in the cleaned mask (RETR_EXTERNAL). For each contour:
       - compute area, circularity, minEnclosingCircle
       - if circularity >= threshold and containing circle radius in expected range,
         accept as single ball
       - otherwise attempt to split: crop ROI -> distanceTransform -> threshold ->
         connectedComponents -> prepare markers -> watershed -> extract sub-contours
    4) Smooth detections over time to avoid flicker (exponential moving average)
    5) Draw results and ArUco on the display frame.

    Each OpenCV call is used deliberately: their exact semantics are documented in inline comments
    where it matters.
    """
    global latest_frame, latest_corners, latest_ids, balls_tracked, next_local_ball_id, ball_count, stop_requested, debug_1, debug_2, debug_3

    # Local function for smoothing / updating tracked balls
    def update_tracked_ball(local_id, x, y, r, color):
        now = time.time()
        entry = balls_tracked.get(local_id)
        if entry is None:
            balls_tracked[local_id] = {
                'center_x': x, 'center_y': y, 'radius': r, 'color': color, 'last_seen': now,
                # Separate storage for smoothed values coming later
                'smoothed_center_x': x, 'smoothed_center_y': y, 'smoothed_radius': r
            }
            return
        # exponential smoothing: new_smoothed = old * (1-alpha) + new * alpha
        sx = entry['smoothed_center_x'] * (1.0 - SMOOTHING_ALPHA) + x * SMOOTHING_ALPHA
        sy = entry['smoothed_center_y'] * (1.0 - SMOOTHING_ALPHA) + y * SMOOTHING_ALPHA
        sr = entry['smoothed_radius'] * (1.0 - SMOOTHING_ALPHA) + r * SMOOTHING_ALPHA
        entry.update({'center_x': x, 'center_y': y, 'radius': r, 'last_seen': now, 'smoothed_center_x': sx, 'smoothed_center_y': sy, 'smoothed_radius': sr})

    def register_new_tracked_ball(x, y, r, color):
        global next_local_ball_id
        lid = next_local_ball_id
        next_local_ball_id += 1
        balls_tracked[lid] = {'center_x': x, 'center_y': y, 'radius': r, 'color': color, 'last_seen': time.time(),
                              'smoothed_center_x': x, 'smoothed_center_y': y, 'smoothed_radius': r}
        return lid

    # For matching detections to previous tracked objects we use a simple greedy nearest match
    def match_detections_to_tracked(detections):
        # detections: list of (x,y,r,color)
        # We compute pairwise distances to current tracked objects and assign if close
        assigned = set()
        used_tracked = set()

        # Build list of tracked ids
        tracked_items = list(balls_tracked.items())  # list of (id, dict)
        for det in detections:
            x, y, r, color = det
            best_id = None
            best_dist = None
            for tid, tdata in tracked_items:
                if tid in used_tracked:
                    continue
                # Only match same color (optional but helpful)
                if tdata['color'] != color:
                    continue
                dx = tdata['center_x'] - x
                dy = tdata['center_y'] - y
                dist = math.hypot(dx, dy)
                # threshold: if centers reasonably near expected radius
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = tid
            # decide accept
            if best_id is not None and best_dist < max(30, r * 1.5):
                update_tracked_ball(best_id, x, y, r, color)
                used_tracked.add(best_id)
            else:
                # new track
                register_new_tracked_ball(x, y, r, color)

        # Cleanup: remove tracks not seen for some time
        now = time.time()
        to_delete = []
        for tid, tdata in list(balls_tracked.items()):
            if now - tdata['last_seen'] > 0.01:  # 0.01 seconds stale -> remove
                to_delete.append(tid)
        for tid in to_delete:
            del balls_tracked[tid]

    # Processing loop
    while not stop_requested:
        with frame_lock:
            frame = latest_frame.copy()

        if frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        all_detections = []  # list of (x,y,r,color)

        ball_count = 0 # restart counter
        
        # For each color: create mask, clean it, find contours
        for color_name, (lower, upper) in HSV_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            c_mask = mask.copy()
            c_contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            """
            for contour in c_contours:
                circularity = compute_circularity(contour)
                # minEnclosingCircle is geometrically smallest circle that contains all contour points
                (circle_center_x, circle_center_y), radius = cv2.minEnclosingCircle(contour)
                # Quick accept condition: sufficiently circular and radius reasonable
                if circularity >= 0.7 and 20 < radius < 200:
                    print("Sufficiently circular: ", color_name)
                    all_detections.append((circle_center_x, circle_center_y, radius, color_name))
                    continue
            """
            # commented lines for the mid sized balls
            #for mid-sized balls gaussianblur(7,7)
            mask = cv2.GaussianBlur(mask, (11,11), 0)
            #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (np.ones((3,3), np.uint8))) mid sized
            # kernel = np.ones((9,9),np.uint8)
            # kernel_d = cv2.getStructuringElement(cv2.MORPH_DIAMOND,(9,9))
           #mask = cv2.erode(mask, kernel_d, iterations=1)
            #mask = cv2.erode(mask, (np.ones((7,7), np.uint8)))
            #mask = cv2.erode(mask, (np.ones((3,3), np.uint8)))
            #mask = cv2.dilate(mask, (np.ones((5,5), np.uint8)))
            #mask = cv2.dilate(mask, kernel, iterations=1)
            # Optional: additional median blur to remove shot noise
            # mask = cv2.medianBlur(mask, 5)
            #mask = cv2.erode(mask, (np.ones((9,9), np.uint8)), iterations=1)
            #mask = cv2.dilate(mask, (np.ones((7,7), np.uint8)), iterations=1)
            #mask = cv2.GaussianBlur(mask, (11, 11), 0)
            
            # Find outer contours only (RETR_EXTERNAL)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                    continue
                (circle_center_x, circle_center_y), radius = cv2.minEnclosingCircle(contour)
                
                
                rect_up_l_x, rect_up_l_y, rect_width, rect_height = cv2.boundingRect(contour)
                # Expand the crop box a bit so we don't lose contour pixels in the crop
                pad = 6
                # expand left and down but don't go out of img - beyond 0 to negative
                # The x and y coordinate values are just img matrix indexes: column(x), row(y)
                column_x0 = max(0, rect_up_l_x - pad) 
                row_y0 = max(0, rect_up_l_y - pad)
                # expand right and up but don't go out of img - beyond frame width/height
                column_x1 = min(frame.shape[1], rect_up_l_x + rect_width + pad) 
                row_y1 = min(frame.shape[0], rect_up_l_y + rect_height + pad)
                # Select every pixel whose row index is in [row_y0, row_y1) 
                # and whose column index is in [column_x0, column_x1) and 
                # return that rectangular sub-image as a new matrix.
                # ROI = Region of interest
                roi_mask = mask[row_y0:row_y1, column_x0:column_x1]
                roi_color_frame = frame[row_y0:row_y1, column_x0:column_x1]

                # distanceTransform expects binary foreground = non-zero and background = 0.
                # However we want peaks inside foreground -> compute distance to background.
                # We invert mask: foreground (ball) = 255 makes distanceTransform measure
                # distance to zero-valued background. We want high values at ball centers.
                # Ensure roi_mask is binary: pixels <=127 become 0 (background black), pixels >127 become 255 (foreground white)
                _, roi_binary_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
                debug_1 = roi_binary_mask
                # Compute distance transform (L2) - dist will hold distance in pixels
                # L2 (Euclidean) distance: sqrt((x1 - x0)^2 + (y1 - y0)^2)
                # For each non-zero (foreground) pixel, computes distance to the nearest zero (background) pixel
                # 2 1 0 <- peak is within 2 pixel distance from background 0
                dist = cv2.distanceTransform(roi_binary_mask, distanceType=cv2.DIST_L2, maskSize=5)

                
                if dist.max() <= 1e-6:
                    print("#distance map degraded, fallback to enclosing circle")
                    all_detections.append((circle_center_x, circle_center_y, radius, color_name))
                    continue
                
                # normalize dist for robust thresholding across resolutions
                dist_norm = dist / dist.max()
                # dist = (dist * 255).astype('uint8')
                # debug = dist
                # debug = dist_norm
                
                # Threshold peaks: 
                # keep pixels whose normalized distance is between [DIST_PEAK_FACTOR, max]
                # Keeps the pixels closest to the circle center because center pixel has dist.max
                # and the pixels around it have only fractions of the max distance.
                # 1 0 0  <- Only peak is preserved and turned into binary
                
                _, peaks = cv2.threshold(dist_norm, 0.03*dist.max(), 1.0, cv2.THRESH_BINARY)
                # All 1 values become white = 255
                peaks = (peaks * 255).astype('uint8')
                #peaks = cv2.dilate(peaks, (3,3),iterations = 1)
                # peaks = cv2.erode(peaks, (3,3),iterations = 1)
                
                # cleaning around peaks e.g. 1 0 1  -> 0 0 0
                # Real ball centers produce peaks large and round 1 1 1 1 1 etc...
                # Noise produces thin, point-like “peaks”
                # → Opening deletes the noise, keeps real peaks
                
                
                
                debug_2 = peaks
                """
                cnts_m, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts_m:
                    continue
                cnt_m = max(cnts_m, key=cv2.contourArea)
                
                area_m = cv2.contourArea(cnt_m)
                for cnt_m in cnts_m:
                    # These are local coords because the contour exists in mask derived from roi_mask
                    (mx, my), mr = cv2.minEnclosingCircle(cnt_m)
                    # convert local coords to frame coords
                    all_detections.append((column_x0 + mx, row_y0 + my, mr, color_name))

                # Marking the labeled pixels:
                # Labels every pixel=marker with int 0...n_label-1
                # Marks these labels to a new 2D matrix
                # The matrix stores labels (integers)
                # so note that none of them are 255 here and its not image
                """
                n_labels, markers = cv2.connectedComponents(peaks)
                if n_labels <= 1:
                    print("no distinct peaks -> fallback")
                    all_detections.append((circle_center_x, circle_center_y, radius, color_name))
                    continue

                # Prepare the markers matrix for watershed:
                markers = markers + 1
                n_labels = n_labels + 1 # prepare for after watershed
                # Previously 1 0 0 now 2 1 1 <- note that 
                # markers are all the pixels and labels group the markers so
                # this ends up temporarily incrementing background pixel +
                # peak pixels + the pixels dismissed from the peak group
                # Mark background pixels (where roi_binary==0) as 0 in markers
                markers[roi_binary_mask == 0] = 0
                markers[roi_binary_mask == 1] = 1
                # Now  2 1 0 <- contains the peak and the original ball pixels lost
                # in the peak threshold, background 0

                # Marker with the value 2 expands through all pixels labeled 1 (unassigned foreground)
                # It stops when reaching other markers or image gradients
                # Boundary pixels become -1
                # Note that here the marker integer matrix is cast on image 
                # and this is how it can tell the coordinates of the peaks etc from color image
                markers = cv2.watershed(roi_color_frame, markers.astype('int32'))
                debug_3 = markers.astype('uint8')
                # Now 2 2 -1
                # 0 = background
                # 2..n_labels = the peak markers
                # 1 = pixels inside ball that weren’t in any peak (unassigned foreground)  
                # Each peak label grows into the nearby unassigned foreground pixels.
                # So pixels that were 1 (unassigned foreground) now take the label of the nearest peak.
                # The boundaries between regions become -1.
                # Resulting matrix (markers) might look like this:

                # -1 -1 -1 -1 -1
                # -1 2 2 3 -1
                # -1 2 2 3 -1
                # -1 4 4 3 -1
                # -1 -1 -1 -1 -1

                #  2 → all pixels that belong to the first peak
                #  3 → all pixels that belong to the second peak
                #  4 → all pixels that belong to the third peak
                # -1 → boundary between regions
                #  0 → background (outside ROI)
                
                # Extract each marker region and compute minEnclosingCircle:
                for mid in range(2, n_labels):
                    # Finally convert the integer matrix to 0 and 255 values
                    # Because e.g. on 1st iteration: 2 == 2 -> true = 1 -> 1 * 255 = 255  
                    # On 2nd iteration: 2 == 3 -> false -> 0 * 255 = 0
                    # On 2nd iteration. 3 == 3 -> true -> 1 * 255 = 255 
                    # Thats why each iteration we save the labeled contour.
                    mask_m = (markers == mid).astype('uint8') * 255 
                    # small cleanup
                    # mask_m = cv2.morphologyEx(mask_m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    cnts_m, _ = cv2.findContours(mask_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts_m:
                        continue
                    cnt_m = max(cnts_m, key=cv2.contourArea)
                    
                    area_m = cv2.contourArea(cnt_m)
                    if area_m < 100:
                        continue
                    
                    # These are local coords because the contour exists in mask derived from roi_mask
                    (mx, my), mr = cv2.minEnclosingCircle(cnt_m)
                    # convert local coords to frame coords
                    all_detections.append((column_x0 + mx, row_y0 + my, mr, color_name))
                
                    

        # Now we have a list of detections; match them to tracked objects with smoothing
        with balls_lock:
            match_detections_to_tracked(all_detections)

        # Build display frame: draw aruco and detected circles (smoothed)
        display = frame.copy()
        with aruco_lock:
            if latest_corners is not None and latest_ids is not None:
                cv2.aruco.drawDetectedMarkers(display, latest_corners, latest_ids)

        with balls_lock:
            for tid, data in balls_tracked.items():
                sx, sy, sr = int(data['smoothed_center_x']), int(data['smoothed_center_y']), int(data['smoothed_radius'])
                color = (0, 255, 0) if data['color'] == 'blue' else (0, 0, 255)
                cv2.circle(display, (sx, sy), sr, color, 2)
                ball_count = ball_count + 1
                cv2.putText(display, f"id={ball_count}", (sx - 10, sy - sr - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Multithreaded Robust Ball And Aruco Detection', display)
        cv2.imshow('Debug 1', debug_1)
        cv2.imshow('Debug 2', debug_2)
        cv2.imshow('Debug 3', debug_3)
        if cv2.waitKey(1) == 27:
            stop_requested = True
            break

    # Cleanups
    if not USE_TEST_IMAGE:
        cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Launch threads
# ---------------------------

capture_thread = threading.Thread(target=get_frame, daemon=True)
aruco_thread = threading.Thread(target=get_aruco, daemon=True)
process_thread = threading.Thread(target=process_and_display, daemon=True)

capture_thread.start()
aruco_thread.start()
process_thread.start()

# Wait until user requests stop (threads are daemonic so program exits once main thread ends)
try:
    while not stop_requested:
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_requested = True

# wait a tiny bit for threads to exit cleanly
time.sleep(0.5)

print('Exiting')
