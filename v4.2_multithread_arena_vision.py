

import cv2
import numpy as np
import threading
import time
import math

USE_TEST_IMAGE = False
TEST_IMAGE_PATH = r"C:\Users\leevi\Desktop\blue_floorballs_all.png" # developer-provided file

CAMERA_INDEX = 0
FRAME_W = 960
FRAME_H = 960
FPS = 60

HSV_RANGES = {
    'blue': ((90, 130, 114), (113, 255, 255)),
    'orange': ((0, 127, 168), (10, 255, 255)),
}

MORPH_CLOSE_KERNEL = np.ones((0, 0), np.uint8)
MORPH_OPEN_KERNEL = np.ones((6, 6), np.uint8)
PEAKS_OPEN_KERNEL = np.ones((7, 7), np.uint8)

MIN_CONTOUR_AREA = 250
MAX_CONTOUR_AREA = 100000

SMOOTHING_ALPHA = 0.25


frame_lock = threading.Lock()
aruco_lock = threading.Lock()
balls_lock = threading.Lock()

latest_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
latest_corners = None
latest_ids = None
stop_requested = False

balls_tracked = {}
next_local_ball_id = 0
ball_count = 0
debug_1 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
debug_2 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
debug_3 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

if not USE_TEST_IMAGE:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera. If you want, set USE_TEST_IMAGE=True and DEBUG on a saved frame.")
    else:
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)


def get_frame():
    global latest_frame, stop_requested
    if USE_TEST_IMAGE:
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
            time.sleep(0.01)
            continue
        frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        with frame_lock:
            latest_frame = frame.copy()


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


def process_and_display():
    global latest_frame, latest_corners, latest_ids, balls_tracked, next_local_ball_id, ball_count, stop_requested, debug_1, debug_2, debug_3

    def update_tracked_ball(local_id, x, y, r, color):
        now = time.time()
        entry = balls_tracked.get(local_id)
        if entry is None:
            balls_tracked[local_id] = {
                'center_x': x, 'center_y': y, 'radius': r, 'color': color, 'last_seen': now,
                'smoothed_center_x': x, 'smoothed_center_y': y, 'smoothed_radius': r
            }
            return
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

    def match_detections_to_tracked(detections):
        assigned = set()
        used_tracked = set()

        tracked_items = list(balls_tracked.items())  
        for det in detections:
            x, y, r, color = det
            best_id = None
            best_dist = None
            for tid, tdata in tracked_items:
                if tid in used_tracked:
                    continue
                if tdata['color'] != color:
                    continue
                dx = tdata['center_x'] - x
                dy = tdata['center_y'] - y
                dist = math.hypot(dx, dy)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None and best_dist < max(30, r * 1.5):
                update_tracked_ball(best_id, x, y, r, color)
                used_tracked.add(best_id)
            else:
                register_new_tracked_ball(x, y, r, color)

        now = time.time()
        to_delete = []
        for tid, tdata in list(balls_tracked.items()):
            if now - tdata['last_seen'] > 0.01:  
                to_delete.append(tid)
        for tid in to_delete:
            del balls_tracked[tid]

    while not stop_requested:
        with frame_lock:
            frame = latest_frame.copy()

        if frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        all_detections = []  

        ball_count = 0
        
        for color_name, (lower, upper) in HSV_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            c_mask = mask.copy()
            c_contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = cv2.GaussianBlur(mask, (11,11), 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                    continue
                (circle_center_x, circle_center_y), radius = cv2.minEnclosingCircle(contour)
                rect_up_l_x, rect_up_l_y, rect_width, rect_height = cv2.boundingRect(contour)
                pad = 6
                column_x0 = max(0, rect_up_l_x - pad) 
                row_y0 = max(0, rect_up_l_y - pad)
                column_x1 = min(frame.shape[1], rect_up_l_x + rect_width + pad) 
                row_y1 = min(frame.shape[0], rect_up_l_y + rect_height + pad)
                roi_mask = mask[row_y0:row_y1, column_x0:column_x1]
                roi_color_frame = frame[row_y0:row_y1, column_x0:column_x1]
                _, roi_binary_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
                debug_1 = roi_binary_mask
                dist = cv2.distanceTransform(roi_binary_mask, distanceType=cv2.DIST_L2, maskSize=5)
                if dist.max() <= 1e-6:
                    print("#distance map degraded, fallback to enclosing circle")
                    all_detections.append((circle_center_x, circle_center_y, radius, color_name))
                    continue
                dist_norm = dist / dist.max()
                _, peaks = cv2.threshold(dist_norm, 0.03*dist.max(), 1.0, cv2.THRESH_BINARY)
                peaks = (peaks * 255).astype('uint8')
                debug_2 = peaks
                n_labels, markers = cv2.connectedComponents(peaks)
                if n_labels <= 1:
                    print("no distinct peaks -> fallback")
                    all_detections.append((circle_center_x, circle_center_y, radius, color_name))
                    continue
                markers = markers + 1
                n_labels = n_labels + 1 
                markers[roi_binary_mask == 0] = 0
                markers[roi_binary_mask == 1] = 1
                markers = cv2.watershed(roi_color_frame, markers.astype('int32'))
                debug_3 = markers.astype('uint8')
                for mid in range(2, n_labels):
                    mask_m = (markers == mid).astype('uint8') * 255 
                    cnts_m, _ = cv2.findContours(mask_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts_m:
                        continue
                    cnt_m = max(cnts_m, key=cv2.contourArea)
                    area_m = cv2.contourArea(cnt_m)
                    if area_m < 100:
                        continue
                    (mx, my), mr = cv2.minEnclosingCircle(cnt_m)
                    all_detections.append((column_x0 + mx, row_y0 + my, mr, color_name))

        with balls_lock:
            match_detections_to_tracked(all_detections)

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
    if not USE_TEST_IMAGE:
        cap.release()
    cv2.destroyAllWindows()

capture_thread = threading.Thread(target=get_frame, daemon=True)
aruco_thread = threading.Thread(target=get_aruco, daemon=True)
process_thread = threading.Thread(target=process_and_display, daemon=True)

capture_thread.start()
aruco_thread.start()
process_thread.start()

try:
    while not stop_requested:
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_requested = True

time.sleep(0.5)

print('Exiting')
