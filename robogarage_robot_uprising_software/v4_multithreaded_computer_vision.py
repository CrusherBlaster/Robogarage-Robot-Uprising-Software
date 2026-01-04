

import cv2
import numpy as np
import threading
import time
import math
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from enum import Enum

USE_TEST_IMAGE = False
TEST_IMAGE_PATH = r"C:\Users\leevi\Desktop\blue_floorballs_all_hard.png" # developer-provided file

CAMERA_INDEX = 0
FRAME_W = 960
FRAME_H = 960
FPS = 60

class RobotState(Enum):
    IDLE = 0 # No specific task
    GOING_FOR_BALL = 1 # Moving towards a ball
    CARRYING_BALL = 2 # Has a ball and trying to score

HSV_RANGES = {
    'blue': ((90, 130, 114), (113, 255, 255)),
    'orange': ((0, 127, 168), (10, 255, 255)),
}

# Store the robot aruco marker corners (coordinates) here by id
# Note: robot ids are 1,2 for team 1 and 3,4 for team 2
# if you want to change robot's team simply change their ids accordingly manually
ROBOT_TEAMS = {
    'team_1': {
      'robot_1': {
          'id': 1,
          'corners': None,
          'bottom_left': None,
          'bottom_right': None,
          'bottom_center': None,
          'center': None,
          'RobotState': RobotState.IDLE,
          'has_ball': False
          
        }, 
      'robot_2': {
          'id': 2,
          'corners': None,
          'bottom_left': None,
          'bottom_right': None,
          'bottom_center': None,
          'center': None,
          'RobotState': RobotState.IDLE,
          'has_ball': False
        }
    },
    'team_2': {
        'robot_3': {
          'id': 3,
          'corners': None,
          'bottom_left': None,
          'bottom_right': None,
          'bottom_center': None,
          'center': None,
          'RobotState': RobotState.IDLE,
          'has_ball': False
        },
        'robot_4': {
          'id': 4,
          'corners': None,
          'bottom_left': None,
          'bottom_right': None,
          'bottom_center': None,
          'center': None,
          'RobotState': RobotState.IDLE,
          'has_ball': False
        }
    }
}

# Define here storage for the fgur corners
ARENA_CORNERS = {
    'id_46': {'corners': None, 'center': None},
    'id_47': {'corners': None, 'center': None},
    'id_48': {'corners': None, 'center': None},
    'id_49': {'corners': None, 'center': None}
}

SMOOTHING_ALPHA = 0.9

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
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

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

def store_corner(marker_id, marker_corners):
    string_id = f"id_{marker_id}"
    marker_corners = marker_corners.reshape((4, 2))
    # Compute center of marker
    center_x = int(np.mean(marker_corners[:, 0]))
    center_y = int(np.mean(marker_corners[:, 1]))
    center = (center_x, center_y)
    
    # Compute plower direction vector
    bottom_center_x = int((marker_corners[2, 0] + marker_corners[3, 0]) / 2)
    bottom_center_y = int((marker_corners[2, 1] + marker_corners[3, 1]) / 2)
    bottom_center = (bottom_center_x, bottom_center_y)
    
    # Update robot teams
    for robots in ROBOT_TEAMS.values():
        for robot in robots.values():
            if marker_id == robot['id']:
                robot['corners'] = marker_corners 
                robot['bottom_left'] = tuple(marker_corners[3].astype(int))
                robot['bottom_right'] = tuple(marker_corners[2].astype(int))
                robot['center'] = center
                robot['bottom_center'] = bottom_center

    # Update arena corners
    if string_id in ARENA_CORNERS:
        ARENA_CORNERS[marker_id]['corners'] = marker_corners
        ARENA_CORNERS[marker_id]['center'] = center


def get_aruco():
    global latest_corners, latest_ids, latest_frame, stop_requested
    while not stop_requested:
        with frame_lock:
            aruco_frame = latest_frame.copy()
        gray = cv2.cvtColor(aruco_frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejected = aruco_detector.detectMarkers(gray)
        if ids is not None and len(corners) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                store_corner(marker_id, corners[i])
        with aruco_lock:
            latest_corners, latest_ids = corners, ids
        time.sleep(0.005)

def compute_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1e-6:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)

def check_ball_possession(robot_parameters, balls_tracked, max_distance=30):
    rx, ry = robot_parameters['bottom_center']
    for ball_data in balls_tracked.values():
        bx, by = ball_data['smoothed_center_x'], ball_data['smoothed_center_y']
        distance = math.hypot(rx - bx, ry - by)
        if distance < max_distance:
            return True
    return False

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
        # Initialize ball counter for this ball color
        ball_count = 0
        
        # Detecting different ball colors and sizes
        for color_name, (lower, upper) in HSV_RANGES.items():
            # Threshold the hsv image with the help of lower and upper hue value ranges
            mask = cv2.inRange(hsv, lower, upper)
            only_selected_color_frame = cv2.bitwise_and(frame, frame, mask=mask)
            only_selected_color_gray = cv2.cvtColor(only_selected_color_frame, cv2.COLOR_BGR2GRAY)
            # Threshold
            _, binary_mask = cv2.threshold(only_selected_color_gray, 127, 255, cv2.THRESH_BINARY)
            # Find contours and approximate them
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            poly = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]
            for i in range(len(poly)):
                cv2.drawContours(binary_mask, poly, i, 255, thickness=cv2.FILLED)

            # clean up noise
            clean_binary_mask = cv2.medianBlur(binary_mask, 3)
            clean_binary_mask = cv2.erode(clean_binary_mask, np.ones((3,3), np.uint8), iterations=1)

            # empty canvas for small contours
            small_mask = np.zeros_like(clean_binary_mask)

            # find cleaned contours
            clean_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in clean_contours:
                area = cv2.contourArea(cnt)
                if area < 100:
                    if area < 1:
                        continue
                    cv2.drawContours(small_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                    continue
                # normal big balls
                (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
                all_detections.append((center_x, center_y, radius, color_name))
                ball_count += 1

            # ----------------------------
            # New segmentation for small balls
            # ----------------------------
            s_contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            small_filled = np.zeros_like(binary_mask)
            s_poly = [cv2.approxPolyDP(cnt, 3, True) for cnt in s_contours]
            for i in range(len(s_poly)):
                cv2.drawContours(small_filled, s_poly, i, 255, cv2.FILLED)

            # Distance transform
            dist = cv2.distanceTransform(small_filled, cv2.DIST_L2, 5)
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
                (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
                all_detections.append((center_x, center_y, radius, color_name))
                ball_count += 1

        with balls_lock:
            match_detections_to_tracked(all_detections)

        display = frame.copy()
        # Drawing arucos to display frame
        with aruco_lock:
            if latest_ids is not None and latest_corners is not None and len(latest_ids) > 0:
                
                for i, marker_id in enumerate(latest_ids.flatten()):
                    
                    key = f"id_{marker_id}"
                    
                    # ------------ ROBOTS ------------
                    for team_name, robots in ROBOT_TEAMS.items():
                        for robot_name, robot_parameters in robots.items():
                            if marker_id == robot_parameters['id']:
                                
                                # Draw plower direction vector
                                if robot_parameters['bottom_center'] is not None and robot_parameters['center'] is not None:
                                    cv2.arrowedLine(
                                        display,
                                        robot_parameters['center'],
                                        robot_parameters['bottom_center'],
                                        (0, 255, 255),
                                        2,
                                        tipLength=0.3
                                    )
                                
                                # update robot state
                                if check_ball_possession(robot_parameters, balls_tracked):
                                    robot_parameters['has_ball'] = True
                                    robot_parameters['state'] = RobotState.CARRYING_BALL
                                else:
                                    robot_parameters['has_ball'] = False
                                    robot_parameters['state'] = RobotState.GOING_FOR_BALL
                                
                                # visualize ball possession
                                if robot_parameters['center'] is not None:
                                    text = "Ball" if robot_parameters['has_ball'] else "No Ball"
                                    color = (0,255,0) if robot_parameters['has_ball'] else (0,0,255)
                                    cv2.putText(display, text, (robot_parameters['center'][0], robot_parameters['center'][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    
                                cv2.putText(
                                    display,
                                    f"{team_name} robot {marker_id}",
                                    (robot_parameters['center']),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2
                                )

                    # ------------ ARENA CORNERS ------------
                    if key in ARENA_CORNERS:
                        cv2.putText(
                            display,
                            f"Arena corner {marker_id}",
                            (ARENA_CORNERS[key]['center']),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2
                        )
                    
                cv2.aruco.drawDetectedMarkers(display, latest_corners, latest_ids)
                

        # Drawing different size and color balls to the display frame
        with balls_lock:
            for tid, data in balls_tracked.items():
                # TODO: If detected balls outside the drawn borderline - leave them out of the drawing
                sx, sy, sr = int(data['smoothed_center_x']), int(data['smoothed_center_y']), int(data['smoothed_radius'])
                color = (0, 255, 0) if data['color'] == 'blue' else (0, 0, 255)
                cv2.circle(display, (sx, sy), sr, color, 2)
                ball_count = ball_count + 1
                cv2.putText(display, f"id={ball_count}", (sx - 10, sy - sr - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Displaying the results in a new window
        cv2.imshow('Multithreaded Robust Ball And Aruco Detection', display)
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
