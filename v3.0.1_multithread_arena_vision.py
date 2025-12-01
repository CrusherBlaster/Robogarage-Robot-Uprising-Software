"""
The camera captures video frames and the computer is connected to the camera.
The computer finds the camera from its devices and requests the frames.
The frames include the entire square-shaped 1500 mm x 1500 mm arena, its corners,
borders and the ball, goal and robot objects. This is achieved with the camera
looking down into the robot arena (mounted with 2450 mm height from the floor
and aligned so that its optical axis is perpendicular to the floor and aimed
at the center of the arena).

In this multithreaded code we split the video frame processing into multiple threads:
- Thread A: Capture frames from camera
- Thread B: Detect ArUcos
- Thread C: Mask creation, ball recognition, drawing the arucos and balls plus displaying them

This approach allows efficient use of single CPU core by sharing memory between threads.
Threading allows for concurrent execution of tasks within a single process making it
suitable for I/O-bound operations like video processing. The main advantage is
near-simultaneous detection of colored balls, corners and ArUco markers, because
the CPU rapidly switches between the threads. This form of multitasking is particularly
useful in robot arenas where both colored balls and ArUco markers can appear in the
same frame, and where the types of objects entering the arena can be predefined.
"""

# ======================================================================
# 0. Library imports, thread locks, constants and global variables
# ======================================================================

import cv2
import threading
import numpy as np

# Global variables to store the latest frame and detection results
latest_frame = np.zeros((960, 960, 3), np.uint8)
latest_corners = None
latest_ids = None
balls_detected = {}
stop_requested = False

# Only one thread can access the resource at a time
frame_lock = threading.Lock()
aruco_lock = threading.Lock()
mask_lock = threading.Lock()
balls_lock = threading.Lock()

# ======================================================================
# 1. Camera setup
# ======================================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# ======================================================================
# 2. Aruco and color detection setup
# ======================================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ======================================================================
# 3. Camera and arena object detection functions
# ======================================================================

# 3.1 Capture frames from camera
def get_frame():
    global latest_frame, stop_requested
    while not stop_requested:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()

# 3.2 Corner and robot recognition
def get_aruco():
    global latest_corners, latest_ids, latest_frame, stop_requested
    while not stop_requested:
        with frame_lock:
            aruco_frame = latest_frame.copy()
        gray_aruco_frame = cv2.cvtColor(aruco_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco_detector.detectMarkers(gray_aruco_frame)
        with aruco_lock:
            latest_corners, latest_ids = corners, ids

# 3.3 Ball recognition and drawing
def get_results():
    global latest_frame, balls_detected, stop_requested
    while not stop_requested:
        # COLOR MASK CREATION
        with frame_lock:
            color_frame = latest_frame.copy()
        hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)

        masks = {
            'blue_ball_1': cv2.inRange(hsv_frame, (90, 130, 114), (113, 255, 255)),
            'orange_ball_1': cv2.inRange(hsv_frame, (0, 127, 168), (10, 255, 255))
        }

        for ball, mask in masks.items():
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            masks.update({ball: clean_mask})

        # BALL RECOGNITION
        all_circles = []

        for color, mask in masks.items():
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            binary_contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            object_mask = np.zeros_like(binary_mask)

            for binary_contour in binary_contours:
                area = cv2.contourArea(binary_contour)
                if area > 80:
                    cv2.drawContours(object_mask, [binary_contour], -1, 255, cv2.FILLED)
            
            object_mask = cv2.medianBlur(object_mask, 5)
            circles = cv2.HoughCircles(
                object_mask,
                cv2.HOUGH_GRADIENT,
                dp=1.4,
                minDist=25,
                param1=50,
                param2=7.5,
                minRadius=30,
                maxRadius=35
            )

            if circles is not None:
                all_circles.extend(circles[0])

        # DRAWING AND DISPLAYING
        with frame_lock:
            display_frame = latest_frame.copy()

        with aruco_lock:
            cv2.aruco.drawDetectedMarkers(display_frame, latest_corners, latest_ids)

        for x, y, r in all_circles:
            cv2.circle(display_frame, (int(x), int(y)), int(r), (0, 255, 0), 2)

        cv2.imshow("Multithreaded Colored Ball and ArUco Recognition", display_frame)
        if cv2.waitKey(1) == 27:
            stop_requested = True
            cap.release()
            cv2.destroyAllWindows()
            break

# ======================================================================
# 4. Separating the functions to threads
# ======================================================================

capture_thread = threading.Thread(target=get_frame)
aruco_recognition_thread = threading.Thread(target=get_aruco)
drawing_thread = threading.Thread(target=get_results)

capture_thread.start()
aruco_recognition_thread.start()
drawing_thread.start()

capture_thread.join()
aruco_recognition_thread.join()
drawing_thread.join()
