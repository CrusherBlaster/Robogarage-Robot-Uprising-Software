import queue
import threading
import time

import cv2
import numpy as np

# ================================================================
# GLOBAL SETTINGS AND STRUCTURE
# ================================================================


latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

# Create queues (bounded to 1 or 2 elements to drop slow frames)
hsv_queue = queue.Queue(maxsize=2)
gray_queue = queue.Queue(maxsize=2)
mask_queue = queue.Queue(maxsize=2)
contour_queue = queue.Queue(maxsize=2)
aruco_queue = queue.Queue(maxsize=2)
draw_queue = queue.Queue(maxsize=2)
display_queue = queue.Queue(maxsize=2)


# ================================================================
# INITIALIZATION
# ================================================================


def cap_init(device, codec, fps, width, height):
    cap = cv2.VideoCapture(device, codec)
    if not cap.isOpened():
        raise RuntimeError("Camera not found.")
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# ================================================================
# THREAD FUNCTIONS
# ================================================================

def capture_frames(cap):
    """Continuously capture frames and push to queues"""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[capture] Frame read failed.")
            break
        with frame_lock:
            latest_frame = frame.copy()

        hsv_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2HSV)
        gray_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)

        # Send frames to processing queues
        for q, data in [(hsv_queue, hsv_frame), (gray_queue, gray_frame), (draw_queue, latest_frame)]:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    cap.release()
    stop_event.set()


def aruco_worker():
    """Detect ArUco markers from gray frames"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(
        aruco_dict, cv2.aruco.DetectorParameters())
    detections = {}

    while not stop_event.is_set():
        try:
            gray = gray_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        corners, ids, _ = detector.detectMarkers(gray)
        detections["aruco"] = (corners, ids)
        try:
            aruco_queue.put_nowait(detections.copy())
        except queue.Full:
            pass


def mask_worker(object_info):
    """Generate color masks"""
    while not stop_event.is_set():
        try:
            hsv = hsv_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        result = build_mask(hsv, object_info)
        try:
            mask_queue.put_nowait(result)
        except queue.Full:
            pass


def ball_worker():
    """Find contours + balls"""
    while not stop_event.is_set():
        try:
            info = mask_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        info = find_contours(info)
        info = find_balls(info)
        try:
            contour_queue.put_nowait(info)
        except queue.Full:
            pass


def draw_worker():
    """Draw combined outputs"""
    latest_balls = None
    latest_arucos = None

    while not stop_event.is_set():
        try:
            frame = draw_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        if not contour_queue.empty():
            latest_balls = contour_queue.get_nowait()
        if not aruco_queue.empty():
            latest_arucos = aruco_queue.get_nowait()

        if latest_balls is None or latest_arucos is None:
            continue

        frame = draw_ball_center(latest_balls, frame)
        frame = draw_arucos(latest_arucos, frame)
        try:
            display_queue.put_nowait(frame)
        except queue.Full:
            pass


def display_worker():
    """Show final output"""
    while not stop_event.is_set():
        try:
            frame = display_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()


# ================================================================
# DETECTOR UTILITIES (from your code)
# ================================================================

def build_mask(hsv_frame, object_information):
    for role, colors in object_information["arena_objects"]["balls"].items():
        for color, shades in colors.items():
            for shade, info in shades.items():
                low, high = info["masks"]["hsv_ranges"]
                mask = cv2.inRange(hsv_frame, low, high)
                morphed = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                info["masks"].update({"mask": morphed})
    return object_information


def find_contours(obj_info):
    for role, colors in obj_info["arena_objects"]["balls"].items():
        for color, shades in colors.items():
            for shade, info in shades.items():
                mask = info["masks"]["mask"]
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                info["contours"] = contours
    return obj_info


def find_balls(obj_info):
    for role, colors in obj_info["arena_objects"]["balls"].items():
        for color, shades in colors.items():
            for shade, info in shades.items():
                info["ball_coordinates"] = {}
                for i, c in enumerate(info["contours"]):
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                        info["ball_coordinates"][f"ball{i+1}"] = {
                            "center_x": cx, "center_y": cy}
    return obj_info


def draw_ball_center(object_info, frame):
    for role, colors in object_info["arena_objects"]["balls"].items():
        for color, shades in colors.items():
            for shade, info in shades.items():
                for ball, pos in info["ball_coordinates"].items():
                    cv2.circle(
                        frame, (pos["center_x"], pos["center_y"]), 20, (0, 255, 0), 2)
    return frame


def draw_arucos(aruco_info, frame):
    if "aruco" in aruco_info:
        corners, ids = aruco_info["aruco"]
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    return frame


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    cap = cap_init(0, cv2.CAP_DSHOW, 60, 1280, 720)

    object_information = {
        "arena_objects": {
            "balls": {
                "good_balls": {
                    "blue": {
                        "blue_1": {
                            "masks": {
                                "hsv_ranges": [(90, 130, 114), (113, 255, 255)]
                            },
                            "contours": {},
                            "ball_coordinates": {}
                        }
                    },
                    "orange": {
                        "orange_1": {
                            "masks":  {
                                "hsv_ranges": [(0, 127, 168), (10, 255, 255)]
                            },
                            "contours": {},
                            "ball_coordinates": {}
                        }
                    }
                }
            },
        }
    }

    # Define threads
    threads = [
        threading.Thread(target=capture_frames, args=(cap,)),
        threading.Thread(target=aruco_worker),
        threading.Thread(target=mask_worker, args=(object_information,)),
        threading.Thread(target=ball_worker),
        threading.Thread(target=draw_worker),
        threading.Thread(target=display_worker)
    ]

    # Start them
    for t in threads:
        t.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    for t in threads:
        t.join()

    print("All threads stopped cleanly.")
