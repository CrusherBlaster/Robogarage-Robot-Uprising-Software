

"""
Multithreading for five threads running five functions:
Functions:
1. Capturing frames 
2. Detecting & categorizing arucos from gray filter
3. Detecting colors and making color filtered frames
4. Detecting ball shapes, categorizing the balls from color masks
5. Drawing the detections to the display frame. 
Threads run these separate functions at the same time to get 
real time feedback and faster processing.
"""


# ======================================================================
# 0. Importing predefined libraries of classes, methods and functions
# ======================================================================


# 0.1 Camera, multiprocessing and math controls
# ----------------------------------------------------------------------

import cv2
import threading
import numpy as np


# ======================================================================
# 1. Camera initialization
# ======================================================================


# 1.1 Choosing different video codec / backend to reduce data
# ----------------------------------------------------------------------
 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# 1.2 Check if camera opened successfully
# ----------------------------------------------------------------------

if not cap.isOpened():
    print("Error: Camera connection lost.")
    exit()


# 1.3 Higher framerate for less latency in recognition
# ----------------------------------------------------------------------

cap.set(cv2.CAP_PROP_FPS, 144)


# 1.4 Setting lower resolution to get less data per frame
# ----------------------------------------------------------------------

# Set the frame width lower to cut data & the view to the arena
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1536)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)


# ======================================================================
# 2. Defining the five classes
# ======================================================================


# 2.1 this guy recognises colors and sorts them to different color groups that hold color masks

# 2.2 this aruco recognition guy takes the gray colored frame and detects arucos from it and classifies them to different object classes: corners, robots, 

# 2.3 this ball guy takes the orange and blue colors and take specific right orange and blue masks from them and then classify ball recognitions for example to the orange ball class and inside that the specific orange masks and objects that match with them  


latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
latest_corners = []
latest_ids = None
latest_colors = {'blue': np.zeros((480,640), dtype=np.uint8), 
                 'orange':np.zeros((480,640), dtype=np.uint8)}

frame_lock = threading.Lock()
aruco_lock = threading.Lock()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def detect_colors(hsv):
    blue_mask = cv2.inRange( hsv, (90, 130, 114), (113,255,255) )
    orange_mask = cv2.inRange( hsv, (0,127,168), (10,255,255) )
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return {'blue': blue_mask, 'orange': orange_mask}


def capture_frames():
    global latest_frame
    while True:
        ret, frame = cap.read()
        # ret is boolean. It's True if read succeeded
        # frame is actual image data (NumPy array)
        if ret:
            with frame_lock:
                latest_frame = frame.copy()


def aruco_recognition():
    global latest_corners, latest_ids
    while True:
        with frame_lock:
            aruco_frame = latest_frame.copy()
        gray_aruco_frame = cv2.cvtColor(aruco_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco_detector.detectMarkers(gray_aruco_frame)
        with aruco_lock:
            latest_corners, latest_ids = corners, ids


def color_recognition():
    global latest_colors
    while True:
        with frame_lock:
            color_frame = latest_frame.copy()
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        latest_colors = detect_colors(hsv)

def display_thread():
    while True:
        with frame_lock:
            display_frame = latest_frame.copy()
        
        
        # Draw ArUco detections
        with aruco_lock:
            cv2.aruco.drawDetectedMarkers(display_frame, latest_corners, latest_ids)

        # Draw color detenctions
        for color, mask in latest_colors.items():

            # Make sure mask is binary
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours (including holes)
            binary_contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create empty mask
            object_mask = np.zeros_like(binary_mask)

            for i, binary_contour in enumerate(binary_contours):
                area = cv2.contourArea(binary_contour)
                if area > 300:
                    # Fill the contour
                    cv2.drawContours(object_mask, [binary_contour], -1, 255, cv2.FILLED)
            
                #leaving only objects with closed contours

            kernel = np.ones((25,25), np.uint8)
            clean_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
            
            clean_contours, rejected = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for clean_contour in clean_contours:
                area = cv2.contourArea(clean_contour)
                perimeter = cv2.arcLength(clean_contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.6:
                    continue
                moments = cv2.moments(clean_contour)
                if moments["m00"] != 0:
                    center_x = int( moments["m10"] / moments["m00"] )
                    center_y = int( moments["m01"] / moments["m00"] )
                    cv2.circle( display_frame, (center_x, center_y), 25, (0,255,0), 3 )
                    cv2.putText(display_frame, color, (center_x, center_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Multithreaded Colored Ball and ArUco Recognition", display_frame)
        if cv2.waitKey(1) == 27: # ESC
            cv2.destroyAllWindows()
            break


# ======================================================================
# 3. Assigning functions to the threads and running the five threads
# ======================================================================


# 3.1 Assigning threads to run functions
# ----------------------------------------------------------------------

# Thread A: capture frames
capture_thread = threading.Thread(target=capture_frames) 

# Thread B: making color filtered frames
color_recognition_thread = threading.Thread(target=color_recognition)

# Thread D: aruco recognition from gray frames
aruco_recognition_thread = threading.Thread(target=aruco_recognition)

# Thread E: drawing results from the threads A-D to one frame
display_thread = threading.Thread(target=display_thread) 


# 3.2 Running the threads simultaneously
# ----------------------------------------------------------------------

capture_thread.start()
color_recognition_thread.start()
aruco_recognition_thread.start()
display_thread.start()


