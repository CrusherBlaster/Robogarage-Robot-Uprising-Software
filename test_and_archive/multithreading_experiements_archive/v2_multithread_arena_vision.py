

"""
The camera captures video frames
and the computer is connected to the camera.
The computer finds the camera from its
devices and requests the frames. 
The frames include the entire square-shaped
1500 mm x 1500 mm arena, its corners,
borders and the ball, goal and robot objects. 
This is achieved with the camera looking down 
into the robot arena (mounted with 2450 mm height 
from the floor and aligned so that it's optical axis
is perpendicular to the floor and aimed at the
center of the arena).   

In this multithreaded code we split 
the video frame processing into multiple threads:
    - Thread A: Capture frames from camera
    - Thread B: Detect ArUcos 
    - Thread C: Mask creation, ball recognition, 
                drawing the arucos and balls
                plus displaying them

This approach allows efficient use of single CPU core
by sharing memory between threads. Threading allows
for concurrent execution of tasks within a single process
making it suitable for I/O-bound operations like video processing.

The main advantage is near-simultaneous detection of
colored balls, corners and ArUco markers, because the
CPU rapidly switches between the threads.

This form of multitasking is particularly useful in robot
arenas where both colored balls and ArUco markers can appear
in the same frame, and where the types of objects entering the 
arena can be predefined.
"""


# ======================================================================
# 0. Library imports, thread locks, constants and global variables
# ======================================================================

# All image processing and computer vision is done with OpenCV
import cv2
# Threading is used to run multiple tasks "simultaneously"
import threading
# NumPy is used for matrix operations on image matrices
import numpy as np

# Global variables to store the latest frame and detection results
latest_frame = np.zeros((960,960,3), np.uint8)
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

# Choosing lighter video codec / backend 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
    
# Requesting sufficient frame rate
cap.set(cv2.CAP_PROP_FPS, 60)

# Setting the frame to just have the entire arena visible
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)


# ======================================================================
# 2. Aruco and color detection setup
# ======================================================================

# The used ArUco dictionary can be changed here
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# The parameters for the ArUco detection can be tuned here
aruco_params = cv2.aruco.DetectorParameters()

# Create the ArUco detector with the specified dictionary and parameters
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


# ======================================================================
# 3. Camera and arena object detection functions
# ======================================================================

# 3.1 Capture frames from camera
# ----------------------------------------------------------------------    
def get_frame():
    
    global latest_frame, stop_requested
    
    while not stop_requested:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                # Copy for the other threads to use.
                # This prevents race conditions.
                # While new frame is read,
                # the old frame is still stored.
                # After this the old frame gets 
                # overwritten with the new frame            
                latest_frame = frame.copy()

# 3.2 Corner and robot recognition
# ----------------------------------------------------------------------  
def get_aruco():
    
    global latest_corners, latest_ids, latest_frame, stop_requested
    
    while not stop_requested:
        
        with frame_lock:
            aruco_frame = latest_frame.copy()
            # TODO: Regions of Interest (ROIs) can be used here
        
        gray_aruco_frame = cv2.cvtColor(aruco_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco_detector.detectMarkers(gray_aruco_frame)
        
        # TODO: Dividing the id's into 
        # different categories can be done here
        # Such as corner ids and robot teams and their player ids
        
        with aruco_lock:
            latest_corners, latest_ids = corners, ids

# 3.3 Ball recognition from color masks and drawing arucos + balls
# ----------------------------------------------------------------------
# All in one thread to get faster results with less frame copying
# This solution fits this small object count application  
def get_results():
    
    # TODO: Regions of interest: cut the frame to the object
    
    global latest_frame, balls_detected, stop_requested
    
    while not stop_requested:
        
        # COLOR MASK CREATION
        # --------------------------------------
        with frame_lock:
            color_frame = latest_frame.copy()
        hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        
        # Put here the color ranges for the colored balls to be detected
        # These values were found with color_finder.py and they proved
        # to work reasonably well in the arena environment lighting.
        # well = less noise, more relevant colors
        # Put here your different balls in the arena.
        # If you make subsets or classify "good/bad balls", 
        # change the for loop too.
        masks = {
            'blue_ball_1': cv2.inRange( hsv_frame, (90, 130, 114), (113,255,255) ),
            'orange_ball_1': cv2.inRange( hsv_frame, (0,127,168), (10,255,255) )
        }
          
        # Morphological operations to clean up the masks
        for ball, mask in masks.items():
            # Closing small holes inside the foreground objects
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            # Opening (reducing) small noise outside the foreground objects
            clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            masks.update({ball: clean_mask})
        
        
        # BALL RECOGNITION
        # --------------------------------------
        # Clear previous detections
        balls_detected = {}  
        
        # Iterate through each color mask
        for color, mask in masks.items():
            
            # Initialize list for detected balls of this color
            balls_detected[color] = []
            
            # Converting the mask to binary colors black and white
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours of objects (including holes inside objects)
            binary_contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create empty mask
            object_mask = np.zeros_like(binary_mask)

            for i, binary_contour in enumerate(binary_contours):
                area = cv2.contourArea(binary_contour)
                if area > 80:
                    # Fill the contour
                    cv2.drawContours(object_mask, [binary_contour], -1, 255, cv2.FILLED)
            
                #leaving only objects with closed contours
                
            # Optional Morphological operations to clean up the mask
            # kernel = np.ones((3,3), np.uint8)
            # clean_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
            # clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours on the cleaned mask
            # Change the mask if you want to debug the ball recognition 
            clean_contours, rejected = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find balls from the clean contours
            for clean_contour in clean_contours:
                
                # The circularity test
                area = cv2.contourArea(clean_contour)
                perimeter = cv2.arcLength(clean_contour, True)
                # Avoiding some math errors
                if perimeter == 0:
                    continue
                # Let's turn the contour area into perfect circle and
                # check how much the current perimeter matches with it
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.3:
                    continue
                
                # Find the center of the ball using image moments
                moments = cv2.moments(clean_contour)
                if moments["m00"] != 0:
                    center_x = int( moments["m10"] / moments["m00"] )
                    center_y = int( moments["m01"] / moments["m00"] )
                    
                    # Classify and store them for display
                    balls_detected[color].append( (center_x, center_y) )
            
            
        # DRAWING AND DISPLAYING
        # --------------------------------------
        with frame_lock:
            display_frame = latest_frame.copy()
    
        # Draw ArUco detections
        with aruco_lock:
            cv2.aruco.drawDetectedMarkers(display_frame, latest_corners, latest_ids)

        # Draw all the classified balls
        for color, ball_list in balls_detected.items():
            ball_count = 1
            for (center_x, center_y) in ball_list:
                cv2.circle( display_frame, (center_x, center_y), 25, (0,255,0), 3 )
                cv2.putText(display_frame, color + f"_{ball_count}", (center_x, center_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                ball_count += 1
    
        cv2.imshow("Multithreaded Colored Ball and ArUco Recognition", display_frame)
        if cv2.waitKey(1) == 27: # ESC
            stop_requested = True
            cap.release()
            cv2.destroyAllWindows()
            break


# ======================================================================
# 4. Separating the functions to threads
# ======================================================================

# Thread A: capture frames
capture_thread = threading.Thread(target=get_frame) 

# Thread B: gray frames and aruco recognition
aruco_recognition_thread = threading.Thread(target=get_aruco)

# Thread C: colored frames and making color masks out of them
# + recognizing the balls from them
# + drawing the recognized arucos and balls
drawing_thread = threading.Thread(target=get_results)


capture_thread.start()
aruco_recognition_thread.start()
drawing_thread.start()

capture_thread.join()
aruco_recognition_thread.join()
drawing_thread.join()