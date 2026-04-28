

"""
    
"""
    

class WorldStatePublisher:
    def __init__(self, tracking_queue, world_state_queue, stop_event):
        self.tracking_queue = tracking_queue
        self.world_state_queue = world_state_queue
        self.stop_event = stop_event
    








































"""

# ============================================================
# 0. Initializing libraries, constants, and global variables
# ============================================================


# --- 0.1. Libraries  ---

import cv2
import numpy as np
import threading
import time
import math
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from enum import Enum


# --- 0.2. Camera and images  ---

# Set to True to use a static test image instead of live camera feed for debugging.
USE_TEST_IMAGE = False
TEST_IMAGE_PATH = r"C:\Users\leevi\Desktop\blue_floorballs_all_hard.png" 

# Camera settings - adjust as needed for your camera and environment
CAMERA_INDEX = 0
FRAME_W = 960
FRAME_H = 960
FPS = 60

# Initializing camera capture
if not USE_TEST_IMAGE:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera. If you want, set USE_TEST_IMAGE=True and DEBUG on a saved frame.")
    else:
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)


# --- 0.3. Detection parameters  ---

# Aruco
# Aruco dictionary selection here by size, etc.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# adjust parameters for better detection in your environment if needed
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Balls on detection
balls_tracked = {}
next_local_ball_id = 0
ball_count = 0


# --- 0.4. Data structures for detected physical objects  ---

# Store detected ArUco marker corners for the arena corner markers
ARENA_CORNERS = {
    'id_46': {'corners': None, 'center': None},
    'id_47': {'corners': None, 'center': None},
    'id_48': {'corners': None, 'center': None},
    'id_49': {'corners': None, 'center': None}
}

# Define robot states for defining their behavior based on ball possession and movement
class RobotState(Enum):
    IDLE = 0 # No specific task
    GOING_FOR_BALL = 1 # Moving towards a ball
    CARRYING_BALL = 2 # Has a ball and trying to score

# Define robot teams and robot parameters, store their data here etc. 
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

# Add more ball colors and their HSV ranges as needed
HSV_RANGES = {
    'blue': ((90, 130, 114), (113, 255, 255)),
    'orange': ((0, 127, 168), (10, 255, 255)),
}


# --- 0.5. Detection and tracking visualization parameters  ---

# Initial values before analysis starts, will be updated by threads
latest_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
latest_corners = None
latest_ids = None

# Locks for synchronizing access to shared data between threads
frame_lock = threading.Lock()
aruco_lock = threading.Lock()
balls_lock = threading.Lock()

# lower values smooth more but react slower to ball detection changes
SMOOTHING_ALPHA = 0.9


# --- 0.6. Error handling, debugging, misc  ---

# For stopping threads gracefully
stop_requested = False

"""