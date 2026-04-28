



"""
This capture_layer module is for:
- capture thread
- controlling how the frames are captured and sent
"""


# ============================================================
# 0. Library imports, constants, and global variables
# ============================================================

import cv2
import numpy as np
import time
import queue    


# ============================================================
# 1. Capture thread function and capture settings
# ============================================================

class FrameCapturer():
    
    def __init__(self):

        self.stop_event = False
        
        # Set to True to use a static test image instead of live camera feed for debugging.
        self.USE_TEST_IMAGE = False
        self.TEST_IMAGE_PATH = r"C:\Users\leevi\Desktop\blue_floorballs_all_hard.png"
        
        # Camera settings - adjust as needed for your camera and environment
        self.CAMERA_INDEX = 0
        self.FRAME_W = 960
        self.FRAME_H = 960
        self.FPS = 60
        
        self.cap = None
    
    # This is the function that will run this entire layer and this function will be run in thread by the conductor in the orchestration_layer.
    def get_frame(self):
        if self.USE_TEST_IMAGE:
            img = cv2.imread(self.TEST_IMAGE_PATH)
            if img is None:
                print('Warning: test image not found at', self.TEST_IMAGE_PATH)
                return None
            
            img = cv2.resize(img, (self.FRAME_W, self.FRAME_H), interpolation=cv2.INTER_AREA)
            
            while not self.stop_event.is_set():
                self.frame_queue.put(img.copy())  # share the latest frame with other layers
                time.sleep(1.0 / max(1, self.FPS))
            return
        
        # Real camera mode
        self.cap = cv2.VideoCapture(self.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Error: Could not open camera. If you want, set USE_TEST_IMAGE=True and DEBUG on a saved frame.")
            return
        
        self.cap.set(cv2.CAP_PROP_FPS, self.FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        
        try:
            while not self.stop_event == True:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                frame = cv2.resize(frame, (self.FRAME_W, self.FRAME_H), interpolation=cv2.INTER_AREA)
                cv2.imshow("websocket_test", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event = True
            self.cap.release()
            cv2.destroyAllWindows() 
                    
        finally:    
            self.cap.release()    


# =======================================================================
# 2. Continuous Frame Capture 
# =======================================================================

capturer = FrameCapturer()
capturer.get_frame()
