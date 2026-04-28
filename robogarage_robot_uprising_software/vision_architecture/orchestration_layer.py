

"""
This orchestration_layer module is for coordinating:
- Conductor = lifecycle manager, each module/thread owns its loop, stop is cooperative
- starting and stopping the threads for the different layers
- defining and passing the queues for the layers 
- queues are used for data transfer between the layers and the main loop)
"""


# ============================================================
# 0. Library imports, constants, and global variables
# ============================================================

import threading
from capture_layer import FrameCapturer
from detection_layer import Detector
from tracking_layer import Tracker
from world_state_layer import WorldStatePublisher
from robogarage_robot_uprising_software.vision_architecture.shared.utilities.queues import FreshestDataQueue


# ============================================================
# 1. Orchestration class to manage threads and data flow
# ============================================================

class Conductor:
    
    def __init__(self):
        self.frame_queue = FreshestDataQueue(maxsize=1)
        self.frame_store_queue = FreshestDataQueue(maxsize=1)
        self.detection_queue = FreshestDataQueue(maxsize=1)
        self.detection_store_queue = FreshestDataQueue(maxsize=1)
        self.tracking_queue = FreshestDataQueue(maxsize=1)
        self.world_state_queue = FreshestDataQueue(maxsize=1)
        self.stop_event = threading.Event()
        # Initialize all the layers with the necessary data queues and stop
        self.capturer = FrameCapturer(self.frame_queue, self.frame_store_queue, self.stop_event)
        self.detector = Detector(self.frame_queue, self.detection_queue, self.stop_event)
        self.tracker = Tracker(self.detection_queue, self.tracking_queue, self.stop_event)
        self.world_state_publisher = WorldStatePublisher(self.tracking_queue, self.world_state_queue, self.stop_event)
        self.threads = []
        
    
    def start(self):
        self.threads = [
            threading.Thread(target=self.capturer.get_frame),
            threading.Thread(target=self.detector.detect_everything),
            threading.Thread(target=self.tracker.track_everything),
            threading.Thread(target=self.world_state_publisher.publish_everything)
        ]
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.stop_event.set() # signal all threads to stop
        for thread in self.threads:
            thread.join(timeout=2.0)
    
        