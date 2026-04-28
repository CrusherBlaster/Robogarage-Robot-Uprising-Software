

"""
this layer module reads from the world_state
- produces the meaningful visualizations of the world_state

"""


# ============================================================
# 0. Library imports, constants, and global variables
# ============================================================

import cv2
import numpy as np


# ============================================================
# 1. Visualization class to handle all visualization tasks
# ============================================================

class Visualizer:
    def __init__(self, world_state):
        self.state = world_state
    
    def visualize(self):


        """
        Add here the visualization parts from v4_multithreaded_computer_vision.py
        lines 432-503 are about drawing. 
        
        """