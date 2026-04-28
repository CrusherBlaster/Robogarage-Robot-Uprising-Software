"""
This module is for:

    - creating infra such as queues for passing shared
      data synchronously between threads/modules
    - queues can be used for passing data 
    - Usually in a queue first data coming into the queue
      comes out first out of the queue.
      But here the freshest data / last data comes out first 
      so remember: last in first out. 
      Why? We want to work with the freshest data in robotics
      for real time decisions such as determining the position
      of a ball object in a image/frame from a video stream. 
    - queues can be different sizes. If we 
      have fresher data that tries to come in to the queue
      but there is old data in the way, 
      we just tackle and remove the old data 
      out of the way so the freshest can get out first.
      This means queue size will be 1 and the blocking
      data is removed with queue.get_nowait and .put_nowait. 
    - If we later need that historical data we can just
      make the program save this freshest data somewhere.
      but for now - freshest is the bestest.   
      
"""


# ============================================================
# 0. Library imports, constants, and global variables
# ============================================================

import queue


# ============================================================
# 1. Data queues for inputting and outputting data between layers
# ============================================================

class FreshestDataQueue:
    def __init__(self, maxsize=1):
        self.queue = queue.Queue(maxsize=maxsize)
    
    def put(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(item)
            except queue.Empty:
                pass
            
    def get(self, timeout=None):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None  # Return None if queue is empty or timeout occurs

    