

""" 
This module is the main entry point for the entire software.
The module basically runs and terminates the software and it
should not have any other roles.

    - initializes the conductor object from the class 
      brought from orchestration_layer.py
    - conductor.start() 
      tells the conductor to start "conducting its orchestra"
      which means to start all its threads. This effectively
      runs the capturer, detector, etc. simultaneously and
      the software has started. 
    - next main.py waits for KeyboardInterrupt such as ctrl+c
      as a signal for telling the conductor to stop 
      conducting the damn beautiful orchestra.
    - the conductor.stop() method then gracefully stops the
      threads and the software exits/terminates/quits.
"""


# =============================================================
# 0. Library imports, constants, and global variables
# =============================================================

from orchestration_layer import Conductor
import time


# =============================================================
# 1. Main entry point to start the vision system
# =============================================================

if __name__ == "__main__":
    conductor = Conductor()
    try:
        conductor.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        conductor.stop()
        
        