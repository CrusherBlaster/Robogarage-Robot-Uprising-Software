

"""
This tracking_layer module is for:
- produces the world_state
- tracking thread
- controlling how the detected objects are tracked and sent to the next module 3. World state    

This next text is for the future update so ignore it for now:
THIS IS THE WRITER AND INITIATOR FOR THE WORLD STATE
THIS CONTROLS AND OWNS THE WORLDSTATE
THIS MODULE APPLIES MEANING TO THE GEOMETRIC REPRESENTATIONS the detection_layer GIVES
IN THE FUTURE THIS MUST READ THE STATIC ROBOTSTATE DEFINITIONS AND CREATION OF BALL GROUPS AND TEAMS FROM THE CONFIG FILES
THIS IS DUE TO THE FACT THAT TRACKING STATE ONLY HANDLES DYNAMIC DATA
"""

# TODO: CONTINUE FROM HERE NEXT!!!!!!!!!!!!

import numpy as np
    
class Tracker:
    def __init__(self, detection_queue, tracking_queue, stop_event):
        # getting the outputs of detection_layer
        self.detection_queue = detection_queue
        self.tracking_queue = tracking_queue
        self.stop_event = stop_event
         
        # arena corners structured by id, corners, center, etc.
        self.arena_corners = {}
        # teams structured by team name, score, etc.
        # Robots structured by team, id, position, state, ball possession, scores etc.
        self.teams = {}
        # Balls structured by color, position, id, radius, etc.
        self.balls = {}


    # --- 0.6. Error handling, debugging, misc  ---
    
    def create_arena_corner(self, marker_id, corners, center):
        self.arena_corners[marker_id] = {
            'corners': corners,
            'center': center
        }
    
    def create_team(self, team_name):
        self.teams[team_name] = {
            'robots': {}
        }
    
    def create_robot(self, team_name, robot_id, corners, bottom_left, bottom_right, bottom_center, center):
        self.teams[team_name]['robots'][robot_id] = {
            'corners': corners,
            'bottom_left': bottom_left,
            'bottom_right': bottom_right,
            'bottom_center': bottom_center,
            'center': center,
            'state': RobotState.IDLE,
            'has_ball': False
        }
        
    def ball_category(self, category):
        self.balls[category] = {}
    
    def create_ball(self, category, ball_id, color, center, radius):
        self.balls[category][ball_id] = {
            'color': color,
            'center': center,
            'radius': radius
        }
    
    # Deliver and sort corner info to storage for all aruco objects
    def update_corner(self, marker_id, marker_corners):
        string_id = f"id_{marker_id}"
        marker_corners = marker_corners.reshape((4, 2))
        # Compute center of marker
        center_x = int(np.mean(marker_corners[:, 0]))
        center_y = int(np.mean(marker_corners[:, 1]))
        center = (center_x, center_y)
        
        # Compute plower direction vector
        bottom_center_x = int((marker_corners[2, 0] + marker_corners[3, 0]) / 2)
        bottom_center_y = int((marker_corners[2, 1] + marker_corners[3, 1]) / 2)
        bottom_center = (bottom_center_x, bottom_center_y)
        
        # Update robot teams
        for robots in self.teams.values():
            for robot in robots.values():
                if marker_id == robot['id']:
                    robot['corners'] = marker_corners 
                    robot['bottom_left'] = tuple(marker_corners[3].astype(int))
                    robot['bottom_right'] = tuple(marker_corners[2].astype(int))
                    robot['center'] = center
                    robot['bottom_center'] = bottom_center

        # Update arena corners
        if string_id in self.arena_corners:
            self.arena_corners[marker_id]['corners'] = marker_corners
            self.arena_corners[marker_id]['center'] = center
    
    def update_robot_state(self, team_name, robot_id, new_state):
        if team_name in self.teams and robot_id in self.teams[team_name]['robots']:
            self.teams[team_name]['robots'][robot_id]['state'] = new_state
    
    def update_ball_position(self, category, ball_id, new_center):
        if category in self.balls and ball_id in self.balls[category]:
            self.balls[category][ball_id]['center'] = new_center
    
    # This is the function that will run this entire layer and this function will be run in thread by the conductor in the orchestration_layer.
    def track_everything(self):
        # Create the teams, robots, ball - initialize all tracking data structures
        
        # update positions, states, etc constantly.
        while not self.stop_event.is_set():
            aruco_data = self.aruco_queue.get(timeout=0.1)
            ball_data = self.ball_queue.get(timeout=0.1)
            if aruco_data is None or ball_data is None:
                continue
        
            # tracking logic here
            self.world_state_queue.put()
            
    
    
    """        
    def update_everything(self):
        while not self.stop_requested:
            if ids is not None and len(corners) > 0:
                with self.aruco_lock:
                    # Enable iteration for pairing ids with their correct corner coordinates
                    for i, marker_id in enumerate(ids.flatten()):
                        self.update_corner(marker_id, corners[i])
            # This function can be used to update the entire tracking state based on the latest detections and world state information
            pass
    """