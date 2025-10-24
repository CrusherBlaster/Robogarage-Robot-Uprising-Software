

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
# 0. Imports and global variables
# ======================================================================


import cv2
import threading
import numpy as np
import multiprocessing as mp
import queue
import time

# Shared frame between threads only one thread at time can access
cap = None
latest_frame = None
frame_lock = threading.Lock()


# ======================================================================
# 1. Camera initialization
# ======================================================================


# 1.1 Creating cap object for getting camera frames
# ----------------------------------------------------------------------

def cap_init(device, codec, fps, width, height):
    global cap
    # Choosing different video codec / backend to reduce data
    # Check if camera opened successfully
    try:
        cap = cv2.VideoCapture(device, codec)
        if not cap.isOpened():
            raise RuntimeError("Camera not found.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    # Higher framerate for less latency in recognition
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Set the frame width lower to cut data & the view to the arena
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# 1.2 Capturing frames to share them with detectors
# ----------------------------------------------------------------------

def capture_frames(**queues):
    global latest_frame
    while not stop_event_mp.is_set():
        ret, frame = cap.read()
        if not ret:
            print("The frame reading from capture object failed")
            break
        with frame_lock:
            latest_frame = frame.copy()
        
        hsv_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2HSV)
        gray_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)

        for key, data in [("hsv_queue", hsv_frame), ("gray_queue", gray_frame), ("draw_queue", latest_frame)]:
            try:
                queues[key].put_nowait(data)
            except queue.Full:
                continue  # drop frame if processing is slow            

def display_frames(display_queue):
    
    while not stop_event_mp.is_set():
        try:
            display_frame = display_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if display_frame is None:
            break
        
        if not cap.isOpened():
            stop_event_mp.set()
            break

        cv2.imshow("Detections", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event_mp.set()
            break
    
        
    cap.release()
    cv2.destroyAllWindows()



# ======================================================================
# 2. Detector classes for detecting specific objects from the frame
# ======================================================================
        
def mask_worker(hsv_queue, mask_queue, object_information, stop_event_mp):
    while True:
        start = time.time()
        try:
            hsv_frame = hsv_queue.get(timeout=0.1)
        except queue.Empty:
            if stop_event_mp.is_set(): break
            continue

        if hsv_frame is None:
            break
    
        updated_info = build_mask(hsv_frame, object_information)
        mask_queue.put(updated_info)
        print(f"[mask_worker] {time.time() - start:.3f}s")
         
def contour_worker(mask_queue, contour_queue, stop_event_mp):
    while True:
        start = time.time()
        try:
            object_information = mask_queue.get(timeout=0.1)
        except queue.Empty:
            if stop_event_mp.is_set(): break
            continue

        if object_information is None:
            break
        
        updated_info = find_contours(object_information)
        updated_info = find_balls(updated_info)
        contour_queue.put(updated_info)
        print(f"[contour_worker] {time.time() - start:.3f}s")

def aruco_worker(gray_queue, aruco_queue, stop_event_mp):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    aruco_detections = {
        "robots": {
            "team_1": {}, 
            "team_2": {}
        },
        "corners": {

        }
    }

    while True:
        start = time.time()
        try:
            gray_frame = gray_queue.get(timeout=0.1)
        except queue.Empty:
            if stop_event_mp.is_set(): break
            continue
        if gray_frame is None:
            break
        corners, ids, rejected = aruco_detector.detectMarkers(gray_frame)

        updated_detections = find_arucos(aruco_detections, ids, corners)
        aruco_queue.put(updated_detections)
        print(f"[aruco_worker] {time.time() - start:.3f}s")
        
def draw_worker(contour_queue, aruco_queue, draw_queue, display_queue, stop_event_mp):
    latest_ball_info = None
    latest_aruco = None
    
    while not stop_event_mp.is_set():
        start = time.time()
        try:
            display_frame = draw_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        if not contour_queue.empty():
            latest_ball_info = contour_queue.get_nowait()
        if not aruco_queue.empty():
            latest_aruco = aruco_queue.get_nowait()

        if latest_ball_info is None or latest_aruco is None:
            continue

        frame_with_balls = draw_ball_center(latest_ball_info, display_frame)
        finished_drawing = draw_arucos(latest_aruco, frame_with_balls)
        display_queue.put(finished_drawing)
        print(f"[draw_worker] {time.time() - start:.3f}s")


      
def build_mask(hsv_frame, object_information):
    for role, colors in object_information["arena_objects"]["balls"].items():
        for color, shades in colors.items():
            for shade, info in shades.items():
                hsv_ranges = info["masks"]["hsv_ranges"]
                # Color filtering the frame
                mask = cv2.inRange(hsv_frame, hsv_ranges[0], hsv_ranges[1])
                # Converting the remaining colors to black & white just for debugging
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                # Closing smoothing and closing gaps inside the object: dilation - erosion
                morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
                # Removing noise: erosion - dilation
                morphed_mask = cv2.morphologyEx(morphed_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                # Converting to black & white for data reduction & debugging
                _, morphed_binary_mask = cv2.threshold(morphed_mask, 127, 255, cv2.THRESH_BINARY)
                info["masks"].update({
                    "mask": mask, 
                    "binary_mask": binary_mask, 
                    "morphed_mask": morphed_mask, 
                    "morphed_binary_mask": morphed_binary_mask
                })
    return object_information
                
    
def find_contours(object_information):
    for role, colors in object_information["arena_objects"]["balls"].items():
        for color, shades in colors.items():
            for shade, info in shades.items():
                morphed_binary_mask = info["masks"]["morphed_binary_mask"]
                # Find contours (including holes)
                binary_contours, hierarchy = cv2.findContours(morphed_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # Create empty black canvas with the same size as reference mask
                object_mask = np.zeros_like(morphed_binary_mask)
                
                for i, binary_contour in enumerate(binary_contours):
                    area = cv2.contourArea(binary_contour)
                    # Filter out small noise contours
                    if area > 300:
                        # Fill the contour for 100% filling of the insides of ball contours 
                        # and draw the detection to canvas
                        cv2.drawContours(object_mask, [binary_contour], -1, 255, cv2.FILLED)
        
                # Leaving only objects with closed contours by aggressively
                # adding 25x25 pixels between gaps and removing lone pixels.
                # Just to make sure the ball contours are closed and not open/glitched.
                kernel = np.ones((25,25), np.uint8)
                clean_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
                clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

                # Confirming the contour of the cleaned up ball detection     
                clean_contours, rejected = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                info["contours"].update({
                    "binary_contours": binary_contours,
                    "object_mask": object_mask,
                    "clean_mask": clean_mask,
                    "clean_contours": clean_contours
                })
    return object_information

def find_balls(object_information):
    for role, colors in object_information["arena_objects"]["balls"].items():
        for color, shades in colors.items():
            for shade, info in shades.items():
                i = 1
                for clean_contour in info["contours"]["clean_contours"]:
                    area = cv2.contourArea(clean_contour)
                    perimeter = cv2.arcLength(clean_contour, True)
                    # Avoiding some math errors
                    if perimeter == 0:
                        continue
                    # Let's turn the contour area into perfect circle and
                    # check how much the current perimeter matches with it
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # Let's ignore the uncircular misdetection contours
                    if circularity < 0.6:
                        continue

                    # Finding the center point (NOT COORDINATE) of the detected ball contour
                    moments = cv2.moments(clean_contour)
                    if moments["m00"] != 0:
                        center_x = int( moments["m10"] / moments["m00"] )
                        center_y = int( moments["m01"] / moments["m00"] )
                        info["ball_coordinates"][f"ball{i}"] = {
                            "center_x": center_x,
                            "center_y": center_y
                        }
                        i += 1
    return object_information

def find_arucos(aruco_detections, ids, corners):
    if ids is not None:
            # Sort (id, corner) pairs by id
            sorted_pairs = sorted(zip(ids.flatten(), corners), key=lambda x: x[0])
            
            for marker_id, marker_corners in sorted_pairs:
                
                # Compute center
                c = marker_corners[0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))

                if marker_id == 1 or marker_id == 2:
                    aruco_detections["robots"]["team_1"].update({
                        f"robot_{marker_id}": {
                            "id": marker_id,
                            "center": (center_x, center_y),
                            "corners": marker_corners
                        }   
                    })

                if marker_id == 3 or marker_id == 4:
                    aruco_detections["robots"]["team_2"].update({
                        f"robot_{marker_id}": {
                            "id": marker_id,
                            "center": (center_x, center_y),
                            "corners": marker_corners
                        }   
                    })

                if 46 <= marker_id <= 49:
                    aruco_detections["corners"].update({
                        f"corner_{marker_id}": {
                            "id": marker_id,
                            "center": (center_x, center_y),
                            "corners": marker_corners
                        }   
                    }) 
    return aruco_detections

def draw_ball_center(object_information, display_frame):
    for role, colors in object_information["arena_objects"]["balls"].items():
        for color, shades in colors.items():          # e.g. "blue", "orange"
            for shade, info in shades.items():        # e.g. "blue_1"
                for ball_name, ball_data in info["ball_coordinates"].items():
                    center_x = ball_data["center_x"]
                    center_y = ball_data["center_y"]
                    
                    color_map = {
                        "blue": (255, 0, 0),
                        "orange": (0, 165, 255)
                    }

                    # Color selection based on the handled ball
                    draw_color = color_map.get(color, (0, 255, 0))

                    # Draw a circle at the detected ball’s center
                    cv2.circle(display_frame, (center_x, center_y), 25, draw_color, 3)

                    # Label text includes the color and ball name, e.g. "blue_1 - ball1"
                    label = f"{shade}_{ball_name}"
                    cv2.putText(
                        display_frame, 
                        label, 
                        (center_x - 20, center_y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
    return display_frame

def draw_arucos(aruco_detections, display_w_balls):
    
    latest_corners = []
    latest_ids = []
    
    if "robots" in aruco_detections:
        for team_key, team_data in aruco_detections["robots"].items():
            for robot_key, robot_info in team_data.items():
                if "corners" in robot_info and "id" in robot_info:
                    latest_corners.append(robot_info["corners"])
                    latest_ids.append(robot_info["id"])

    if "corners" in aruco_detections:
        for corner_key, corner_info in aruco_detections["corners"].items():
            if "corners" in corner_info and "id" in corner_info:
                latest_corners.append(corner_info["corners"])
                latest_ids.append(corner_info["id"])

    # If nothing to draw, just return the original frame
    if not latest_corners or not latest_ids:
        return display_w_balls

    # Convert to NumPy arrays as required by OpenCV
    latest_ids = np.array(latest_ids, dtype=np.int32)

    # Draw the markers
    cv2.aruco.drawDetectedMarkers(display_w_balls, latest_corners, latest_ids)

    # Optionally, you can also draw marker centers
    for i, corners in enumerate(latest_corners):
        c = corners[0]
        center_x = int(np.mean(c[:, 0]))
        center_y = int(np.mean(c[:, 1]))
        cv2.circle(display_w_balls, (center_x, center_y), 6, (0, 255, 255), -1)
        cv2.putText(display_w_balls, f"ID {latest_ids[i]}", (center_x + 10, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    final_display = display_w_balls

    return final_display


           

# ======================================================================
# 3. Main code 
# ======================================================================


# Only processed on the main CPU core running the main script 
if __name__ == "__main__":
    
    
    hsv_queue = mp.Queue(maxsize=1)
    gray_queue = mp.Queue(maxsize=1)
    mask_queue = mp.Queue(maxsize=1)
    contour_queue = mp.Queue(maxsize=1)
    aruco_queue = mp.Queue(maxsize=1)
    draw_queue = mp.Queue(maxsize=1)
    display_queue = mp.Queue(maxsize=1)
    queues = {
        "hsv_queue": hsv_queue, 
        "gray_queue": gray_queue,
        "mask_queue": mask_queue,
        "contour_queue": contour_queue,
        "aruco_queue": aruco_queue,  
        "draw_queue": draw_queue,
        "display_queue": display_queue
        }   

    object_information = {
        "arena_objects": {
            "balls": {
                "good_balls": {
                    "blue": {
                        "blue_1": {
                            "masks": {
                                "hsv_ranges": [(90, 130, 114), (113,255,255)]
                            },
                            "contours": {},
                            "ball_coordinates": {}    
                        }
                    },
                    "orange": {
                        "orange_1": {
                            "masks":  {  
                                "hsv_ranges": [(0,127,168), (10,255,255)]
                            },
                            "contours": {},
                            "ball_coordinates": {}  
                        }
                    }
                }
            },
        }
    }

    stop_event_mp = mp.Event()

    # Defining processes for multiprocessing in different cores
    p_mask = mp.Process(target=mask_worker, args=(hsv_queue, mask_queue, object_information, stop_event_mp))
    p_contour = mp.Process(target=contour_worker, args=(mask_queue, contour_queue, stop_event_mp))
    p_aruco = mp.Process(target=aruco_worker, args=(gray_queue, aruco_queue, stop_event_mp))
    p_draw = mp.Process(target=draw_worker, args=(contour_queue, aruco_queue, draw_queue, display_queue, stop_event_mp))
    processes = [p_mask, p_contour, p_aruco, p_draw]

    # Initializing capture object
    cap_init(0, cv2.CAP_DSHOW, 60, 1536, 1536)

    # Defining I/O processes to run on 1 core but two separate threads

    capture_thread = threading.Thread(target=capture_frames, kwargs=queues)
    display_thread = threading.Thread(target=display_frames, args=(display_queue, ))
    capture_thread.start()
    display_thread.start()
    # Excecuting processes simultaneosly
    for p in processes:
        p.start()

    try:
        # Main loop simply waits until 'q' is pressed in the display thread
        while not stop_event_mp.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event_mp.set()
        # Join threads
        capture_thread.join()
        display_thread.join()
        print("KeyboardInterrupt detected, stopping threads and processes...")

    # Signal all queues to stop
    for q in queues.values():
        try:
            q.put_nowait(None)
        except queue.Full:
            pass

    # Join processes
    for p in processes:
        p.join()

    print("All threads and processes have been stopped cleanly.")
    
    

