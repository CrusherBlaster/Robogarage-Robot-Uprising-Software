

"""
ArUco Marker Detection Script using OpenCV.

- Captures frames from the default camera
- Converts to grayscale for detection
- Detects ArUco markers using predefined dictionary
- Draws markers and displays IDs
- Exits on pressing 'q'
"""


import cv2


# ======================================================================
# 1. Camera Initialization
# ======================================================================
 
# 1.1 Open the default camera (0) and store the capture object in 'cap' 
cap = cv2.VideoCapture(0)

# 1.2 Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
    

# ======================================================================
# 2. OpenCV AruCo Marker Detection Parameters
# ======================================================================

# 2.1 Load a predefined ArUco marker dictionary
# ----------------------------------------------------------------------
# The physical markers use a 6x6 dictionary with 250 markers.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 2.2 Use default detector parameters for marker detection
# ----------------------------------------------------------------------
parameters = cv2.aruco.DetectorParameters()


# =======================================================================
# 3. Continuous Frame Capture and ArUco Marker Detection
# =======================================================================

while True:
    # 3.1 Get boolean and frame from cap
    # ----------------------------------------------------------------------
    ret, frame = cap.read()
    # Frame capture error handling
    if not ret: 
        print(f"Failed to capture frame! ret = {ret}")
        break

    # 3.2 Convert the captured frame to grayscale (improves detection) 
    # ----------------------------------------------------------------------
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3.3 Detect ArUco markers in the grayscale image 
    # ----------------------------------------------------------------------
    # Inputs:
    #   gray_frame -> grayscale frame from camera
    #   aruco_dict -> predefined marker dictionary
    #   parameters -> detection parameters
    # Outputs:
    #   corners  -> list of detected marker corners
    #   ids      -> array of detected marker IDs
    #   rejected -> list of rejected candidate marker regions
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray_frame, aruco_dict, parameters=parameters
    )

    print("Detected ids: ", ids)

    # 3.4 Draw the detected corners and display id on the frame
    # ----------------------------------------------------------------------
    edited_frame = frame.copy()
    if ids is not None:
        edited_frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0,255,0))

    # 3.5 Display the resulting frame - edited_frame
    # ----------------------------------------------------------------------
    cv2.imshow("ArUco Marker Detection", edited_frame)

    # 3.6 Exit loop if 'q' key is pressed
    # ----------------------------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =======================================================================
# 4. Release Camera and Close Windows
# =======================================================================

# 4.1 Release the camera
cap.release()

# 4.2 Close all OpenCV windows
cv2.destroyAllWindows()

