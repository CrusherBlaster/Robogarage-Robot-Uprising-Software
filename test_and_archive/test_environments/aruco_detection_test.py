import cv2
import numpy as np
import math
import time

# -------------------------
# Helpers
# -------------------------
def nothing(x):
    pass

def ensure_odd(x):
    x = max(3, int(x))
    return x if x % 2 == 1 else x + 1

# -------------------------
# Setup ArUco dictionary & params
# -------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Build list of available corner refinement constants on this OpenCV build
corner_refine_names = []
corner_refine_vals = []
for name in dir(cv2.aruco):
    if name.startswith("CORNER_REFINE"):
        corner_refine_names.append(name)
        corner_refine_vals.append(getattr(cv2.aruco, name))
# Fallback if none found
if not corner_refine_names:
    corner_refine_names = ["CORNER_REFINE_NONE"]
    corner_refine_vals = [0]

# -------------------------
# GUI trackbars (Full Pro panel)
# -------------------------
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 520, 640)

# Adaptive thresholding window sizes & constant
cv2.createTrackbar("adaptiveThreshWinSizeMin", "Trackbars", 3, 50, nothing)
cv2.createTrackbar("adaptiveThreshWinSizeMax", "Trackbars", 23, 200, nothing)
cv2.createTrackbar("adaptiveThreshWinSizeStep", "Trackbars", 10, 50, nothing)
cv2.createTrackbar("adaptiveThreshConstant", "Trackbars", 7, 50, nothing)

# Marker geometry (rates are scaled by trackbar conversions)
cv2.createTrackbar("minMarkerPerimeterRate x100", "Trackbars", 3, 100, nothing)   # divide by 100
cv2.createTrackbar("maxMarkerPerimeterRate x10", "Trackbars", 40, 200, nothing)   # divide by 10
cv2.createTrackbar("polygonalApproxAccuracy x100", "Trackbars", 3, 100, nothing)  # divide by 100
cv2.createTrackbar("minCornerDistanceRate x100", "Trackbars", 5, 100, nothing)    # divide by 100
cv2.createTrackbar("minDistanceToBorder", "Trackbars", 3, 50, nothing)            # integer

# Corner refinement controls
cv2.createTrackbar("cornerRefineMethod", "Trackbars", 0, max(0, len(corner_refine_vals)-1), nothing)
cv2.createTrackbar("cornerRefinementWinSize", "Trackbars", 5, 30, nothing)
cv2.createTrackbar("cornerRefinementMaxIterations", "Trackbars", 30, 200, nothing)
cv2.createTrackbar("cornerRefinementMinAccuracy x1000", "Trackbars", 0, 1000, nothing)  # divide by 1000

# Misc / robustness
cv2.createTrackbar("errorCorrectionRate x100", "Trackbars", 60, 100, nothing)  # divide by 100
cv2.createTrackbar("detectInvertedMarker", "Trackbars", 0, 1, nothing)
cv2.createTrackbar("minMarkerDistanceRate x100", "Trackbars", 0, 50, nothing)  # divide by 100

# -------------------------
# Video capture
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera (index 0).")

# Optional resizing for performance
FRAME_W, FRAME_H = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

last_print = 0.0
while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed, retrying...")
        continue

    # convert to gray (more stable than color)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # Read GUI values and apply conversions
    # -------------------------
    aw_min = cv2.getTrackbarPos("adaptiveThreshWinSizeMin", "Trackbars")
    aw_max = cv2.getTrackbarPos("adaptiveThreshWinSizeMax", "Trackbars")
    aw_step = cv2.getTrackbarPos("adaptiveThreshWinSizeStep", "Trackbars")
    aConst = cv2.getTrackbarPos("adaptiveThreshConstant", "Trackbars")

    minPerim = cv2.getTrackbarPos("minMarkerPerimeterRate x100", "Trackbars") / 100.0
    maxPerim = cv2.getTrackbarPos("maxMarkerPerimeterRate x10", "Trackbars") / 10.0
    polyAcc = cv2.getTrackbarPos("polygonalApproxAccuracy x100", "Trackbars") / 100.0
    minCornerDist = cv2.getTrackbarPos("minCornerDistanceRate x100", "Trackbars") / 100.0
    minDistToBorder = cv2.getTrackbarPos("minDistanceToBorder", "Trackbars")

    cr_method_idx = cv2.getTrackbarPos("cornerRefineMethod", "Trackbars")
    cr_method = corner_refine_vals[cr_method_idx]
    cr_win = cv2.getTrackbarPos("cornerRefinementWinSize", "Trackbars")
    cr_iters = cv2.getTrackbarPos("cornerRefinementMaxIterations", "Trackbars")
    cr_minacc = cv2.getTrackbarPos("cornerRefinementMinAccuracy x1000", "Trackbars") / 1000.0

    errCorr = cv2.getTrackbarPos("errorCorrectionRate x100", "Trackbars") / 100.0
    detectInv = bool(cv2.getTrackbarPos("detectInvertedMarker", "Trackbars"))
    minMarkerDistRate = cv2.getTrackbarPos("minMarkerDistanceRate x100", "Trackbars") / 100.0

    # ensure adaptive window settings make sense (and are odd where needed for some visualizations)
    aw_min = max(3, aw_min)
    aw_max = max(aw_min, aw_max)
    aw_step = max(1, aw_step)

    # -------------------------
    # Map GUI -> DetectorParameters
    # -------------------------
    try:
        aruco_params.adaptiveThreshWinSizeMin = aw_min
        aruco_params.adaptiveThreshWinSizeMax = aw_max
        aruco_params.adaptiveThreshWinSizeStep = aw_step
    except Exception:
        pass

    try:
        aruco_params.adaptiveThreshConstant = aConst
    except Exception:
        pass

    try:
        aruco_params.minMarkerPerimeterRate = minPerim
        aruco_params.maxMarkerPerimeterRate = maxPerim
        aruco_params.polygonalApproxAccuracyRate = polyAcc
        aruco_params.minCornerDistanceRate = minCornerDist
        aruco_params.minDistanceToBorder = minDistToBorder
    except Exception:
        pass

    try:
        aruco_params.cornerRefinementMethod = cr_method
        aruco_params.cornerRefinementWinSize = cr_win
        aruco_params.cornerRefinementMaxIterations = cr_iters
        # some builds support this
        if hasattr(aruco_params, "cornerRefinementMinAccuracy"):
            aruco_params.cornerRefinementMinAccuracy = cr_minacc
    except Exception:
        pass

    try:
        aruco_params.errorCorrectionRate = errCorr
    except Exception:
        pass

    try:
        aruco_params.detectInvertedMarker = detectInv
    except Exception:
        pass

    try:
        if hasattr(aruco_params, "minMarkerDistanceRate"):
            aruco_params.minMarkerDistanceRate = minMarkerDistRate
    except Exception:
        pass

    # -------------------------
    # Recreate detector with updated params
    # -------------------------
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # -------------------------
    # Optionally compute a visualization of adaptive threshold (approx)
    # Note: OpenCV's internal adaptive thresholding in detector may differ slightly;
    #       this is only to help you see what adaptiveThreshConstant & window sizes do.
    # -------------------------
    # choose a blockSize for visualization (odd)
    vis_block = ensure_odd(min(max(3, aw_min), 101))
    try:
        vis_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, vis_block, aConst)
    except Exception:
        # fallback to simple threshold visualization
        _, vis_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # scale to 3-channels for overlay views
    vis_thresh_color = cv2.cvtColor(vis_thresh, cv2.COLOR_GRAY2BGR)

    # -------------------------
    # Detect markers (use grayscale for stability)
    # -------------------------
    corners, ids, rejected = detector.detectMarkers(gray)

    # draw detected markers and ids
    out = frame.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)

        # draw ids with blue text above each marker
        for i, c in enumerate(corners):
            if ids is None: break
            id_val = int(ids[i])
            # compute center
            pts = c.reshape((-1, 2))
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(out, f"id={id_val}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # visualize rejected candidates
    rej_vis = frame.copy()
    if rejected is not None and len(rejected) > 0:
        try:
            cv2.aruco.drawDetectedMarkers(rej_vis, rejected, borderColor=(0,0,255))
        except Exception:
            # fallback: draw each rejected polygon in red
            for rc in rejected:
                pts = np.array(rc).reshape((-1,2)).astype(int)
                cv2.polylines(rej_vis, [pts], True, (0,0,255), 2)

    # -------------------------
    # Compose debugging display
    # -------------------------
    # Top-left: final output, Top-right: adaptive threshold visualization
    # Bottom-left: rejected candidates, Bottom-right: small info panel
    h, w = frame.shape[:2]
    vis_w = 640
    vis_h = int(h * (vis_w / w))
    out_small = cv2.resize(out, (vis_w, vis_h))
    thresh_small = cv2.resize(vis_thresh_color, (vis_w, vis_h))
    rej_small = cv2.resize(rej_vis, (vis_w, vis_h))

    # Info panel
    panel = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 30
    info_lines = [
        f"adaptiveThreshWinSizeMin: {aw_min}",
        f"adaptiveThreshWinSizeMax: {aw_max}",
        f"adaptiveThreshWinSizeStep: {aw_step}",
        f"adaptiveThreshConstant: {aConst}",
        f"minMarkerPerimeterRate: {minPerim:.3f}",
        f"maxMarkerPerimeterRate: {maxPerim:.3f}",
        f"polygonalApproxAccuracyRate: {polyAcc:.3f}",
        f"minCornerDistanceRate: {minCornerDist:.3f}",
        f"minDistanceToBorder: {minDistToBorder}",
        f"cornerRefineMethod: {corner_refine_names[cr_method_idx]}",
        f"cornerRefWin: {cr_win}",
        f"cornerRefIters: {cr_iters}",
        f"cornerRefMinAcc: {cr_minacc:.3f}",
        f"errorCorrectionRate: {errCorr:.2f}",
        f"detectInverted: {detectInv}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(panel, line, (8, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    top_row = np.hstack([out_small, thresh_small])
    bottom_row = np.hstack([rej_small, panel])
    debug_canvas = np.vstack([top_row, bottom_row])

    # show windows
    cv2.imshow("ArUco - Debug (press ESC to quit)", debug_canvas)

    # Also show rejected and threshold as separate large windows optionally
    # cv2.imshow("Adaptive Threshold (visual)", vis_thresh)
    # cv2.imshow("Rejected candidates", rej_vis)

    # quit key
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # occasional FPS print
    t1 = time.time()
    if t1 - last_print > 2.0:
        fps = 1.0 / max(1e-6, t1 - t0)
        print(f"FPS ~ {fps:.1f}  |  Chosen cornerRefine methods: {corner_refine_names}")
        last_print = t1

cap.release()
cv2.destroyAllWindows()
