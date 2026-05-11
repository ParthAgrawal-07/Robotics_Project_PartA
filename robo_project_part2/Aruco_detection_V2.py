#!/usr/bin/env python
"""
Main ArUco-based vision pick-and-place control loop.
Camera: DroidCam app (Android or iPhone) over Wi-Fi.

Setup:
  1. Install DroidCam on your phone:
       Android : https://www.dev47apps.com/
       iPhone  : search "DroidCam Webcam & OBS Camera" on App Store
  2. Open DroidCam on your phone — note the IP and Port shown on screen.
  3. Connect your phone and PC to the SAME Wi-Fi network.
  4. Set DROIDCAM_IP below to match the IP shown in the app.

Controls:
  p  ->  trigger pick-and-place on the detected object
  q  ->  quit
"""

import time
import cv2
import numpy as np
from ArucoDetection_definitions import (
    getMarkerCoordinates, getMarkerCenter_foam,
    draw_corners, draw_field, four_point_transform
)
import braccio_control_python
import keyboard

# =============================================================================
# DroidCam configuration — edit ONLY these two lines
# =============================================================================
DROIDCAM_IP   = "10.120.215.199"   # <-- replace with IP shown in DroidCam app
DROIDCAM_PORT = "4747"          # default port (same for Android AND iPhone)
# =============================================================================
# The MJPEG stream URL is identical for Android and iPhone DroidCam.
# Just set the correct IP above — nothing else needs to change.
DROIDCAM_URL = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/mjpegfeed"

desired_aruco_dictionary1 = "DICT_4X4_50"   # workspace boundary markers (IDs 1-4)
desired_aruco_dictionary2 = "DICT_6X6_50"   # object markers

ARUCO_DICT = {
    "DICT_4X4_50":         cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":        cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":        cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000":       cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50":         cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":        cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":        cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000":       cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50":         cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":        cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250":        cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000":       cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50":         cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100":        cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250":        cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000":       cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def get_markers(vid_frame, detector):
    """Detect ArUco markers using OpenCV 4.7+ ArucoDetector API."""
    bboxs, ids, _ = detector.detectMarkers(vid_frame)
    if ids is not None:
        ids_sorted = [i[0] for i in ids]
    else:
        ids_sorted = []
    return bboxs, ids_sorted


# Fallback corner positions (pixels) before any marker is first detected
init_locs             = [[10, 400], [400, 400], [400, 10], [10, 10]]
current_square_points = init_locs.copy()
current_center_Corner = [[0, 0]]
marker_location_hold  = True   # keep last known corners if markers briefly disappear


def connect_droidcam():
    """Open DroidCam MJPEG stream and verify a frame can be read."""
    print(f"[INFO] Connecting to DroidCam at {DROIDCAM_URL} ...")
    cap = cv2.VideoCapture(DROIDCAM_URL)
    time.sleep(2)   # give IP stream time to negotiate

    if not cap.isOpened():
        print("[ERROR] Could not open DroidCam stream. Check:")
        print(f"          - DROIDCAM_IP is correct  (currently '{DROIDCAM_IP}')")
        print(f"          - DroidCam app is open and running on your phone")
        print(f"          - Phone and PC are on the SAME Wi-Fi network")
        print(f"          - Port {DROIDCAM_PORT} is not blocked by Windows Firewall")
        return None

    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] DroidCam opened but could not read a frame.")
        print("        Try restarting the DroidCam app on your phone.")
        cap.release()
        return None

    h, w = frame.shape[:2]
    print(f"[INFO] DroidCam connected successfully — resolution {w}x{h}")
    return cap


def main():
    print("[INFO] Initialising ArUco detectors...")

    # ArUco detectors — OpenCV 4.7+ API
    dict1     = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary1])
    params1   = cv2.aruco.DetectorParameters()
    detector1 = cv2.aruco.ArucoDetector(dict1, params1)

    dict2     = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary2])
    params2   = cv2.aruco.DetectorParameters()
    detector2 = cv2.aruco.ArucoDetector(dict2, params2)

    # Connect to DroidCam
    cap = connect_droidcam()
    if cap is None:
        return

    print(f"[INFO] Detecting '{desired_aruco_dictionary1}' workspace markers...")
    print("[INFO] Press 'p' to pick object, 'q' to quit.")

    warped = None   # keep last warped frame in scope for the 'p' trigger
    center = [[0, 0]]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Lost frame — retrying...")
            time.sleep(0.1)
            continue

        # Workspace boundary detection
        markers, ids        = get_markers(frame, detector1)
        frame_clean         = frame.copy()
        corners, corner_ids = getMarkerCoordinates(markers, ids, 0)

        # Hold last known corner positions when markers briefly leave the frame
        if marker_location_hold and corner_ids:
            for idx, cid in enumerate(corner_ids):
                if cid <= 4:
                    current_square_points[cid - 1] = corners[idx]

        corners    = current_square_points
        corner_ids = [1, 2, 3, 4]

        frame_viz, found = draw_field(frame, corners, corner_ids)
        cv2.imshow('workspace', frame_viz)

        # Object detection in warped bird's-eye view
        if found:
            warped = four_point_transform(frame_clean, np.array(corners))

            foam_markers, foam_ids = get_markers(warped, detector2)
            lf, fid = getMarkerCoordinates(foam_markers, foam_ids, 0)

            center = getMarkerCenter_foam(foam_markers)
            draw_corners(warped, center)
            cv2.imshow('object view', warped)

        # Pick trigger — press 'p'
        if keyboard.is_pressed('p') and found and warped is not None:
            h, w = warped.shape[:2]
            u, v = center[0][0], center[0][1]

            # Map pixel centroid -> robot-frame coordinates (mm)
            x = int((v / w) * 600) - 300
            y = int((u / h) * 300)

            x_corr, y_corr = braccio_control_python.camera_compensation(x, y)
            print(f"[PICK] pixel=({u},{v})  robot=({x_corr},{y_corr}) mm")
            braccio_control_python.pick_and_place_cartesian(x_corr, y_corr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return current_center_Corner


if __name__ == '__main__':
    braccio_control_python.home()
    main()