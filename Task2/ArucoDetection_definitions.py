import cv2
import numpy as np


# NOTE: Detection is now handled via ArucoDetector objects in the main script.
# These utility functions operate on already-detected marker bboxs and ids.

def getMarkerCoordinates(markers, ids, point=0):
    """
    Extract specified corner point from each detected marker.
    :param markers: list of bbox arrays
    :param ids:     list of marker IDs
    :param point:   which corner (0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left)
    :return:        list of [x, y] and corresponding ids
    """
    marker_array = []
    for marker in markers:
        x = int(marker[0][point][0])
        y = int(marker[0][point][1])
        marker_array.append([x, y])
    return marker_array, ids


def getMarkerCenter_foam(markers):
    """
    Compute centroid of a single foam marker by averaging its four corners.
    """
    pts,  _ = getMarkerCoordinates(markers, [1], point=0)
    pt0 = pts[0] if pts else [0, 0]
    pts1, _ = getMarkerCoordinates(markers, [1], point=1)
    pts2, _ = getMarkerCoordinates(markers, [1], point=2)
    pts3, _ = getMarkerCoordinates(markers, [1], point=3)

    if pts and pts1 and pts2 and pts3:
        cx = (pt0[0] + pts1[0][0] + pts2[0][0] + pts3[0][0]) * 0.25
        cy = (pt0[1] + pts1[0][1] + pts2[0][1] + pts3[0][1]) * 0.25
        return [[int(cx), int(cy)]]
    else:
        return [[0, 0]]


def draw_corners(img, corners):
    """Draw small circles at each corner point."""
    for (x, y) in corners:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)


def draw_numbers(img, corners, ids):
    """Draw the ID number next to each corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (x, y) in enumerate(corners):
        cv2.putText(img, str(ids[i]), (x + 5, y + 5), font, 0.5, (0, 0, 0), 2)


def show_spec(img, corners):
    """Overlay count of detected corners."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{len(corners)} markers found."
    cv2.putText(img, text, (10, 20), font, 0.6, (0, 0, 255), 2)


def draw_field(img, corners, ids):
    """
    When exactly 4 workspace corners are found,
    fill the quadrilateral with a transparent overlay.
    """
    if len(corners) == 4:
        pts = [None] * 4
        for idx, cid in enumerate(ids):
            pts[cid - 1] = corners[idx]
        pts_np = np.array(pts, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts_np], (255, 215, 0))
        img_new = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        return img_new, True
    return img, False


def order_points(pts):
    """
    Order four points as TL, TR, BR, BL for homography.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
    Warp the image so the quadrilateral defined by pts
    becomes a rectangle (bird's-eye view).
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([
        [0,      0],
        [maxW-1, 0],
        [maxW-1, maxH-1],
        [0,      maxH-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))
