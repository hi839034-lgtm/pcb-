"""캔버스 규격, ArUco 마커 배치, 호모그래피/박스 변환 공통 함수"""
import cv2
import numpy as np

CLASSES = ["Cap1", "Cap2", "Cap3", "Cap4", "MOSFET", "Mov", "Resistor", "Transformer"]

CANVAS_W, CANVAS_H = 1920, 1080
MARKER_PX = 150      # 마커 한 변
QUIET = 20           # 마커 둘레 흰 여백 (어두운 배경에서도 검출되도록)
EDGE = 40            # 캔버스 가장자리와 마커 블록 사이 간격
PCB_AREA = (250, 60, 1670, 1020)   # PCB 이미지를 놓는 영역 (x1, y1, x2, y2), 마커와 겹치지 않음

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def marker_corners():
    """마커 id -> 캔버스 좌표계에서의 4개 꼭짓점 (TL, TR, BR, BL 순서, ArUco 검출 순서와 동일)"""
    s = MARKER_PX
    tl = {
        0: (EDGE + QUIET, EDGE + QUIET),
        1: (CANVAS_W - EDGE - QUIET - s, EDGE + QUIET),
        2: (CANVAS_W - EDGE - QUIET - s, CANVAS_H - EDGE - QUIET - s),
        3: (EDGE + QUIET, CANVAS_H - EDGE - QUIET - s),
    }
    return {i: np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], np.float32) for i, (x, y) in tl.items()}


def draw_markers(canvas):
    for i, pts in marker_corners().items():
        x, y = int(pts[0, 0]), int(pts[0, 1])
        canvas[y - QUIET: y + MARKER_PX + QUIET, x - QUIET: x + MARKER_PX + QUIET] = 255
        m = cv2.aruco.generateImageMarker(ARUCO_DICT, i, MARKER_PX)
        canvas[y: y + MARKER_PX, x: x + MARKER_PX] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return canvas


_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())


def find_homography(frame, min_markers=3):
    """웹캠 프레임에서 마커를 찾아 캔버스 -> 프레임 호모그래피를 구한다.
    반환: (H 또는 None, 검출된 마커 수)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return None, 0
    layout = marker_corners()
    src, dst = [], []
    for c, i in zip(corners, ids.flatten()):
        if int(i) in layout:
            src.append(layout[int(i)])
            dst.append(c.reshape(4, 2))
    n = len(src)
    if n < min_markers:
        return None, n
    H, _ = cv2.findHomography(np.concatenate(src), np.concatenate(dst), cv2.RANSAC, 5.0)
    return H, n


def transform_boxes(M, boxes):
    """3x3 변환행렬 M 으로 박스 (K,5)[cls,x1,y1,x2,y2] 를 옮기고, 변환된 네 꼭짓점을 감싸는 정축 박스를 반환"""
    boxes = np.asarray(boxes, float).reshape(-1, 5)
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2 = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    pts = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                    np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)    # (K,4,2)
    flat = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), np.asarray(M, np.float64)).reshape(-1, 4, 2)
    out = np.stack([boxes[:, 0], flat[..., 0].min(1), flat[..., 1].min(1), flat[..., 0].max(1), flat[..., 1].max(1)], 1)
    return out
