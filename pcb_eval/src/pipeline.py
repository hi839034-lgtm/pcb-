"""
전처리(HSV 기판 분리 / CLAHE) + 검출 파이프라인.
YOLO 에 의존하지 않고 predict_fn(img) -> (N,6)[cls,conf,x1,y1,x2,y2] 만 받는다.
"""
import cv2
import numpy as np

MODES = ["none", "hsv", "clahe", "hsv_clahe"]


def detect_pcb_by_hsv(frame, min_area_ratio=0.10, margin=0.10):
    """HSV 녹색+갈색 마스크의 가장 큰 덩어리를 기판으로 보고 (x,y,w,h) 를 반환. 없으면 None"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([20, 15, 20]), np.array([100, 255, 255]))
    mask_brown = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([30, 255, 200]))
    mask = cv2.bitwise_or(mask_green, mask_brown)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) / (frame.shape[0] * frame.shape[1]) <= min_area_ratio:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    x0 = max(0, int(x - w * margin))
    y0 = max(0, int(y - h * margin))
    x1 = min(frame.shape[1], int(x + w * (1 + margin)))
    y1 = min(frame.shape[0], int(y + h * (1 + margin)))
    return x0, y0, x1 - x0, y1 - y0


def apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    """LAB 의 L 채널에만 CLAHE 적용"""
    l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def run(frame, predict_fn, mode="none", fallback="none"):
    """
    mode     : none | hsv | clahe | hsv_clahe
    fallback : HSV 가 기판을 못 찾았을 때 'none'(검출 0건, 시연 코드와 동일) / 'full'(전체 프레임으로 검출)
    반환     : (dets (N,6) 원본 프레임 좌표, pcb_found 또는 None)
    """
    use_hsv = mode in ("hsv", "hsv_clahe")
    use_clahe = mode in ("clahe", "hsv_clahe")
    ox = oy = 0
    img = frame
    found = None
    if use_hsv:
        box = detect_pcb_by_hsv(frame)
        found = box is not None
        if found:
            x, y, w, h = box
            img = frame[y:y + h, x:x + w]
            ox, oy = x, y
        elif fallback == "none":
            return np.zeros((0, 6)), False
    if use_clahe:
        img = apply_clahe(img)
    dets = np.asarray(predict_fn(img), float).reshape(-1, 6).copy()
    dets[:, [2, 4]] += ox
    dets[:, [3, 5]] += oy
    return dets, found
