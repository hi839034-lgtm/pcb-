"""
웹캠 없이 캔버스를 "가상 촬영"한다 (원근 기울임 + 조명 + 블러/노이즈).
  1) 파이프라인(마커 -> 호모그래피 -> GT 변환 -> 평가) 동작 검증용
  2) 실촬영 전에 요인별 열화 경향을 보는 보조 실험(합성 열화)
capture.py 와 같은 형식(capture_manifest.csv)으로 저장한다.

  python simulate_capture.py --canvases data/canvases --out data/sim/light_dim --light dim --tilt 20
"""
import argparse
import csv
import os

import cv2
import numpy as np

import common as C

LIGHTS = {  # (gamma, 밝기 배율, 눈부심)
    "normal": (1.0, 1.0, False),
    "dim": (1.6, 0.45, False),
    "glare": (1.0, 1.0, True),
}


def perspective_frame(canvas, tilt, out_w=1280, out_h=720, rng=None):
    """캔버스를 out 프레임 중앙에 두고 tilt(도) 만큼 한쪽을 멀리 기울여 촬영한 것처럼 만든다"""
    h, w = canvas.shape[:2]
    scale = 0.62
    cw, ch = w * scale, h * scale
    cx, cy = out_w / 2, out_h / 2
    shrink = 1 - np.sin(np.radians(tilt)) * 0.35   # 오른쪽 변이 멀어져 작아짐
    dst = np.float32([[cx - cw / 2, cy - ch / 2],
                      [cx + cw / 2, cy - ch / 2 * shrink],
                      [cx + cw / 2, cy + ch / 2 * shrink],
                      [cx - cw / 2, cy + ch / 2]])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    bg = np.full((out_h, out_w, 3), (60, 70, 80), np.uint8)   # 책상/베젤 색
    return cv2.warpPerspective(canvas, M, (out_w, out_h), dst=bg, borderMode=cv2.BORDER_TRANSPARENT)


def apply_light(img, light, rng):
    gamma, gain, glare = LIGHTS[light]
    x = (img.astype(np.float32) / 255) ** gamma * gain
    if glare:
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        cx, cy = rng.integers(300, 900), rng.integers(150, 500)
        x += 0.9 * np.exp(-(((xx - cx) / 150) ** 2 + ((yy - cy) / 100) ** 2))[..., None]
    x += rng.normal(0, 0.015, x.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--light", default="normal", choices=list(LIGHTS))
    ap.add_argument("--tilt", type=float, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.canvases, "canvas_manifest.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    with open(os.path.join(args.out, "capture_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["capture_file", "light", "brightness", "tilt", "n_markers"])
        w.writeheader()
        for i, r in enumerate(rows):
            canvas = cv2.imread(os.path.join(args.canvases, r["canvas_file"]))
            fr = perspective_frame(canvas, args.tilt)
            fr = cv2.GaussianBlur(fr, (0, 0), 0.8)
            fr = apply_light(fr, args.light, rng)
            name = f"{i:04d}.jpg"
            cv2.imwrite(os.path.join(args.out, name), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            _, n = C.find_homography(fr, min_markers=1)
            w.writerow(dict(r, capture_file=name, light=args.light, brightness="sim", tilt=args.tilt, n_markers=n))
    print("저장:", args.out)


if __name__ == "__main__":
    main()
