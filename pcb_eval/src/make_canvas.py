"""
테스트 이미지를 모니터에 띄울 "캔버스"로 합성한다.

  캔버스 = 배경(단색/나무결/복잡) + PCB 이미지(지정 각도로 회전) + 모서리 ArUco 마커 4개

GT 박스는 합성에 쓴 변환을 그대로 적용해 캔버스 좌표로 계산한다.
(회전된 박스는 네 꼭짓점을 감싸는 정축 박스를 쓰므로 비스듬한 각도에서는 GT 가 약간 느슨해진다.)

실험 설계 (--design)
  ofat : 기준(white, 0도)에서 요인을 하나씩만 바꾼다 -> 요인별 영향 분리
  full : 배경 x 각도 전체 조합

사용:  python make_canvas.py --test-dir data/pcb_yolo/test --out data/canvases
"""
import argparse
import csv
import glob
import json
import os
import zlib

import cv2
import numpy as np

import common as C

BACKGROUNDS = ["white", "dark", "wood", "clutter"]


def make_background(kind, seed):
    rng = np.random.default_rng(seed)
    H, W = C.CANVAS_H, C.CANVAS_W
    if kind == "white":
        return np.full((H, W, 3), 235, np.uint8)
    if kind == "dark":
        return np.full((H, W, 3), 30, np.uint8)
    if kind == "wood":
        low = cv2.resize(rng.normal(size=(6, 40)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        x = np.arange(W, dtype=np.float32)[None, :]
        pattern = 0.5 + 0.5 * np.sin(x * 0.03 + low * 6)
        pattern += 0.08 * rng.normal(size=(H, W)).astype(np.float32)
        base = np.array([55, 95, 140], np.float32)   # BGR 갈색
        return np.clip(base[None, None, :] * (0.65 + 0.35 * pattern[..., None]), 0, 255).astype(np.uint8)
    if kind == "clutter":
        bg = np.full((H, W, 3), rng.integers(90, 170), np.uint8)
        for _ in range(60):   # 녹색/갈색 포함 -> HSV 기판 분리에 대한 스트레스
            color = tuple(int(v) for v in rng.integers(0, 256, 3))
            if rng.random() < 0.3:
                color = (int(rng.integers(0, 80)), int(rng.integers(120, 220)), int(rng.integers(0, 90)))
            p = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            t = rng.integers(0, 3)
            if t == 0:
                cv2.circle(bg, p, int(rng.integers(20, 120)), color, -1)
            elif t == 1:
                q = (p[0] + int(rng.integers(40, 300)), p[1] + int(rng.integers(40, 200)))
                cv2.rectangle(bg, p, q, color, -1)
            else:
                q = (int(rng.integers(0, W)), int(rng.integers(0, H)))
                cv2.line(bg, p, q, color, int(rng.integers(3, 20)))
        return cv2.GaussianBlur(bg, (0, 0), 2)
    raise ValueError(kind)


def place_pcb(canvas, img, boxes, angle):
    """img 를 angle 도 회전 -> PCB_AREA 에 맞게 축소 -> 캔버스에 합성. 박스를 캔버스 좌표로 반환"""
    h, w = img.shape[:2]
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(R[0, 0]), abs(R[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    R[0, 2] += nw / 2 - w / 2
    R[1, 2] += nh / 2 - h / 2
    rot = cv2.warpAffine(img, R, (nw, nh))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), R, (nw, nh))

    ax1, ay1, ax2, ay2 = C.PCB_AREA
    s = min((ax2 - ax1) / nw, (ay2 - ay1) / nh)
    sw, sh = int(nw * s), int(nh * s)
    rot = cv2.resize(rot, (sw, sh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    ox, oy = ax1 + (ax2 - ax1 - sw) // 2, ay1 + (ay2 - ay1 - sh) // 2
    roi = canvas[oy: oy + sh, ox: ox + sw]
    a = (mask.astype(np.float32) / 255)[..., None]
    canvas[oy: oy + sh, ox: ox + sw] = (rot * a + roi * (1 - a)).astype(np.uint8)

    M = np.eye(3)
    M[:2] = R
    S = np.array([[s, 0, ox], [0, s, oy], [0, 0, 1.0]])
    return C.transform_boxes(S @ M, boxes)


def read_yolo(label_path, w, h):
    rows = []
    for line in open(label_path):
        c, x, y, bw, bh = line.split()
        c, x, y, bw, bh = int(c), float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
        rows.append([c, x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2])
    return np.array(rows)


def conditions(design, backgrounds, angles):
    if design == "full":
        return [(b, a) for b in backgrounds for a in angles]
    conds = [("white", 0)]
    conds += [(b, 0) for b in backgrounds if b != "white"]
    conds += [("white", a) for a in angles if a != 0]
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True, help="test 폴더 (images/, labels/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", choices=["ofat", "full"], default="ofat")
    ap.add_argument("--backgrounds", default=",".join(BACKGROUNDS))
    ap.add_argument("--angles", default="0,15,30,45")
    ap.add_argument("--n-images", type=int, default=0, help="0 이면 전부 사용")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.test_dir, "images", "*.jpg")))
    if args.n_images:
        imgs = imgs[:args.n_images]
    conds = conditions(args.design, args.backgrounds.split(","), [int(a) for a in args.angles.split(",")])
    os.makedirs(args.out, exist_ok=True)

    gt, rows = {}, []
    for n, ip in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(ip))[0]
        img = cv2.imread(ip)
        boxes = read_yolo(os.path.join(args.test_dir, "labels", stem + ".txt"), img.shape[1], img.shape[0])
        for bg, ang in conds:
            seed = zlib.crc32(f"{stem}{bg}".encode())
            canvas = make_background(bg, seed)
            cb = place_pcb(canvas, img, boxes, ang)
            C.draw_markers(canvas)
            name = f"{bg}_{ang:03d}/{n:03d}.png"   # 짧은 파일명 (Windows 경로 길이 제한 회피)
            os.makedirs(os.path.join(args.out, os.path.dirname(name)), exist_ok=True)
            cv2.imwrite(os.path.join(args.out, name), canvas)
            gt[name] = cb.tolist()
            rows.append(dict(canvas_file=name, image=stem, background=bg, angle=ang))

    json.dump(gt, open(os.path.join(args.out, "canvas_gt.json"), "w"))
    with open(os.path.join(args.out, "canvas_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"이미지 {len(imgs)}장 x 조건 {len(conds)}개 = 캔버스 {len(rows)}장 -> {args.out}")
    print("조건:", conds)


if __name__ == "__main__":
    main()
