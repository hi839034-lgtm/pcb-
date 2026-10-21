"""
웹캠(재촬영) 프레임에서 모델 x 전처리 구성별 mAP 를 조건별로 계산한다.

흐름
  capture_manifest.csv 의 각 프레임
    -> ArUco 마커 검출 -> 호모그래피 -> GT 박스(캔버스 좌표)를 프레임 좌표로 변환
    -> 구성(none/hsv/clahe/hsv_clahe)별 검출 -> mAP
  전체 / 요인별(배경, 각도, 조명, ...) 로 묶어 mAP50, mAP50-95, 95% 신뢰구간(bootstrap) 저장

사용 예
  python evaluate.py --captures data/captures/run1 --canvas-gt data/canvases/canvas_gt.json \
      --model base=runs/base/weights/best.pt --model rot=runs/rot/weights/best.pt \
      --out results/results.csv
  (--captures 를 여러 번 지정하면 조명 등 run 별 폴더를 합쳐서 평가)
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

import common as C
import metrics
import pipeline

DEFAULT_FACTORS = ["background", "angle", "light", "brightness", "tilt"]


def load_captures(dirs):
    rows = []
    for d in dirs:
        with open(os.path.join(d, "capture_manifest.csv"), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_path"] = os.path.join(d, r["capture_file"])
                rows.append(r)
    return rows


def make_yolo_predict(path, imgsz, conf, iou, device):
    from ultralytics import YOLO   # torch 가 필요하므로 여기서만 import
    model = YOLO(path)

    def predict(img):
        res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
        if len(res.boxes) == 0:
            return np.zeros((0, 6))
        b = res.boxes
        return np.concatenate([b.cls.cpu().numpy()[:, None], b.conf.cpu().numpy()[:, None], b.xyxy.cpu().numpy()], 1)
    return predict


def prepare_gt(rows, canvas_gt, debug_dir=None):
    """프레임마다 (frame, gt_in_frame, n_markers) 를 만든다. 마커 실패 프레임은 gt=None"""
    out = []
    for i, r in enumerate(rows):
        frame = cv2.imread(r["_path"])
        H, n = C.find_homography(frame)
        gt = None
        if H is not None:
            gt = C.transform_boxes(H, canvas_gt[r["canvas_file"]])
            h, w = frame.shape[:2]
            gt[:, [1, 3]] = gt[:, [1, 3]].clip(0, w)
            gt[:, [2, 4]] = gt[:, [2, 4]].clip(0, h)
            if debug_dir and i < 40:
                dbg = frame.copy()
                for c, x1, y1, x2, y2 in gt:
                    cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"{i:04d}.jpg"), dbg)
        out.append((frame, gt, n))
    return out


def summarize(entries, n_cls, n_boot):
    imgs = [(d, g) for d, g, _ in entries]
    m = metrics.compute_map(imgs, n_cls)
    lo, hi = metrics.bootstrap_map50(imgs, n_cls, n_boot) if len(imgs) > 1 and n_boot else (np.nan, np.nan)
    found = [f for _, _, f in entries if f is not None]
    return m, lo, hi, (float(np.mean(found)) if found else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", action="append", required=True)
    ap.add_argument("--canvas-gt", required=True)
    ap.add_argument("--model", action="append", required=True, help="이름=가중치경로 (여러 번 지정 가능)")
    ap.add_argument("--modes", default=",".join(pipeline.MODES))
    ap.add_argument("--fallback", choices=["none", "full"], default="none")
    ap.add_argument("--factors", default=",".join(DEFAULT_FACTORS))
    ap.add_argument("--conf", type=float, default=0.001, help="mAP 계산용(낮게 고정)")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--debug-dir", default=None, help="변환된 GT 를 프레임에 그려 저장 (정렬 확인용)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    canvas_gt = json.load(open(args.canvas_gt))
    rows = load_captures(args.captures)
    prepared = prepare_gt(rows, canvas_gt, args.debug_dir)
    ok = [(r, p) for r, p in zip(rows, prepared) if p[1] is not None]
    print(f"프레임 {len(rows)}장 중 마커 인식 성공 {len(ok)}장 (실패 {len(rows) - len(ok)}장은 제외)")

    # 이 데이터에 실제로 값이 여러 개 있는 요인만 사용
    factors = []
    for fac in args.factors.split(","):
        vals = {r.get(fac, "") for r, _ in ok}
        if fac in rows[0] and len(vals - {""}) > 1:
            factors.append(fac)
    print("분석 요인:", factors)

    result_rows = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        predict = make_yolo_predict(path, args.imgsz, args.conf, args.iou, args.device)
        for mode in args.modes.split(","):
            entries = []
            for (r, (frame, gt, _)) in ok:
                dets, found = pipeline.run(frame, predict, mode, args.fallback)
                entries.append((dets, gt, found))
            groups = [("overall", "all", list(range(len(ok))))]
            for fac in factors:
                for lv in sorted({r[fac] for r, _ in ok}):
                    groups.append((fac, lv, [i for i, (r, _) in enumerate(ok) if r[fac] == lv]))
            for fac, lv, idx in groups:
                sub = [entries[i] for i in idx]
                m, lo, hi, found_rate = summarize(sub, len(C.CLASSES), args.n_boot)
                result_rows.append(dict(model=name, mode=mode, factor=fac, level=lv, n_images=len(sub),
                                        mAP50=round(m["mAP50"], 4), mAP50_95=round(m["mAP50_95"], 4),
                                        mAP50_ci_lo=round(lo, 4), mAP50_ci_hi=round(hi, 4),
                                        pcb_found_rate=round(found_rate, 3) if found_rate == found_rate else ""))
            ov = result_rows[-len(groups)]
            print(f"[{name:>6} / {mode:<9}] mAP50={ov['mAP50']:.3f} mAP50-95={ov['mAP50_95']:.3f} "
                  f"CI=({ov['mAP50_ci_lo']:.3f},{ov['mAP50_ci_hi']:.3f}) pcb_found={ov['pcb_found_rate']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        w.writeheader()
        w.writerows(result_rows)
    print("저장:", args.out)


if __name__ == "__main__":
    main()
