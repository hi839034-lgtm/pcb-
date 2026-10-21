"""
캔버스를 모니터에 전체화면으로 띄우고 웹캠으로 촬영한다.

주의: 노트북 내장 웹캠은 자기 화면을 볼 수 없다. 외장 웹캠으로 모니터를 보거나,
      보조 모니터에 캔버스를 띄우고(--win-x 로 창 위치 지정) 웹캠을 그쪽으로 향하게 한다.

한 번 실행 = 한 가지 조명 조건(run). 조명/화면밝기를 바꿔 여러 번 실행한다.
  python capture.py --canvases data/canvases --out data/captures/light_normal --light normal --brightness 100
  python capture.py --canvases data/canvases --out data/captures/light_dim --light dim --brightness 30

시작 전 정렬 화면: 웹캠 영상 위에 인식된 마커 수가 표시된다 (4개가 목표). SPACE 로 촬영 시작, ESC 로 중단.
"""
import argparse
import csv
import os
import time

import cv2

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", type=int, default=0, help="웹캠 인덱스")
    ap.add_argument("--light", default="normal", help="방 조명 라벨 (예: normal, dim, glare)")
    ap.add_argument("--brightness", default="100", help="화면 밝기 라벨 (예: 100, 50, 30)")
    ap.add_argument("--tilt", default="0", help="카메라/모니터를 기울였다면 라벨 (예: 0, 20)")
    ap.add_argument("--win-x", type=int, default=0, help="캔버스 창을 놓을 모니터의 x 오프셋 (보조 모니터용)")
    ap.add_argument("--win-y", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0.8, help="화면 전환 후 대기(초)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.canvases, "canvas_manifest.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit("웹캠을 열 수 없음")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "canvas"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, args.win_x, args.win_y)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, cv2.imread(os.path.join(args.canvases, rows[0]["canvas_file"])))

    # 정렬 단계
    print("정렬 중: 'view' 창에서 마커 4개가 인식되도록 웹캠/모니터를 맞추고 SPACE 를 누르세요 (ESC 종료)")
    while True:
        ok, frame = cap.read()
        if ok:
            _, n = C.find_homography(frame, min_markers=1)
            cv2.putText(frame, f"markers: {n}/4", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0) if n == 4 else (0, 0, 255), 3)
            cv2.imshow("view", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            return
        if k == 32:
            break

    manifest_path = os.path.join(args.out, "capture_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["capture_file", "light", "brightness", "tilt", "n_markers"])
        w.writeheader()
        for i, r in enumerate(rows):
            cv2.imshow(win, cv2.imread(os.path.join(args.canvases, r["canvas_file"])))
            t0 = time.time()
            while time.time() - t0 < args.settle:     # 화면 갱신/오토노출 안정화 + 버퍼 비우기
                cap.read()
                cv2.waitKey(1)
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"{i:04d}.jpg"
            cv2.imwrite(os.path.join(args.out, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _, n = C.find_homography(frame, min_markers=1)
            w.writerow(dict(r, capture_file=name, light=args.light, brightness=args.brightness, tilt=args.tilt, n_markers=n))
            f.flush()
            cv2.imshow("view", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if i % 20 == 0:
                print(f"{i + 1}/{len(rows)}")
    cap.release()
    cv2.destroyAllWindows()
    print("저장:", args.out)


if __name__ == "__main__":
    main()
