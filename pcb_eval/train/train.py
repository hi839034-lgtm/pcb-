"""
영상 단위로 분할된 train 으로 YOLO 를 학습한다. (test 는 학습/검증에 절대 쓰지 않음)

  base : 회전 증강 없음 (degrees=0)         -> 보완 전 모델
  rot  : 회전 증강 (degrees=<값>, flipud)   -> 보완 후 모델
  ※ 원본 데이터셋에는 Roboflow 증강본(회전/기울임)이 이미 들어 있으므로 base 도 완전한 무회전은 아님.

  python train.py --data ../data/pcb_yolo/data.yaml --name base --degrees 0
  python train.py --data ../data/pcb_yolo/data.yaml --name rot  --degrees 30 --flipud 0.5
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--degrees", type=float, default=0.0)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="../runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true", help="반정밀도 끄기 (MX450 등에서 loss 가 nan 이 될 때)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, name=args.name, project=args.project, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, degrees=args.degrees, flipud=args.flipud, device=args.device, seed=args.seed, workers=args.workers,
                amp=not args.no_amp)


if __name__ == "__main__":
    main()
