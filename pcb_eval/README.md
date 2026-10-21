# PCB 부품 검출 - 재촬영(모니터→웹캠) 평가

```
src/prepare_dataset.py   DatasetNinja -> YOLO, 영상 단위 train/val/test 분할 (완료: data/pcb_yolo)
src/make_canvas.py       테스트 이미지 + 배경/각도 + ArUco 마커 캔버스 합성, GT 좌표 계산
src/capture.py           캔버스를 모니터에 띄우고 웹캠 촬영 (조명 run 별로 반복)
src/simulate_capture.py  웹캠 없이 가상 촬영 (검증/합성 열화용)
src/pipeline.py          HSV 기판분리 + CLAHE + 좌표 복원
src/evaluate.py          모델 x 전처리 x 조건별 mAP (+95% CI)
src/analyze.py           요인별 하락폭 표
train/train.py           base(회전증강 X) / rot(회전증강 O) 학습
tests/selftest.py        웹캠/torch 없이 파이프라인 검증
```

## 환경
`conda activate yolo` (python 3.9, torch 2.6+cu124, ultralytics, opencv 5.0, pandas 설치됨). 모든 명령은 이 환경에서 실행한다.

## 실행 순서
```bash
# 0) (이미 완료) 데이터 분할
python src/prepare_dataset.py --raw <압축푼폴더> --out data/pcb_yolo

# 1) 학습 - test 는 사용되지 않음 (torch/ultralytics 필요, GPU 권장)
cd train
python train.py --data ../data/pcb_yolo/data.yaml --name base --degrees 0
python train.py --data ../data/pcb_yolo/data.yaml --name rot  --degrees 30 --flipud 0.5
cd ..

# 2) 캔버스 생성 (7개 조건 x 50장 = 350장)
python src/make_canvas.py --test-dir data/pcb_yolo/test --out data/canvases

# 3) 촬영: 조명/화면밝기를 바꿔 여러 번
python src/capture.py --canvases data/canvases --out data/captures/normal --light normal --brightness 100
python src/capture.py --canvases data/canvases --out data/captures/dim    --light dim    --brightness 30

# 4) 평가
python src/evaluate.py --captures data/captures/normal --captures data/captures/dim \
    --canvas-gt data/canvases/canvas_gt.json \
    --model base=runs/base/weights/best.pt --model rot=runs/rot/weights/best.pt \
    --out results/results.csv --debug-dir results/gt_overlay
python src/analyze.py results/results.csv --model base --mode none --reference background=white angle=0 light=normal
```
`results/gt_overlay` 의 이미지에서 빨간 박스가 부품에 맞는지 반드시 눈으로 확인하세요.

## 해석 시 주의
- 모니터 재촬영이므로 "실제 PCB 환경"이 아니라 "테스트셋 재촬영 환경"의 결과이다.
- 각도 조건의 GT 는 회전 박스를 감싼 정축 박스라 느슨해진다 (mAP50-95 는 불리, mAP50 위주로 해석).
- 마커 인식 실패 프레임은 평가에서 제외되며 개수가 출력된다.
- 조건당 이미지가 적으므로 신뢰구간이 겹치면 차이를 단정하지 않는다.


## Colab 학습 (GPU)
`pcb_eval/colab/colab_train.ipynb` 를 Colab 에서 열면 된다
([열기](https://colab.research.google.com/github/hi839034-lgtm/pcb-/blob/main/pcb_eval/colab/colab_train.ipynb)).
필요한 것은 원본 `pcb-component-detection-DatasetNinja.tar` 하나뿐이며, 노트북이 코드를 GitHub 에서 받고
영상 단위 분할(`prepare_dataset.py`)을 직접 재현한다. PC 의 test 영상 7개와 일치하는지 자동 검증한다.
