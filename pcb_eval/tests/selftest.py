"""웹캠/torch 없이 파이프라인을 검증: 캔버스 -> 가상촬영 -> 마커/호모그래피 -> GT 변환 -> mAP
   (GT 로 만든 가짜 검출기를 사용하므로 mAP 가 ~1 이어야 하고, 박스를 밀면 떨어져야 한다)"""
import json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import common as C, metrics, pipeline, evaluate

tmp = tempfile.mkdtemp(prefix="pcbtest_")
py = sys.executable
def run(*a): subprocess.run([py, *a], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
canv = os.path.join(tmp, "canv")
run("src/make_canvas.py", "--test-dir", "data/pcb_yolo/test", "--out", canv, "--n-images", "4")
run("src/simulate_capture.py", "--canvases", canv, "--out", os.path.join(tmp, "a"), "--light", "normal", "--tilt", "0")
run("src/simulate_capture.py", "--canvases", canv, "--out", os.path.join(tmp, "b"), "--light", "dim", "--tilt", "25")

gt_canvas = json.load(open(os.path.join(canv, "canvas_gt.json")))
rows = evaluate.load_captures([os.path.join(tmp, "a"), os.path.join(tmp, "b")])
prep = evaluate.prepare_gt(rows, gt_canvas, os.path.join(ROOT, "..", "tmp", "dbg_overlay"))
ok = [p for p in prep if p[1] is not None]
print("마커 인식:", len(ok), "/", len(rows))
assert len(ok) == len(rows), "마커 인식 실패 프레임 존재"

def oracle(shift):
    imgs = []
    for _, gt, _ in ok:
        d = np.concatenate([gt[:, :1], np.full((len(gt), 1), 0.9), gt[:, 1:]], 1)
        d[:, [2, 4]] += shift
        imgs.append((d, gt))
    return imgs
for s in (0, 10, 40):
    m = metrics.compute_map(oracle(s), len(C.CLASSES))
    print(f"박스 이동 {s:>2}px -> mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50_95']:.3f}")
assert metrics.compute_map(oracle(0), 8)["mAP50"] > 0.99
assert metrics.compute_map(oracle(40), 8)["mAP50"] < metrics.compute_map(oracle(10), 8)["mAP50"] + 1e-9

# 전처리: HSV 기판 분리 성공률
for mode in ("hsv", "hsv_clahe"):
    f = [pipeline.run(fr, lambda im: np.zeros((0, 6)), mode)[1] for fr, _, _ in ok]
    print(mode, "기판 검출률", np.mean(f))
# 크롭 좌표 복원 검증: 전체 프레임 GT 를 크롭 후 되돌려 예측했다고 가정
fr, gt, _ = ok[0]
box = pipeline.detect_pcb_by_hsv(fr)
print("HSV bbox:", box)
def fake(img):  # 크롭 좌표계에서 GT 를 그대로 돌려주는 가짜 검출기
    x, y, w, h = box
    d = np.concatenate([gt[:, :1], np.ones((len(gt), 1)), gt[:, 1:] - [x, y, x, y]], 1)
    return d
dets, _ = pipeline.run(fr, fake, "hsv")
assert np.allclose(dets[:, 2:], gt[:, 1:]), "크롭 좌표 복원 오류"
print("OK")
