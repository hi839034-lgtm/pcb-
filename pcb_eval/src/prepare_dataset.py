"""
DatasetNinja(PCB component detection) -> YOLO 형식 변환 + 영상(video) 단위 분할

원본 데이터셋의 문제점
  * 원본 test 폴더는 라벨이 비어 있어(111장 모두 objects=[]) 평가에 쓸 수 없다.
  * 같은 프레임이 Roboflow 증강(회전/기울임)으로 3장씩 복제돼 있고,
    같은 영상의 프레임이 train/test 에 섞여 있다 (영상 36개 중 18개 겹침).
  -> 이미지 단위로 나누면 누수가 생기므로 "영상 ID" 단위로 train/val/test 를 나눈다.

처리
  * Resestor(오타) -> Resistor 로 병합해 8개 클래스로 만든다.
  * train/val : 증강본을 모두 사용
  * test      : 같은 원본 프레임당 1장만 남긴다 (검은 여백이 가장 적은 복사본 = 증강이 가장 약한 것)

사용:  python prepare_dataset.py --raw <압축 푼 폴더> --out <출력 폴더>
"""
import argparse
import collections
import glob
import json
import os
import random
import shutil

import cv2
import numpy as np

CLASS_MERGE = {"Resestor": "Resistor"}
CLASSES = ["Cap1", "Cap2", "Cap3", "Cap4", "MOSFET", "Mov", "Resistor", "Transformer"]


def video_id(ann_path):
    # VID20210601144029-63_jpg.rf.<hash>.jpg.json -> VID20210601144029
    return os.path.basename(ann_path).split("_jpg")[0].rsplit("-", 1)[0]


def frame_id(ann_path):
    # 같은 원본 프레임의 증강본을 묶는 키: VID20210601144029-63
    return os.path.basename(ann_path).split("_jpg")[0]


def load_ann(path):
    d = json.load(open(path, encoding="utf-8"))
    w, h = d["size"]["width"], d["size"]["height"]
    rows = []
    for o in d["objects"]:
        name = CLASS_MERGE.get(o["classTitle"], o["classTitle"])
        (x1, y1), (x2, y2) = o["points"]["exterior"]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        rows.append((CLASSES.index(name), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return rows


def black_ratio(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return float(np.mean(im < 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-videos", type=int, default=7)
    ap.add_argument("--val-videos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 라벨이 있는 이미지만 수집 (원본 train + validation)
    items = []
    for split in ("train", "validation"):
        for a in sorted(glob.glob(os.path.join(args.raw, split, "ann", "*.json"))):
            rows = load_ann(a)
            if not rows:
                continue
            img = os.path.join(args.raw, split, "img", os.path.basename(a)[:-5])
            items.append(dict(ann=a, img=img, rows=rows, vid=video_id(a), frame=frame_id(a)))

    vids = sorted({it["vid"] for it in items})
    print(f"라벨 있는 이미지 {len(items)}장 / 영상 {len(vids)}개")

    # 테스트셋에 8개 클래스가 모두 들어가도록 시드를 바꿔가며 분할
    by_vid = collections.defaultdict(list)
    for it in items:
        by_vid[it["vid"]].append(it)
    for attempt in range(1000):
        rng = random.Random(args.seed + attempt)
        order = vids[:]
        rng.shuffle(order)
        test_v = set(order[: args.test_videos])
        val_v = set(order[args.test_videos: args.test_videos + args.val_videos])
        cls_in_test = {r[0] for v in test_v for it in by_vid[v] for r in it["rows"]}
        if len(cls_in_test) == len(CLASSES):
            break
    else:
        raise SystemExit("모든 클래스를 포함하는 test 분할을 찾지 못함")

    split_of = {v: ("test" if v in test_v else "val" if v in val_v else "train") for v in vids}

    # test 는 프레임당 1장만 (증강이 가장 약한 복사본)
    keep = []
    frames = collections.defaultdict(list)
    for it in items:
        if split_of[it["vid"]] == "test":
            frames[it["frame"]].append(it)
        else:
            keep.append(it)
    for f, group in frames.items():
        keep.append(min(group, key=lambda it: black_ratio(it["img"])))

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    counts = collections.Counter()
    inst = {s: collections.Counter() for s in ("train", "val", "test")}
    manifest = []
    for it in keep:
        s = split_of[it["vid"]]
        os.makedirs(os.path.join(args.out, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, s, "labels"), exist_ok=True)
        stem = os.path.basename(it["img"])[:-4]
        shutil.copy2(it["img"], os.path.join(args.out, s, "images", stem + ".jpg"))
        with open(os.path.join(args.out, s, "labels", stem + ".txt"), "w") as f:
            for c, x, y, w, h in it["rows"]:
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                inst[s][CLASSES[c]] += 1
        counts[s] += 1
        manifest.append(dict(split=s, image=stem + ".jpg", video=it["vid"], frame=it["frame"]))

    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out).replace(os.sep, '/')}\n")
        f.write("train: train/images\nval: val/images\ntest: test/images\n")
        f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    json.dump(manifest, open(os.path.join(args.out, "split_manifest.json"), "w"), indent=1)

    print("분할 결과 (이미지 수):", dict(counts))
    print("분할 영상 수: train", sum(v == "train" for v in split_of.values()),
          "val", len(val_v), "test", len(test_v))
    for s in ("train", "val", "test"):
        print(f"  {s} 인스턴스:", dict(inst[s]))
    assert not (set(v for v in vids if split_of[v] == "train") & test_v)
    print("영상 단위 누수 없음 확인 완료")


if __name__ == "__main__":
    main()
