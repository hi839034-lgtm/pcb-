"""
COCO 방식 mAP (101점 보간) 구현. numpy 만 사용한다.

images: 이미지별 (dets, gts) 리스트
  dets: (N,6) [cls, conf, x1, y1, x2, y2]
  gts : (M,5) [cls, x1, y1, x2, y2]
"""
import numpy as np

IOU_THRS = np.arange(0.5, 0.96, 0.05)


def _iou(a, b):
    """a:(N,4) b:(M,4) -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _prepare(images, n_cls):
    """클래스별로 (conf, iou행렬, 이미지idx) 를 미리 계산"""
    per_cls = [[] for _ in range(n_cls)]
    n_gt = np.zeros(n_cls, int)
    for i, (dets, gts) in enumerate(images):
        dets = np.asarray(dets, float).reshape(-1, 6)
        gts = np.asarray(gts, float).reshape(-1, 5)
        for c in range(n_cls):
            d = dets[dets[:, 0] == c]
            g = gts[gts[:, 0] == c]
            n_gt[c] += len(g)
            if len(d) == 0:
                continue
            d = d[np.argsort(-d[:, 1])]
            per_cls[c].append((i, d[:, 1], _iou(d[:, 2:6], g[:, 1:5]), len(g)))
    return per_cls, n_gt


def _ap_from_matches(tp, conf, n_gt):
    if n_gt == 0:
        return np.nan
    if len(conf) == 0:
        return 0.0
    order = np.argsort(-conf, kind="stable")
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.concatenate([[0.0], prec, [0.0]])
    rec = np.concatenate([[0.0], rec, [1.0]])
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, pts, side="left")
    return float(np.mean(prec[np.minimum(idx, len(prec) - 1)]))


def compute_map(images, n_cls):
    """-> dict(mAP50, mAP50_95, ap50_per_class)"""
    per_cls, n_gt = _prepare(images, n_cls)
    ap = np.full((n_cls, len(IOU_THRS)), np.nan)
    for c in range(n_cls):
        for t, thr in enumerate(IOU_THRS):
            tps, confs = [], []
            for _, conf, iou, ng in per_cls[c]:
                tp = np.zeros(len(conf))
                used = np.zeros(ng, bool)
                for k in range(len(conf)):  # conf 내림차순으로 정렬돼 있음
                    if ng == 0:
                        break
                    cand = np.where(used, -1, iou[k])
                    j = int(np.argmax(cand))
                    if cand[j] >= thr:
                        used[j] = True
                        tp[k] = 1
                tps.append(tp)
                confs.append(conf)
            tp_all = np.concatenate(tps) if tps else np.zeros(0)
            conf_all = np.concatenate(confs) if confs else np.zeros(0)
            ap[c, t] = _ap_from_matches(tp_all, conf_all, n_gt[c])
    valid = ~np.isnan(ap[:, 0])
    return dict(
        mAP50=float(np.mean(ap[valid, 0])) if valid.any() else float("nan"),
        mAP50_95=float(np.mean(ap[valid])) if valid.any() else float("nan"),
        ap50_per_class=ap[:, 0],
    )


def bootstrap_map50(images, n_cls, n_boot=200, seed=0):
    """이미지를 복원추출해 mAP50 의 95% 신뢰구간을 구한다."""
    rng = np.random.default_rng(seed)
    n = len(images)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(compute_map([images[i] for i in idx], n_cls)["mAP50"])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
