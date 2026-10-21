"""
evaluate.py 가 만든 results.csv 에서
  1) 보완 전(base/none) vs 보완 후(rot/hsv_clahe 등) 전체 mAP
  2) 요인별 하락폭: 기준 수준 대비 각 수준의 mAP50 하락
를 표로 출력한다.

  python analyze.py results/results.csv --model base --mode none \
      --reference background=white angle=0 light=normal
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default=None, help="요인 분석에 쓸 모델 (기본: 첫 번째)")
    ap.add_argument("--mode", default="none")
    ap.add_argument("--reference", nargs="*", default=[], help="요인=기준수준 (예: background=white angle=0)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"level": str})
    pd.set_option("display.width", 160)

    print("\n[1] 구성별 전체 성능")
    ov = df[df.factor == "overall"][["model", "mode", "n_images", "mAP50", "mAP50_ci_lo", "mAP50_ci_hi", "mAP50_95", "pcb_found_rate"]]
    print(ov.to_string(index=False))

    model = args.model or df.model.iloc[0]
    ref = dict(r.split("=", 1) for r in args.reference)
    sub = df[(df.model == model) & (df["mode"] == args.mode) & (df.factor != "overall")]
    print(f"\n[2] 요인별 mAP50 하락폭  (model={model}, mode={args.mode})")
    for fac, g in sub.groupby("factor"):
        base_lv = ref.get(fac)
        g = g.copy()
        if base_lv is not None and base_lv in set(g.level):
            b = float(g[g.level == base_lv].mAP50.iloc[0])
            g["drop_vs_ref"] = (b - g.mAP50).round(4)
        print(f"\n  <{fac}>" + (f"  (기준: {base_lv})" if base_lv else ""))
        print(g[[c for c in ["level", "n_images", "mAP50", "mAP50_ci_lo", "mAP50_ci_hi", "mAP50_95", "drop_vs_ref", "pcb_found_rate"] if c in g]].to_string(index=False))
    print("\n주의: 신뢰구간이 겹치는 차이는 '차이가 있다'고 단정하지 말 것.")


if __name__ == "__main__":
    main()
