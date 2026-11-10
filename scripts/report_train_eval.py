"""Evaluasi deskriptif Model B pada subset data LATIH (1.000 gambar acak, seed 42) dengan konfigurasi beku. Untuk laporan UTS (kesenjangan train/val/test)."""
import sys, json, random, pathlib
sys.path.insert(0, "football_detection/scripts")
from eval_lib import run_preds, metrics

R = pathlib.Path("football_detection/results")


def main():
    cfg = json.load(open(R / "day3_frozen_config.json"))
    train = [l.strip() for l in open("football_detection/data/work/splits/train.txt") if l.strip()]
    sub = random.Random(42).sample(train, 1000)
    per = run_preds(cfg["weights_B"], sub)
    m = metrics(per, op_conf=cfg["op_conf_B"])
    out = dict(run="B_train_subset1000", n_images=len(sub), seed=42, note="gambar latih tanpa augmentasi; conf beku per kelas dari val", metrics=m)
    json.dump(out, open(R / "runs/B_report_train_subset.json", "w"), indent=1)
    for k in ("pemain", "bola", "wasit"):
        print(k, {x: (round(v, 3) if isinstance(v, float) else v) for x, v in m[k].items()})
    print("mAP50", round(m["_mean"]["ap50"], 3), "mAP50-95", round(m["_mean"]["ap50_95"], 3))


if __name__ == "__main__":
    main()
