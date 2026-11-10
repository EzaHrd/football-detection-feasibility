"""Deskriptif (bukan tuning): metrik test per sumber + indikasi augmentasi rotasi di tiap sumber (porsi kotak pemain lebih lebar dari tinggi)."""
import sys, json, re, pathlib, collections, csv
sys.path.insert(0, "football_detection/scripts")
from eval_lib import run_preds, metrics
from PIL import Image

R = pathlib.Path("football_detection/results")
base = lambda s: re.sub(r"_sahi_\d+", "", re.sub(r"_\d+$", "", s))


def main():
    cfg = json.load(open(R / "day3_frozen_config.json"))
    test = [l.strip() for l in open("football_detection/data/work/splits/test.txt") if l.strip()]
    out = {"metrics_per_source": {}, "wide_player_boxes": {}}
    for tag, w, cm, cls in [("A", cfg["weights_A"], {0: 0, 32: 1}, [0, 32]), ("B", cfg["weights_B"], None, None)]:
        per = run_preds(w, test, class_map=cm, classes=cls)
        groups = collections.defaultdict(list)
        for d in per: groups[base(pathlib.Path(d["img"]).stem)].append(d)
        for g, L in groups.items():
            m = metrics(L, op_conf=cfg[f"op_conf_{tag}"])
            out["metrics_per_source"][f"{tag}_{g}"] = dict(n_images=len(L), map50=m["_mean"]["ap50"],
                **{f"{k}_{x}": m[k].get(x) for k in ("pemain", "bola", "wasit") for x in ("ap50", "precision", "recall", "n_gt")})
    # porsi kotak pemain melintang (w > 1.2*h dalam piksel) per sumber, semua split
    rows = list(csv.DictReader(open("football_detection/data/work/splits/split_by_source_v1.csv")))
    V1 = pathlib.Path("football_detection/data/work/soccana/V1")
    c = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        p = V1 / r["image"]; lp = V1 / (r["image"].replace("images", "labels", 1).rsplit(".", 1)[0] + ".txt")
        lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
        if not lines: continue
        W, H = Image.open(p).size
        for s in lines:
            if s[0] == "0":
                c[r["source_group"]][1] += 1
                c[r["source_group"]][0] += float(s[3]) * W > 1.2 * float(s[4]) * H
    out["wide_player_boxes"] = {g: dict(frac=v[0] / v[1], n=v[1]) for g, v in c.items()}
    json.dump(out, open(R / "day3_per_source.json", "w"), indent=1)
    for k, v in out["metrics_per_source"].items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    print({g: round(v["frac"], 3) for g, v in out["wide_player_boxes"].items()})


if __name__ == "__main__":
    main()
