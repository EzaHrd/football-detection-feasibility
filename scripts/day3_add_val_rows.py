"""Tambah baris val (conf beku Hari 3) ke summary.csv."""
import json, csv

f = lambda v: "tidak_tersedia" if v is None else round(v, 4)
with open("football_detection/results/summary.csv", "a", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    for t, mn in [("A", "yolov8n_COCO"), ("B", "yolov8n_finetuned")]:
        d = json.load(open(f"football_detection/results/runs/{t}_day3_val_opconf.json")); m = d["metrics"]; op = d["op_conf"]
        g = lambda k, x: f(m[k].get(x)) if op.get(k) else "tidak_tersedia"
        opr = json.dumps({k: (round(v, 3) if v else v) for k, v in op.items()})
        w.writerow([f"{t}_day3_val_opconf", mn, f"Hari3 val 797 img; P/R pada conf beku per kelas {opr}; mAP=AP50 rata-rata {m['_mean']['n_classes']} kelas",
                    f(m["_mean"]["ap50"]), g("pemain", "precision"), g("pemain", "recall"), g("bola", "precision"), g("bola", "recall"),
                    g("wasit", "precision"), g("wasit", "recall"), "lihat Hari 2 (val)"])
print(json.load(open("football_detection/results/day3_frozen_config.json"))["sha256_B"])
