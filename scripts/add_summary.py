"""Tambah satu baris ke results/summary.csv dari results/runs/<nama>.json (tidak menimpa baris lama).
Pemakaian: add_summary.py NAMA_RUN MODEL PARAMETER"""
import sys, json, csv, pathlib

name, model, param = sys.argv[1:4]
d = json.load(open(f"football_detection/results/runs/{name}.json"))
m = d["metrics"]
f = lambda v: "tidak_tersedia" if v is None else round(v, 4)
na = d.get("referee_available", True) is False
cols = ["nama_run", "model", "parameter", "mAP", "precision_pemain", "recall_pemain", "precision_bola", "recall_bola",
        "precision_wasit", "recall_wasit", "ms_per_frame"]
if na:
    mAP = round((m["pemain"]["ap50"] + m["bola"]["ap50"]) / 2, 4)
    param += " | mAP=AP50 rata-rata 2 kelas (pemain,bola); wasit tidak tersedia"
    pw = rw = "tidak_tersedia"
else:
    mAP = round(m["_mean"]["ap50"], 4); param += " | mAP=AP50 rata-rata 3 kelas"
    pw, rw = f(m["wasit"]["precision"]), f(m["wasit"]["recall"])
row = [name, model, param, mAP, f(m["pemain"]["precision"]), f(m["pemain"]["recall"]), f(m["bola"]["precision"]), f(m["bola"]["recall"]),
       pw, rw, f"GPU {d['speed_gpu']['median_ms']:.1f} / CPU {d['speed_cpu']['median_ms']:.1f} (median)"]
p = pathlib.Path("football_detection/results/summary.csv")
new = not p.exists()
with open(p, "a", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    if new: w.writerow(cols)
    w.writerow(row)
print(row)
