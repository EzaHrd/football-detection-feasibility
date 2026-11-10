"""T9: nilai kriteria lulus EVAL_PLAN §5 (dibekukan 2026-11-10) dari berkas hasil. Ambang di bawah disalin dari EVAL_PLAN dan tidak diubah."""
import json, csv, pathlib

R = pathlib.Path("football_detection/results")
A = json.load(open(R / "runs/A_day3_test.json")); B = json.load(open(R / "runs/B_day3_test.json"))
BR = json.load(open(R / "runs/B_day3_rf100.json"))
rob = {r["gangguan"]: r for r in csv.DictReader(open(R / "day3_robustness.csv"))}
mb, ma = B["metrics"], A["metrics"]
f = lambda k: float(rob[k]["turun_relatif_map50"])
light = max(f("brightness=-0.3"), f("brightness=+0.3"))
C = [
    (1, "AP50 pemain (test)", mb["pemain"]["ap50"], ">=", 0.70, True),
    (2, "AP50 wasit (test)", mb["wasit"]["ap50"], ">=", 0.50, True),
    (3, "Recall bola (test, conf beku)", mb["bola"]["recall"], ">=", 0.50, True),
    (4, "Precision bola (test, conf beku)", mb["bola"]["precision"], ">=", 0.70, True),
    (5, "mAP50 3 kelas (RF100)", BR["metrics"]["_mean"]["ap50"], ">=", 0.50, True),
    ("6a", "Turun relatif mAP50, blur sigma=2", f("gauss_sigma=2"), "<=", 0.25, False),
    ("6b", "Turun relatif mAP50, resolusi x0.5", f("down_up=0.5"), "<=", 0.25, False),
    ("6c", "Turun relatif mAP50, kecerahan +-30% (terburuk)", light, "<=", 0.15, False),
    ("7a", "ms/frame GPU median (test)", B["speed"]["gpu"]["median_ms"], "<=", 50, False),
    ("7b", "ms/frame CPU median (test)", B["speed"]["cpu"]["median_ms"], "<=", 200, False),
    ("8a", "Kenaikan AP50 pemain B-A (test)", mb["pemain"]["ap50"] - ma["pemain"]["ap50"], ">=", 0.10, True),
    ("8b", "Kenaikan AP50 bola B-A (test)", mb["bola"]["ap50"] - ma["bola"]["ap50"], ">=", 0.10, True),
]
rows = []
for no, name, val, op, thr, core in C:
    ok = val >= thr if op == ">=" else val <= thr
    rows.append(dict(no=no, kriteria=name, nilai=round(val, 4), ambang=f"{op} {thr}", lulus=ok, inti=core))
core_ok = all(r["lulus"] for r in rows if r["inti"])
out = dict(kriteria=rows, kelayakan_terbukti=core_ok, aturan="terbukti jika kriteria inti (1-5, 8) semua lulus; 6-7 dilaporkan sebagai risiko")
json.dump(out, open(R / "day3_criteria.json", "w"), indent=1)
for r in rows: print(r)
print("KELAYAKAN TERBUKTI:", core_ok)
