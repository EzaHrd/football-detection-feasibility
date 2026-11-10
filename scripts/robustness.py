"""T7: uji ketahanan Model B pada test internal. Gangguan diterapkan DI MEMORI (file asli tidak diubah), satu per run.
Tingkat gangguan diambil dari day3_frozen_config.json (ditetapkan sebelum run)."""
import sys, json, csv, pathlib, functools
import numpy as np, cv2
sys.path.insert(0, "football_detection/scripts")
from eval_lib import run_preds, metrics

R = pathlib.Path("football_detection/results")


def gauss(img, s): return cv2.GaussianBlur(img, (0, 0), s)
def motion(img, k):
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, ker)
def down_up(img, f):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
def bright(img, b): return np.clip(img.astype(np.float32) * (1 + b), 0, 255).astype(np.uint8)
def contrast(img, c):
    m = img.mean(); return np.clip((img.astype(np.float32) - m) * c + m, 0, 255).astype(np.uint8)
def jpeg(img, q): return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
_rng = {}
def noise(img, s):
    g = _rng.setdefault(s, np.random.default_rng(42))
    return np.clip(img.astype(np.float32) + g.normal(0, s, img.shape), 0, 255).astype(np.uint8)


def main():
    cfg = json.load(open(R / "day3_frozen_config.json")); rb = cfg["robustness"]
    test = [l.strip() for l in open("football_detection/data/work/splits/test.txt") if l.strip()]
    op = cfg["op_conf_B"]
    runs = [("bersih", None)]
    runs += [(f"gauss_sigma={s}", functools.partial(gauss, s=s)) for s in rb["gauss_sigma"]]
    runs += [(f"motion_kernel={k}", functools.partial(motion, k=k)) for k in rb["motion_kernel"]]
    runs += [(f"down_up={f}", functools.partial(down_up, f=f)) for f in rb["down_up"]]
    runs += [(f"brightness={b:+}", functools.partial(bright, b=b)) for b in rb["brightness"]]
    runs += [(f"contrast={c}", functools.partial(contrast, c=c)) for c in rb["contrast"]]
    runs += [(f"jpeg_quality={q}", functools.partial(jpeg, q=q)) for q in rb["jpeg_quality"]]
    runs += [(f"gauss_noise_sigma={s}", functools.partial(noise, s=s)) for s in rb["gauss_noise_sigma"]]
    rows = []; base = None
    for name, tf in runs:
        m = metrics(run_preds(cfg["weights_B"], test, transform=tf), op_conf=op)
        r = dict(gangguan=name, map50=m["_mean"]["ap50"], ap50_pemain=m["pemain"]["ap50"], ap50_bola=m["bola"]["ap50"],
                 ap50_wasit=m["wasit"]["ap50"], recall_bola=m["bola"]["recall"], precision_bola=m["bola"]["precision"])
        if base is None: base = r
        r["turun_relatif_map50"] = (base["map50"] - r["map50"]) / base["map50"]
        r["turun_relatif_recall_bola"] = (base["recall_bola"] - r["recall_bola"]) / base["recall_bola"]
        rows.append(r); print(json.dumps(r), flush=True)
    with open(R / "day3_robustness.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("ROBUST_DONE")


if __name__ == "__main__":
    main()
