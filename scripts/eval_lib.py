"""Evaluator bersama (A dan B memakai kode yang sama).
Kelas target: 0 pemain, 1 bola, 2 wasit. class_map: id kelas model -> id target (kelas lain dibuang).
Hari 3: ditambah transform (uji ketahanan, di memori saja), P/R pada ambang beku per kelas, recall bola per ukuran, bootstrap."""
import pathlib
import numpy as np, torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou, ap_per_class
from ultralytics.utils.patches import imread

NAMES = ["pemain", "bola", "wasit"]
IOUV = np.linspace(0.5, 0.95, 10)


def label_path(img):
    s = str(img).replace("\\", "/")
    s = s.replace("/images/", "/labels/")
    return pathlib.Path(s).with_suffix(".txt")


def read_gt(img):
    p = label_path(img)
    w, h = Image.open(img).size
    rows = [list(map(float, l.split())) for l in p.read_text().splitlines() if l.strip()]
    if not rows: return np.zeros((0, 4)), np.zeros(0, int), (w, h), np.zeros(0)
    a = np.array(rows); c = a[:, 0].astype(int)
    xy = a[:, 1:5] * [w, h, w, h]
    b = np.c_[xy[:, 0] - xy[:, 2] / 2, xy[:, 1] - xy[:, 3] / 2, xy[:, 0] + xy[:, 2] / 2, xy[:, 1] + xy[:, 3] / 2]
    return b, c, (w, h), a[:, 3] * a[:, 4]  # area ternormalisasi


def _pairs(gb, gc, pb, pc, t):
    """Pencocokan greedy berbasis IoU (gaya Ultralytics). Kembalikan array [gt_idx, pred_idx]."""
    if len(pb) == 0 or len(gb) == 0: return np.zeros((0, 2), int)
    iou = box_iou(torch.tensor(gb, dtype=torch.float32), torch.tensor(pb, dtype=torch.float32)).numpy()
    iou = iou * (gc[:, None] == pc[None, :])
    m = np.array(np.nonzero(iou >= t)).T
    if not len(m): return np.zeros((0, 2), int)
    m = m[np.argsort(-iou[m[:, 0], m[:, 1]])]
    m = m[np.unique(m[:, 1], return_index=True)[1]]  # urut per indeks prediksi = conf tertinggi dulu
    m = m[np.unique(m[:, 0], return_index=True)[1]]  # per GT: prediksi conf tertinggi (sama dengan Ultralytics / Hari 2)
    return m.astype(int)


def match(gb, gc, pb, pc):
    correct = np.zeros((len(pb), len(IOUV)), bool)
    for i, t in enumerate(IOUV):
        m = _pairs(gb, gc, pb, pc, t)
        if len(m): correct[m[:, 1], i] = True
    return correct


def run_preds(weights, images, class_map=None, classes=None, imgsz=640, device=0, bs=16, transform=None):
    """Prediksi per gambar. transform(img_bgr)->img_bgr diterapkan di memori (file asli tidak diubah)."""
    model = YOLO(weights)
    per = []
    for i in range(0, len(images), bs):
        chunk = images[i:i + bs]
        src = chunk if transform is None else [transform(imread(p)) for p in chunk]
        res = model.predict(src, imgsz=imgsz, conf=0.001, iou=0.6, max_det=300, device=device, verbose=False, classes=classes)
        for img, r in zip(chunk, res):
            gb, gc, _, area = read_gt(img)
            b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int); s = r.boxes.conf.cpu().numpy()
            if class_map is not None:
                keep = np.array([k in class_map for k in c], bool)
                b, c, s = b[keep], np.array([class_map[k] for k in c[keep]], int), s[keep]
            per.append(dict(img=str(img), tp=match(gb, gc, b, c), conf=s, pc=c, tc=gc, gb=gb, pb=b, area=area))
    return per


def _cat(per):
    return (np.concatenate([d["tp"] for d in per]), np.concatenate([d["conf"] for d in per]),
            np.concatenate([d["pc"] for d in per]), np.concatenate([d["tc"] for d in per]))


def metrics(per, op_conf=None, classes_expected=(0, 1, 2)):
    """AP per kelas (conf 0,001), conf F1-optimal per kelas, dan P/R pada op_conf {kelas: ambang} jika diberikan."""
    tp, conf, pc, tc = _cat(per)
    out = ap_per_class(tp, conf, pc, tc, plot=False)
    ap, uc, f1c, x = out[5], out[6], out[9], out[10]
    res = {}
    for k in classes_expected:
        n_gt = int((tc == k).sum()); n_pred = int((pc == k).sum())
        if k in uc and n_pred > 0:
            j = list(uc).index(k)
            res[NAMES[k]] = dict(ap50=float(ap[j, 0]), ap50_95=float(ap[j].mean()), n_gt=n_gt,
                                 f1_opt_conf=float(x[int(np.argmax(f1c[j]))]), f1_max=float(f1c[j].max()))
        else:
            res[NAMES[k]] = dict(ap50=None, ap50_95=None, n_gt=n_gt, f1_opt_conf=None, f1_max=None)
        if op_conf is not None and op_conf.get(NAMES[k]) is not None:
            t = op_conf[NAMES[k]]
            msk = (pc == k) & (conf >= t)
            TP = int(tp[msk, 0].sum())
            res[NAMES[k]].update(op_conf=t, precision=TP / max(int(msk.sum()), 1) if msk.sum() else None,
                                 recall=TP / n_gt if n_gt else None, tp=TP, n_pred_at_op=int(msk.sum()))
    av = [res[NAMES[k]]["ap50"] for k in classes_expected if res[NAMES[k]]["ap50"] is not None]
    av95 = [res[NAMES[k]]["ap50_95"] for k in classes_expected if res[NAMES[k]]["ap50_95"] is not None]
    res["_mean"] = dict(ap50=float(np.mean(av)) if av else None, ap50_95=float(np.mean(av95)) if av95 else None, n_classes=len(av))
    return res


def ball_recall_by_size(per, t_ball, size_thr):
    """Recall bola (IoU 0,5, conf >= t_ball) untuk bola kecil (area ternorm < size_thr) dan besar."""
    hit = {"kecil": [0, 0], "besar": [0, 0]}
    for d in per:
        g = d["tc"] == 1
        if not g.any(): continue
        gb, ga = d["gb"][g], d["area"][g]
        p = (d["pc"] == 1) & (d["conf"] >= t_ball)
        m = _pairs(gb, np.ones(len(gb), int), d["pb"][p], np.ones(int(p.sum()), int), 0.5)
        matched = set(m[:, 0].tolist())
        for i, a in enumerate(ga):
            key = "kecil" if a < size_thr else "besar"
            hit[key][1] += 1; hit[key][0] += int(i in matched)
    return {k: dict(recall=(v[0] / v[1]) if v[1] else None, n_gt=v[1], tp=v[0]) for k, v in hit.items()}


def bootstrap(per, op_conf, n=1000, seed=42):
    """Resampling per gambar. Interval 95% untuk AP50 bola, recall bola (ambang beku), dan mAP50."""
    rng = np.random.default_rng(seed)
    vals = {"ap50_bola": [], "recall_bola": [], "map50": []}
    N = len(per)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        r = metrics([per[i] for i in idx], op_conf=op_conf)
        vals["ap50_bola"].append(r["bola"]["ap50"]); vals["recall_bola"].append(r["bola"].get("recall"))
        vals["map50"].append(r["_mean"]["ap50"])
    out = {}
    for k, v in vals.items():
        v = np.array([x for x in v if x is not None], float)
        out[k] = dict(lo95=float(np.percentile(v, 2.5)), hi95=float(np.percentile(v, 97.5)), n_valid=len(v))
    out["n_boot"] = n; out["seed"] = seed
    return out


def evaluate(weights, images, class_map=None, imgsz=640, device=0, bs=16, classes=None):
    """Kompatibel dengan Hari 2."""
    return metrics(run_preds(weights, images, class_map, classes, imgsz, device, bs))


def speed(weights, images, imgsz=640, device=0, n=100, warm=10):
    """ms per frame (pra-proses+inferensi+NMS), batch 1, setelah pemanasan. Median dan p95."""
    model = YOLO(weights)
    for im in images[:warm]: model.predict(im, imgsz=imgsz, device=device, verbose=False)
    tot = []
    for im in images[warm:warm + n]:
        r = model.predict(im, imgsz=imgsz, device=device, verbose=False)[0]
        tot.append(sum(r.speed.values()))
    return dict(median_ms=float(np.median(tot)), p95_ms=float(np.percentile(tot, 95)), n=len(tot))
