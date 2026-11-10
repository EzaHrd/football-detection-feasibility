"""Analisis error. Pemakaian: error_analysis.py BOBOT NAMA_RUN [conf=0.25] [DAFTAR_GAMBAR=val.txt]
Hari 3: conf dapat berupa 'beku' = ambang per kelas dari day3_frozen_config.json (Model B)."""
import sys, json, random, pathlib, collections
import numpy as np, torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
sys.path.insert(0, "football_detection/scripts")
from eval_lib import read_gt, NAMES

w, run = sys.argv[1], sys.argv[2]
carg = sys.argv[3] if len(sys.argv) > 3 else "0.25"
lst = sys.argv[4] if len(sys.argv) > 4 else "football_detection/data/work/splits/val.txt"
if carg == "beku":
    opc = json.load(open("football_detection/results/day3_frozen_config.json"))["op_conf_B"]
    cls_conf = np.array([opc["pemain"], opc["bola"], opc["wasit"]]); conf = float(cls_conf.min())
else:
    conf = float(carg); cls_conf = None
val = [l.strip() for l in open(lst) if l.strip()]
out = pathlib.Path("football_detection/results/error_analysis") / run
out.mkdir(parents=True, exist_ok=True)
model = YOLO(w)
errs = []  # (kategori, img, box, info)
for i in range(0, len(val), 16):
    ch = val[i:i + 16]
    for img, r in zip(ch, model.predict(ch, imgsz=640, conf=conf, iou=0.6, device=0, verbose=False)):
        gb, gc = read_gt(img)[:2]
        pb = r.boxes.xyxy.cpu().numpy(); pc = r.boxes.cls.cpu().numpy().astype(int)
        if cls_conf is not None:
            k = r.boxes.conf.cpu().numpy() >= cls_conf[pc]
            pb, pc = pb[k], pc[k]
        iou = box_iou(torch.tensor(gb, dtype=torch.float32), torch.tensor(pb, dtype=torch.float32)).numpy() if len(gb) and len(pb) else np.zeros((len(gb), len(pb)))
        used = set()
        gmatched = set()
        # cocokkan kelas sama IoU>=0.5 (greedy berdasarkan IoU)
        pairs = sorted([(iou[g, p], g, p) for g in range(len(gb)) for p in range(len(pb)) if iou[g, p] >= 0.5 and gc[g] == pc[p]], reverse=True)
        for _, g, p in pairs:
            if g not in gmatched and p not in used: gmatched.add(g); used.add(p)
        for g in range(len(gb)):
            if g in gmatched: continue
            k = NAMES[gc[g]]
            cand = [p for p in range(len(pb)) if iou[g, p] >= 0.5 and pc[p] != gc[g] and p not in used]
            if cand:
                p = max(cand, key=lambda q: iou[g, q]); used.add(p)
                errs.append((f"salah_kelas_{k}_jadi_{NAMES[pc[p]]}", img, gb[g], "gt"))
            elif len(pb) and any(iou[g, p] >= 0.1 and pc[p] == gc[g] for p in range(len(pb))):
                errs.append((f"lokalisasi_buruk_{k}", img, gb[g], "gt"))
            else:
                errs.append((f"{k}_tidak_terdeteksi", img, gb[g], "gt"))
        for p in range(len(pb)):
            if p in used: continue
            k = NAMES[pc[p]]
            same = [g for g in range(len(gb)) if iou[g, p] >= 0.5 and gc[g] == pc[p]]
            if same: errs.append((f"{k}_ganda", img, pb[p], "pred"))
            elif len(gb) and iou[:, p].max() >= 0.1: errs.append((f"lokalisasi_buruk_{k}", img, pb[p], "pred"))
            else: errs.append((f"latar_jadi_{k}", img, pb[p], "pred"))
cnt = collections.Counter(e[0] for e in errs)
print(json.dumps(cnt.most_common(), indent=1))
random.seed(42)
# sampel 40 error, stratified proporsional (min 1 per jenis), simpan contoh gambar (maks 3 per jenis)
bytype = collections.defaultdict(list)
for e in errs: bytype[e[0]].append(e)
sample = []
tot = len(errs)
for k, v in bytype.items(): sample += random.sample(v, min(len(v), max(1, round(40 * len(v) / tot))))
for k, v in bytype.items():
    d = out / k; d.mkdir(exist_ok=True)
    nex = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    pool = [s for s in sample if s[0] == k] if nex <= 3 else random.Random(42).sample(v, min(len(v), nex))
    for j, e in enumerate(pool[:nex]):
        im = Image.open(e[1]).convert("RGB"); dr = ImageDraw.Draw(im)
        b = e[2]; col = (255, 0, 0) if e[3] == "gt" else (0, 160, 255)
        dr.rectangle(list(b), outline=col, width=3)
        m = 40; crop = im.crop((max(0, b[0] - m * 3), max(0, b[1] - m * 3), min(im.width, b[2] + m * 3), min(im.height, b[3] + m * 3)))
        if min(crop.size) < 200: crop = crop.resize((crop.width * 2, crop.height * 2))
        crop.save(d / f"{j}_{pathlib.Path(e[1]).stem}.jpg", quality=85)
res = dict(run=run, conf=conf, n_val=len(val), total_errors=tot, counts=dict(cnt), n_sampled=len(sample),
           sample=[dict(tipe=s[0], img=pathlib.Path(s[1]).name, asal="merah=GT / biru=prediksi" if False else s[3]) for s in sample])
json.dump(res, open(out / "errors.json", "w"), indent=1)
