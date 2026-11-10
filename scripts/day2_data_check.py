"""Hari 2 tugas 1: buat daftar train/val (TANPA test), hitung statistik, cek duplikat train vs val."""
import csv, pathlib, collections, json, numpy as np, imagehash
from PIL import Image

W = pathlib.Path("football_detection/data/work")
V1 = (W / "soccana/V1").resolve()
rows = list(csv.DictReader(open(W / "splits/split_by_source_v1.csv")))
sp = collections.defaultdict(list)
for r in rows:
    sp[r["new_split"]].append(r)
out = W / "splits"
# daftar test sengaja TIDAK dibuat hari ini
for s in ["train", "val"]:
    (out / f"{s}.txt").write_text("\n".join(str(V1 / r["image"]) for r in sp[s]) + "\n")
(out / "data_day2.yaml").write_text(
    f"train: {(out/'train.txt').resolve().as_posix()}\nval: {(out/'val.txt').resolve().as_posix()}\n"
    "names:\n  0: pemain\n  1: bola\n  2: wasit\n")


def lab(r):
    return V1 / (r["image"].replace("images", "labels", 1).rsplit(".", 1)[0] + ".txt")


stats = {}
for s in ["train", "val"]:
    c = collections.Counter(); ib = collections.Counter(); empty = 0
    for r in sp[s]:
        rr = [x.split()[0] for x in lab(r).read_text().splitlines() if x.strip()]
        if not rr: empty += 1
        c.update(rr)
        for k in set(rr): ib[k] += 1
    stats[s] = dict(images=len(sp[s]), empty=empty, player=c["0"], ball=c["1"], referee=c["2"],
                    img_with_ball=ib["1"], img_with_ref=ib["2"])
print(json.dumps(stats, indent=1))


def H(rs):
    return np.array([imagehash.phash(Image.open(V1 / r["image"]).convert("L")).hash.flatten() for r in rs]).astype(np.uint8)


ht, hv = H(sp["train"]), H(sp["val"])
d = np.array([(ht != h).sum(1).min() for h in hv])
same_name = len({pathlib.Path(r["image"]).name for r in sp["val"]} & {pathlib.Path(r["image"]).name for r in sp["train"]})
print("val vs train pHash<=4:", int((d <= 4).sum()), "<=8:", int((d <= 8).sum()), "exact:", int((d == 0).sum()), "nama sama:", same_name)
json.dump(dict(stats=stats, val_vs_train_le4=int((d <= 4).sum()), le8=int((d <= 8).sum()), exact=int((d == 0).sum()), same_name=same_name),
          open("football_detection/results/day2_data_check.json", "w"), indent=1)
(out / "val_near_dup_train.txt").write_text("\n".join(sp["val"][i]["image"] for i in np.where(d <= 4)[0]))
