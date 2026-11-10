"""T2: buat daftar test internal, cek kebocoran pHash test vs train+val SEBELUM inferensi. Gambar test dengan jarak <= 4 dikeluarkan."""
import csv, pathlib, collections, json, numpy as np, imagehash
from PIL import Image

W = pathlib.Path("football_detection/data/work"); V1 = (W / "soccana/V1").resolve(); S = W / "splits"


def main():
    rows = list(csv.DictReader(open(S / "split_by_source_v1.csv")))
    sp = collections.defaultdict(list)
    for r in rows: sp[r["new_split"]].append(str(V1 / r["image"]))
    H = lambda ps: np.array([imagehash.phash(Image.open(p).convert("L")).hash.flatten() for p in ps]).astype(np.uint8)
    href = H(sp["train"] + sp["val"]); ref = sp["train"] + sp["val"]
    ht = H(sp["test"])
    d = np.array([(href != h).sum(1).min() for h in ht])
    arg = [int((href != h).sum(1).argmin()) for h in ht]
    excl = [i for i in np.where(d <= 4)[0]]
    keep = [p for i, p in enumerate(sp["test"]) if i not in set(excl)]
    (S / "test.txt").write_text("\n".join(keep) + "\n")
    (S / "test_excluded_near_dup.txt").write_text("\n".join(f"{sp['test'][i]}\t{ref[arg[i]]}\tdist={int(d[i])}" for i in excl))
    (S / "data_test.yaml").write_text(f"train: {(S/'train.txt').resolve().as_posix()}\nval: {(S/'test.txt').resolve().as_posix()}\n"
                                      "names:\n  0: pemain\n  1: bola\n  2: wasit\n")
    cnt = collections.Counter(); empty = 0; big = 0
    for p in keep:
        lp = pathlib.Path(p.replace("\\", "/").replace("/images/", "/labels/")).with_suffix(".txt")
        rr = [l.split()[0] for l in lp.read_text().splitlines() if l.strip()]
        if not rr: empty += 1
        cnt.update(rr)
        with Image.open(p) as im: big += min(im.size) >= 640
    tr_names = {pathlib.Path(p).name for p in sp["train"] + sp["val"]}
    out = dict(n_test_before=len(sp["test"]), excluded_pHash_le4=len(excl), le8=int((d <= 8).sum()), exact=int((d == 0).sum()),
               same_filename_as_trainval=sum(pathlib.Path(p).name in tr_names for p in keep),
               n_test_after=len(keep), empty=empty, min_side_ge640=big, player=cnt["0"], ball=cnt["1"], referee=cnt["2"],
               excluded_by_source=dict(collections.Counter(pathlib.Path(sp["test"][i]).stem.rsplit("_", 1)[0] for i in excl)))
    json.dump(out, open("football_detection/results/day3_test_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
