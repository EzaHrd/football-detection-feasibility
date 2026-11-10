"""T3: konversi RF100 soccer-players-5fuqs (parquet, semua split) ke YOLO di data/work/rf100_yolo. Data asli tidak diubah.
Peta: football->1 bola, player->0 pemain, referee->2 wasit, soccer-players (kelas induk) dibuang."""
import glob, io, json, pathlib, collections
import pandas as pd
from PIL import Image

MAP = {1: 1, 2: 0, 3: 2}  # 0 soccer-players dibuang
out = pathlib.Path("football_detection/data/work/rf100_yolo"); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
paths = []; cnt = collections.Counter(); dropped = 0
for f in sorted(glob.glob("football_detection/data/raw/soccer-players-5fuqs/data/*.parquet")):
    sp = pathlib.Path(f).name.split("-")[0]
    for _, r in pd.read_parquet(f).iterrows():
        im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"); w, h = im.size
        name = f"rf100_{sp}_{int(r['image_id'])}"
        p = out / "images" / f"{name}.jpg"; im.save(p, quality=95)
        lines = []
        for c, (x, y, bw, bh) in zip(r["objects"]["category"], r["objects"]["bbox"]):
            if c not in MAP: dropped += 1; continue
            cnt[MAP[c]] += 1
            lines.append(f"{MAP[c]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
        paths.append(str(p.resolve()))
(out / "rf100_all.txt").write_text("\n".join(paths) + "\n")
(out / "data_rf100.yaml").write_text(f"train: {(out/'rf100_all.txt').resolve().as_posix()}\nval: {(out/'rf100_all.txt').resolve().as_posix()}\nnames:\n  0: pemain\n  1: bola\n  2: wasit\n")
s = dict(n_images=len(paths), pemain=cnt[0], bola=cnt[1], wasit=cnt[2], dropped_soccer_players_boxes=dropped)
json.dump(s, open("football_detection/results/day3_rf100_convert.json", "w"), indent=1); print(s)
