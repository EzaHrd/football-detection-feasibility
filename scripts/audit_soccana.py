import pathlib, collections, json, re, sys
import numpy as np
from PIL import Image
root=pathlib.Path("football_detection/data/work/soccana/V1")
out={}
for sp in ["train","test"]:
    imgs=sorted((root/"images"/sp).iterdir()); lbls=list((root/"labels"/sp).glob("*.txt"))
    cnt=collections.Counter(); empty=0; sizes=collections.Counter(); ballarea=[]
    for l in lbls:
        rows=[r.split() for r in l.read_text().splitlines() if r.strip()]
        if not rows: empty+=1
        for r in rows:
            cnt[int(r[0])]+=1
            if r[0]=="1": ballarea.append(float(r[3])*float(r[4]))
    for p in imgs[::50]:
        with Image.open(p) as im: sizes[im.size]+=1
    stems=collections.Counter(re.sub(r"\.rf\..*$","",p.stem) for p in imgs)
    out[sp]=dict(n_images=len(imgs),n_label_files=len(lbls),empty_labels=empty,boxes_by_class=dict(cnt),
      sizes_sampled_every50=dict((f"{k[0]}x{k[1]}",v) for k,v in sizes.items()),
      example_names=[p.name for p in imgs[:3]+imgs[len(imgs)//2:len(imgs)//2+3]],
      ball_box_area_median=float(np.median(ballarea)) if ballarea else None)
print(json.dumps(out,indent=1))
tr={p.name for p in (root/"images/train").iterdir()}; te={p.name for p in (root/"images/test").iterdir()}
print("same filename in train&test:",len(tr&te))
json.dump(out,open("football_detection/results/audit_soccana_stats.json","w"),indent=1)
