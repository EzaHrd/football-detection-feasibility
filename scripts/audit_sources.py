import pathlib, re, collections, json, io, glob, numpy as np, pandas as pd, imagehash
from PIL import Image
root=pathlib.Path("football_detection/data/work/soccana/V1")
base=lambda s: re.sub(r"_sahi_\d+","",re.sub(r"_\d+$","",s))
st=collections.defaultdict(lambda: collections.Counter())
for sp in ["train","test"]:
    for l in (root/"labels"/sp).glob("*.txt"):
        b=base(l.stem); st[b]["images"]+=1
        rows=[r.split()[0] for r in l.read_text().splitlines() if r.strip()]
        for c in rows: st[b]["cls"+c]+=1
        if "1" in rows: st[b]["img_with_ball"]+=1
df=pd.DataFrame(st).T.fillna(0).astype(int).rename(columns={"cls0":"player","cls1":"ball","cls2":"referee"}); print(df)
df.to_csv("football_detection/results/audit_soccana_by_source.csv")
# cross-check RF100 vs soccana (external-set overlap)
H=[];names=[]
for p in sorted(root.glob("images/*/*.jpg")):
    with Image.open(p) as im: 
        if min(im.size)>=320: H.append(imagehash.phash(im.convert("L")).hash.flatten()); names.append(p.name)
H=np.array(H).astype(np.uint8); hit=collections.Counter(); n=0; tot=0
for f in glob.glob("football_detection/data/raw/soccer-players-5fuqs/data/*.parquet"):
    for x in pd.read_parquet(f)["image"]:
        h=imagehash.phash(Image.open(io.BytesIO(x["bytes"])).convert("L")).hash.flatten().astype(np.uint8); tot+=1
        d=(H!=h).sum(1); 
        if d.min()<=6: n+=1; hit[base(names[int(d.argmin())][:-4])]+=1
print("RF100 imgs with Soccana img (>=320px) at pHash<=6:",n,"/",tot,dict(hit))
