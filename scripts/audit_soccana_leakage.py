import pathlib, collections, re, random, json
import numpy as np, imagehash
from PIL import Image, ImageDraw
root=pathlib.Path("football_detection/data/work/soccana/V1")
def pref(n): return re.sub(r"_\d+$","",n)
res={}
for sp in ["train","test"]:
    c=collections.Counter(pref(p.stem) for p in (root/"images"/sp).iterdir()); res[sp]=dict(c)
print(json.dumps(res,indent=0))
H={}
for sp in ["train","test"]:
    ps=sorted((root/"images"/sp).iterdir())
    H[sp]=(ps,np.array([imagehash.phash(Image.open(p).convert("L")).hash.flatten() for p in ps]))
tr=H["train"][1].astype(np.uint8); 
d=[]
for h in H["test"][1].astype(np.uint8):
    d.append(int((tr!=h).sum(1).min()))
d=np.array(d)
print("test imgs with a train img at pHash dist<=4:",int((d<=4).sum()),"of",len(d),"| <=8:",int((d<=8).sum()))
# per-source
ps=H["test"][0]; bysrc=collections.defaultdict(list)
for p,x in zip(ps,d): bysrc[pref(p.stem)].append(bool(x<=4))
print({k:f"{sum(v)}/{len(v)}" for k,v in bysrc.items()})
json.dump({"near_dup_le4":int((d<=4).sum()),"n_test":len(d),"near_dup_le8":int((d<=8).sum()),"by_source_le4":{k:[sum(v),len(v)] for k,v in bysrc.items()}},open("football_detection/results/audit_soccana_leakage.json","w"))
# contact sheet w/ boxes
random.seed(0); names=["Player","Ball","Referee"]; col=[(0,255,0),(255,0,0),(255,255,0)]
sel=random.sample(sorted((root/"images/train").iterdir()),12)
tiles=[]
for p in sel:
    im=Image.open(p).convert("RGB"); w,h=im.size; dr=ImageDraw.Draw(im)
    lp=root/"labels/train"/(p.stem+".txt")
    for r in lp.read_text().split("\n"):
        if r.strip():
            c,x,y,bw,bh=map(float,r.split()); dr.rectangle([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h],outline=col[int(c)],width=max(2,w//200))
    tiles.append(im.resize((480,480)))
sheet=Image.new("RGB",(480*4,480*3))
for i,t in enumerate(tiles): sheet.paste(t,((i%4)*480,(i//4)*480))
sheet.save("football_detection/results/audit_soccana_sample_sheet.jpg",quality=80)
print([p.name for p in sel])

