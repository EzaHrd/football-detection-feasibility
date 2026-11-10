import pandas as pd, io, collections, imagehash, numpy as np, json, glob
from PIL import Image, ImageDraw
names=["soccer-players","football","player","referee"]
R={};H={};cnt={}
for sp in ["train","validation","test"]:
    f=glob.glob(f"football_detection/data/raw/soccer-players-5fuqs/data/{sp}-*.parquet")[0]
    df=pd.read_parquet(f); R[sp]=df
    c=collections.Counter(); sz=collections.Counter(); ar=[]
    for _,r in df.iterrows():
        o=r["objects"]
        for cat,b in zip(o["category"],o["bbox"]): c[names[cat]]+=1
        sz[(int(r["width"]),int(r["height"]))]+=1
    cnt[sp]=dict(n=len(df),boxes=dict(c),sizes=dict((f"{k[0]}x{k[1]}",v) for k,v in sz.most_common(4)))
    H[sp]=np.array([imagehash.phash(Image.open(io.BytesIO(x["bytes"])).convert("L")).hash.flatten() for x in df["image"]]).astype(np.uint8)
print(json.dumps(cnt,indent=1))
for a,b in [("train","test"),("train","validation"),("validation","test")]:
    d=np.array([(H[a]!=h).sum(1).min() for h in H[b]])
    print(b,"vs",a,": pHash<=4:",int((d<=4).sum()),"/",len(d),"; <=8:",int((d<=8).sum()))
# sheet from test
col={"soccer-players":(255,0,255),"football":(255,0,0),"player":(0,255,0),"referee":(255,255,0)}
tiles=[]
for _,r in R["test"].head(8).iterrows():
    im=Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"); dr=ImageDraw.Draw(im)
    for cat,(x,y,w,h) in zip(r["objects"]["category"],r["objects"]["bbox"]): dr.rectangle([x,y,x+w,y+h],outline=col[names[cat]],width=3)
    tiles.append(im.resize((480,480)))
sh=Image.new("RGB",(1920,960))
for i,t in enumerate(tiles): sh.paste(t,((i%4)*480,(i//4)*480))
sh.save("football_detection/results/audit_rf100_sample_sheet.jpg",quality=80)
json.dump(cnt,open("football_detection/results/audit_rf100_stats.json","w"),indent=1)
