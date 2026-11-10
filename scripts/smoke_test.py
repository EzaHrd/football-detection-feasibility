import sys, time, json, urllib.request, pathlib
from ultralytics import YOLO
out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
imgs=[]
for n in ["bus.jpg","zidane.jpg"]:
    f=out/n
    if not f.exists(): urllib.request.urlretrieve("https://ultralytics.com/images/"+n,f)
    imgs.append(str(f))
m=YOLO(str(out/"yolov8n.pt"))
res=[]
for dev in ["cpu",0]:
    m.predict(imgs[0],device=dev,verbose=False)
    t=time.perf_counter()
    for _ in range(10): r=m.predict(imgs[1],device=dev,verbose=False)
    ms=(time.perf_counter()-t)/10*1000
    res.append({"device":str(dev),"ms_per_img_avg10":round(ms,1)})
for i in imgs:
    r=m.predict(i,device=0,verbose=False)[0]
    r.save(str(out/("pred_"+pathlib.Path(i).name)))
    res.append({"img":i,"detections":[m.names[int(c)] for c in r.boxes.cls]})
print(json.dumps(res,indent=1)); json.dump(res,open(out/"smoke_result.json","w"),indent=1)
