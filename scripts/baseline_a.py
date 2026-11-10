"""Baseline A: YOLOv8n COCO pretrained tanpa fine-tune, evaluasi di VAL. person->pemain, sports ball->bola, wasit tidak tersedia."""
import sys, json, csv, pathlib, time
sys.path.insert(0, "football_detection/scripts")
from eval_lib import evaluate, speed

val = [l.strip() for l in open("football_detection/data/work/splits/val.txt") if l.strip()]
W = "football_detection/results/smoke_test/yolov8n.pt"
name = "A_yolov8n_pretrained_val"
res = evaluate(W, val, class_map={0: 0, 32: 1}, classes=[0, 32])
spd_gpu = speed(W, val, device=0); spd_cpu = speed(W, val, device="cpu", n=40)
out = dict(run=name, split="val", n_images=len(val), imgsz=640, conf_eval=0.001, mapping="person->pemain, sports ball->bola, wasit: tidak tersedia",
           metrics=res, speed_gpu=spd_gpu, speed_cpu=spd_cpu)
pathlib.Path("football_detection/results/runs").mkdir(parents=True, exist_ok=True)
json.dump(out, open(f"football_detection/results/runs/{name}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
