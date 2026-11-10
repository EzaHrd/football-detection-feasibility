"""Evaluasi model fine-tune di VAL (evaluator bersama + confusion matrix Ultralytics). Pemakaian: eval_model.py BOBOT NAMA_RUN"""
import sys, json, pathlib
sys.path.insert(0, "football_detection/scripts")
from eval_lib import evaluate, speed
from ultralytics import YOLO


def main():
    w, name = sys.argv[1], sys.argv[2]
    val = [l.strip() for l in open("football_detection/data/work/splits/val.txt") if l.strip()]
    res = evaluate(w, val)
    sg = speed(w, val, device=0); sc = speed(w, val, device="cpu", n=40)
    out = dict(run=name, split="val", n_images=len(val), imgsz=640, conf_eval=0.001, weights=w, referee_available=True,
               metrics=res, speed_gpu=sg, speed_cpu=sc)
    pathlib.Path("football_detection/results/runs").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"football_detection/results/runs/{name}.json", "w"), indent=1)
    # confusion matrix resmi Ultralytics, split val saja
    v = YOLO(w).val(data=str(pathlib.Path("football_detection/data/work/splits/data_day2.yaml").resolve()), split="val", imgsz=640, batch=8,
                    device=0, plots=True, workers=2, project=str(pathlib.Path("football_detection/results/val_plots").resolve()), name=name)
    out["ultralytics_val"] = dict(mAP50=float(v.box.map50), mAP50_95=float(v.box.map))
    json.dump(out, open(f"football_detection/results/runs/{name}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
