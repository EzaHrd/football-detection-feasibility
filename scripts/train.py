"""Latih YOLO pada train, validasi HANYA pada val. Pemakaian: train.py NAMA_RUN BOBOT_AWAL EPOCH IMGSZ BATCH"""
import sys, time, json, pathlib
from ultralytics import YOLO


def main():
    name, w0, epochs, imgsz, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    project = pathlib.Path("football_detection/results/train").resolve()
    assert not (project / name).exists(), "nama run sudah ada; pilih nama unik"
    cfg = dict(data=str(pathlib.Path("football_detection/data/work/splits/data_day2.yaml").resolve()), epochs=epochs, imgsz=imgsz,
               batch=batch, seed=42, deterministic=True, workers=4, device=0, project=str(project), name=name, exist_ok=False,
               patience=100, plots=True, cache=False)
    t = time.time()
    m = YOLO(w0)
    m.train(**cfg)
    cfg.update(weights_init=w0, train_seconds=round(time.time() - t, 1))
    json.dump(cfg, open(project / name / "run_config.json", "w"), indent=1)
    print("TRAIN_DONE", cfg["train_seconds"])


if __name__ == "__main__":
    main()
