"""T1: bekukan konfigurasi Hari 3 HANYA dari val + train. Test tidak dibaca."""
import sys, json, pathlib, hashlib, numpy as np
sys.path.insert(0, "football_detection/scripts")
from eval_lib import run_preds, metrics, NAMES

R = pathlib.Path("football_detection/results")
WA = "football_detection/results/smoke_test/yolov8n.pt"
WB = "football_detection/results/train/B_yolov8n_e25_s42_b8/weights/best.pt"
A_MAP = {0: 0, 32: 1}


def main():
    val = [l.strip() for l in open("football_detection/data/work/splits/val.txt") if l.strip()]
    train = [l.strip() for l in open("football_detection/data/work/splits/train.txt") if l.strip()]
    cfg = dict(created="2026-11-10", note="dibekukan dari val/train sebelum test disentuh", imgsz=640, nms_iou=0.6, eval_conf=0.001,
               match_iou=0.5, weights_B=WB, sha256_B=hashlib.sha256(open(WB, "rb").read()).hexdigest(),
               weights_A=WA, sha256_A=hashlib.sha256(open(WA, "rb").read()).hexdigest(),
               class_map_A={"person(0)": "pemain", "sports ball(32)": "bola", "wasit": "tidak tersedia"})
    out = {}
    for tag, w, cm, cls in [("A", WA, A_MAP, [0, 32]), ("B", WB, None, None)]:
        per = run_preds(w, val, class_map=cm, classes=cls)
        m0 = metrics(per)
        op = {NAMES[k]: m0[NAMES[k]]["f1_opt_conf"] for k in range(3)}
        if tag == "A": op["wasit"] = None
        m = metrics(per, op_conf=op)
        cfg[f"op_conf_{tag}"] = op
        out[tag] = m
        json.dump(dict(run=f"{tag}_val_opconf", split="val", n_images=len(val), op_conf=op, metrics=m),
                  open(R / f"runs/{tag}_day3_val_opconf.json", "w"), indent=1)
    # ambang ukuran bola: median area ternormalisasi bola di TRAIN (bukan test)
    areas = []
    for p in train:
        lp = pathlib.Path(p.replace("\\", "/").replace("/images/", "/labels/")).with_suffix(".txt")
        for l in lp.read_text().splitlines():
            s = l.split()
            if s and s[0] == "1": areas.append(float(s[3]) * float(s[4]))
    cfg["ball_size_threshold_norm_area"] = float(np.median(areas)); cfg["ball_size_threshold_source"] = f"median {len(areas)} bola train"
    cfg["subset_min_side_px"] = 640
    cfg["robustness"] = dict(gauss_sigma=[1, 2, 3], motion_kernel=[7, 15], down_up=[0.5, 0.25], brightness=[-0.3, 0.3],
                             contrast=[0.6, 1.4], jpeg_quality=[30, 10], gauss_noise_sigma=[10, 25],
                             judged=dict(blur="gauss_sigma=2", resolution="down_up=0.5", light="brightness=+-0.3 (terburuk dari keduanya)"))
    json.dump(cfg, open(R / "day3_frozen_config.json", "w"), indent=1)
    print(json.dumps(cfg, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
