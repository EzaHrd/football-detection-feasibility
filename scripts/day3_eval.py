"""T4-T6: evaluasi SEKALI pada test internal dan RF100 dengan konfigurasi beku (day3_frozen_config.json).
Tidak ada tuning di sini. Keluaran: results/runs/*_test.json, *_rf100.json, baris baru summary.csv."""
import sys, json, csv, pathlib
sys.path.insert(0, "football_detection/scripts")
from PIL import Image
from eval_lib import run_preds, metrics, ball_recall_by_size, bootstrap, speed
from ultralytics import YOLO

R = pathlib.Path("football_detection/results")


def lst(p): return [l.strip() for l in open(p) if l.strip()]


def summary_row(name, model, param, m, sp):
    f = lambda v: "tidak_tersedia" if v is None else round(v, 4)
    g = lambda k, x: f(m[k].get(x)) if m[k].get("op_conf") is not None else "tidak_tersedia"
    row = [name, model, param, f(m["_mean"]["ap50"]), g("pemain", "precision"), g("pemain", "recall"), g("bola", "precision"), g("bola", "recall"),
           g("wasit", "precision"), g("wasit", "recall"), f"GPU {sp['gpu']['median_ms']:.1f} / CPU {sp['cpu']['median_ms']:.1f} (median)" if sp else "lihat run test"]
    with open(R / "summary.csv", "a", newline="", encoding="utf-8") as fh: csv.writer(fh).writerow(row)


def main():
    cfg = json.load(open(R / "day3_frozen_config.json"))
    test = lst("football_detection/data/work/splits/test.txt")
    rf = lst("football_detection/data/work/rf100_yolo/rf100_all.txt")
    big = [p for p in test if min(Image.open(p).size) >= cfg["subset_min_side_px"]]
    models = {"A": (cfg["weights_A"], {0: 0, 32: 1}, [0, 32], "yolov8n_COCO"),
              "B": (cfg["weights_B"], None, None, "yolov8n_finetuned")}
    for tag, (w, cm, cls, mname) in models.items():
        op = cfg[f"op_conf_{tag}"]
        sp = dict(gpu=speed(w, test, device=0), cpu=speed(w, test, device="cpu", n=40))
        for dname, imgs in [("test", test), ("rf100", rf)]:
            per = run_preds(w, imgs, class_map=cm, classes=cls)
            out = dict(run=f"{tag}_day3_{dname}", dataset=dname, n_images=len(imgs), op_conf=op, metrics=metrics(per, op_conf=op),
                       ball_recall_by_size=ball_recall_by_size(per, op["bola"], cfg["ball_size_threshold_norm_area"]))
            if dname == "test":
                bigset = set(big)
                out["subset_ge640"] = dict(n_images=len(big), metrics=metrics([d for d in per if d["img"] in bigset], op_conf=op))
                out["bootstrap"] = bootstrap(per, op)
                out["speed"] = sp
            json.dump(out, open(R / f"runs/{out['run']}.json", "w"), indent=1)
            m = out["metrics"]
            param = (f"Hari3 {dname} {len(imgs)} img; conf beku per kelas dari val {json.dumps({k: (round(v, 3) if v else v) for k, v in op.items()})}; "
                     f"imgsz640; mAP=AP50 rata-rata {m['_mean']['n_classes']} kelas" + ("; wasit tidak tersedia" if tag == "A" else ""))
            summary_row(out["run"], mname, param, m, sp if dname == "test" else None)
            print(out["run"], json.dumps({k: {x: v.get(x) for x in ("ap50", "precision", "recall")} for k, v in m.items() if k != "_mean"}), m["_mean"])
    # confusion matrix resmi Ultralytics untuk B (test & rf100)
    for dname, y in [("test", "football_detection/data/work/splits/data_test.yaml"), ("rf100", "football_detection/data/work/rf100_yolo/data_rf100.yaml")]:
        YOLO(cfg["weights_B"]).val(data=str(pathlib.Path(y).resolve()), split="val", imgsz=640, batch=8, device=0, plots=True, workers=2,
                                   project=str((R / "val_plots").resolve()), name=f"B_day3_{dname}")
    print("DAY3_EVAL_DONE")


if __name__ == "__main__":
    main()
