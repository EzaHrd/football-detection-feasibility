"""Buat semua gambar laporan UTS dari berkas hasil (tidak ada angka yang diketik tangan)."""
import sys, json, random, pathlib, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw
sys.path.insert(0, "football_detection/scripts")
from eval_lib import run_preds, read_gt

R = pathlib.Path("football_detection/results"); F = pathlib.Path("football_detection/report/figures"); F.mkdir(parents=True, exist_ok=True)
J = lambda n: json.load(open(R / "runs" / n))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = {"pemain": "#2e7d32", "bola": "#c62828", "wasit": "#f9a825", "A": "#9e9e9e", "B": "#1565c0"}
CLS = ["pemain", "bola", "wasit"]; EN = {"pemain": "Player", "bola": "Ball", "wasit": "Referee"}


def fig_dist():
    chk = json.load(open(R / "day2_data_check.json")); tc = json.load(open(R / "day3_test_check.json")); rf = json.load(open(R / "day3_rf100_convert.json"))
    d = {"Train": [chk["stats"]["train"][k] for k in ("player", "ball", "referee")], "Validation": [chk["stats"]["val"][k] for k in ("player", "ball", "referee")],
         "Test (internal)": [tc["player"], tc["ball"], tc["referee"]], "RF100 (external)": [rf["pemain"], rf["bola"], rf["wasit"]]}
    fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(d)); w = 0.26
    for i, c in enumerate(CLS):
        v = [d[k][i] for k in d]; b = ax.bar(x + (i - 1) * w, v, w, label=EN[c], color=C[c])
        for r, val in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, val * 1.08, f"{val:,}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(d.keys()); ax.set_ylabel("Bounding boxes (log scale)"); ax.legend(frameon=False)
    ax.set_title("Annotated objects per class and split"); fig.tight_layout(); fig.savefig(F / "fig_class_distribution.png"); plt.close(fig)


def box(ax, x, y, w, h, t, col, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=col, ec="#333", lw=0.8)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def fig_arch():
    fig, ax = plt.subplots(figsize=(9, 4.4)); ax.set_xlim(0, 20); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.2, 4.2, 1.8, 1.6, "Input\n640x640x3", "#eeeeee")
    bb = ["Conv 3x3 s2\n(16)", "Conv 3x3 s2\n(32) + C2f", "Conv 3x3 s2\n(64) + C2f -> P3", "Conv 3x3 s2\n(128) + C2f -> P4", "Conv 3x3 s2\n(256) + C2f\n+ SPPF -> P5"]
    for i, t in enumerate(bb): box(ax, 2.6, 8.3 - i * 1.7, 3.2, 1.3, t, "#bbdefb", 7)
    ax.text(4.2, 9.85, "Backbone (CSPDarknet-style, SiLU)", ha="center", fontsize=8, weight="bold")
    nk = ["Upsample(P5) + Concat(P4)\n+ C2f", "Upsample + Concat(P3)\n+ C2f  ->  N3", "Conv s2 + Concat\n+ C2f  ->  N4", "Conv s2 + Concat(P5 path)\n+ C2f  ->  N5"]
    for i, t in enumerate(nk): box(ax, 7.4, 7.6 - i * 1.9, 4.4, 1.4, t, "#c8e6c9", 7)
    ax.text(9.6, 9.3, "Neck (PAN-FPN)", ha="center", fontsize=8, weight="bold")
    for i, (t, s) in enumerate([("Detect head\n80x80 (stride 8)", "small objects"), ("Detect head\n40x40 (stride 16)", ""), ("Detect head\n20x20 (stride 32)", "large objects")]):
        box(ax, 12.8, 7.4 - i * 2.4, 3.3, 1.5, t, "#ffe0b2", 7)
    ax.text(14.45, 9.3, "Decoupled anchor-free head\n(box: DFL reg_max=16, class: 3 logits)", ha="center", fontsize=7, weight="bold")
    box(ax, 17.0, 4.2, 2.8, 1.6, "NMS (IoU 0.6)\n-> player / ball /\nreferee boxes", "#f8bbd0", 7)
    arr = lambda x0, y0, x1, y1, ls="-": ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.8, ls=ls))
    arr(2.0, 5.0, 2.6, 8.9)                                             # input -> first conv
    for i in range(4): arr(4.2, 8.3 - i * 1.7, 4.2, 7.9 - i * 1.7)       # backbone chain
    arr(5.8, 2.15, 7.4, 8.4)                                            # P5 -> neck 0
    arr(5.8, 3.85, 7.4, 8.1)                                            # P4 -> neck 0 (concat)
    arr(5.8, 5.55, 7.4, 6.5)                                            # P3 -> neck 1 (concat)
    arr(5.8, 1.9, 7.4, 2.6, "--")                                       # P5 path -> neck 3
    for i in range(3): arr(9.6, 7.6 - i * 1.9, 9.6, 7.1 - i * 1.9)       # neck chain
    arr(11.8, 6.4, 12.8, 8.1); arr(11.8, 4.5, 12.8, 5.7); arr(11.8, 2.6, 12.8, 3.3)  # N3,N4,N5 -> heads
    arr(16.1, 8.1, 17.0, 5.6); arr(16.1, 5.7, 17.0, 5.1); arr(16.1, 3.3, 17.0, 4.4)  # heads -> NMS
    fig.tight_layout(); fig.savefig(F / "fig_architecture.png"); plt.close(fig)


def fig_curves():
    df = pd.read_csv(R / "train/B_yolov8n_e25_s42_b8/results.csv"); df.columns = [c.strip() for c in df.columns]; e = df["epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for k, l in [("train/box_loss", "train box"), ("train/cls_loss", "train cls"), ("train/dfl_loss", "train dfl")]: ax[0].plot(e, df[k], label=l)
    for k, l in [("val/box_loss", "val box"), ("val/cls_loss", "val cls"), ("val/dfl_loss", "val dfl")]: ax[0].plot(e, df[k], "--", label=l)
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(fontsize=6, frameon=False, ncol=2)
    ax[1].plot(e, df["metrics/mAP50(B)"], label="mAP@0.5"); ax[1].plot(e, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95"); ax[1].set_title("Validation mAP"); ax[1].set_xlabel("Epoch"); ax[1].legend(frameon=False)
    ax[2].plot(e, df["metrics/precision(B)"], label="precision"); ax[2].plot(e, df["metrics/recall(B)"], label="recall"); ax[2].set_title("Validation precision / recall (all classes)"); ax[2].set_xlabel("Epoch"); ax[2].legend(frameon=False)
    ax[0].axvline(11.5, color="#999", lw=0.6, ls=":"); ax[1].axvline(11.5, color="#999", lw=0.6, ls=":")
    ax[1].text(11.7, ax[1].get_ylim()[0] + 0.02, "resume after crash", fontsize=6, color="#666")
    fig.tight_layout(); fig.savefig(F / "fig_training_curves.png"); plt.close(fig)


def fig_ab():
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for a, (title, na, nb) in zip(ax, [("Validation (797 img)", "A_day3_val_opconf.json", "B_day3_val_opconf.json"), ("Internal test (2,288 img)", "A_day3_test.json", "B_day3_test.json"), ("External RF100 (163 img)", "A_day3_rf100.json", "B_day3_rf100.json")]):
        A, B = J(na)["metrics"], J(nb)["metrics"]; x = np.arange(3); w = 0.36
        va = [A[c]["ap50"] or 0 for c in CLS]; vb = [B[c]["ap50"] or 0 for c in CLS]
        a1 = a.bar(x - w / 2, va, w, color=C["A"], label="A: pretrained COCO"); a2 = a.bar(x + w / 2, vb, w, color=C["B"], label="B: fine-tuned")
        for r, v, c in zip(a1, va, CLS): a.text(r.get_x() + r.get_width() / 2, v + 0.01, "n/a" if c == "wasit" else f"{v:.2f}", ha="center", fontsize=7)
        for r, v in zip(a2, vb): a.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        a.set_xticks(x); a.set_xticklabels([EN[c] for c in CLS]); a.set_title(title, fontsize=9); a.set_ylim(0, 1.08)
    ax[0].set_ylabel("AP@0.5"); ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(F / "fig_baseline_vs_finetuned.png"); plt.close(fig)


def fig_gap():
    sets = [("Train subset\n(1,000 img)", "B_report_train_subset.json"), ("Validation\n(797)", "B_day3_val_opconf.json"), ("Internal test\n(2,288)", "B_day3_test.json"), ("RF100\n(163)", "B_day3_rf100.json")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); x = np.arange(len(sets)); w = 0.26
    for i, c in enumerate(CLS):
        v = [J(n)["metrics"][c]["ap50"] for _, n in sets]; b = ax.bar(x + (i - 1) * w, v, w, color=C[c], label=EN[c])
        for r, val in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in sets]); ax.set_ylabel("AP@0.5"); ax.set_ylim(0, 1.1); ax.legend(frameon=False, ncol=3)
    ax.set_title("Model B: AP@0.5 on seen vs. unseen data"); fig.tight_layout(); fig.savefig(F / "fig_generalization_gap.png"); plt.close(fig)


def fig_robust():
    df = pd.read_csv(R / "day3_robustness.csv"); fig, ax = plt.subplots(figsize=(8, 3.8))
    col = ["#1565c0" if g == "bersih" else "#c62828" if t > 0.15 else "#78909c" for g, t in zip(df.gangguan, df.turun_relatif_map50)]
    lab = [g.replace("bersih", "clean").replace("gauss_sigma", "blur s").replace("motion_kernel", "motion k").replace("down_up", "res x").replace("brightness", "bright").replace("contrast", "contrast").replace("jpeg_quality", "JPEG q").replace("gauss_noise_sigma", "noise s") for g in df.gangguan]
    b = ax.bar(range(len(df)), df.map50, color=col)
    for r, v in zip(b, df.map50): ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=7); ax.set_ylabel("mAP@0.5 (3 classes)")
    ax.set_title("Robustness of Model B on the internal test set (red = drop > 15%)"); fig.tight_layout(); fig.savefig(F / "fig_robustness.png"); plt.close(fig)


def fig_pred():
    cfg = json.load(open(R / "day3_frozen_config.json")); op = cfg["op_conf_B"]
    test = [l.strip() for l in open("football_detection/data/work/splits/test.txt") if l.strip()]
    cand = []
    for p in test:
        gb, gc, (w, h), _ = read_gt(p)
        if min(w, h) >= 640 and (gc == 1).any(): cand.append(p)
    sel = random.Random(42).sample(cand, 3); per = run_preds(cfg["weights_B"], sel); colr = {0: "#43a047", 1: "#e53935", 2: "#fdd835"}
    fig, ax = plt.subplots(3, 2, figsize=(8.2, 11))
    for i, (p, d) in enumerate(zip(sel, per)):
        im = Image.open(p).convert("RGB"); g = im.copy(); q = im.copy(); dg = ImageDraw.Draw(g); dq = ImageDraw.Draw(q); lw = max(3, im.width // 300)
        for b, c in zip(d["gb"], d["tc"]): dg.rectangle(list(b), outline=colr[int(c)], width=lw)
        thr = np.array([op["pemain"], op["bola"], op["wasit"]]); ball_hit = False
        for b, c, s in zip(d["pb"], d["pc"], d["conf"]):
            if s >= thr[int(c)]: dq.rectangle(list(b), outline=colr[int(c)], width=lw)
        bb = d["gb"][d["tc"] == 1][0]; cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; half = max(160, im.width // 8)
        x0 = int(min(max(cx - half, 0), im.width - 2 * half)); y0 = int(min(max(cy - half, 0), im.height - 2 * half)); box_ = (x0, y0, x0 + 2 * half, y0 + 2 * half)
        ax[i, 0].imshow(g.crop(box_)); ax[i, 1].imshow(q.crop(box_))
        ax[i, 0].set_title(f"Ground truth (crop around the ball): {pathlib.Path(p).name}", fontsize=7); ax[i, 1].set_title("Model B prediction (frozen thresholds)", fontsize=7)
        for a in ax[i]: a.axis("off")
    fig.suptitle("Random internal-test images containing a ball (seed 42), cropped. Green = player, red = ball, yellow = referee", fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fig_predictions.png"); plt.close(fig); json.dump(sel, open(F / "fig_predictions_images.json", "w"), indent=1)


def copy_ultra():
    base = R / "train/B_yolov8n_e25_s42_b8"
    for src, dst in [(base / "labels.jpg", "ultra_labels_distribution.jpg"), (base / "train_batch0.jpg", "ultra_train_batch_mosaic.jpg"),
                     (R / "val_plots/B_day3_test/confusion_matrix_normalized.png", "ultra_cm_test.png"), (R / "val_plots/B_day3_test/BoxPR_curve.png", "ultra_pr_test.png"),
                     (R / "val_plots/B_day3_rf100/confusion_matrix_normalized.png", "ultra_cm_rf100.png"),
                     (R / "val_plots/B_yolov8n_e25_val/confusion_matrix_normalized.png", "ultra_cm_val.png")]:
        shutil.copy(src, F / dst)


if __name__ == "__main__":
    for f in (fig_dist, fig_arch, fig_curves, fig_ab, fig_gap, fig_robust, fig_pred, copy_ultra): f(); print("ok", f.__name__)
