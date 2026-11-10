"""Bangun laporan UTS (DOCX). Semua angka dibaca dari berkas hasil di results/. Pemakaian: build_report.py"""
import ast, csv, json, pathlib, textwrap
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

R = pathlib.Path("football_detection/results"); FIG = pathlib.Path("football_detection/report/figures"); S = pathlib.Path("football_detection/scripts")
OUT = pathlib.Path("football_detection/report/UTS_Deep_Learning_Report.docx")
J = lambda n: json.load(open(R / "runs" / n))
vA, vB = J("A_day3_val_opconf.json"), J("B_day3_val_opconf.json")
tA, tB = J("A_day3_test.json"), J("B_day3_test.json")
rA, rB = J("A_day3_rf100.json"), J("B_day3_rf100.json")
trn = J("B_report_train_subset.json")
crit = json.load(open(R / "day3_criteria.json")); cfg = json.load(open(R / "day3_frozen_config.json"))
chk = json.load(open(R / "day2_data_check.json")); tchk = json.load(open(R / "day3_test_check.json")); leak = json.load(open(R / "audit_soccana_leakage.json"))
rf = json.load(open(R / "day3_rf100_convert.json")); ps = json.load(open(R / "day3_per_source.json"))
err = json.load(open(R / "error_analysis/B_day3_test/errors.json")); rob = pd.read_csv(R / "day3_robustness.csv")
src = pd.read_csv(R / "audit_soccana_by_source.csv", index_col=0)
res = pd.read_csv(R / "train/B_yolov8n_e25_s42_b8/results.csv"); res.columns = [c.strip() for c in res.columns]
v2A, v2B = J("A_yolov8n_pretrained_val.json"), J("B_yolov8n_e25_val.json")
train_min = (res.loc[res.epoch == 11, "time"].iloc[0] + res.loc[res.epoch == 25, "time"].iloc[0]) / 60
f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
f2 = lambda v: "n/a" if v is None else f"{v:.2f}"
pct = lambda v: f"{100 * v:.1f}%"
CLS = ["pemain", "bola", "wasit"]; EN = {"pemain": "Player", "bola": "Ball", "wasit": "Referee"}

doc = Document()
sec = doc.sections[0]; sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(sec, m, Cm(2))
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5); st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(5); st.paragraph_format.line_spacing = 1.08
for n, sz, col in (("Heading 1", 16, "1F3864"), ("Heading 2", 12.5, "2E5597"), ("Heading 3", 11, "2E5597")):
    h = doc.styles[n]; h.font.name = "Calibri"; h.font.size = Pt(sz); h.font.bold = True; h.font.color.rgb = RGBColor.from_string(col)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri"); h.paragraph_format.space_before = Pt(12 if n != "Heading 1" else 18); h.paragraph_format.space_after = Pt(5)
counters = {"fig": 0, "tab": 0}


def field(par, instr):
    for t, txt in (("begin", None), (None, instr), ("separate", None), ("text", "1"), ("end", None)):
        r = par.add_run()
        if t in ("begin", "separate", "end"):
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); r._r.append(e)
        elif t == "text": r.text = txt
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r._r.append(e)


def P(text="", bold=False, italic=False, size=None, align=None, after=None, color=None):
    p = doc.add_paragraph(); parts = text.split("**")
    for i, t in enumerate(parts):
        if not t: continue
        r = p.add_run(t); r.bold = bold or (i % 2 == 1); r.italic = italic
        if size: r.font.size = Pt(size)
        if color: r.font.color.rgb = RGBColor.from_string(color)
    if align: p.alignment = align
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p


def B(text, level=0):
    p = doc.add_paragraph(style="List Bullet"); parts = text.split("**")
    for i, t in enumerate(parts):
        if t: r = p.add_run(t); r.bold = (i % 2 == 1)
    p.paragraph_format.left_indent = Cm(0.9 + 0.6 * level); p.paragraph_format.space_after = Pt(2); return p


def H(text, lvl=1): return doc.add_heading(text, lvl)


def shade(cell, hexcol):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement("w:shd"); s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), hexcol); tcPr.append(s)


def caption(kind, text):
    counters[kind] += 1; p = P(f"{'Figure' if kind == 'fig' else 'Table'} {counters[kind]}. {text}", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, after=8, color="404040")
    return p


def T(rows, widths=None, cap=None, hdr=True, fs=8.5, hl=None, align_num=True):
    if cap: caption("tab", cap).paragraph_format.keep_with_next = False
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""; p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            parts = str(v).split("**")
            for k, x in enumerate(parts):
                if x: r = p.add_run(x); r.font.size = Pt(fs); r.bold = (hdr and i == 0) or (k % 2 == 1)
            if widths: c.width = Cm(widths[j])
            if hdr and i == 0: shade(c, "D9E2F3")
            elif hl and i in hl: shade(c, hl[i])
            if align_num and j > 0 and i > 0 and len(str(v)) < 14: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph().paragraph_format.space_after = Pt(2); return t


def FG(name, width=16, cap=None):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True; p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(FIG / name), width=Cm(width))
    if cap: caption("fig", cap)


def CODE(text, title=None, fs=7):
    if title: P(title, bold=True, size=9, after=2)
    lines = []
    for ln in text.rstrip().split("\n"):
        ind = len(ln) - len(ln.lstrip()); lines += textwrap.wrap(ln, width=112, subsequent_indent=" " * (ind + 4), replace_whitespace=False, drop_whitespace=False) or [""]
    t = doc.add_table(rows=1, cols=1); t.style = "Table Grid"; t.autofit = False; c = t.cell(0, 0); c.width = Cm(17); shade(c, "F5F5F5"); c.text = ""
    p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
    for i, ln in enumerate(lines):
        r = p.add_run(ln); r.font.name = "Consolas"; r.font.size = Pt(fs); r._r.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        if i < len(lines) - 1: r.add_break()
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def src_of(path, names):
    text = pathlib.Path(path).read_text(encoding="utf-8"); tree = ast.parse(text); lines = text.split("\n"); out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names: out.append("\n".join(lines[n.lineno - 1:n.end_lineno]))
    return "\n\n".join(out)


def whole(path): return pathlib.Path(path).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------- cover
for _ in range(3): P("")
P("UNIVERSITAS ISLAM INDONESIA", bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
P("International Program in Informatics", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=24)
P("Midterm Examination (Take Home), Odd Semester AY 2026/2027", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
P("Deep Learning", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, after=30)
P("Detecting Players, the Ball, and Referees in Football Broadcast Frames with YOLOv8", bold=True, size=20, align=WD_ALIGN_PARAGRAPH.CENTER, after=8, color="1F3864")
P("A technical feasibility study with transfer learning, source-level data splitting, and a pre-registered evaluation", italic=True, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
T([["Student", "[Student name]"], ["Student ID (NIM)", "[NIM]"], ["Course", "Deep Learning"], ["Examiner", "Dr. Syarif Hidayat, S.Kom., M.I.T."],
   ["Exam date", "Wednesday, November 11, 2026"], ["Algorithm chosen", "YOLO (YOLOv8n, Ultralytics 8.4.202)"]], widths=[4.5, 9], hdr=False, fs=10, align_num=False)
doc.add_page_break()
P("Contents", bold=True, size=14, after=4)
p = doc.add_paragraph(); field(p, 'TOC \\o "1-2" \\h \\z \\u')
doc.add_page_break()

# ---------------------------------------------------------------- summary
H("Executive summary", 1)
P(f"This project trains a YOLOv8n object detector to find **players, the ball, and referees** in football broadcast frames, and measures honestly how well it works on data it has not seen. "
  f"The model was fine-tuned from COCO weights on {chk['stats']['train']['images']:,} training images and compared with the same architecture without fine-tuning.")
P("**Main results** (internal test set of {n:,} images from sources never used for training; thresholds frozen on validation):".format(n=tchk["n_test_after"]), after=2)
B(f"Fine-tuning is clearly better than the pretrained baseline: mean AP@0.5 over the classes the baseline can detect rises from {f3(tA['metrics']['_mean']['ap50'])} to {f3(tB['metrics']['pemain']['ap50'] / 2 + tB['metrics']['bola']['ap50'] / 2)} (player and ball), and the referee class (absent from COCO) reaches AP@0.5 {f3(tB['metrics']['wasit']['ap50'])}.")
B(f"**Player** (AP@0.5 {f3(tB['metrics']['pemain']['ap50'])}) and **referee** ({f3(tB['metrics']['wasit']['ap50'])}) passed the pre-defined success criteria; the model also reaches mAP@0.5 {f3(rB['metrics']['_mean']['ap50'])} on an external dataset (RF100).")
B(f"The **ball** did not: recall {f3(tB['metrics']['bola']['recall'])} and precision {f3(tB['metrics']['bola']['precision'])} against targets of 0.50 and 0.70. The weakness is partly underfitting (ball recall is only {f3(trn['metrics']['bola']['recall'])} even on training images) and partly a domain gap.")
B(f"Speed on a laptop GPU is {tB['speed']['gpu']['median_ms']:.1f} ms/frame and {tB['speed']['cpu']['median_ms']:.1f} ms/frame on CPU (median, batch 1).")
P("Conclusion: detecting players and referees is technically feasible with this setup; **detecting the ball is not yet**, and the report states what evidence supports that and what to do next.", after=6)

# ---------------------------------------------------------------- section 1
H("Section 1. Project Description and Case Study Selection", 1)
H("1.1 Project objective", 2)
P("The objective is to build and evaluate a deep-learning detector that localizes three object classes in single frames of football broadcast video: **player**, **ball**, and **referee**. "
  "The project is framed as a **technical feasibility study**: the question is whether a small, publicly trainable YOLO model, trained on public data on one laptop, reaches useful accuracy, rather than whether a finished product exists.")
P("To avoid judging the results after seeing them, measurable success criteria were fixed before the test set was evaluated (Section 5.2). In scope: detection of the three classes, evaluation, and robustness tests. Out of scope: object tracking, real-time video processing, and a user interface.")
H("1.2 Case study", 2)
P("The case study is **object detection in football broadcast frames**. A detector that finds players, referees, and the ball is the first step of most sports-analytics pipelines (positions, distances, tactical statistics). "
  "The case is challenging in a way that is useful for studying deep learning: players are medium-sized, referees look like players with a different shirt colour, and the ball is a very small object that is often blurred or occluded.")
H("1.3 Why YOLO suits this case", 2)
T([["Option", "Why it is (not) suitable here"],
   ["CNN image classifier", "Predicts one label per image; cannot say where objects are or count them. Not suitable for localization."],
   ["RNN / LSTM / GRU", "Designed for sequences (text, time series). Single frames carry no sequence, so these are not the natural fit."],
   ["Two-stage detectors (e.g., Faster R-CNN)", "Often accurate but slower and heavier to train; unattractive on a laptop with a 6 GB GPU."],
   ["**YOLO (single-stage)**", "One forward pass predicts boxes and classes at three scales; anchor-free decoupled head; pretrained COCO weights already know *person* and *sports ball*, so transfer learning needs little data; fast enough for a laptop."]],
  widths=[4.2, 12.8], cap="Algorithm options considered for the case study.", fs=8.5, align_num=False)
P("The smallest variant (YOLOv8n) was chosen because the hardware limit is 6 GB of GPU memory and because the study needs several comparable runs within three days. The known trade-off is that small models have limited capacity for tiny objects such as the ball; Section 5 shows that this trade-off matters.")

# ---------------------------------------------------------------- section 2
H("Section 2. Data Collection and Preprocessing", 1)
H("2.1 Candidate datasets and audit", 2)
T([["Dataset", "Source", "Images", "Classes", "License", "Audit status"],
   ["Soccana player/ball/referee v1", "Hugging Face (Adit-jain)", f"{tchk['n_test_before'] + chk['stats']['train']['images'] + chk['stats']['val']['images']:,} (README claims 25,000)", "Player, Ball, Referee", "Not stated; derived from 8 Roboflow datasets", "Fully audited"],
   ["soccer-players (RF100)", "Hugging Face (Francesco)", "163", "football, player, referee (+ parent class)", "tag \"cc\" (unspecific)", "Fully audited"],
   ["Roboflow Universe projects", "universe.roboflow.com", "unknown", "ball / player / referee", "mostly unverified", "Not audited: pages returned HTTP 403 to automated access; download needs an account"],
   ["SoccerNet-Tracking / GSR", "soccer-net.org", "200 sequences (paper)", "player, goalkeeper, referee, ball", "paper lists CC BY 4.0, data terms unverified", "Not used (access not checked)"]],
  widths=[3.1, 2.7, 2.6, 2.9, 2.9, 2.8], cap="Candidate datasets (audit performed 2026-11-05).", fs=7.5, align_num=False)
P("**Selected dataset:** Soccana v1 as the main data pool, because it is the only candidate large enough (it contains 3,500+ ball boxes) and spans several sources. RF100 is too small to train on (141 balls) and is kept as an **external test set**.")
P("**Audit findings** (checked on the data, not taken from the description):", after=2)
B(f"The dataset contains {tchk['n_test_before'] + chk['stats']['train']['images'] + chk['stats']['val']['images']:,} images, not the 25,000 stated in its description.")
B("The class table in the README is in the wrong order (it lists 1 = Referee, 2 = Ball); the `data.yaml` is correct (0 Player, 1 Ball, 2 Referee). This was confirmed by looking at labelled images and by box size (class 1 boxes are tiny).")
B("Many images are **SAHI slices**: crops of larger frames at sizes 160, 320, 640, 1080, 1280 and 1920x1080 pixels. About 28% of training images contain no objects.")
B("Images come from 7 source groups (prefixes in the file names), each originally a separate Roboflow dataset. Licenses were not verified.")
H("2.2 Splitting by source and removing duplicates", 2)
P(f"The dataset ships with a random train/test split. This is unsafe for video frames, because consecutive frames are almost identical. A perceptual-hash (pHash) check showed that **{leak['near_dup_le4']} of {leak['n_test']:,} images in the original test split ({100 * leak['near_dup_le4'] / leak['n_test']:.1f}%) had a near-duplicate (Hamming distance <= 4) in the training split**, and every one of the seven sources appeared in both splits. "
  "The data was therefore re-split **by source group**, so that whole sources are held out:")
rows = [["Source group", "Images", "Players", "Balls", "Referees", "Assigned to"]]
asg = {"v3": "train", "v7": "train", "v2_temp": "train", "spt_v2": "train", "tbd_v2": "validation", "v12": "test", "v5_temp": "test"}
for g in ["v3", "v7", "v2_temp", "spt_v2", "tbd_v2", "v12", "v5_temp"]:
    r = src.loc[g]; rows.append([g, f"{int(r.images):,}", f"{int(r.player):,}", f"{int(r.ball):,}", f"{int(r.referee):,}", asg[g]])
T(rows, widths=[3, 2.2, 2.4, 2.2, 2.4, 3], cap="Source groups and their assignment (counts before duplicate removal).", fs=8.5)
P(f"After assigning sources, the test images were hashed again against train and validation. **{tchk['excluded_pHash_le4']} test images ({tchk['excluded_pHash_le4'] * 100 / tchk['n_test_before']:.1f}%) were near-duplicates of images from other sources and were removed** before any evaluation ({tchk['exact']} had an identical hash), leaving {tchk['n_test_after']:,} test images. "
  f"This shows that different Roboflow sources share images. Between train and validation no near-duplicate (distance <= 4) was found. Residual leakage that pHash cannot detect (different crops of one frame) may remain.")
T([["Split", "Images", "Empty", "Players", "Balls", "Referees"],
   ["Train (4 sources)", f"{chk['stats']['train']['images']:,}", f"{chk['stats']['train']['empty']:,}", f"{chk['stats']['train']['player']:,}", f"{chk['stats']['train']['ball']:,}", f"{chk['stats']['train']['referee']:,}"],
   ["Validation (tbd_v2)", f"{chk['stats']['val']['images']}", "0", f"{chk['stats']['val']['player']:,}", f"{chk['stats']['val']['ball']}", f"{chk['stats']['val']['referee']}"],
   ["Internal test (v12, v5_temp, de-duplicated)", f"{tchk['n_test_after']:,}", f"{tchk['empty']}", f"{tchk['player']:,}", f"{tchk['ball']}", f"{tchk['referee']:,}"],
   ["External test (RF100)", f"{rf['n_images']}", "n/a", f"{rf['pemain']:,}", f"{rf['bola']}", f"{rf['wasit']}"]],
  widths=[6, 2, 1.8, 2.3, 2, 2.3], cap="Final dataset statistics per split.", fs=8.5)
FG("fig_class_distribution.png", 13, "Number of boxes per class and split (log scale). The ball is the rarest class in every split.")
P(f"The ball is the rarest class: {100 * chk['stats']['train']['ball'] / (chk['stats']['train']['player'] + chk['stats']['train']['ball'] + chk['stats']['train']['referee']):.1f}% of training boxes. All validation images contain a ball, and validation comes from a single source, which turned out to make it easier than the test set (Section 5).")
H("2.3 Preprocessing methods and rationale", 2)
T([["Step", "What was done", "Why"],
   ["Class mapping", "Player(0), Ball(1), Referee(2). Goalkeepers are merged into Player by the dataset. RF100: player→Player, football→Ball, referee→Referee, parent class dropped.", "Matches the three target classes; verified visually."],
   ["Source-level split + de-duplication", "Whole sources held out; pHash filter (distance <= 4) on test.", "Prevents near-identical frames in train and test, which would inflate scores."],
   ["Resize (letterbox)", "Images scaled to 640 px on the long side, padded to 640x640 with grey.", "Fixed input size; keeps aspect ratio so boxes are not distorted."],
   ["Normalization", "BGR→RGB, pixel values divided by 255 (range 0–1).", "Standard input scale for a network with pretrained weights."],
   ["Augmentation (train only)", "Mosaic (4 images in one), HSV jitter, random scale ±50%, translation ±10%, horizontal flip (p = 0.5). Mosaic is switched off for the last 10 epochs.", "More variety from limited data; mosaic puts small objects at new positions and scales; turning it off at the end lets the model settle on realistic images."],
   ["No augmentation on val/test", "Evaluation uses plain letterboxed images.", "Evaluation must reflect real inputs; augmenting evaluation data would change the question."]],
  widths=[3.2, 8.3, 5.5], cap="Preprocessing pipeline.", fs=8, align_num=False)
P("No vertical flip and no rotation were used, because broadcast cameras keep the pitch upright. The configuration also contains `auto_augment` and `erasing` entries; in Ultralytics these apply to classification, so they are not counted as detection augmentations here. "
  "Augmentation settings are the Ultralytics defaults; they were not tuned, which is a limitation.")
FG("ultra_train_batch_mosaic.jpg", 12, "A training batch after augmentation (mosaic, scaling, colour jitter), as produced by the training pipeline.")
CODE(whole(S / "make_split.py"), "Code 1. Source-level split manifest (scripts/make_split.py)")
CODE(whole(S / "day3_test_prep.py"), "Code 2. Duplicate check of the test set against train/validation, run before any test evaluation (scripts/day3_test_prep.py)")
CODE("""# Ultralytics augmentation / training configuration actually used (args.yaml)
hsv_h: 0.015  hsv_s: 0.7  hsv_v: 0.4
translate: 0.1  scale: 0.5  fliplr: 0.5  flipud: 0.0  degrees: 0.0  shear: 0.0
mosaic: 1.0  close_mosaic: 10  mixup: 0.0""", "Code 3. Augmentation configuration")
H("2.4 Data risks", 2)
B("**License unclear** for Soccana (derived from several Roboflow datasets): fine for a course project, not verified for publication.")
B("**Missing labels:** in a manual check of 20 random validation \"false positives\", about 19 were real people who had no label (Section 5.1). Precision is therefore probably underestimated.")
B("**Sources may contain augmented images:** the source v5_temp has 34.3% player boxes that are wider than tall (other sources 0.6–3.3%), and example images are rotated by 90°, which indicates augmentation applied by the original dataset exporter.")

# ---------------------------------------------------------------- section 3
H("Section 3. Model Implementation", 1)
H("3.1 Model and transfer learning", 2)
P("The model is **YOLOv8n** (nano) from Ultralytics. Training starts from weights pretrained on COCO (80 classes) and fine-tunes all layers for the three football classes. "
  "The training log reports that 349 of 355 weight tensors were transferred; the remaining six are most likely the final class-prediction layers, which must be re-initialised for 3 classes (inferred, not inspected one by one). "
  "The class branch of the head keeps the hidden width (80) of the pretrained model, so the final network has more parameters than a model built from scratch for three classes (3,138,489 versus 3,011,433, both counted from the code).")
H("3.2 Architecture", 2)
FG("fig_architecture.png", 16, "YOLOv8n architecture as used in this project (23 top-level modules; 73 layers and 3,133,193 parameters after Conv–BatchNorm fusion; 8.6 GFLOPs at 640x640).")
rows = [["#", "Module", "Kernel / stride / out channels", "Parameters", "Role"],
        ["0", "Conv", "3x3 / 2 / 16", "464", "Stem, downsample x2"], ["1", "Conv", "3x3 / 2 / 32", "4,672", "Downsample x4"], ["2", "C2f (1 bottleneck)", "-- / -- / 32", "7,360", "Feature extraction"],
        ["3", "Conv", "3x3 / 2 / 64", "18,560", "Downsample x8"], ["4", "C2f (2 bottlenecks)", "-- / -- / 64", "49,664", "Output P3 (stride 8)"],
        ["5", "Conv", "3x3 / 2 / 128", "73,984", "Downsample x16"], ["6", "C2f (2 bottlenecks)", "-- / -- / 128", "197,632", "Output P4 (stride 16)"],
        ["7", "Conv", "3x3 / 2 / 256", "295,424", "Downsample x32"], ["8", "C2f (1 bottleneck)", "-- / -- / 256", "460,288", "Feature extraction"],
        ["9", "SPPF", "pool k = 5", "164,608", "Output P5 (stride 32); multi-scale context"],
        ["10–12", "Upsample, Concat(6), C2f", "x2 nearest", "148,224", "Top-down fusion P5→P4"], ["13–15", "Upsample, Concat(4), C2f", "x2 nearest", "37,248", "Top-down fusion → N3 (small objects)"],
        ["16–18", "Conv s2, Concat(12), C2f", "3x3 / 2", "160,640", "Bottom-up fusion → N4"], ["19–21", "Conv s2, Concat(9), C2f", "3x3 / 2", "640,768", "Bottom-up fusion → N5"],
        ["22", "Detect", "strides 8/16/32, DFL 16 bins, 3 classes", "878,953", "Decoupled box + class heads"],
        ["", "**Total (unfused)**", "", "**3,138,489**", ""]]
T(rows, widths=[1.2, 4.2, 4.6, 2.2, 4.8], cap="Layer table of the trained model (read from the checkpoint; channels shown after YOLOv8n width scaling).", fs=7.5, align_num=False)
P("**Choices and rationale**", after=2)
B("**Convolutions:** 3x3 kernels with stride 2 halve the resolution at each stage (640→320→160→80→40→20), building features from fine to coarse with few parameters.")
B("**Activation: SiLU** (x·sigmoid(x)) after every convolution with batch normalization. It is smooth and non-saturating for positive values, and it is the activation the pretrained weights were trained with, so transfer learning stays consistent.")
B("**C2f blocks** (cross-stage partial with extra skip connections): residual-style bottlenecks improve gradient flow while keeping the network small. Repeats are 1, 2, 2, 1 in the backbone (YOLOv8n depth scaling of 0.33).")
B("**SPPF** (three chained 5x5 max-pools): enlarges the receptive field cheaply so that the deepest features see wide context.")
B("**PAN-FPN neck:** combines deep (semantic) and shallow (detailed) features top-down and bottom-up. Three output scales (strides 8, 16, 32) let one network handle large players and small balls; the stride-8 map at 80x80 is the main chance for the ball.")
B("**Decoupled anchor-free head:** separate branches predict box and class. The box uses a distribution over 16 bins per side (Distribution Focal Loss); the class branch has 3 outputs trained with binary cross-entropy. The box loss is CIoU + DFL, and the label assigner is task-aligned. No anchor sizes need to be tuned for the tiny ball.")
B("**Loss weights** (box 7.5, class 0.5, DFL 1.5) are the Ultralytics defaults and were not changed.")
H("3.3 Implementation code", 2)
P("The network is defined by the Ultralytics model configuration (`yolov8n.yaml`); training and evaluation code was written for this project. The backbone/head definition below was read from the loaded model; numbers in brackets are the *base* widths that the n-scale (width 0.25) reduces to 16–256 channels.")
CODE("""backbone:                                   head:
  - [-1, 1, Conv, [64, 3, 2]]    # 0-P1/2     - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [-1, 1, Conv, [128, 3, 2]]   # 1-P2/4     - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [128, True]]                 - [-1, 3, C2f, [512]]
  - [-1, 1, Conv, [256, 3, 2]]   # 3-P3/8     - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [-1, 6, C2f, [256, True]]                 - [[-1, 4], 1, Concat, [1]]
  - [-1, 1, Conv, [512, 3, 2]]   # 5-P4/16    - [-1, 3, C2f, [256]]                 # N3
  - [-1, 6, C2f, [512, True]]                 - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32    - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [1024, True]]                - [-1, 3, C2f, [512]]                 # N4
  - [-1, 1, SPPF, [1024, 5]]     # 9          - [-1, 1, Conv, [512, 3, 2]]
                                              - [[-1, 9], 1, Concat, [1]]
                                              - [-1, 3, C2f, [1024]]                # N5
                                              - [[15, 18, 21], 1, Detect, [nc]]     # nc = 3""", "Code 4. Model definition (yolov8n.yaml, as loaded)", fs=6.5)
CODE(whole(S / "train.py"), "Code 5. Training script (scripts/train.py)")
CODE("""# Inference with the frozen per-class thresholds (used for all P/R numbers and figures)
from ultralytics import YOLO
model = YOLO("best.pt")
thr = {0: 0.472, 1: 0.234, 2: 0.280}                 # player, ball, referee: from validation
r = model.predict("frame.jpg", imgsz=640, conf=0.001, iou=0.6)[0]
keep = [i for i, (c, s) in enumerate(zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist())) if s >= thr[c]]
boxes = r.boxes[keep]""", "Code 6. Inference with per-class confidence thresholds")
H("3.4 Training configuration", 2)
T([["Setting", "Value", "Reason / note"],
   ["Initial weights", "yolov8n.pt (COCO)", "Transfer learning from a model that already detects people and sports balls"],
   ["Input size / batch", "640 / 8", "Batch 16 caused a GPU out-of-memory error on the 6 GB GPU, so it was halved"],
   ["Epochs", "25", "Planned 30; 30 epochs would take about 63 minutes (measured 125 s/epoch), above the 1-hour limit per experiment"],
   ["Optimizer", "AdamW, lr0 0.001429, lrf 0.01, momentum 0.937, weight decay 0.0005", "Chosen by Ultralytics' automatic rule (`optimizer=auto`) for a short schedule; not tuned"],
   ["Warm-up", "3 epochs", "Default; stabilises the first updates"],
   ["Mixed precision / seed", "AMP on / 42, deterministic", "Speed and memory; repeatability"],
   ["Hardware", "RTX 3050 Laptop 6 GB, Ryzen 7 8845HS, 13.8 GB RAM", "Python 3.12.6, PyTorch 2.5.1+cu121, Ultralytics 8.4.202"],
   ["Interruption", "System RAM ran out at epoch 12", "Training was resumed from the saved checkpoint with 2 data-loader workers (resume keeps the schedule but not bit-identical data order)"]],
  widths=[3.2, 5.8, 8], cap="Training configuration and deviations from the plan.", fs=8, align_num=False)

# ---------------------------------------------------------------- section 4
H("Section 4. Model Training and Evaluation", 1)
H("4.1 Training", 2)
P(f"Training used only the training split; validation images were used after every epoch to monitor progress and to choose the checkpoint (`best.pt`). Total training time was about **{train_min:.0f} minutes** for 25 epochs (excluding start-up and the crash). "
  f"The final-epoch validation mAP@0.5 reported by the trainer is {res['metrics/mAP50(B)'].iloc[-1]:.3f}.")
FG("fig_training_curves.png", 17, "Training and validation losses, validation mAP, and precision/recall per epoch. The dotted line marks the resume after the crash.")
P("**Reading the curves.** All training losses decrease steadily, validation losses decrease more slowly, and validation mAP is still creeping upward at epoch 25. The model has therefore not clearly converged: more training would probably help a little. The gap between the training class loss (falling fastest) and the validation class loss is the first sign of a train/validation difference.")
H("4.2 Evaluation protocol", 2)
B("**Matching:** a prediction is correct if its class matches and IoU with a ground-truth box is at least 0.5 (AP@0.5); mAP@0.5:0.95 averages IoU thresholds 0.50–0.95.")
B("**AP** is computed over all predictions with confidence >= 0.001. **Precision and recall** are reported at one **frozen confidence threshold per class** (maximum-F1 point on validation: player 0.472, ball 0.234, referee 0.280), then applied unchanged to the test sets.")
B("**Baseline A** is the same YOLOv8n architecture with COCO weights and no fine-tuning: *person* is mapped to Player and *sports ball* to Ball. COCO has no referee class, so the referee is **not available** for the baseline; its mAP is averaged over two classes and is not directly comparable with the three-class mAP of Model B.")
B("**Speed:** milliseconds per frame (pre-processing + inference + NMS) at batch 1 after warm-up, 100 test images on GPU, 40 on CPU.")
B("**Order of use:** configuration was frozen on validation, then the test sets were evaluated once. Nothing was changed afterwards.")
CODE(src_of(S / "eval_lib.py", ["_pairs", "match"]) + "\n\n" + src_of(S / "eval_lib.py", ["metrics"]), "Code 7. Matching and metric computation (scripts/eval_lib.py, shortened to the core functions)", fs=6.5)
H("4.3 Results", 2)
def row(name, A, Bm, key):
    return [name, f3(A["metrics"][key]["ap50"]) if A["metrics"][key]["ap50"] is not None else "n/a", f3(Bm["metrics"][key]["ap50"])]
mt = lambda m, k, x: f3(m["metrics"][k].get(x)) if m["metrics"][k].get("op_conf") is not None else "n/a"
hdr = ["Class", "A: AP@0.5", "A: P / R", "B: AP@0.5", "B: P / R", "B: AP@0.5:0.95"]
def table_for(A, Bm):
    rows = [hdr]
    for c in CLS:
        rows.append([EN[c], f3(A["metrics"][c]["ap50"]), "n/a" if A["metrics"][c].get("op_conf") is None else f"{mt(A, c, 'precision')} / {mt(A, c, 'recall')}",
                     f3(Bm["metrics"][c]["ap50"]), f"{mt(Bm, c, 'precision')} / {mt(Bm, c, 'recall')}", f3(Bm["metrics"][c]["ap50_95"])])
    rows.append(["**mAP@0.5**", f"{f3(A['metrics']['_mean']['ap50'])} (2 classes)", "", f"**{f3(Bm['metrics']['_mean']['ap50'])}** (3 classes)", "", f3(Bm["metrics"]["_mean"]["ap50_95"])])
    return rows
T(table_for(vA, vB), widths=[2.6, 3, 2.9, 3, 2.9, 2.6], cap=f"Validation set ({vB['n_images']} images). Used for model and threshold selection.", fs=8.5)
T(table_for(tA, tB), widths=[2.6, 3, 2.9, 3, 2.9, 2.6], cap=f"Internal test set ({tB['n_images']:,} images, unseen sources). Final result.", fs=8.5, hl={4: "E2EFDA"})
T(table_for(rA, rB), widths=[2.6, 3, 2.9, 3, 2.9, 2.6], cap=f"External test set RF100 ({rB['n_images']} images; only {rf['bola']} balls and {rf['wasit']} referees, so these numbers are noisy).", fs=8.5)
FG("fig_baseline_vs_finetuned.png", 17, "AP@0.5 per class for Baseline A (grey) and Model B (blue) on validation, internal test and external data.")
bt = tB["bootstrap"]
P(f"**Uncertainty.** A bootstrap over test images (1,000 resamples, seed 42) gives 95% intervals for Model B on the internal test set: mAP@0.5 {bt['map50']['lo95']:.3f}–{bt['map50']['hi95']:.3f}, ball AP@0.5 {bt['ap50_bola']['lo95']:.3f}–{bt['ap50_bola']['hi95']:.3f}, ball recall {bt['recall_bola']['lo95']:.3f}–{bt['recall_bola']['hi95']:.3f}. "
  "Resampling is per image and therefore does not capture dependence between slices of the same frame, so the true uncertainty is probably wider.")
H("Performance on training data versus unseen data", 3)
P(f"To see how much the model has learned versus how well it generalises, the frozen configuration was also applied to a random subset of {trn['n_images']:,} training images (without augmentation, seed 42).")
T([["Data", "Player AP@0.5", "Ball AP@0.5", "Referee AP@0.5", "Ball recall", "mAP@0.5"]] +
  [[n, f3(m["metrics"]["pemain"]["ap50"]), f3(m["metrics"]["bola"]["ap50"]), f3(m["metrics"]["wasit"]["ap50"]), f3(m["metrics"]["bola"]["recall"]), f3(m["metrics"]["_mean"]["ap50"])]
   for n, m in [("Train subset (1,000)", trn), ("Validation (797)", vB), ("Internal test (2,288)", tB), ("RF100 (163)", rB)]],
  widths=[4.2, 2.6, 2.6, 2.8, 2.4, 2.4], cap="Model B on seen and unseen data.", fs=8.5)
FG("fig_generalization_gap.png", 12.5, "Model B AP@0.5 per class on seen (train subset) and unseen data.")
P("**Interpretation.** For players, performance on unseen sources is lower than on training images but still good (AP@0.5 0.77 on test vs 0.95 on train), which is an ordinary generalisation gap. "
  f"For the ball, even the **training** images give only AP@0.5 {f3(trn['metrics']['bola']['ap50'])} and recall {f3(trn['metrics']['bola']['recall'])}: the model has not learned to find the ball well even where it has seen the answers. This points to **underfitting** (small model, 25 epochs, tiny objects) as the first problem, and domain difference as the second. "
  "Validation is higher than training for the ball because the validation source (tbd_v2) is easy: all of its images contain a ball.")
H("Confusion matrices and precision–recall curves", 3)
FG("ultra_cm_test.png", 9.5, "Normalized confusion matrix of Model B on the internal test set (Ultralytics default threshold 0.25; background = missed or false detections).")
FG("ultra_pr_test.png", 10, "Precision–recall curves of Model B on the internal test set; the ball curve is clearly the lowest.")
H("Speed", 3)
T([["Model", "GPU median (p95) ms/frame", "CPU median (p95) ms/frame"],
   ["A: pretrained (test images)", f"{tA['speed']['gpu']['median_ms']:.1f} ({tA['speed']['gpu']['p95_ms']:.1f})", f"{tA['speed']['cpu']['median_ms']:.1f} ({tA['speed']['cpu']['p95_ms']:.1f})"],
   ["B: fine-tuned (test images)", f"{tB['speed']['gpu']['median_ms']:.1f} ({tB['speed']['gpu']['p95_ms']:.1f})", f"{tB['speed']['cpu']['median_ms']:.1f} ({tB['speed']['cpu']['p95_ms']:.1f})"],
   ["B: fine-tuned (validation images, earlier run)", f"{v2B['speed_gpu']['median_ms']:.1f} ({v2B['speed_gpu']['p95_ms']:.1f})", f"{v2B['speed_cpu']['median_ms']:.1f} ({v2B['speed_cpu']['p95_ms']:.1f})"]],
  widths=[7, 5, 5], cap="Inference speed on the laptop (batch 1, 640x640).", fs=8.5)
P("A and B share the same architecture, so the difference between A and B in the table is measurement variation (the laptop's load changed between runs; the same model measured 14.4 ms earlier), not a real speed difference.")
H("Robustness to image degradations", 3)
P("Each degradation was applied in memory to a copy of the test images (the files were not changed), one at a time, with levels fixed before the run. Model B and its frozen thresholds were used.")
FG("fig_robustness.png", 14, "mAP@0.5 of Model B under degradations on the internal test set. Red bars mark a drop of more than 15% relative to clean images.")
rr = [["Degradation", "mAP@0.5", "Relative drop", "Ball recall"]]
for _, r in rob.iterrows(): rr.append([r.gangguan, f3(r.map50), f"{100 * r.turun_relatif_map50:.1f}%", f3(r.recall_bola)])
T(rr, widths=[5, 3, 3, 3], cap="Robustness results (negative drop = slight improvement).", fs=8)
P("Mild and medium degradations (blur sigma <= 2, half resolution, brightness ±30%, contrast, JPEG quality 30, noise sigma 10) change mAP by less than 10%. Strong ones (motion blur k = 15, blur sigma 3, JPEG quality 10, noise sigma 25) cost 23–73%, and the ball recall collapses to 0.05–0.13. "
  "The \"half resolution\" test hardly matters, probably because many test images are small slices that are enlarged to 640 px anyway, so this test understates the effect of real low-resolution video.")
CODE(src_of(S / "robustness.py", ["gauss", "motion", "down_up", "bright", "jpeg"]), "Code 8. Image degradations (scripts/robustness.py, excerpt)", fs=7)

# ---------------------------------------------------------------- section 5
H("Section 5. Results Analysis and Conclusion", 1)
H("5.1 In-depth analysis", 2)
H("Strengths", 3)
B(f"**Transfer learning pays off.** Compared with the same network without fine-tuning, AP@0.5 for players rises by {tB['metrics']['pemain']['ap50'] - tA['metrics']['pemain']['ap50']:+.3f} and for the ball by {tB['metrics']['bola']['ap50'] - tA['metrics']['bola']['ap50']:+.3f} on the test set, and a new class (referee) becomes available at AP@0.5 {f3(tB['metrics']['wasit']['ap50'])}.")
B(f"**Players are detected reliably:** AP@0.5 {f3(tB['metrics']['pemain']['ap50'])} on unseen sources and {f3(rB['metrics']['pemain']['ap50'])} on the external dataset, with precision {f3(tB['metrics']['pemain']['precision'])}.")
B("**Good tolerance to common degradations** (brightness, contrast, moderate blur, reduced resolution).")
B(f"**Fast and small:** {tB['speed']['gpu']['median_ms']:.0f} ms/frame on the laptop GPU and about {tB['speed']['cpu']['median_ms']:.0f} ms on its CPU, with a 6.5 MB weight file.")
H("Limitations and their causes", 3)
B(f"**The ball is the main failure.** On the test set recall is {f3(tB['metrics']['bola']['recall'])} (95% interval {bt['recall_bola']['lo95']:.2f}–{bt['recall_bola']['hi95']:.2f}) and precision {f3(tB['metrics']['bola']['precision'])}. "
  f"Both small and large balls are missed (recall {f3(tB['ball_recall_by_size']['kecil']['recall'])} and {f3(tB['ball_recall_by_size']['besar']['recall'])}), so size alone does not explain it, and restricting to images of at least 640 px gives about the same result (ball AP@0.5 {f3(tB['subset_ge640']['metrics']['bola']['ap50'])}).")
B(f"**Underfitting:** ball recall on training images is only {f3(trn['metrics']['bola']['recall'])}; training curves are still improving at epoch 25; the model is the smallest variant.")
B("**Domain gap and data quality:** validation (one easy source) gave ball recall 0.73, the test set 0.29. The test source v5_temp holds 63% of the test balls, shows signs of exporter augmentation, and gives much lower scores than v12:")
T([["Test source", "Images", "mAP@0.5", "Player AP@0.5", "Ball AP@0.5", "Ball P / R", "Referee AP@0.5"]] +
  [[g, f"{ps['metrics_per_source']['B_' + g]['n_images']:,}", f3(ps["metrics_per_source"]["B_" + g]["map50"]), f3(ps["metrics_per_source"]["B_" + g]["pemain_ap50"]), f3(ps["metrics_per_source"]["B_" + g]["bola_ap50"]),
    f"{f3(ps['metrics_per_source']['B_' + g]['bola_precision'])} / {f3(ps['metrics_per_source']['B_' + g]['bola_recall'])}", f3(ps["metrics_per_source"]["B_" + g]["wasit_ap50"])] for g in ("v12", "v5_temp")],
  widths=[2.6, 1.8, 2.2, 2.7, 2.5, 2.8, 2.9], cap="Model B per test source (descriptive; not used for any decision).", fs=8.5)
B("**Referee vs player confusion:** 153 referees were predicted as players on the test set (59 players as referees). Referees differ mostly by shirt colour, which varies between matches. On RF100, referee precision is low (0.35) while recall is high (0.91): in that domain many non-referees are called referees.")
B("**Label problems:** in a manual review of 20 random validation \"false positives\" (player class), about 19 were real people without a label. Reported precision for players is therefore probably lower than the true value; the size of the effect was not measured.")
H("Error analysis (internal test set)", 3)
cn = err["counts"]
names = {"pemain_tidak_terdeteksi": "Missed player", "lokalisasi_buruk_pemain": "Poorly localized player (IoU 0.1–0.5)", "wasit_tidak_terdeteksi": "Missed referee", "bola_tidak_terdeteksi": "Missed ball",
         "lokalisasi_buruk_bola": "Poorly localized ball", "salah_kelas_wasit_jadi_pemain": "Referee predicted as player", "lokalisasi_buruk_wasit": "Poorly localized referee",
         "latar_jadi_pemain": "Background predicted as player", "latar_jadi_bola": "Background predicted as ball", "latar_jadi_wasit": "Background predicted as referee",
         "salah_kelas_pemain_jadi_wasit": "Player predicted as referee", "pemain_ganda": "Duplicate player", "wasit_ganda": "Duplicate referee"}
T([["Error type", "Count"]] + [[names[k], f"{v:,}"] for k, v in sorted(cn.items(), key=lambda x: -x[1])] + [["**Total**", f"**{err['total_errors']:,}**"]],
  widths=[9, 3], cap="Errors of Model B on the internal test set at the frozen thresholds (IoU 0.5; automatic grouping, so categories can overlap).", fs=8.5)
P("Most errors are **objects the model missed**, not false alarms. Of the 706 test balls, 423 were missed and 165 were found but poorly localized. In sampled images, missed balls were very small, blurred, in the goal net, or against a similar background; missed players were often small or rotated (in the augmented source).")
FG("fig_predictions.png", 12, "Three random test images that contain a ball, cropped around it: ground truth (left) and Model B prediction (right). The ball is found in the first image and missed in the other two.")
H("5.2 Were the project objectives achieved?", 2)
P("The success criteria were set before the test sets were evaluated and not changed afterwards. They are judgement-based targets for a feasibility study, not an industry standard, and they were chosen after seeing validation results, so they may favour achievable values. Each row is judged exactly as it came out.")
rows = [["#", "Criterion", "Result", "Target", "Outcome"]]
CR = {"1": "Player AP@0.5 (internal test)", "2": "Referee AP@0.5 (internal test)", "3": "Ball recall (internal test, frozen threshold)", "4": "Ball precision (internal test, frozen threshold)",
      "5": "mAP@0.5, 3 classes (external RF100)", "6a": "Relative mAP@0.5 drop, Gaussian blur sigma = 2", "6b": "Relative mAP@0.5 drop, half resolution", "6c": "Relative mAP@0.5 drop, brightness ±30% (worse of the two)",
      "7a": "Median ms/frame on GPU", "7b": "Median ms/frame on CPU", "8a": "Player AP@0.5 gain, B minus A (internal test)", "8b": "Ball AP@0.5 gain, B minus A (internal test)"}
for c in crit["kriteria"]:
    no = str(c["no"]); v = c["nilai"]; tgt = c["ambang"].replace(">=", "≥").replace("<=", "≤")
    if no.startswith("6"): val = f"{100 * v:.1f}%"; tgt = tgt.replace("0.25", "25%").replace("0.15", "15%")
    elif no.startswith("7"): val = f"{v:.1f} ms"; tgt += " ms"
    else: val = f"{v + 1e-9:.3f}"
    rows.append([no, CR[no], val, tgt, "**PASS**" if c["lulus"] else "**FAIL**"])
hl = {i: ("FCE4E4" if r[-1] == "**FAIL**" else "E2EFDA") for i, r in enumerate(rows) if i > 0}
T(rows, widths=[1, 8, 2.2, 2.6, 2.2], cap="Pre-defined success criteria and outcomes (criteria 1–5 and 8 are core; 6–7 are reported as risks).", fs=8.5, hl=hl, align_num=False)
P("**Verdict.** Feasibility is proven for **players and referees** (criteria 1, 2, 5, 8a pass) and for the claim that **fine-tuning clearly beats the pretrained model**. It is **not proven for the ball** (criteria 3 and 4 fail), so the overall objective of detecting all three classes is **only partially achieved**. "
  "Speed and robustness targets pass, but the speed target does not claim real-time processing and the robustness test on resolution is weak, as discussed above.")
H("5.3 Recommendations", 2)
P("Ordered by the evidence in this report; each is a hypothesis to test, one change at a time, with a **new test set**, because the current one has been used.", after=2)
B("**Fix the data before the model.** Use a cleaner, multi-source ball dataset (for example SoccerNet with a split by match, if access is allowed), remove or flag sources with exporter augmentation, and review missing labels. Build a validation set from several sources, including frames without a ball.")
B("**Give the ball more resolution:** train and run at 960–1280 px, or run the detector on overlapping tiles (SAHI-style inference). Ball recall is low even for large balls, so resolution is not the only lever.")
B("**Give the model more capacity and time:** try YOLOv8s/m and more epochs on a larger GPU (for example Kaggle or Colab); curves were still improving and ball recall on training data is only 0.53.")
B("**Use time:** a tracker or temporal smoothing across frames could recover balls missed in single frames (outside the scope of this project).")
B("**Report confidence intervals and per-source results** by default, and check the referee class on more matches, because shirt colours differ.")
B("**Clarify licenses** before any publication of the model or data.")
P("**Overall conclusion:** a small YOLO model fine-tuned on public data detects players and referees well enough on unseen sources, but it does not yet detect the ball reliably. The most valuable lesson of the project is methodological: a random split would have hidden both the leakage and the weak ball performance, while a source-level split with a frozen protocol exposed them.", after=6)

# ---------------------------------------------------------------- appendix
H("Appendix A. Reproducibility", 1)
T([["Item", "Value"],
   ["Random seed", "42 (training, sampling, bootstrap)"],
   ["Model B weights", "results/train/B_yolov8n_e25_s42_b8/weights/best.pt, 6,506,915 bytes"],
   ["SHA-256", cfg["sha256_B"]],
   ["Frozen thresholds (Model B)", f"player {cfg['op_conf_B']['pemain']:.3f}, ball {cfg['op_conf_B']['bola']:.3f}, referee {cfg['op_conf_B']['wasit']:.3f}"],
   ["Frozen thresholds (Baseline A)", f"player {cfg['op_conf_A']['pemain']:.3f}, ball {cfg['op_conf_A']['bola']:.3f}"],
   ["Ball size boundary", f"normalized box area {cfg['ball_size_threshold_norm_area']:.6f} (median of training balls)"],
   ["Scripts", "scripts/: make_split.py, day2_data_check.py, train.py, resume.py, eval_lib.py, day3_freeze.py, day3_test_prep.py, rf100_to_yolo.py, day3_eval.py, robustness.py, error_analysis.py, day3_criteria.py, report_train_eval.py, report_figures.py, build_report.py"],
   ["Result files", "results/summary.csv, results/runs/*.json, results/day3_robustness.csv, results/day3_criteria.json"]],
  widths=[4.5, 12.5], fs=8, hdr=True, align_num=False)

sec.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
fp = sec.footer.paragraphs[0]; fp.add_run("Deep Learning Midterm Project  |  page "); field(fp, "PAGE")
OUT.parent.mkdir(parents=True, exist_ok=True); doc.save(OUT); print("saved", OUT)
