# SETUP — Lingkungan & cara menjalankan ulang

## Spesifikasi laptop (fakta, dicek 2026-11-05)
| Item | Nilai |
|---|---|
| CPU | AMD Ryzen 7 8845HS, 8 core / 16 thread |
| RAM | 13,8 GB |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 6 GB VRAM (driver 610.62) + Radeon 780M iGPU |
| OS | Windows 11 Home |
| Disk kosong | sekitar 12 GB di C: saat dicek (dataset kandidat memakai ± 1,5 GB zip + hasil ekstraksi) |
| Python | 3.12.6 (venv di `football_detection/.venv`) |

## Pustaka utama (lihat `requirements.lock.txt`)
torch 2.5.1+cu121, torchvision 0.20.1+cu121, ultralytics 8.4.202, numpy <2.3, opencv (via ultralytics), imagehash, pandas, pyarrow, huggingface_hub, roboflow, supervision. `torch.cuda.is_available()` = True.

## Uji lingkungan (fakta)
`scripts/smoke_test.py`: YOLOv8n pretrained (COCO) pada `bus.jpg` dan `zidane.jpg` — terdeteksi bus, person, stop sign, tie. Hasil di `results/smoke_test/`.
Waktu rata-rata 10 pemanggilan pada 1 gambar (termasuk baca file & pra-proses, tanpa pemanasan berulang): CPU ≈ 62,8 ms, GPU ≈ 43,6 ms. **Hanya indikasi kasar**; pengukuran resmi ada di EVAL_PLAN.md (belum dijalankan).

## Struktur folder
`data/raw` (asli, jangan diubah) · `data/work` (salinan/ekstraksi + manifest split) · `docs` · `scripts` · `notebooks` · `results`

## Menjalankan ulang
```bash
python -m venv football_detection/.venv
football_detection/.venv/Scripts/python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
football_detection/.venv/Scripts/python.exe -m pip install ultralytics imagehash pandas pyarrow huggingface_hub "numpy<2.3"
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/smoke_test.py football_detection/results/smoke_test
```
Dataset: unduh `Adit-jain/Soccana_player_ball_detection_v1` (V1.zip) dan `Francesco/soccer-players-5fuqs` dari Hugging Face ke `data/raw`, ekstrak V1.zip ke `data/work/soccana`, lalu jalankan `scripts/audit_soccana.py`, `audit_sources.py`, `audit_soccana_leakage.py`, `audit_rf100.py`, `make_split.py`.
Catatan: `dataset.tar.gz` pada RF100 gagal diekstrak penuh (hard-link JSON rusak); audit memakai file parquet.

## Mengulang training Hari 2 (ditambahkan)
Pembuatan daftar split dan cek data: `python -I scripts/day2_data_check.py` → `data/work/splits/{train,val}.txt` dan `data_day2.yaml` (tanpa test).
```bash
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/baseline_a.py
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/train.py B_yolov8n_e25_s42_b8 football_detection/results/smoke_test/yolov8n.pt 25 640 8
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/eval_model.py football_detection/results/train/B_yolov8n_e25_s42_b8/weights/best.pt B_yolov8n_e25_val
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/error_analysis.py football_detection/results/train/B_yolov8n_e25_s42_b8/weights/best.pt B_yolov8n_e25_val 0.25
```
Jika proses mati (mis. RAM habis), lanjutkan: `scripts/resume.py B_yolov8n_e25_s42_b8 2`. `add_summary.py` menambah baris ke `results/summary.csv`.
Catatan Windows: `train.py` memakai guard `__main__` (wajib karena dataloader multiproses). Pengaturan: seed 42, imgsz 640, batch 8 (16 → OOM GPU 6 GB), 25 epoch (± 125 dtk/epoch; 30 epoch melewati batas 1 jam). Tutup aplikasi lain agar RAM tidak habis (13,8 GB total). Daftar percobaan dan deviasi: DEVELOP_BRIEF.md.

## Hari 3: evaluasi akhir (ditambahkan)
Bobot Model B tidak disimpan di git: `football_detection/results/train/B_yolov8n_e25_s42_b8/weights/best.pt` (6.506.915 byte), SHA-256 `a189bcdb3fd729dd2885b1f7e494bd084c3013b304eb4b221a4218f5f4d8a542`.
Urutan (semua dengan `football_detection/.venv/Scripts/python.exe -I`):
```bash
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/day3_freeze.py
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/day3_test_prep.py
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/rf100_to_yolo.py
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/day3_eval.py
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/day3_per_source.py
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/robustness.py
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/error_analysis.py football_detection/results/train/B_yolov8n_e25_s42_b8/weights/best.pt B_day3_test beku football_detection/data/work/splits/test.txt
football_detection/.venv/Scripts/python.exe -I football_detection/scripts/day3_criteria.py
```
`day3_eval.py` dan `robustness.py` masing-masing butuh sekitar 10–20 menit di laptop ini (perkiraan dari run 2026-11-10). Menjalankan ulang akan **menambah** baris `summary.csv`, tidak menimpanya.
