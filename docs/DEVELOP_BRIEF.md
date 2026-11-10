# DEVELOP_BRIEF — Hari 2: membangun dan membandingkan model

Semua angka diambil dari `results/runs/*.json`, `results/summary.csv`, dan `results/train/B_yolov8n_e25_s42_b8/results.csv`. **Hanya val (797 gambar, sumber tbd_v2) yang dipakai. Test (v12, v5_temp) dan RF100 tidak disentuh.** Daftar test bahkan tidak dibuat hari ini.

## 1. Verifikasi data (fakta)
| Split | Gambar | Kosong | Pemain | Bola | Wasit | Gambar berbola |
|---|---|---|---|---|---|---|
| train (v3, v7, v2_temp, spt_v2) | 8.451 | 2.862 | 35.673 | 2.133 | 3.244 | 2.129 |
| val (tbd_v2) | 797 | 0 | 3.133 | 653 | 351 | 653 |

Train vs val: nama berkas sama 0; pHash ≤ 4: 0; pHash ≤ 8: 3 dari 797; identik 0 (`results/day2_data_check.json`). Kelas paling sedikit: **bola** (2.133 di train, 1 kotak per gambar berbola) dan wasit. Catatan: val seluruhnya gambar berbola dan satu sumber (tbd_v2), sehingga val **tidak mewakili** gambar tanpa bola maupun sumber lain.

## 2. Perbandingan model (val, imgsz 640, evaluator sama untuk A dan B)
Presisi/recall pada **satu** ambang kepercayaan yang memaksimalkan rata-rata F1 semua kelas (perilaku bawaan `ap_per_class` Ultralytics). AP dihitung pada conf 0,001. *(Koreksi 2026-11-10: versi awal dokumen ini salah menulis "F1 per kelas". Hari 3 memakai ambang per kelas; lihat TEST_REPORT.md §1.)*
| Metrik | A: YOLOv8n pretrained (tanpa fine-tune) | B: YOLOv8n fine-tune (25 epoch) |
|---|---|---|
| Pemain P / R | 0,617 / 0,690 | 0,779 / 0,915 |
| Bola P / R | 0,681 / 0,311 | 0,891 / 0,683 |
| Wasit P / R | tidak tersedia | 0,802 / 0,775 |
| AP50 pemain / bola / wasit | 0,638 / 0,381 / tidak tersedia | 0,878 / 0,781 / 0,810 |
| AP50-95 pemain / bola / wasit | 0,283 / 0,131 / tidak tersedia | 0,460 / 0,384 / 0,391 |
| mAP50 (kelas tersedia) | 0,510 (2 kelas) | 0,823 (3 kelas); 0,830 jika hanya pemain+bola |
| mAP50-95 | tidak dihitung rata-rata (2 kelas) | 0,412 |
| ms/frame GPU (median / p95) | 14,2 / 17,1 | 14,4 / 19,0 |
| ms/frame CPU (median) | 47,4 | 48,6 |
Cek silang Ultralytics `val` untuk B: mAP50 0,835, mAP50-95 0,422 (sedikit berbeda dari evaluator saya, 0,823 / 0,412, karena implementasi pencocokan berbeda; angka saya dipakai konsisten untuk A dan B). Kecepatan: batch 1, pra-proses + inferensi + NMS, 100 gambar val (CPU 40), setelah pemanasan, RTX 3050 / Ryzen 7 8845HS.
Confusion matrix B: `results/val_plots/B_yolov8n_e25_val/confusion_matrix.png` (plus kurva P/R/F1). Untuk A tidak dibuat (wasit tidak ada).

## 3. Percobaan yang dilakukan (kronologis)
| # | Percobaan | Hasil |
|---|---|---|
| 0 | Baseline A | lihat tabel |
| 1 | B, 30 epoch, batch 16 | **Gagal**: CUDA OOM pada epoch 1 (error training #1). Log: `results/train_B_fail1_oom_b16.*` |
| 2 | B, 30 epoch, batch 8 | **Dihentikan manual** setelah epoch 2: laju ≈ 125 dtk/epoch → ≈ 63 menit, melebihi batas 1 jam. Log: `train_B_aborted_e30_toolong.*` |
| 3 | B, **25 epoch**, batch 8, seed 42 | Crash di epoch 12: RAM sistem habis saat memuat gambar (error training #2; RAM bebas ≈ 2,9 GB, aplikasi lain ikut memakai memori). Dilanjutkan dengan `resume=True` dan workers 2 dari `last.pt` (epoch 11) → selesai 25 epoch. Total waktu training ≈ 3.280 dtk (≈ 55 menit) |
| 4 | Perbaikan terbatas (langkah 5) | **Dilewati**: perubahan tunggal yang masuk akal (model s, imgsz 960, 40 epoch) diperkirakan > 1 jam per percobaan di laptop ini. Kurva mAP50 val mendatar pelan (epoch 20: 0,833; 25: 0,837), jadi kenaikan dari epoch tambahan kemungkinan kecil — **ini perkiraan, tidak diuji** |

Deviasi dari default (dicatat): 25 epoch bukan 30 (batas waktu), batch 8 bukan 16 (OOM), workers diturunkan 4→2 setelah epoch 11 (RAM). Total waktu training Hari 2 termasuk percobaan gagal ≈ 1 jam 5 menit dari batas 3 jam (perkiraan). Pengaturan lain: init `yolov8n.pt` (COCO), imgsz 640, optimizer auto (AdamW lr 0,001429), augmentasi default Ultralytics (hanya pada train), AMP, `deterministic=True`. Konfigurasi: `results/train/B_yolov8n_e25_s42_b8/args.yaml`.
Catatan: proses resume mengubah urutan data acak sehingga hasil tidak identik bit-per-bit jika dijalankan ulang tanpa crash.

## 4. Model terpilih untuk Hari 3
**B_yolov8n_e25_s42_b8, `weights/best.pt`** (satu-satunya model fine-tune). Alasan: fine-tuning **jelas lebih baik** daripada baseline pada val di semua metrik deteksi (pemain AP50 0,638→0,878; bola AP50 0,381→0,781, recall bola 0,311→0,683) dan menambah kelas wasit yang tidak ada pada baseline; kecepatan setara (arsitektur sama). Peringatan: `best.pt` dipilih berdasarkan val, jadi skor val sedikit optimistik; hanya satu run/seed; val hanya satu sumber. Belum ada bukti ketahanan pada sumber lain (itulah tugas test Hari 3).

## 5. Kesimpulan awal (hati-hati)
- Bola masih titik terlemah: recall 0,683 (artinya ≈ 1 dari 3 bola di val terlewat pada ambang F1-optimal).
- Wasit sering tertukar dengan pemain (lihat ERROR_NOTES.md).
- Skor val belum membuktikan generalisasi. Tidak ada kesimpulan "lulus/gagal" karena kriteria masih TBD.

## Pertanyaan terbuka
1. **Kriteria lulus di EVAL_PLAN.md masih "TBD". Pemilik proyek harus menetapkannya SEBELUM Hari 3** (saya tidak mengisinya).
2. Apakah 25 epoch/batch 8 dapat diterima, atau ingin eksperimen lebih besar (model s / 960 px) di Kaggle/Colab?
3. Val hanya satu sumber dan semua gambar berbola; perlukah menambah val dari sumber lain? (Tidak diubah sendiri.)
4. Lisensi dataset masih belum jelas (lihat DATA_AUDIT.md).
