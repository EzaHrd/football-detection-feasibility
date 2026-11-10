# TEST_REPORT: Hari 3, evaluasi akhir pada test

Tanggal: 2026-11-10. Semua angka berasal dari berkas di `results/` (sumber disebut per bagian). Test dijalankan **satu kali** dengan konfigurasi yang dibekukan sebelumnya. Tidak ada pelatihan ulang, perubahan ambang, atau pemilihan model setelah hasil test terlihat.

## 1. Konfigurasi beku (`results/day3_frozen_config.json`)
- Model B: `results/train/B_yolov8n_e25_s42_b8/weights/best.pt`, SHA-256 `a189bcdb3fd729dd2885b1f7e494bd084c3013b304eb4b221a4218f5f4d8a542`. Pembanding A: `yolov8n.pt` COCO (person→pemain, sports ball→bola, wasit tidak tersedia).
- imgsz 640, NMS IoU 0,6, AP pada conf 0,001, pencocokan IoU 0,5.
- Ambang conf operasional per kelas = F1 maksimum per kelas di **val**:
  - B: pemain 0,472; bola 0,234; wasit 0,280.
  - A: pemain 0,338; bola 0,146.
- Pada ambang ini, P/R B di val adalah: pemain 0,828/0,879; bola 0,827/0,734; wasit 0,792/0,801 (`runs/B_day3_val_opconf.json`).
- Batas bola kecil/besar: luas kotak ternormalisasi < 0,000361 (median 2.133 bola **train**).
- Kriteria lulus: EVAL_PLAN.md §5, dibekukan sebelum test disentuh.
- Catatan evaluator: saat menyiapkan Hari 3, saya sempat mengubah cara pencocokan sehingga AP val turun sekitar 0,03. Perubahan itu dikembalikan ke cara Ultralytics/Hari 2 sebelum test dijalankan, dan AP val kembali sama persis dengan Hari 2.

## 2. Cek kebocoran test (`results/day3_test_check.json`), sebelum inferensi
| | Jumlah |
|---|---|
| Gambar test awal (v12 + v5_temp) | 2.425 |
| pHash ≤ 4 terhadap train/val → **dikeluarkan** | 137 (100 di antaranya pHash identik). v12: 79, v5_temp: 58 |
| pHash ≤ 8 | 195 |
| Nama berkas sama dengan train/val | 0 |
| **Test akhir** | **2.288 gambar**: 14.666 pemain, **706 bola**, 1.527 wasit; 638 tanpa label; 1.189 bersisi ≥ 640 px |

Temuan: sumber Roboflow yang berbeda **berbagi gambar yang sama**. Risiko ini sudah dicatat di Hari 1 dan sekarang terbukti. Daftar yang dikeluarkan ada di `data/work/splits/test_excluded_near_dup.txt`. Kebocoran yang tidak tertangkap pHash (irisan berbeda dari frame yang sama) tetap mungkin ada.

## 3. Hasil utama: A vs B (`runs/*_day3_test.json`, `runs/*_day3_rf100.json`, `summary.csv`)
P/R pada ambang beku. AP pada conf 0,001.

| Metrik | A test | **B test** | A RF100 | **B RF100** |
|---|---|---|---|---|
| AP50 pemain | 0,394 | **0,770** | 0,402 | **0,954** |
| AP50 bola | 0,078 | **0,267** | 0,013 | **0,222** |
| AP50 wasit | tidak tersedia | **0,543** | tidak tersedia | **0,608** |
| mAP50 | 0,236 (2 kelas) | **0,527** (3 kelas) | 0,208 (2 kelas) | **0,595** (3 kelas) |
| mAP50-95 | 0,106 (2 kelas) | 0,272 | 0,074 (2 kelas) | 0,237 |
| Pemain P / R | 0,675 / 0,355 | 0,912 / 0,680 | 0,810 / 0,232 | 0,956 / 0,890 |
| **Bola P / R** | 0,342 / 0,092 | **0,521 / 0,288** | 0,190 / 0,028 | **0,300 / 0,191** |
| Wasit P / R | tidak tersedia | 0,776 / 0,531 | tidak tersedia | 0,347 / 0,906 |
| ms/frame GPU median (p95) | 16,7 (20,9) | 22,3 (26,9) | | |
| ms/frame CPU median (p95) | 43,5 (47,7) | 59,3 (65,6) | | |

- **Kecepatan:** diukur pada 100 gambar test (GPU) dan 40 gambar (CPU), batch 1, setelah pemanasan. A dan B punya arsitektur yang sama. Selisih 16,7 vs 22,3 ms kemungkinan variasi beban laptop saat pengukuran. Di val Hari 2 keduanya ≈ 14 ms. Angka ini tidak saya jelaskan lebih jauh.
- **Jumlah objek RF100:** 1.766 pemain, 141 bola, 96 wasit. Interval kepercayaan di RF100 lebar karena bola dan wasit sedikit.
- **Confusion matrix B:** `results/val_plots/B_day3_test/confusion_matrix*.png` dan `results/val_plots/B_day3_rf100/` (Ultralytics, conf bawaan matriks 0,25).
- **Wasit di RF100:** precision rendah (0,347) dengan recall tinggi (0,906), artinya banyak prediksi wasit yang salah di domain ini. Penyebabnya belum diselidiki.

## 4. Bootstrap test internal (1.000×, seed 42, resampling per gambar) — `runs/*_day3_test.json` → `bootstrap`
| Interval 95% | A | B |
|---|---|---|
| AP50 bola | 0,058 – 0,103 | 0,232 – 0,304 |
| Recall bola (ambang beku) | 0,072 – 0,114 | 0,258 – 0,321 |
| mAP50 | 0,221 – 0,253 | 0,507 – 0,548 |
Batas atas recall bola B (0,321) jauh di bawah ambang 0,50. Kegagalan ini bukan kebetulan sampel.

## 5. Subset
**Gambar bersisi ≥ 640 px** (1.189 gambar, 665 bola):
- B: AP50 pemain 0,752, bola 0,249, wasit 0,526; mAP50 0,509; bola P/R 0,500/0,266.
- A: pemain 0,410, bola 0,075.
Hasil hampir sama dengan test penuh, jadi irisan kecil SAHI bukan penyebab utama.

**Recall bola per ukuran** (ambang beku, IoU 0,5):

| | Kecil (< median train) | Besar |
|---|---|---|
| B test | 0,259 (111/428) | 0,331 (92/278) |
| A test | 0,075 | 0,119 |
| B RF100 | 0,20 (2/10) | 0,191 (25/131) |

Bola besar pun hanya sepertiga terdeteksi. Ukuran bukan satu-satunya penyebab.

**Per sumber** (deskriptif, bukan dasar penilaian; `results/day3_per_source.json`):

| B | Gambar | mAP50 | AP50 pemain | AP50 bola | Bola P / R | AP50 wasit |
|---|---|---|---|---|---|---|
| v12 | 1.646 | 0,712 | 0,974 | 0,421 | 0,694 / 0,414 (263 bola) | 0,740 |
| v5_temp | 642 | 0,366 | 0,550 | 0,198 | 0,403 / 0,212 (443 bola) | 0,351 |

Porsi kotak pemain "melintang" (lebar > 1,2 × tinggi): **v5_temp 34,3%**, sumber lain 0,6–3,3%. Bersama contoh visual (gambar diputar 90°, tempelan potongan/mosaik), ini indikasi kuat bahwa v5_temp berisi **augmentasi bawaan ekspor Roboflow**, bukan frame video asli. v5_temp memegang 63% bola test (443/706). Bahkan pada v12 yang framenya normal, bola tetap di bawah ambang (recall 0,414; precision 0,694). Penilaian resmi tetap memakai test penuh sesuai rencana yang dibekukan.

## 6. Uji ketahanan Model B (`results/day3_robustness.csv`)
Gangguan diterapkan di memori pada 2.288 gambar test (file asli tidak diubah). Kolom "turun" = penurunan relatif mAP50 dibanding gambar bersih (negatif = naik).

| Gangguan | mAP50 | Turun | Recall bola |
|---|---|---|---|
| bersih | 0,527 | – | 0,288 |
| Gaussian blur σ 1 / **2** / 3 | 0,534 / **0,489** / 0,377 | −1% / **7%** / 28% | 0,280 / 0,242 / 0,133 |
| Motion blur k 7 / 15 | 0,477 / 0,143 | 9% / 73% | 0,212 / 0,054 |
| Resolusi ×**½** / ×¼ | **0,534** / 0,481 | **−1%** / 9% | 0,285 / 0,228 |
| Kecerahan **−30% / +30%** | **0,529 / 0,520** | **0% / 1%** | 0,285 / 0,290 |
| Kontras ×0,6 / ×1,4 | 0,511 / 0,525 | 3% / 0% | 0,279 / 0,293 |
| JPEG kualitas 30 / 10 | 0,522 / 0,406 | 1% / 23% | 0,261 / 0,125 |
| Noise σ 10 / 25 | 0,510 / 0,383 | 3% / 27% | 0,261 / 0,089 |

Tingkat yang dinilai (tebal) semuanya lulus. Gangguan berat (motion blur k15, blur σ3, noise σ25, JPEG 10) menurunkan mAP 23–73%, dan recall bola jatuh ke 0,05–0,13.
Catatan: "resolusi ×½" hampir tidak berpengaruh, kemungkinan karena banyak gambar test adalah irisan kecil yang memang di-upscale ke 640. Uji ini jadi kurang mewakili video beresolusi rendah sungguhan.

## 7. Penilaian kriteria (`results/day3_criteria.json`, dihitung otomatis oleh `scripts/day3_criteria.py`)
| # | Kriteria | Nilai | Ambang | Hasil |
|---|---|---|---|---|
| 1 | AP50 pemain (test) | 0,770 | ≥ 0,70 | **LULUS** |
| 2 | AP50 wasit (test) | 0,543 | ≥ 0,50 | **LULUS** |
| 3 | Recall bola (test) | 0,288 | ≥ 0,50 | **GAGAL** |
| 4 | Precision bola (test) | 0,521 | ≥ 0,70 | **GAGAL** |
| 5 | mAP50 RF100 | 0,595 | ≥ 0,50 | **LULUS** |
| 6a | Turun mAP50, blur σ 2 | 7,1% | ≤ 25% | LULUS |
| 6b | Turun mAP50, resolusi ×½ | −1,4% | ≤ 25% | LULUS (lihat catatan §6) |
| 6c | Turun mAP50, kecerahan ±30% (terburuk) | 1,3% | ≤ 15% | LULUS |
| 7a | ms/frame GPU | 22,3 | ≤ 50 | LULUS |
| 7b | ms/frame CPU | 59,3 | ≤ 200 | LULUS |
| 8a | Kenaikan AP50 pemain B − A | +0,376 | ≥ 0,10 | **LULUS** |
| 8b | Kenaikan AP50 bola B − A | +0,188 | ≥ 0,10 | **LULUS** |

**Kesimpulan menurut aturan yang dibekukan: kelayakan teknis untuk 3 kelas BELUM terbukti**, karena kedua kriteria inti untuk bola (3 dan 4) gagal. Pemain dan wasit lulus di test internal dan eksternal, dan fine-tuning jelas lebih baik daripada model umum.

## 8. Kesenjangan val → test
B di val: AP50 bola 0,781, recall 0,734. Di test: 0,267 / 0,288. Pemain turun lebih sedikit (0,878 → 0,770).

Penyebab yang didukung data:
1. Val hanya satu sumber (tbd_v2) dan semua gambarnya berbola. Val terlalu mudah dan tidak mewakili.
2. Test v5_temp berisi augmentasi bawaan (rotasi/mosaik).
3. Bola tetap lemah di frame normal v12.

Ambang conf yang dipilih di val juga terlalu optimistik untuk test. **Tidak ada yang diubah berdasarkan temuan ini.**
