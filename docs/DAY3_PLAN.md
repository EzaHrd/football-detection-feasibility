# DAY3_PLAN: rencana implementasi Hari 3 (testing, finishing, pelaporan)

> Status: **SELESAI DIJALANKAN 2026-11-10.** Hasil: TEST_REPORT.md dan FINAL_REPORT.md. Deviasi dari rencana: (1) ditambah analisis per sumber (`day3_per_source.py`, deskriptif saja) setelah melihat contoh error; (2) evaluator sempat diubah lalu dikembalikan sebelum test (TEST_REPORT §1); (3) commit belum dilakukan, menunggu persetujuan pemilik.
> Ditulis 2026-11-10 setelah Hari 2.
> Semua estimasi waktu adalah perkiraan, bukan hasil ukur.
> Model yang diuji: `results/train/B_yolov8n_e25_s42_b8/weights/best.pt` (Model B). Pembanding: Baseline A (`yolov8n.pt` COCO).

## 0. Apa yang masih kurang (inventaris dari dokumen Hari 1–2)

| # | Kekurangan | Sumber kewajiban | Status sekarang |
|---|---|---|---|
| K1 | Kriteria lulus masih 12 baris "TBD" | EVAL_PLAN §5 | Belum ditetapkan pemilik proyek. Pemilik memilih "lanjut Hari 3 dulu" |
| K2 | Cek duplikat pHash **train↔test** dan **val↔test** sebelum evaluasi | EVAL_PLAN §1 ("Pemeriksaan wajib") | Baru train↔val yang dicek (Hari 2) |
| K3 | Daftar `test.txt` dan yaml test | — | Sengaja belum dibuat |
| K4 | Evaluasi A dan B pada test internal (v12 + v5_temp, 2.425 gambar) | EVAL_PLAN §1–2 | Belum |
| K5 | Evaluasi pada test eksternal RF100 (163 gambar, format parquet/COCO) | EVAL_PLAN §1, §3 | Belum. Perlu konversi ke YOLO di `data/work` |
| K6 | Subset metrik: gambar ≥ 640 px; bola kecil vs besar | EVAL_PLAN §2 | Belum |
| K7 | Interval kepercayaan (bootstrap) untuk metrik bola | EVAL_PLAN §2 | Belum |
| K8 | Uji ketahanan: blur, motion blur, resolusi turun, kecerahan/kontras, JPEG, noise | EVAL_PLAN §3 | Belum ada skrip |
| K9 | Ambang kepercayaan operasional dipilih di **val**, lalu dipakai tetap di test | EVAL_PLAN §2 | Belum dipilih eksplisit. P/R Hari 2 memakai ambang F1-optimal per kelas |
| K10 | Confusion matrix pada test | EVAL_PLAN §2 | Hanya ada untuk val (Model B) |
| K11 | Cek manual "latar_jadi_pemain": label hilang atau salah model? | ERROR_NOTES (pertanyaan terbuka) | Baru 3 contoh dilihat |
| K12 | Laporan akhir kelayakan | DEFINE_BRIEF | Belum ada |
| K13 | Commit ke git | — | 47 berkas belum di-commit (hanya di-stage sebagian) |
| K14 | Lisensi dataset | DATA_AUDIT risiko #1 | Masih tidak diketahui. Di luar kendali teknis |

## 1. Gerbang sebelum menyentuh test (WAJIB)

Test set dipakai **satu kali** untuk angka akhir. Urutan yang tidak boleh dibalik:

1. **Bekukan semua pilihan di val dulu:** model (`best.pt` B), imgsz 640, ambang conf operasional (K9), NMS IoU 0,6, pemetaan kelas A. Catat di `results/day3_frozen_config.json`.
2. **Kriteria lulus (K1).** **Keputusan pemilik (2026-11-10): Jalur 1.** Angka menunggu diisi pemilik; test tidak dijalankan sebelum itu. Dua jalur yang tersedia:
   - **Jalur 1 (disarankan):** pemilik mengisi angka di EVAL_PLAN §5 sebelum langkah 3, lalu hasil dinilai lulus/gagal.
   - **Jalur 2 (sesuai keputusan "lanjut dulu"):** test dijalankan dan hasilnya **dilaporkan tanpa status lulus/gagal**. Laporan menulis jelas bahwa ambang belum ada, dan ambang yang ditetapkan sesudah melihat skor test tidak boleh disebut "kriteria lulus" untuk run ini.
3. **Cek kebocoran test (K2)** sebelum inferensi apa pun pada test. Gambar test dengan pHash ≤ 4 terhadap train/val dikeluarkan dan dicatat jumlahnya. Ini berdasarkan hash gambar, bukan hasil model, jadi tidak termasuk "tuning pada test".
4. Setelah angka test keluar: **tidak ada** pelatihan ulang, perubahan ambang, atau pemilihan model berdasarkan test. Bila hasil mengecewakan, tulis apa adanya.

## 2. Urutan tugas Hari 3

| Langkah | Isi | Keluaran | Estimasi |
|---|---|---|---|
| T1 | Bekukan konfigurasi. Pilih ambang conf operasional dari val (mis. ambang F1-optimal gabungan atau per kelas), lalu hitung P/R val pada ambang itu | `day3_frozen_config.json`, baris val baru di `summary.csv` | 20 mnt |
| T2 | Buat `test.txt` dari `split_by_source_v1.csv`. Cek pHash test↔train dan test↔val. Hitung gambar/objek per kelas di test | `scripts/day3_test_prep.py`, `results/day3_test_check.json` | 20 mnt |
| T3 | Konversi RF100 (parquet → YOLO) ke `data/work/rf100_yolo/`. Peta: player→pemain, football→bola, referee→wasit, `soccer-players` dibuang. Data asli tidak diubah | `scripts/rf100_to_yolo.py` | 20 mnt |
| T4 | Evaluasi **A dan B** pada test internal dan RF100 dengan `eval_lib.evaluate` yang sama: mAP50, mAP50-95, P/R per kelas pada ambang beku, confusion matrix B (Ultralytics `val`), ms/frame median/p95 (GPU 100 gambar, CPU 40) | `results/runs/*_test.json`, `*_rf100.json`, baris baru `summary.csv` | 40 mnt |
| T5 | Subset (K6): gambar ≥ 640 px; bola kecil vs besar menurut luas kotak (batas ditetapkan sebelum melihat hasil, mis. median luas bola di **train**) | bagian di `TEST_REPORT.md` | 30 mnt |
| T6 | Bootstrap 1.000× per gambar (seed 42) untuk AP50 & recall bola dan mAP50, interval 95% | `results/day3_bootstrap.json` | 30 mnt |
| T7 | Uji ketahanan (K8) pada **salinan** gambar test internal, satu gangguan per run. Tingkat ditetapkan di skrip sebelum dijalankan: Gaussian blur (σ 1, 2, 3), motion blur (kernel 7, 15), turun-naik resolusi (×½, ×¼), kecerahan ±30%, kontras ×0,6 / ×1,4, JPEG kualitas 30 dan 10, noise Gaussian σ 10 dan 25. Laporkan Δ mAP50 dan Δ recall bola vs gambar bersih. Hanya Model B (A opsional bila waktu ada) | `scripts/robustness.py`, `results/day3_robustness.csv`, grafik | 60–90 mnt |
| T8 | Analisis error test (skrip `error_analysis.py` diarahkan ke test, tanpa mengubah apa pun), 30–50 contoh. Cek manual 20 contoh "latar_jadi_pemain" di **val** untuk K11 (label hilang vs salah model) | `results/error_analysis/B_*_test/`, tambahan ERROR_NOTES | 45 mnt |
| T9 | Dokumentasi akhir (bagian 4) | 3 dokumen | 60 mnt |
| T10 | Rapikan repo dan commit (bagian 5) | 1 commit | 15 mnt |

Total ≈ 6–7 jam (perkiraan). Tidak ada training baru, jadi batas 3 jam training Hari 2 tidak relevan. Jika waktu kurang, urutan prioritas: T1–T4 → T9 → T7 (sebagian gangguan) → T6 → T5 → T8.

**Risiko teknis yang sudah diketahui:**
- RAM sempat habis di Hari 2 (13,8 GB total). Tutup aplikasi lain, pakai `workers=2`, dan proses gangguan per batch, bukan disimpan semua di memori.
- Skrip multiproses di Windows wajib memakai guard `if __name__ == "__main__":`.
- Gambar test campur ukuran (160–1920 px). Irisan 160 px yang di-upscale ke 640 bisa membuat uji "resolusi turun" kurang bermakna. Laporkan juga subset ≥ 640 px.

## 3. Aturan Hari 3 (dibawa dari Hari 1–2)

- Jangan mengarang angka. Setiap angka harus bisa dilacak ke berkas di `results/`.
- Jangan mengubah data asli. Gangguan dan konversi hanya di `data/work`.
- Nama run unik, jangan menimpa hasil lama. `summary.csv` hanya ditambah baris.
- Jika kriteria masih TBD, jangan diisi sendiri.
- Berhenti dan tanya jika: unduhan gagal, error berulang > 2 kali, bola nyaris tidak terdeteksi di test, atau ada dorongan mengubah apa pun berdasarkan hasil test.

## 4. Dokumen yang akan ditulis/diperbarui

| Dokumen | Isi |
|---|---|
| `TEST_REPORT.md` (baru) | Konfigurasi beku. Hasil cek kebocoran test. Tabel A vs B pada test internal dan RF100 (per kelas, mAP, kecepatan). Subset ≥ 640 px dan ukuran bola. Interval bootstrap. Tabel ketahanan per gangguan. Confusion matrix. Status kriteria: lulus/gagal per baris jika ambang ada, atau "tidak dinilai: ambang TBD" |
| `FINAL_REPORT.md` (baru) | Ringkasan kelayakan untuk pemilik proyek: pertanyaan teknis dari DEFINE_BRIEF, jawaban berdasarkan bukti, apa yang **tidak** terbukti (tracking, real-time, kamera amatir, generalisasi luas), keterbatasan (lisensi, label hilang, val/test satu dataset induk, satu seed), dan rekomendasi langkah lanjut. Bukan klaim produk |
| `ERROR_NOTES.md` | Tambah bagian test dan hasil cek manual K11 |
| `SETUP.md` | Perintah menjalankan ulang T1–T8 |
| `EVAL_PLAN.md` | Tidak diubah, kecuali pemilik mengisi kriteria. Catat tanggal pengisian |

## 5. Finishing repo

- `.gitignore`: data, `.venv`, cache, dan bobot besar tetap di luar git. Simpan `best.pt` (≈ 6 MB) hanya jika pemilik setuju. Jika tidak, catat lokasi dan hash SHA-256 di SETUP.md.
- Hapus `scripts/__pycache__` dari stage.
- Satu commit di branch `claude/football-detection-feasibility-819175` setelah pemilik setuju. Tidak push atau membuat PR tanpa diminta.

## 6. Kriteria selesai Hari 3

- [ ] Konfigurasi dibekukan di val sebelum test disentuh
- [ ] Cek kebocoran test↔train/val tercatat
- [ ] A dan B dievaluasi pada test internal dan RF100, dengan metrik per kelas dan kecepatan
- [ ] Uji ketahanan minimal blur, resolusi turun, dan pencahayaan
- [ ] TEST_REPORT.md dan FINAL_REPORT.md ada. Status kriteria ditulis jujur (dinilai / TBD)
- [ ] summary.csv lengkap, semua angka dapat dilacak

## Keputusan (2026-11-10, pemilik mendelegasikan ke Claude)

1. Kriteria lulus: **Jalur 1**. Angka diisi dan dibekukan di EVAL_PLAN.md §5, sebelum test disentuh.
2. Ambang conf operasional: **per kelas**, dari F1 terbaik di val.
3. `best.pt` **tidak** disimpan di git. Lokasi dan hash SHA-256 dicatat di SETUP.md.
4. Commit dilakukan setelah Hari 3 selesai, dengan persetujuan pemilik.

## Pertanyaan terbuka

- Lisensi dataset Soccana: siapa yang memverifikasi sebelum hasil dipublikasikan?
