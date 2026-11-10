# FINAL_REPORT: Studi kelayakan deteksi pemain, bola, dan wasit

> Studi kelayakan **teknis** selama 3 hari dengan data publik dan satu laptop. Ini bukan produk dan bukan klaim untuk pengguna tertentu (lihat DEFINE_BRIEF.md: pengguna sasaran masih hipotesis).

## Pertanyaan
Apakah detektor YOLO terkecil yang di-fine-tune pada data publik dapat mendeteksi **pemain, bola, dan wasit** pada frame siaran sepak bola dengan akurasi layak, diuji pada sumber yang tidak dilihat saat latih, dan berjalan di laptop?

## Jawaban singkat
**Sebagian.** Diukur dengan kriteria yang dibekukan sebelum test (EVAL_PLAN §5, ditetapkan atas delegasi pemilik):

| Kelas | Status | Bukti (test internal, 2.288 gambar) |
|---|---|---|
| Pemain | **Layak** | AP50 0,770 (≥ 0,70). Di RF100 eksternal 0,954 |
| Wasit | **Layak, minimum** | AP50 0,543 (≥ 0,50). Sering tertukar dengan pemain. Precision rendah di RF100 (0,347) |
| Bola | **Belum layak** | Recall 0,288 (≥ 0,50 gagal), precision 0,521 (≥ 0,70 gagal). Interval 95% recall 0,258–0,321 |
| Fine-tuning vs model umum | **Jelas lebih baik** | AP50 pemain +0,376, bola +0,188. Model umum tidak punya kelas wasit |
| Kecepatan laptop | **Cukup untuk offline** | 22 ms/frame GPU, 59 ms/frame CPU (median, YOLOv8n 640). Bukan klaim real-time |
| Ketahanan | Tahan gangguan ringan–sedang | Blur σ2, resolusi ×½, kecerahan ±30%: mAP turun ≤ 7%. Gangguan berat: turun 23–73% |

Menurut aturan yang dibekukan, **kelayakan 3 kelas belum terbukti** karena bola gagal. Rincian lengkap ada di TEST_REPORT.md.

## Apa yang dipelajari
1. **Bola adalah masalah inti.** Bola kecil dan besar sama-sama lemah (recall 0,26 / 0,33). Di frame siaran normal (sumber v12), recall bola hanya 0,41.
2. **Validasi Hari 2 terlalu optimistik.** Recall bola di val 0,734 vs test 0,288. Val hanya satu sumber dan selalu berbola, jadi pemilihan model dan ambang di val tidak mewakili.
3. **Data publik gabungan Roboflow bermasalah:**
   - 137 gambar test (5,6%) ternyata duplikat dari sumber lain dan dikeluarkan.
   - Satu sumber test (v5_temp) berisi augmentasi bawaan (rotasi 90°, mosaik).
   - Banyak orang tidak dilabel: 19 dari 20 "false positive pemain" yang saya periksa adalah orang sungguhan.
   - Lisensi masih tidak diketahui.
4. **Split per sumber penting.** Tanpa itu, kebocoran akan menaikkan skor secara semu.

## Yang TIDAK dibuktikan
- Tracking, real-time, antarmuka: di luar cakupan.
- Kamera taktis/amatir, pertandingan non-siaran: tidak ada datanya.
- Generalisasi luas: hanya 2 domain uji (sumber Soccana yang disisihkan dan RF100), satu model, satu seed, satu run.
- Kualitas label test sendiri: label yang hilang ikut menurunkan precision dan AP. Besarnya tidak diukur.

## Keterbatasan dan risiko
- **Lisensi:** dataset Soccana tidak mencantumkan lisensi. Belum aman untuk publikasi atau komersial.
- **Kriteria:** angka ambang adalah usulan Claude atas delegasi pemilik, ditetapkan setelah melihat skor val. Bukan standar industri.
- **Proses training:**
  - Batch 8 dan 25 epoch, menyesuaikan batas waktu dan memori.
  - Training sempat crash karena RAM habis, lalu dilanjutkan dengan resume; urutan data jadi tidak identik bit per bit.
- **Bias test:** 63% bola test berasal dari sumber yang teraugmentasi.

## Rekomendasi langkah lanjut (belum dikerjakan, untuk diputuskan pemilik)
1. **Data bola lebih baik dulu, baru model.** Contoh:
   - SoccerNet-Tracking/GSR (split per pertandingan, ada bola), jika aksesnya diizinkan.
   - Membersihkan sumber yang teraugmentasi.
   - Audit label yang hilang.
2. **Bangun ulang val** dari beberapa sumber, dengan gambar tanpa bola, agar pemilihan model dan ambang lebih jujur.
3. **Eksperimen khusus bola** (satu perubahan per percobaan), di GPU yang lebih besar (Kaggle/Colab):
   - imgsz 960–1280 atau inferensi berpotongan (SAHI).
   - Model s/m.
   - Epoch lebih banyak.
4. **Test baru yang belum pernah dilihat.** Test sekarang sudah "terpakai"; hasil di atas tidak boleh dipakai untuk memilih perbaikan, dan angka setelah perbaikan harus diukur pada test baru.

## Berkas
- Dokumen: DEFINE_BRIEF, DATA_AUDIT, EVAL_PLAN, SETUP, DEVELOP_BRIEF, ERROR_NOTES, DAY3_PLAN, TEST_REPORT (folder `docs`).
- Hasil: `results/summary.csv`, `results/day3_criteria.json`, `results/runs/`, `results/day3_robustness.csv`.
- Bobot: `results/train/B_yolov8n_e25_s42_b8/weights/best.pt` (tidak di git; SHA-256 di SETUP.md).
