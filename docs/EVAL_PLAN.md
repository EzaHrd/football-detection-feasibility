# EVAL_PLAN — Rencana pembagian data & evaluasi

Semua angka kriteria lulus **TBD** (ditetapkan pemilik proyek). Tidak ada hasil model pada dokumen ini; belum ada model yang dilatih.

## 1. Rencana split (berbasis sumber, bukan frame acak)
Split bawaan dataset A dibuang karena acak per gambar (bocor). Unit pemisah = **kelompok sumber** (prefiks nama berkas tanpa `_sahi_N` dan indeks). Manifest: `data/work/splits/split_by_source_v1.csv` dibuat oleh `scripts/make_split.py` (tidak mengubah data asli; skrip memverifikasi tiap kelompok hanya di satu split).

| Split | Kelompok sumber | Gambar | Catatan |
|---|---|---|---|
| train | v3, v7, v2_temp, spt_v2 | 8.451 | augmentasi hanya di sini |
| val | tbd_v2 | 797 | untuk pemilihan epoch/hiperparameter; kaya bola (653 kotak) |
| test (internal, domain-berbeda) | v12, v5_temp | 2.425 | tidak disentuh sampai evaluasi akhir |
| test eksternal | seluruh RF100 soccer-players-5fuqs | 163 | tidak dipakai latih/val; pemetaan football→bola |

Jumlah kotak per sumber ada di `results/audit_soccana_by_source.csv`. Distribusi bola per split belum dicek menyeluruh; v12+v5_temp punya 814 kotak bola di test, tbd_v2 653 di val (dihitung dari tabel).

**Risiko kebocoran yang tersisa (jujur):** (a) satu kelompok sumber bisa berisi banyak video, dan beberapa sumber bisa memakai pertandingan/siaran yang sama — tidak diketahui; (b) kelompok sumber mungkin berisi duplikat dari augmentasi; (c) split per sumber membuat test lebih sulit daripada split acak, sehingga skor lebih rendah dan **tidak sebanding** dengan angka dari proyek lain yang memakai split bawaan.
**Pemeriksaan wajib sebelum melatih (Hari 2):** hitung pHash antar split lintas-sumber (train↔val, train↔test, val↔test); gambar dengan jarak ≤ 4 dikeluarkan dari test/val dan dicatat; pastikan tidak ada nama/hash identik di train dan test. Jika SoccerNet (D) dapat diakses, ganti dengan split per pertandingan.

## 2. Metrik
- mAP@0.5 dan mAP@0.5:0.95 (keseluruhan dan per kelas) — Ultralytics `val`.
- Precision & recall per kelas, dengan sorotan **bola**; kurva P-R dan F1 untuk memilih ambang kepercayaan (dipilih di val, bukan di test).
- Confusion matrix (termasuk latar belakang). Perhatian khusus: wasit↔pemain.
- Kecepatan: ms per frame (pra-proses + inferensi + NMS) di laptop ini, CPU dan GPU, resolusi 640, batch 1, N frame setelah pemanasan; laporkan median dan p95.
- Subset: (i) semua gambar test, (ii) hanya gambar ≥ 640 px (mendekati frame penuh), (iii) bola kecil vs besar menurut luas kotak.
- Karena gambar test bervariasi dan bola sedikit, laporkan jumlah kotak per kelas bersama tiap metrik; pertimbangkan bootstrap untuk interval kepercayaan bola.

## 3. Uji ketahanan
Diterapkan hanya pada **salinan** gambar test (tidak pada train), tiap gangguan satu per satu dengan beberapa tingkat (tingkat ditetapkan di skrip, ditulis di hasil):
Gaussian blur; motion blur; penurunan resolusi (turun lalu naik kembali, mis. faktor 2 dan 4); kecerahan/kontras naik-turun; kompresi JPEG berat; noise. Laporkan penurunan mAP dan recall bola relatif terhadap gambar bersih.
Uji domain shift: (1) test internal v12/v5_temp (sumber tak terlihat); (2) test eksternal RF100; (3) bila memungkinkan, beberapa klip video publik lain yang dilabel/diperiksa manual (jumlah dan sumber TBD).

## 4. Pembanding
| Model | Keterangan |
|---|---|
| P0 | YOLO pretrained COCO tanpa fine-tune: `person`→ (pemain/wasit tidak terbedakan, dilaporkan sebagai kelas gabungan), `sports ball`→bola. Pemetaan dicatat; wasit tidak dapat dinilai |
| P1 | Model yang sama di-fine-tune pada train 3 kelas |
Ukuran model kandidat: YOLOv8n/s (muat di 6 GB VRAM; ukuran akhir diputuskan Hari 2). Seed dan hiperparameter dicatat.

## 5. Kriteria lulus (DIBEKUKAN 2026-11-10, sebelum test disentuh)

**Asal angka:** pemilik proyek memilih Jalur 1 lalu mendelegasikan penetapan angka ke Claude ("saya ikut claude saja", 2026-11-10). Angka di bawah adalah **usulan Claude**, ditetapkan sebelum inferensi apa pun pada test internal atau RF100. Ini penilaian (judgment) untuk studi kelayakan, bukan standar industri, dan tidak divalidasi dengan pengguna. Pada saat menetapkan, Claude sudah melihat hasil val Hari 2 (lihat DEVELOP_BRIEF.md), jadi ada risiko bias ke arah angka yang "bisa dicapai". Angka sengaja dibuat lebih rendah dari skor val, karena test berasal dari sumber yang belum pernah dilihat model. **Angka tidak boleh diubah setelah hasil test terlihat.**

P/R dihitung pada ambang conf **per kelas** yang dibekukan dari F1 terbaik di val (DAY3_PLAN T1). AP dihitung pada conf 0,001. IoU pencocokan 0,5.

| # | Kriteria | Ambang | Data | Alasan singkat |
|---|---|---|---|---|
| 1 | AP50 pemain | ≥ 0,70 | test internal | Objek besar dan banyak. Harus jelas di atas detektor umum (A val: 0,64) |
| 2 | AP50 wasit | ≥ 0,50 | test internal | Kelas minoritas, sering mirip pemain. Cukup untuk membuktikan kelas ini bisa dipelajari |
| 3 | Recall bola | ≥ 0,50 | test internal; subset ≥ 640 px dilaporkan dengan ambang yang sama | Minimal separuh bola per frame terdeteksi. Celah bisa ditutup tracking di tahap lanjut (di luar cakupan) |
| 4 | Precision bola | ≥ 0,70 | test internal | Positif palsu bola merusak analisis lanjut. Dibuat lebih ketat daripada recall |
| 5 | mAP50 3 kelas | ≥ 0,50 | RF100 (eksternal) | Domain lain (siaran Liga Inggris, dataset berbeda). Bukti generalisasi minimum |
| 6 | Penurunan relatif mAP50 maksimum | blur Gaussian σ = 2: ≤ 25%; resolusi ×½: ≤ 25%; kecerahan ±30%: ≤ 15% | test internal yang diberi gangguan | Dinilai pada tingkat gangguan sedang. Tingkat lain hanya dilaporkan |
| 7 | ms per frame (median, batch 1, 640) | GPU ≤ 50; CPU ≤ 200 | laptop ini | Setara ≥ 20 fps GPU untuk pemrosesan offline. **Bukan klaim real-time** |
| 8 | Kenaikan B vs A | AP50 pemain dan AP50 bola masing-masing naik ≥ 0,10 (absolut) | test internal | Fine-tuning harus memberi nilai tambah yang jelas, bukan selisih kecil |

**Aturan penilaian:** kelayakan teknis disebut **terbukti** jika kriteria 1–5 dan 8 lulus. Kriteria 6–7 dilaporkan sebagai lulus/gagal tetapi tidak membatalkan kesimpulan; kegagalannya dicatat sebagai risiko. Setiap baris dilaporkan apa adanya, termasuk yang gagal.

## Pertanyaan terbuka
- Siapa yang menetapkan ambang, dan kapan?
- Apakah bola dihitung "lulus" pada recall atau mAP?
- Izin menambah klip video publik untuk uji domain shift?
