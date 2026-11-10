# DATA_AUDIT — Audit kandidat dataset

Tanggal audit: 2026-11-05. **Fakta** = dicek langsung pada data; **Asumsi/klaim** = dari deskripsi dataset, belum diverifikasi.

## Tabel kandidat
| # | Dataset | Sumber | Gambar | Kelas | Format | Lisensi | Dari video? | Status audit |
|---|---|---|---|---|---|---|---|---|
| A | Soccana_player_ball_detection_v1 | [HF Adit-jain](https://huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1) | **11.673** terhitung (train 8.677, test 2.996). README menyebut 25.000 → **tidak cocok** | Player, Ball, Referee (goalkeeper digabung ke Player menurut README) | YOLO txt + COCO json | **Tidak tertera** pada kartu/tag HF. Gabungan 8 sumber Roboflow dengan lisensi masing-masing (tidak diperiksa) | Frame siaran (klaim README: SoccerNet, render game, rekaman publik); video/pertandingan asal **tidak diketahui** per gambar | Diaudit penuh |
| B | soccer-players-5fuqs (RF100) | [HF Francesco](https://huggingface.co/datasets/Francesco/soccer-players-5fuqs) | 163 (114/16/33), semua 640x640 | soccer-players (kelas induk, tak terpakai), football, player, referee | Parquet + COCO | Tag HF "cc" (tidak spesifik; kartu merujuk halaman Roboflow) | Terlihat frame siaran, sebagian besar satu pertandingan (Burnley–Arsenal, sebagian kecil Man City–Newcastle, dari overlay skor) | Diaudit penuh |
| C | Kandidat Roboflow Universe (FutVAR, football-players-ball-referee-detection, football-player-detection yolotest, dll.) | universe.roboflow.com | tidak diketahui | ada ball/player/referee (dari hasil pencarian) | YOLO/COCO | Sebagian tertera CC BY 4.0 di hasil pencarian, belum diverifikasi | tidak diketahui | **Tidak diaudit**: halaman memberi HTTP 403 pada pengambilan otomatis, dan unduhan memerlukan akun/API key (pembuatan akun tidak saya lakukan). Tidak dicari jalan pintas |
| D | SoccerNet-Tracking / GSR | soccer-net.org | 200 sekuens 30 dtk, 12 pertandingan (dari paper) | player, goalkeeper, referee, ball, other | MOT | arXiv mencantumkan CC BY 4.0 (itu lisensi paper, bukan data) | Ya, per pertandingan | **Tidak diunduh/diaudit** hari ini (akses data & syarat belum diperiksa). Opsi terbaik untuk split per pertandingan jika dapat diakses |

## Hasil audit A (Soccana)
- Pemetaan kelas sesuai `data.yaml`: 0 Player, 1 Ball, 2 Referee. **Tabel kelas di README salah urutan** (menulis 1=Referee, 2=Ball); pemeriksaan visual & ukuran kotak (kelas 1 median luas ≈ 0,0004 dari gambar = kecil) memastikan 1 = Ball.
- Jumlah kotak (train/test): Player 43.438/12.334; Ball 2.863/737; Referee 4.234/1.123.
- 28% gambar train (2.460) dan 35% test (1.041) tanpa label (latar saja/irisan tanpa objek).
- Ukuran gambar sangat beragam: 160², 320², 640², 1080², 1280², 1920x1080, 640x360 — banyak gambar adalah **irisan SAHI** dari frame yang lebih besar, bukan frame penuh.
- Cakupan per sumber (gabungan train+test): spt_v2 140 gambar; tbd_v2 797; v12 1.725; v2_temp 1.290; v3 3.475; v5_temp 700; v7 3.546. Bola per sumber: 50 s/d 945 (`results/audit_soccana_by_source.csv`).
- **Kebocoran pada split bawaan (fakta):** semua 7 sumber muncul di train sekaligus test (split acak per gambar). Sebanyak 119 dari 2.996 gambar test (4%) punya gambar train dengan jarak pHash ≤ 4, 203 (7%) dengan ≤ 8; pada sumber v3 25%, v5_temp 12,5%, v2_temp 22,5% (≤4). pHash tidak menangkap irisan SAHI dari frame yang sama pada area berbeda, jadi kebocoran sebenarnya kemungkinan **lebih besar**.
- Sampel visual (12 gambar acak train, `results/audit_soccana_sample_sheet.jpg`): kotak pemain, bola, wasit terlihat wajar pada sampel itu; ada irisan latar kosong dan pemain terpotong di tepi irisan. Hanya 12 gambar — **bukan jaminan konsistensi label**. Konsistensi pelabelan bola/wasit antar sumber belum terukur.
- Sumber kemungkinan mengandung augmentasi bawaan Roboflow (tidak diketahui).

## Hasil audit B (RF100)
- Kotak: player 1.766 total, football 141, referee 96. Label pada 8 sampel uji terlihat benar dan konsisten (bola kecil, wasit berbaju gelap).
- Terlalu kecil (163 gambar, 141 bola) untuk melatih; cocok sebagai **uji eksternal**. Overlap pHash ≤ 6 dengan gambar Soccana ≥ 320 px: 0 dari 163 (fakta, tetapi pengecekan terbatas pada gambar ≥ 320 px).
- Split bawaannya juga acak antar frame satu pertandingan: test vs train pHash ≤ 4 = 3/33.

## Dataset terpilih
**Utama: A (Soccana), dipakai sebagai kumpulan gambar yang di-split ulang per sumber** (bukan split bawaan). **Uji eksternal domain-shift: B (RF100)**, tidak dipakai untuk latih/validasi.
Alasan: hanya A cukup besar (2.863 + 737 bola) dan multi-sumber; B terlalu kecil tetapi terpisah dari A. Kelas dipetakan: Player→pemain, Ball→bola, Referee→wasit. Untuk B: player→pemain, football→bola, referee→wasit, `soccer-players`→dibuang (kelas induk).

## Risiko
1. **Lisensi tidak jelas (tinggi):** A tidak mencantumkan lisensi; turunan 8 dataset Roboflow. Aman untuk studi internal, belum aman untuk dipublikasikan/dikomersialkan. Perlu diverifikasi per sumber.
2. **Kebocoran data** (lihat atas) — dimitigasi dengan split per sumber, tetapi satu sumber bisa memuat banyak video dan sumber berbeda bisa memuat pertandingan yang sama (tidak diketahui).
3. **Irisan SAHI & resolusi campur:** distribusi latih tidak sama dengan inferensi frame penuh; metrik harus dilaporkan juga pada subset gambar ≥ 640 px.
4. **Bola sedikit & kecil:** hanya 2.863 kotak bola di train bawaan; sumber spt_v2 hanya 50.
5. **Label tidak konsisten antar sumber** (kiper → Player; asal-usul dan kebijakan pelabelan bola tidak diketahui); hanya diperiksa pada 12 + 8 sampel.
6. **Bias domain:** dominan siaran TV; kamera taktis/amatir tidak terwakili.
7. Kartu dataset A memuat ketidaksesuaian (25.000 vs 11.673; tabel kelas) → kualitas dokumentasi rendah.

## Pertanyaan terbuka
- Lisensi tiap sumber Roboflow A; apakah kita boleh memakainya untuk laporan publik?
- Apakah pemilik punya akun/API key Roboflow agar kandidat C bisa diaudit?
- Apakah akses SoccerNet (D) diinginkan?
