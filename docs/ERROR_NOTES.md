# ERROR_NOTES — Analisis error awal (Model B, val)

Setelan: bobot `best.pt` Model B, **conf 0,25**, NMS IoU 0,6, kecocokan IoU ≥ 0,5 (`scripts/error_analysis.py`). Semua 797 gambar val dianalisis; total **1.616 error**. Dari semuanya diambil **44 contoh acak** (seed 42, proporsional, minimal 1 per jenis) dan sampai 3 gambar per jenis disimpan di `results/error_analysis/B_yolov8n_e25_val/<jenis>/` (kotak merah = anotasi GT, biru = prediksi). Daftar sampel: `errors.json`. Hanya val; test tidak disentuh.

## Jumlah per jenis (semua val)
| Jenis | Jumlah | Arti |
|---|---|---|
| latar_jadi_pemain | 623 | prediksi pemain tanpa GT yang tumpang tindih (IoU < 0,1) |
| lokalisasi_buruk_pemain | 464 | kelas benar, IoU 0,1–0,5 (GT atau prediksi) |
| bola_tidak_terdeteksi | 166 | GT bola tanpa prediksi |
| latar_jadi_bola | 77 | prediksi bola tanpa GT |
| pemain_ganda | 63 | prediksi kedua pada pemain yang sudah cocok |
| lokalisasi_buruk_wasit | 53 | |
| salah_kelas_wasit_jadi_pemain | 50 | GT wasit diprediksi pemain |
| pemain_tidak_terdeteksi | 42 | |
| lokalisasi_buruk_bola | 24 | |
| latar_jadi_wasit | 23 | |
| salah_kelas_pemain_jadi_wasit | 11 | |
| wasit_tidak_terdeteksi | 9 | |
| salah_kelas_bola_jadi_pemain | 7 | |
| salah_kelas_pemain_jadi_bola | 2 | |
| wasit_ganda | 2 | |
Catatan hitung: satu GT yang gagal bisa dihitung sekali, dan satu prediksi berlebih sekali; "lokalisasi_buruk" dapat muncul dua kali (sisi GT dan sisi prediksi) untuk satu objek.

## Temuan
1. **Terbanyak: "latar jadi pemain" (623) dan lokalisasi pemain (464).** Pada 3 contoh `latar_jadi_pemain` yang saya lihat, prediksi jatuh pada **pemain sungguhan yang tampaknya tidak dilabel** (pemain terpotong di tepi gambar, pemain di latar belakang). **Hipotesis (belum diuji):** sebagian besar "false positive" ini adalah label yang hilang di data, bukan salah model; jika benar, presisi pemain (0,78) terlalu rendah dari yang sebenarnya. Perlu pengecekan manual lebih banyak sebelum disimpulkan. Per instruksi, label tidak diubah.
2. **Bola: 166 tidak terdeteksi, 77 positif palsu, 24 lokalisasi buruk.** Contoh miss: bola di gawang/jaring yang bercampur latar, bola sangat kecil/buram, objek kuning yang ambigu. Ini sejalan dengan recall bola 0,683.
3. **Wasit ↔ pemain:** 50 wasit diprediksi pemain vs 11 pemain diprediksi wasit; wasit hampir selalu terdeteksi (hanya 9 terlewat) tetapi kelasnya sering salah. Penyebab (misal warna seragam) belum diselidiki.
4. Duplikasi: pemain ganda 63 (relatif kecil dibanding 3.133 pemain).
5. Keterbatasan: hanya 44 contoh terlihat langsung dan hanya sebagian yang saya periksa visual; pengelompokan otomatis berbasis IoU, jadi kategori "latar" dan "lokalisasi" bisa tercampur.

## Pertanyaan terbuka
- ~~Apakah pemilik proyek mau pengecekan manual untuk menilai seberapa banyak "latar_jadi_pemain" sebenarnya label yang hilang?~~ Dijawab di Hari 3 (lihat di bawah).

---

# Hari 3: error pada test internal dan cek label

## Test internal (Model B, ambang conf beku per kelas, 2.288 gambar)
Skrip: `error_analysis.py ... B_day3_test beku .../test.txt`. Total **7.078 error**; 44 contoh acak (seed 42) disimpan di `results/error_analysis/B_day3_test/`.

| Jenis | Jumlah |
|---|---|
| pemain_tidak_terdeteksi | 3.921 |
| lokalisasi_buruk_pemain | 1.364 |
| wasit_tidak_terdeteksi | 523 |
| bola_tidak_terdeteksi | 423 |
| lokalisasi_buruk_bola | 165 |
| salah_kelas_wasit_jadi_pemain | 153 |
| lokalisasi_buruk_wasit | 136 |
| latar_jadi_pemain | 126 |
| latar_jadi_bola | 102 |
| latar_jadi_wasit | 74 |
| salah_kelas_pemain_jadi_wasit | 59 |
| pemain_ganda | 27 |
| wasit_ganda | 5 |

Berbeda dengan val, error terbanyak di test adalah **objek terlewat**, bukan false positive. Dari 706 bola, 423 tidak terdeteksi dan 165 lokalisasinya buruk.

Contoh yang dilihat langsung (9 gambar):
- Gambar v5_temp yang **diputar 90°** dan berisi tempelan potongan gambar lain (mosaik). Pemain "berbaring" sering terlewat.
- Bola sangat kecil di frame normal v12.
- Bola di jaring.

Indikasi augmentasi bawaan di v5_temp: 34,3% kotak pemainnya melintang, sumber lain ≤ 3,3% (`results/day3_per_source.json`).

## Cek manual K11: apakah "latar_jadi_pemain" di val sebenarnya label hilang?
20 contoh acak (seed 42) dari kategori `latar_jadi_pemain` di val (ambang beku), disimpan di `results/error_analysis/B_day3_val_labelcheck/latar_jadi_pemain/`. Lembar ringkas: `_latar_jadi_pemain_20.jpg`.

Hasil (penilaian visual saya, bukan anotator independen):
- **≈ 19 dari 20 adalah orang sungguhan yang tidak dilabel**, kebanyakan pemain terpotong di tepi gambar/irisan.
- 1 orang di luar lapangan (staf/fotografer).
- Tidak ada yang jelas berupa objek latar.

Kesimpulan: hipotesis Hari 2 didukung. Sebagian besar "false positive pemain" adalah label yang hilang, sehingga precision pemain yang dilaporkan cenderung **lebih rendah** dari kemampuan model sebenarnya. Sampel hanya 20, jadi angka pastinya tidak diketahui. Label tidak diubah.
