# DEFINE_BRIEF — Studi kelayakan deteksi pemain, bola, wasit

> Status: Hari 1 dari 3. Ini **studi kelayakan teknis**, bukan produk.

## Masalah
Analisis video sepak bola (statistik posisi, taktik) membutuhkan langkah dasar: mengetahui di mana pemain, bola, dan wasit pada tiap frame. Pertanyaan teknis proyek ini: **apakah detektor objek keluarga YOLO yang di-fine-tune pada data publik dapat mendeteksi 3 kelas (pemain, bola, wasit) dengan akurasi yang layak di video siaran, dan berjalan pada laptop?**

## Pengguna sasaran (HIPOTESIS — belum divalidasi, tidak ada wawancara)
- Hipotesis H1: peneliti/mahasiswa visi komputer yang butuh baseline deteksi sepak bola.
- Hipotesis H2: pengembang aplikasi analitik olahraga yang akan menambahkan tracking di tahap lanjut.
Tidak ada klaim bahwa ini produk untuk pelatih amatir.

## Problem statement
Dengan hanya data publik dan satu laptop, ukur seberapa baik model YOLO (pretrained vs fine-tuned) mendeteksi pemain, bola, dan wasit, terutama **bola** (objek kecil), pada data uji yang dipisah berdasarkan sumber/video, serta seberapa tahan terhadap gangguan citra.

## Batas cakupan
Dalam cakupan: deteksi 3 kelas, evaluasi (mAP, P/R per kelas, confusion matrix, ms/frame), uji ketahanan.
Di luar cakupan: tracking, real-time, antarmuka pengguna, identifikasi tim/nomor punggung, deployment, wawancara pengguna.

## Kendala
3 hari total; data publik; laptop (lihat SETUP.md); angka kriteria lulus ditetapkan pemilik proyek (lihat EVAL_PLAN.md).
