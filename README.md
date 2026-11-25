# DigitLens — Pengenalan Angka Multi-digit dari Gambar

Project akhir Mata Kuliah Pembelajaran Mesin (ML2026).

> **Isi bagian ini sebelum dikumpulkan**
> - Kelas / Nama kelompok: `...`
> - Anggota & NIM: `...` · `...` · `...`
> - Pembagian tugas: lihat bagian *Pembagian tugas* di bawah.

## Ringkasan

DigitLens membaca **rangkaian angka dalam satu gambar**. Alurnya:

```
Gambar → Preprocessing (grayscale, filter, threshold) → Segmentasi digit (connected components)
       → Preprocessing per-digit (gaya MNIST 28×28) → Klasifikasi CNN (0–9) → Gabung kiri→kanan → Angka
```

- **Jenis masalah:** klasifikasi citra multi-kelas (10 kelas) + segmentasi.
- **Dataset pelatihan:** MNIST (LeCun et al., 1998) — `keras.datasets.mnist`.
- **Algoritma yang dibandingkan:** CNN, SVM (RBF + PCA), Random Forest — lihat `notebooks/ML_DigitLens_Training_Evaluation.ipynb`.
- **Model di aplikasi:** CNN (`models/model_cnn.keras`), dengan alasan pemilihan di bagian 8 notebook.

## Struktur proyek

```
app.py                      # entry point Streamlit
digitlens/
  segmentation.py           # segmentasi digit (connected components, + margin otomatis)
  predictor.py              # preprocessing per-digit + prediksi CNN
  storage.py  history_store.py  ui.py
  pages/ recognizer.py history.py about.py
models/model_cnn.keras      # model CNN yang dipakai aplikasi
notebooks/                  # EDA, training, evaluasi, perbandingan model
scripts/
  make_synthetic_testset.py # membuat gambar uji sintetis
  eval_end_to_end.py        # akurasi pipeline lengkap pada examples/testset
examples/sample_253.png     # contoh gambar
examples/testset/           # gambar uji: <angka>__<keterangan>.png
reports/                    # keluaran notebook (CSV metrik, grafik)
```

## Menjalankan aplikasi

Gunakan **Python 3.12** (TensorFlow belum mendukung 3.14).

```bash
py -3.12 -m venv C:\venvs\digitlens
C:\venvs\digitlens\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py     # buka http://localhost:8501
```

Coba dengan `examples/sample_253.png`; hasil yang diharapkan: tiga kandidat digit dan angka terbaca `253`.

## Menjalankan notebook ML

```bash
python -m pip install -r requirements-training.txt
jupyter notebook notebooks/ML_DigitLens_Training_Evaluation.ipynb
```

Atau unggah ke Google Colab (jalankan dari root proyek agar `ROOT` benar). Mode normal membutuhkan beberapa menit
(pelatihan SVM pada seluruh data latih adalah bagian paling lama). Hasil tersimpan di `reports/`.
Model baru disimpan sebagai `models/model_cnn_retrained.keras` agar `model_cnn.keras` tidak tertimpa.

## Pengujian end-to-end

```bash
python scripts/make_synthetic_testset.py     # opsional: 40 gambar sintetis
python scripts/eval_end_to_end.py            # akurasi segmentasi, angka persis, dan per-digit
```

Tambahkan gambar nyata milik kelompok ke `examples/testset/` (format `<angka>__<ket>.png`) agar hasil mewakili penggunaan sebenarnya.

## Cara kerja segmentasi

1. Gambar → grayscale, filter *bilateral* untuk meredam noise dengan tetap menjaga tepi.
2. Beberapa kandidat threshold (Otsu dan dua yang lebih ketat) × dua polaritas (digit gelap/terang) dibentuk, lalu **dipilih yang paling "mirip digit"** lewat skor heuristik.
3. *Connected components* → komponen kecil, terlalu tipis, atau menyentuh tepi gambar disaring.
4. Jika angka tampak menyentuh tepi (crop terlalu rapat), gambar diberi **margin otomatis** dan hasilnya dipakai bila lebih meyakinkan.
5. Kandidat diurutkan per baris lalu kiri→kanan dan dipotong.

## Preprocessing per-digit & model

- Input CNN `28×28×1`, output softmax 10 kelas; normalisasi `÷255` dilakukan di `predictor.py` (tidak ada layer Rescaling di model).
- Per crop: samakan polaritas (digit terang/latar gelap) → regangkan kontras → ambil komponen terbesar → bounding box → tebalkan goresan tipis → resize sisi terpanjang 20 px → letakkan di kanvas 28×28 → pusatkan dengan *center of mass*.
- **Preprocessing ini harus konsisten dengan pipeline training.** Parameter ada di bagian "Parameter preprocessing" pada `predictor.py`.
- Hasil dengan confidence < 80% ditandai agar diperiksa manual.

## Batasan

- Fokus pada satu baris angka dengan jarak antar-digit; digit menempel, tulisan bertumpuk, banyak baris, dan latar kompleks belum ditangani andal. Beberapa baris akan digabung menjadi satu angka (aplikasi memberi peringatan).
- CNN dilatih pada MNIST (tulisan tangan); font cetak atau foto dapat menurunkan akurasi (mis. digit `1` berkaki serif tertukar). Evaluasi pada gambar nyata ada di `reports/end_to_end_results.csv`.
- Riwayat hanya di memori sesi (tombol *Download history* untuk CSV). Gambar yang diunggah disimpan di `storage/uploads/`.
- Hanya dijalankan lokal; tidak ada deployment.

## Pembagian tugas

| Anggota | Tugas |
|---|---|
| `...` | `...` |
| `...` | `...` |
| `...` | `...` |

## Referensi

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11).
- Studi literatur lengkap (min. 5 jurnal): `docs/studi_literatur.md` *(belum diisi)*.
