# Catatan perubahan (hasil review)

| File | Perubahan | Alasan |
|---|---|---|
| `digitlens/segmentation.py` | Tambah `_add_margin()` dan percobaan kedua dengan margin; pilih hasil dengan skor lebih tinggi | Angka yang di-crop rapat kehilangan digit di tepi (90417 → 3 kandidat; 1 digit → 0 kandidat) |
| `digitlens/storage.py` | Nama file = hash isi gambar | Sebelumnya timestamp membuat file yang sama tersimpan berulang |
| `digitlens/history_store.py` | Kolom baru `Angka terbaca`, `Confidence minimum` | Output utama aplikasi sebelumnya hanya ada di teks status |
| `digitlens/pages/recognizer.py` | Peringatan multi-baris; pesan error menyebut tipe error; label "Threshold terpilih"; hapus pesan "Gambar asli tersimpan"; kirim angka+confidence ke riwayat | Akurasi pesan dan kelengkapan riwayat |
| `digitlens/pages/history.py` | Format kolom confidence | — |
| `digitlens/pages/about.py` | Perbaiki salah ketik "Ahmad,"; perbarui status dan teks usang | Teks tidak sesuai kondisi aplikasi |
| `app.py` | Hapus ±300 baris kode lama yang di-comment | Kebersihan kode |
| semua halaman | `use_container_width=True` → `width="stretch"` | Parameter lama deprecated dan dijadwalkan dihapus Streamlit |
| `requirements.txt` | `streamlit>=1.50` | Dibutuhkan oleh `width="stretch"` |
| baru | `notebooks/…ipynb`, `scripts/*`, `requirements-training.txt`, `.gitignore`, README baru | Bagian ML dan dokumentasi yang sebelumnya kosong |

## Hasil uji setelah perubahan (40 gambar sintetis, `scripts/eval_end_to_end.py`)

| Metrik | Segmentasi lama | Segmentasi baru |
|---|---|---|
| Jumlah kandidat = jumlah digit | 77,5% | 92,5% |
| Angka terbaca persis | 60,0% | 75,0% |
| Akurasi per-digit (segmentasi benar) | 92,8% | 94,2% |

Kegagalan yang tersisa: kesalahan klasifikasi pada font berkaki serif (mis. 4→1, 1→2, 6→5) dan digit menempel akibat noise.
Ini data sintetis dengan font DejaVu — bukan pengganti uji pada gambar nyata milik kelompok.
