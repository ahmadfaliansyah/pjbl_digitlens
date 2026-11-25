"""Integrasi model CNN untuk klasifikasi satu digit (0-9).

Modul segmentasi menghasilkan satu potongan gambar (crop) per kandidat digit.
Setiap crop diproses di sini menjadi tensor 28x28x1 lalu diklasifikasikan oleh
``models/model_cnn.keras``.

Spesifikasi model (dibaca dari config.json di dalam file .keras):
    - Input  : (28, 28, 1), grayscale
    - Output : softmax 10 kelas -> indeks 0..9 = digit 0..9
    - Tidak ada layer Rescaling di dalam model, sehingga normalisasi pixel
      dilakukan di sini dan HARUS sama dengan saat training.

Preprocessing mengikuti konvensi MNIST: digit putih di atas latar hitam,
sisi terpanjang digit diskalakan ke 20 px, lalu dipusatkan pada kanvas 28x28
berdasarkan center of mass, dan pixel dinormalisasi ke rentang 0-1.
Jika model dilatih dengan format berbeda, ubah konstanta di bagian
"Parameter preprocessing" di bawah.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Lokasi model
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model_cnn.keras"
MODEL_NAME = "CNN"

# -----------------------------------------------------------------------------
# Parameter preprocessing (harus sama dengan pipeline training)
# -----------------------------------------------------------------------------
IMAGE_SIZE = 28          # ukuran input model (28 x 28)
DIGIT_BOX = 20           # sisi terpanjang digit setelah di-resize (gaya MNIST)
PIXEL_SCALE = 255.0      # pembagi normalisasi -> rentang 0..1
MIN_CONTRAST = 10        # selisih terang-gelap minimum agar crop dianggap berisi digit
STROKE_RATIO = 0.12      # tebal goresan target = 12% sisi terpanjang digit (gaya MNIST)


class ModelNotAvailableError(RuntimeError):
    """Model tidak dapat dimuat (file hilang atau backend Keras belum terpasang)."""


# -----------------------------------------------------------------------------
# Memuat model
# -----------------------------------------------------------------------------
def _select_keras_backend() -> list[str]:
    """Pilih backend Keras 3 yang benar-benar bisa diimpor.

    Jika KERAS_BACKEND sudah diatur, nilainya dipakai apa adanya. Jika belum,
    dicoba berurutan tensorflow -> jax -> torch, dan yang pertama berhasil
    diimpor dipilih. Paket yang ada tetapi rusak (instalasi tidak lengkap)
    dilewati. Mengembalikan daftar backend yang gagal untuk pesan error.
    Harus dipanggil sebelum ``import keras``.
    """
    failed: list[str] = []
    if os.environ.get("KERAS_BACKEND"):
        return failed
    for module_name in ("tensorflow", "jax", "torch"):
        if importlib.util.find_spec(module_name) is None:
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failed.append(f"{module_name} ({exc})")
            continue
        os.environ["KERAS_BACKEND"] = module_name
        break
    return failed


@lru_cache(maxsize=1)
def load_model():
    """Muat model sekali saja, lalu pakai ulang untuk semua prediksi."""
    if not MODEL_PATH.exists():
        raise ModelNotAvailableError(
            f"File model tidak ditemukan: {MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}"
        )

    failed_backends = _select_keras_backend()
    try:
        import keras
    except Exception as exc:  # ImportError, atau backend tidak ditemukan
        raise ModelNotAvailableError(
            "Keras belum bisa dipakai. Pastikan library terpasang "
            "(python -m pip install -r requirements.txt). TensorFlow belum menyediakan "
            "paket untuk Python 3.14, gunakan Python 3.10-3.13, atau pasang backend "
            f"lain (jax/torch). Backend yang gagal diimpor: {failed_backends or 'tidak ada'}. "
            f"Detail: {exc}"
        ) from exc

    try:
        return keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as exc:
        raise ModelNotAvailableError(f"Model gagal dimuat: {exc}") from exc


def is_model_available() -> bool:
    """True jika model dapat dimuat (dipakai untuk menampilkan status di UI)."""
    try:
        load_model()
    except ModelNotAvailableError:
        return False
    return True


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def preprocess_crop(crop: Image.Image) -> np.ndarray:
    """Ubah crop hasil segmentasi menjadi array float32 (28, 28) rentang 0..1.

    Langkah: hitung jarak warna tiap pixel dari warna latar (digit terang, latar
    gelap, digit berwarna aman) -> ambil komponen terbesar (membuang sisa digit tetangga
    dan noise) -> potong ke bounding box -> tebalkan goresan tipis -> resize ke
    20 px -> letakkan di kanvas 28x28 -> pusatkan berdasarkan center of mass ->
    normalisasi.
    """
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32)
    if rgb.shape[0] < 2 or rgb.shape[1] < 2:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # Foreground = seberapa jauh warna sebuah pixel dari warna latar (median
    # pixel di tepi crop). Cara ini bekerja untuk digit gelap di latar terang,
    # terang di latar gelap, maupun digit berwarna. Grayscale biasa tidak cukup:
    # warna terang seperti kuning hampir sama terangnya dengan latar putih,
    # sehingga bagian itu hilang dan digit terpotong.
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=2))

    peak = float(distance.max())
    if peak < MIN_CONTRAST:  # hampir seragam -> tidak ada digit yang jelas
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    # Hasilnya selalu digit terang di atas latar gelap (gaya MNIST).
    gray = (distance * (255.0 / peak)).astype(np.uint8)

    # Hanya pertahankan komponen terbesar. Segmentasi memotong crop dengan
    # padding, sehingga potongan digit tetangga bisa ikut masuk di tepi crop.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8)
    keep = cv2.dilate(component, np.ones((3, 3), np.uint8), iterations=1)  # sisakan tepi anti-alias
    gray = gray * keep

    ys, xs = np.nonzero(gray)
    top_y, bottom_y, left_x, right_x = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    digit = gray[top_y:bottom_y, left_x:right_x]
    component = component[top_y:bottom_y, left_x:right_x]

    # Goresan tipis di gambar beresolusi besar akan memudar saat diperkecil ke
    # 20 px, padahal digit MNIST bergoresan tebal (sekitar 12% dari tinggi digit).
    # Tebalkan goresan secara proporsional sebelum resize.
    height, width = digit.shape
    longest = max(height, width)
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 3)
    stroke_width = 4.0 * float(distance[component > 0].mean())  # perkiraan tebal goresan
    missing = STROKE_RATIO * longest - stroke_width
    if missing >= 1.0:
        kernel_size = int(round(missing)) + 1
        kernel_size += 1 - kernel_size % 2  # harus ganjil
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        pad = kernel_size // 2
        digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        digit = cv2.dilate(digit, kernel)
        height, width = digit.shape

    # Resize dengan menjaga rasio aspek, sisi terpanjang = DIGIT_BOX.
    scale = DIGIT_BOX / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    top = (IMAGE_SIZE - new_h) // 2
    left = (IMAGE_SIZE - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = digit

    # Pusatkan berdasarkan center of mass, seperti pembuatan dataset MNIST.
    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        target = (IMAGE_SIZE - 1) / 2.0
        shift = np.float32([[1, 0, target - center_x], [0, 1, target - center_y]])
        canvas = cv2.warpAffine(canvas, shift, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR)

    # Digit MNIST memiliki pixel paling terang bernilai penuh (255).
    if canvas.max() > 0:
        canvas = (canvas.astype(np.float32) * (255.0 / canvas.max())).astype(np.uint8)

    return canvas.astype(np.float32) / PIXEL_SCALE


# -----------------------------------------------------------------------------
# Prediksi
# -----------------------------------------------------------------------------
def predict_digits(crops: Sequence[Image.Image]) -> list[dict]:
    """Klasifikasikan banyak crop sekaligus (satu pemanggilan model).

    Mengembalikan satu dict per crop, sesuai urutan input:
        {"digit": "8", "confidence": 0.97, "model_name": "CNN",
         "alternatives": [("3", 0.02), ("0", 0.01)]}
    """
    if not crops:
        return []

    model = load_model()
    batch = np.stack([preprocess_crop(crop) for crop in crops])[..., np.newaxis]
    probabilities = np.asarray(model.predict(batch, verbose=0))

    results: list[dict] = []
    for row in probabilities:
        order = np.argsort(row)[::-1]
        best = int(order[0])
        results.append(
            {
                "digit": str(best),
                "confidence": float(row[best]),
                "model_name": MODEL_NAME,
                # Dua tebakan terbaik berikutnya: (digit, probabilitas).
                "alternatives": [(str(int(i)), float(row[i])) for i in order[1:3]],
            }
        )
    return results


def predict_one_digit(crop: Image.Image) -> dict:
    """Klasifikasikan satu digit hasil segmentasi.

    Return:
        {"digit": "8", "confidence": 0.97, "model_name": "CNN"}
    """
    return predict_digits([crop])[0]
