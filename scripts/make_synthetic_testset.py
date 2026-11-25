"""Buat set uji end-to-end sintetis (angka cetak multi-digit) di examples/testset/.

Tujuan: mengukur akurasi pipeline lengkap (segmentasi + preprocessing + CNN)
pada gambar yang BUKAN bagian dari MNIST. Ini hanya pelengkap. Untuk laporan,
tambahkan juga foto/tulisan tangan milik kelompok sendiri ke folder yang sama
dengan format nama file:  <angka>__<keterangan>.png   (contoh: 90417__foto1.png)

Jalankan dari root proyek:  python scripts/make_synthetic_testset.py
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "testset"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONTS = ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf"]
SEED = 7


def render(text: str, font_name: str, size: int, margin: int, noise: float, tilt: float) -> Image.Image:
    font = ImageFont.truetype(str(FONT_DIR / font_name), size)
    left, top, right, bottom = font.getbbox(text)
    w, h = right - left, bottom - top
    img = Image.new("L", (w + 2 * margin, h + 2 * margin), 255)
    ImageDraw.Draw(img).text((margin - left, margin - top), text, font=font, fill=20)
    if tilt:
        img = img.rotate(tilt, expand=True, fillcolor=255, resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    if noise:
        arr += np.random.normal(0, noise, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")


def main(n: int = 40) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    for i in range(n):
        length = random.choice([2, 3, 4, 5])
        digits = "".join(random.choice("0123456789") for _ in range(length))
        font = FONTS[i % len(FONTS)]
        margin = random.choice([0, 12, 40])  # 0 = di-crop rapat
        noise = random.choice([0, 0, 8, 16])
        tilt = random.choice([0, 0, -4, 4])
        size = random.choice([90, 140])
        img = render(digits, font, size, margin, noise, tilt)
        tag = f"{Path(font).stem}_m{margin}_n{int(noise)}_t{int(tilt)}"
        img.save(OUT / f"{digits}__{tag}_{i:02d}.png")
    print(f"{n} gambar uji dibuat di {OUT}")


if __name__ == "__main__":
    main()
