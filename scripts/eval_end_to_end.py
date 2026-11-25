"""Evaluasi end-to-end: gambar -> segmentasi -> CNN -> angka gabungan.

Format nama file pada folder uji:  <angka_sebenarnya>__<keterangan>.png
Contoh: 90417__foto1.png  -> label sebenarnya "90417".

Pemakaian (dari root proyek):
    python scripts/eval_end_to_end.py            # memakai examples/testset
    python scripts/eval_end_to_end.py <folder>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from digitlens.predictor import predict_digits  # noqa: E402
from digitlens.segmentation import segment_digits as _default_segment  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg"}


def evaluate_testset(folder: Path | str, segment_fn=_default_segment) -> pd.DataFrame:
    """Jalankan pipeline pada semua gambar di `folder`, kembalikan tabel hasil."""
    rows = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in IMAGE_EXT or "__" not in path.stem:
            continue
        truth = path.stem.split("__")[0]
        image = Image.open(path).convert("RGB")
        result = segment_fn(image)
        preds = predict_digits([r.crop for r in result.regions]) if result.regions else []
        predicted = "".join(p["digit"] for p in preds)
        min_conf = min((p["confidence"] for p in preds), default=float("nan"))
        same_len = len(predicted) == len(truth)
        digit_hits = sum(a == b for a, b in zip(predicted, truth)) if same_len else 0
        rows.append(
            {
                "file": path.name,
                "truth": truth,
                "predicted": predicted,
                "n_truth": len(truth),
                "n_candidates": len(result.regions),
                "segmentation_ok": len(result.regions) == len(truth),
                "exact_match": predicted == truth,
                "digit_hits": digit_hits,
                "min_confidence": min_conf,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    segmented = df[df["segmentation_ok"]]
    return {
        "jumlah_gambar": len(df),
        "segmentasi_benar (jumlah kandidat = jumlah digit)": f"{df['segmentation_ok'].mean():.1%}",
        "angka_terbaca_persis": f"{df['exact_match'].mean():.1%}",
        "akurasi_per_digit (hanya yg segmentasinya benar)": (
            f"{segmented['digit_hits'].sum() / segmented['n_truth'].sum():.1%}"
            if len(segmented)
            else "-"
        ),
    }


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples" / "testset"
    table = evaluate_testset(folder)
    print(table.to_string(index=False))
    print()
    for key, value in summarize(table).items():
        print(f"{key}: {value}")
