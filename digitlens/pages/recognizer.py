from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

from digitlens.history_store import add_history
from digitlens.predictor import ModelNotAvailableError, predict_digits
from digitlens.segmentation import SegmentationResult, segment_digits
from digitlens.storage import save_uploaded_image
from digitlens.ui import render_header


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOW_CONFIDENCE = 0.90  # di bawah ini: diberi peringatan dan tidak ikut digabung menjadi angka

# Ambang validasi lunak (hasil tetap ditampilkan, hanya diberi peringatan).
MAX_PLAUSIBLE_DIGITS = 8      # kandidat lebih banyak dari ini dicurigai noise/tekstur
MAX_HEIGHT_VARIATION = 0.35   # simpangan baku / rata-rata tinggi kandidat
LARGE_IMAGE_PIXELS = 2_000_000  # gambar sebesar ini sebaiknya di-crop dulu
MIN_CROP_SIDE = 16            # sisi minimum area crop (px)
# Deteksi digit berwarna-warni (gradasi/pelangi).
COLOR_MIN_SATURATION = 90     # pixel dianggap "berwarna" jika saturasi (0-255) >= nilai ini
COLOR_MIN_VALUE = 60          # ...dan kecerahan >= nilai ini (abaikan area gelap)
COLOR_MIN_PIXELS = 150        # minimal pixel berwarna agar dinilai
COLOR_HUE_BINS = 12           # hue (0-179) dibagi 12 kelompok warna
COLOR_BIN_SHARE = 0.10        # sebuah kelompok dihitung jika >= 10% pixel berwarna
COLOR_MIN_BINS = 3            # >= 3 kelompok warna berbeda = berwarna-warni
MAX_ROW_SPREAD = 0.75         # selisih tinggi pusat digit terhadap tinggi digit (baris dianggap horizontal)


def read_uploaded_image(uploaded_file: Any) -> tuple[Image.Image, bytes]:
    """Read an uploaded image and return an RGB PIL image plus original bytes."""
    image_bytes = uploaded_file.getvalue()
    image = Image.open(BytesIO(image_bytes))
    image.load()
    return image.convert("RGB"), image_bytes


def image_metadata(
    image: Image.Image,
    image_bytes: bytes,
    original_format: str | None,
) -> dict[str, Any]:
    return {
        "width": image.width,
        "height": image.height,
        "format": original_format or "Unknown",
        "mode": image.mode,
        "size_kb": len(image_bytes) / 1024,
    }


def crop_box(
    size: tuple[int, int],
    x_range: tuple[int, int],
    y_range: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Ubah rentang persen (kiri-kanan, atas-bawah) menjadi kotak piksel."""
    width, height = size
    left = int(round(width * x_range[0] / 100))
    right = int(round(width * x_range[1] / 100))
    top = int(round(height * y_range[0] / 100))
    bottom = int(round(height * y_range[1] / 100))
    return left, top, right, bottom


def select_digit_area(
    image: Image.Image,
    key_prefix: str,
) -> tuple[Image.Image, bool, bool, tuple[int, int, int, int]]:
    """Biarkan pengguna memilih area angka sebelum segmentasi.

    Mengembalikan (gambar_yang_diproses, apakah_di_crop, area_valid, kotak_crop).
    Tanpa perubahan slider, seluruh gambar dipakai.
    """
    full_box = (0, 0, image.width, image.height)
    with st.expander("✂️ Pilih area angka (disarankan untuk foto yang luas)"):
        st.caption(
            "Geser slider sampai kotak merah hanya mengelilingi angka. "
            "Area di luar kotak tidak diproses."
        )
        x_range = st.slider(
            "Rentang horizontal (%)", 0, 100, (0, 100), key=f"{key_prefix}_x"
        )
        y_range = st.slider(
            "Rentang vertikal (%)", 0, 100, (0, 100), key=f"{key_prefix}_y"
        )
        box = crop_box(image.size, x_range, y_range)
        valid = (box[2] - box[0]) >= MIN_CROP_SIDE and (box[3] - box[1]) >= MIN_CROP_SIDE

        if not valid:
            st.error(f"Area terlalu kecil. Minimal {MIN_CROP_SIDE}×{MIN_CROP_SIDE} px.")
            return image, False, False, box

        overlay = image.copy()
        line = max(3, min(image.size) // 150)
        ImageDraw.Draw(overlay).rectangle(
            (box[0], box[1], max(box[0], box[2] - 1), max(box[1], box[3] - 1)),
            outline=(220, 38, 38),
            width=line,
        )
        before_col, after_col = st.columns(2)
        with before_col:
            st.image(overlay, caption="Area terpilih (kotak merah)", use_container_width=True)
        with after_col:
            st.image(image.crop(box), caption="Yang akan diproses", use_container_width=True)

    is_cropped = box != full_box
    if is_cropped:
        cropped = image.crop(box)
        st.caption(
            f"Area crop aktif: {cropped.width}×{cropped.height} px "
            f"dari {image.width}×{image.height} px."
        )
        return cropped, True, True, box
    return image, False, True, box


def is_multicolor(images: list[Image.Image]) -> bool:
    """True jika gabungan gambar memuat tiga warna (hue) berbeda atau lebih.

    Pixel berwarna dari semua potongan digabung sebelum dinilai, karena
    segmentasi bisa memecah satu digit pelangi menjadi serpihan yang masing-masing
    hanya memuat satu atau dua warna.
    """
    counts = np.zeros(COLOR_HUE_BINS, dtype=np.int64)
    for image in images:
        hsv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
        colored = (hsv[..., 1] >= COLOR_MIN_SATURATION) & (hsv[..., 2] >= COLOR_MIN_VALUE)
        groups = hsv[..., 0][colored].astype(int) * COLOR_HUE_BINS // 180
        counts += np.bincount(groups, minlength=COLOR_HUE_BINS)
    total = int(counts.sum())
    if total < COLOR_MIN_PIXELS:
        return False
    return int((counts / total >= COLOR_BIN_SHARE).sum()) >= COLOR_MIN_BINS


CROP_TIP = "Coba gunakan **Pilih area angka** di atas tombol untuk memotong gambar hingga hanya angkanya yang tersisa."
CROP_TIP_ACTIVE = "Coba sempitkan area crop supaya hanya angkanya yang tersisa."
COLOR_TIP = "Gunakan gambar angka satu warna, misalnya hitam di atas latar putih."


def assess_quality(
    result: SegmentationResult,
    predictions: list[dict] | None,
) -> list[tuple[str, str]]:
    """Daftar (kode, pesan) alasan hasil ini meragukan; kosong jika tidak ada masalah.

    CNN selalu memilih salah satu dari 0-9, bahkan untuk tekstur yang bukan
    angka, jadi kualitas dinilai dari segmentasi, warna, dan confidence-nya.
    """
    issues: list[tuple[str, str]] = []
    count = len(result.regions)

    if count > MAX_PLAUSIBLE_DIGITS:
        issues.append((
            "many",
            f"Terdeteksi {count} kandidat, lebih banyak dari jumlah digit pada umumnya. "
            "Sebagian kemungkinan tekstur atau noise.",
        ))

    if count >= 3:
        heights = np.asarray([region.bbox[3] for region in result.regions], dtype=float)
        if heights.std() / max(heights.mean(), 1.0) > MAX_HEIGHT_VARIATION:
            issues.append((
                "height",
                "Tinggi kandidat sangat tidak seragam, padahal digit dalam satu baris "
                "biasanya setinggi sama.",
            ))

    if count and is_multicolor([region.crop for region in result.regions]):
        issues.append((
            "color",
            "Digit tampak berwarna-warni (gradasi atau pelangi). Model lebih andal "
            "membaca digit satu warna, sehingga hasilnya bisa keliru.",
        ))

    if predictions and all(prediction is not None for prediction in predictions):
        confidences = [prediction["confidence"] for prediction in predictions]
        if float(np.median(confidences)) < LOW_CONFIDENCE:
            issues.append(("confidence", "Model kurang yakin pada sebagian besar potongan digit."))

    return issues


MIN_ALTERNATIVE_SHARE = 0.05  # tebakan alternatif di bawah 5% tidak ditampilkan


def alternatives_text(prediction: dict) -> str:
    """Tebakan alternatif model, mis. '6 (41%), 0 (7%)'; kosong jika tidak ada yang berarti."""
    return ", ".join(
        f"{digit} ({share:.0%})"
        for digit, share in prediction.get("alternatives", [])
        if share >= MIN_ALTERNATIVE_SHARE
    )


def is_horizontal_row(regions: list) -> bool:
    """True jika digit tersusun mendatar dalam satu baris.

    Pusat vertikal semua digit harus berada dalam rentang yang kecil
    dibanding tinggi digit. Susunan menurun (vertikal) atau bertingkat gagal.
    """
    if len(regions) < 2:
        return True
    centers = np.asarray([region.bbox[1] + region.bbox[3] / 2 for region in regions], dtype=float)
    heights = np.asarray([region.bbox[3] for region in regions], dtype=float)
    return float(centers.max() - centers.min()) <= MAX_ROW_SPREAD * float(np.median(heights))


def compose_number(
    result: SegmentationResult,
    predictions: list[dict] | None,
) -> tuple[str, int, str | None]:
    """Gabungkan digit yang meyakinkan menjadi satu angka.

    Mengembalikan (angka, jumlah_digit_disembunyikan, masalah). ``masalah``
    adalah None jika angka berhasil dibentuk, "no_confident" jika tidak ada
    digit dengan confidence cukup, atau "not_horizontal" jika digit yang lolos
    tidak berderet mendatar. Tabel hasil tetap memuat semua kandidat.
    """
    if not predictions or len(predictions) != len(result.regions):
        return "", 0, None
    confident = [
        (region, prediction)
        for region, prediction in zip(result.regions, predictions)
        if prediction["confidence"] >= LOW_CONFIDENCE
    ]
    hidden = len(predictions) - len(confident)
    if not confident:
        return "", hidden, "no_confident"
    if not is_horizontal_row([region for region, _ in confident]):
        return "", hidden, "not_horizontal"
    return "".join(prediction["digit"] for _, prediction in confident), hidden, None


def classify_regions(
    result: SegmentationResult,
) -> tuple[list[dict] | None, str | None]:
    """Klasifikasikan semua crop dengan model CNN.

    Mengembalikan (prediksi, pesan_error). Prediksi berurutan sama dengan
    ``result.regions`` (kiri ke kanan). Jika model tidak dapat dimuat,
    prediksi bernilai None dan pesan_error berisi penjelasannya.
    """
    if not result.regions:
        return [], None
    try:
        return predict_digits([region.crop for region in result.regions]), None
    except ModelNotAvailableError as exc:
        return None, str(exc)


def show_segmentation_result(
    result: SegmentationResult,
    predictions: list[dict] | None = None,
    model_error: str | None = None,
    cropped: bool = False,
) -> None:
    st.markdown("### Hasil segmentasi")
    st.caption(
        "Kotak biru menunjukkan area kandidat digit yang ditemukan."
        + (" Koordinat mengacu pada area crop." if cropped else "")
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Kandidat digit", len(result.regions))
    metric2.metric("Latar terdeteksi", result.background.title())
    metric3.metric("Otsu threshold", f"{result.threshold:.0f}")

    if result.warnings:
        for warning in result.warnings:
            st.warning(warning, icon="⚠️")

    issues = assess_quality(result, predictions)
    if issues:
        codes = {code for code, _ in issues}
        tips: list[str] = []
        if codes & {"many", "height"}:
            tips.append(CROP_TIP_ACTIVE if cropped else CROP_TIP)
            image_pixels = result.original_with_boxes.width * result.original_with_boxes.height
            if not cropped and image_pixels >= LARGE_IMAGE_PIXELS:
                tips[-1] += f" Gambar ini cukup besar ({image_pixels / 1_000_000:.1f} MP)."
        if "color" in codes:
            tips.append(COLOR_TIP)
        st.warning(
            "**Hasil ini meragukan.**\n\n"
            + "\n".join(f"- {message}" for _, message in issues)
            + ("\n\n" + " ".join(tips) if tips else ""),
            icon="🔎",
        )

    image_col, binary_col = st.columns(2, gap="large")
    with image_col:
        st.markdown("**Deteksi area pada gambar**")
        st.image(result.original_with_boxes, use_container_width=True)
    with binary_col:
        st.markdown("**Hasil thresholding**")
        st.image(result.binary_preview, use_container_width=True)

    st.markdown("#### Potongan kandidat digit")
    if not result.regions:
        st.info("Belum ada potongan digit untuk ditampilkan.")
        return

    crop_cols = st.columns(min(5, len(result.regions)))
    for i, region in enumerate(result.regions):
        with crop_cols[i % len(crop_cols)]:
            st.image(
                region.crop,
                caption=f"Digit #{region.index}",
                use_container_width=True,
            )
            x, y, w, h = region.bbox
            st.caption(f"Posisi: x={x}, y={y} · {w}×{h}px")

    st.markdown("#### Hasil klasifikasi")
    if model_error:
        st.error(f"Klasifikasi tidak dapat dijalankan. {model_error}", icon="🧠")
    if predictions is None or len(predictions) != len(result.regions):
        predictions = [None] * len(result.regions)

    if all(prediction is not None for prediction in predictions):
        number, hidden, problem = compose_number(result, predictions)
        threshold_label = f"{LOW_CONFIDENCE:.0%}"
        if problem == "no_confident":
            st.info(
                f"Tidak ada digit dengan confidence ≥ {threshold_label}, jadi angka tidak ditampilkan. "
                "Semua kandidat tetap tercantum di tabel di bawah.",
                icon="ℹ️",
            )
        elif problem == "not_horizontal":
            st.warning(
                "Digit yang terbaca tidak tersusun dalam satu baris mendatar, jadi tidak digabung "
                "menjadi satu angka. DigitLens hanya mendukung angka yang berderet horizontal.",
                icon="↔️",
            )
        elif number:
            model_name = predictions[0]["model_name"]
            note = (
                f" · {hidden} kandidat dengan confidence &lt; {threshold_label} tidak disertakan"
                if hidden
                else ""
            )
            st.markdown(
                "<div class='result-card'>"
                f"<div class='eyebrow'>{'Perkiraan angka (meragukan)' if issues else 'Angka terbaca'}</div>"
                f"<div style='font-family:Manrope,sans-serif;font-size:2.8rem;font-weight:800;"
                f"letter-spacing:.12em;margin:4px 0;'>{number}</div>"
                f"<p class='muted' style='margin:0;'>Model: {model_name} · digit digabung dari kiri ke kanan{note}</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        for region, prediction in zip(result.regions, predictions):
            if prediction["confidence"] < LOW_CONFIDENCE:
                other = alternatives_text(prediction)
                st.warning(
                    f"Digit #{region.index} diprediksi {prediction['digit']} dengan confidence "
                    f"{prediction['confidence']:.0%}."
                    + (f" Model juga menimbang: {other}." if other else "")
                    + " Hasil mungkin keliru, periksa potongan digitnya.",
                    icon="⚠️",
                )

    table = pd.DataFrame(
        [
            {
                "Urutan": region.index,
                "Area gambar": f"{region.bbox[2]} × {region.bbox[3]} px",
                "Prediksi": prediction["digit"] if prediction else "-",
                "Confidence": f"{prediction['confidence']:.1%}" if prediction else "-",
                "Alternatif": (alternatives_text(prediction) or "-") if prediction else "-",
            }
            for region, prediction in zip(result.regions, predictions)
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def _render_empty_upload_state() -> None:
    st.markdown(
        "<div class='soft-card' style='text-align:center;padding:34px 20px;'>"
        "<div style='font-size:2.2rem;margin-bottom:8px;'>▧</div>"
        "<h3 style='margin-bottom:5px;'>Belum ada gambar</h3>"
        "<p class='muted' style='margin:0;'>Unggah gambar yang berisi satu atau beberapa digit dalam satu baris.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Alur penggunaan")
    cols = st.columns(4)
    steps = [
        ("01", "Upload", "Pilih gambar angka."),
        ("02", "Preprocess", "Ubah ke grayscale dan threshold."),
        ("03", "Segment", "Pisahkan area setiap digit."),
        ("04", "Recognize", "Klasifikasikan tiap digit dengan model ML."),
    ]
    for col, (number, title, description) in zip(cols, steps):
        with col:
            st.markdown(
                f"<div class='soft-card' style='height:150px;'><span class='step-number'>{number}</span>"
                f"<h4 style='margin:12px 0 4px 0;'>{title}</h4>"
                f"<p class='muted' style='font-size:.86rem;margin:0;'>{description}</p></div>",
                unsafe_allow_html=True,
            )


def render_page() -> None:
    render_header(
        "Digit Recognition",
        "Unggah satu gambar berisi rangkaian angka. DigitLens akan mencoba memisahkan "
        "setiap digit sebelum klasifikasi ML.",
    )
    st.info(
        "Versi ini memiliki segmentasi dasar untuk satu baris angka yang terpisah jelas, "
        "lalu setiap digit dikenali dengan model CNN. Angka harus berderet mendatar (horizontal).",
        icon="ℹ️",
    )

    uploaded_file = st.file_uploader(
        "Pilih gambar yang ingin diperiksa",
        type=["png", "jpg", "jpeg"],
        help=(
            "Format PNG, JPG, JPEG. Maksimal 10 MB. "
            "Disarankan angka gelap di atas latar polos."
        ),
    )

    if uploaded_file is None:
        _render_empty_upload_state()
        return

    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error("Ukuran file melebihi 10 MB. Silakan gunakan gambar yang lebih kecil.")
        return

    try:
        image, image_bytes = read_uploaded_image(uploaded_file)
        original_format = (
            uploaded_file.type.split("/")[-1].upper()
            if uploaded_file.type
            else None
        )
        meta = image_metadata(image, image_bytes, original_format)
        current_key = f"{uploaded_file.name}:{sha256(image_bytes).hexdigest()}"

        if st.session_state.last_uploaded_key != current_key:
            st.session_state.last_uploaded_key = current_key
            st.session_state.last_segmentation = None

        preview_col, detail_col = st.columns([1.1, 0.9], gap="large")
        with preview_col:
            st.markdown("#### Image Preview")
            st.image(image, use_container_width=True)
            st.caption(f"File: {uploaded_file.name}")

        with detail_col:
            st.markdown("#### Image Information")
            a, b = st.columns(2)
            a.metric("Resolution", f"{meta['width']} × {meta['height']}")
            b.metric("File size", f"{meta['size_kb']:.1f} KB")
            c, d = st.columns(2)
            c.metric("Format", meta["format"])
            d.metric("Color mode", meta["mode"])
            st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
            st.markdown("**Processing status**")
            st.warning("Siap untuk segmentasi", icon="✂️")
            st.caption(
                "Jalankan proses untuk mendeteksi kandidat digit dan mengenali angkanya."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Segmentasi gambar")
        st.caption(
            "Versi awal mengasumsikan digit tersusun dalam satu baris dan memiliki jarak antar-digit."
        )

        source_image, is_cropped, crop_valid, crop_area = select_digit_area(
            image, key_prefix=f"crop_{sha256(current_key.encode()).hexdigest()[:12]}"
        )
        # Hasil lama hanya berlaku untuk area yang sama persis.
        result_key = f"{current_key}:{crop_area}"

        if st.button(
            "Segment & recognize digits",
            type="primary",
            use_container_width=True,
            disabled=not crop_valid,
        ):
            with st.spinner("Memproses gambar dan mencari area digit..."):
                # Save only after segmentation completes successfully.
                result = segment_digits(source_image)
                predictions, model_error = classify_regions(result)
                saved_image_path = save_uploaded_image(
                    image_bytes,
                    uploaded_file.name,
                )

            st.session_state.last_segmentation = {
                "key": result_key,
                "cropped": is_cropped,
                "result": result,
                "image_path": saved_image_path,
                "predictions": predictions,
                "model_error": model_error,
            }

            number, _, _ = compose_number(result, predictions)
            if number:
                status = "Terbaca: " + number
            elif result.regions:
                status = "Kandidat ditemukan"
            else:
                status = "Tidak ada kandidat"
            add_history(
                uploaded_file.name,
                len(result.regions),
                status,
                saved_image_path,
            )
            st.success(f"Gambar asli tersimpan: {saved_image_path}")

        last = st.session_state.last_segmentation
        if last and last["key"] == result_key:
            show_segmentation_result(
                last["result"],
                last.get("predictions"),
                last.get("model_error"),
                last.get("cropped", False),
            )

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"File tidak dapat diproses: {exc}")
    except Exception as exc:
        st.error(f"Terjadi kesalahan saat segmentasi: {exc}")
