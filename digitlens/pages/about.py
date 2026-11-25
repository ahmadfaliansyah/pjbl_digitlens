from __future__ import annotations

import pandas as pd
import streamlit as st

from digitlens.ui import render_header


def render_page() -> None:
    render_header(
        "About DigitLens",
        "Tujuan, alur pemrosesan, algoritma, dan ruang lingkup prototype.",
    )
    st.markdown(
        "DigitLens merupakan prototype aplikasi pengenalan rangkaian digit dari gambar. "
        "Sistem menyiapkan gambar, memisahkan kandidat digit, mengklasifikasikan setiap potongan "
        "dengan model CNN, lalu menyusun hasilnya kembali dari kiri ke kanan."
    )

    st.markdown("### How it works")
    process_cols = st.columns(5)
    process_items = [
        ("01", "Input", "Pengguna mengunggah gambar angka."),
        ("02", "Preprocessing", "Grayscale, thresholding, dan pembersihan noise dasar."),
        ("03", "Segmentation", "Menemukan dan memotong area kandidat digit."),
        ("04", "Classification", "Model ML mengenali setiap potongan sebagai 0–9."),
        ("05", "Sequence", "Hasil digit digabungkan mengikuti urutan kiri ke kanan."),
    ]
    for col, (number, title, description) in zip(process_cols, process_items):
        with col:
            st.markdown(
                f"<div class='soft-card' style='height:185px;'><span class='step-number'>{number}</span>"
                f"<h4 style='margin:12px 0 5px 0;'>{title}</h4>"
                f"<p class='muted' style='font-size:.83rem;'>{description}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Algoritma yang dibandingkan")
    st.caption(
        "Ketiga algoritma dilatih dan dibandingkan pada notebook evaluasi. "
        "Model dengan hasil terbaik (CNN) dipakai pada aplikasi ini."
    )
    algo_cols = st.columns(3)
    algorithms = [
        (
            "CNN",
            "Convolutional Neural Network",
            "Model deep learning yang mempelajari pola visual dari citra digit.",
        ),
        (
            "SVM",
            "Support Vector Machine",
            "Model klasifikasi yang membentuk batas pemisah antar kelas.",
        ),
        (
            "Random Forest",
            "Ensemble Learning",
            "Model klasifikasi yang menggabungkan banyak decision tree.",
        ),
    ]
    for col, (short, full, description) in zip(algo_cols, algorithms):
        with col:
            st.markdown(
                f"<div class='soft-card' style='height:165px;'><div class='eyebrow'>{short}</div>"
                f"<h4 style='margin:8px 0 5px 0;'>{full}</h4>"
                f"<p class='muted' style='font-size:.83rem;'>{description}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Development status")
    status_df = pd.DataFrame(
        [
            {"Komponen": "Upload dan preview gambar", "Status": "Ready"},
            {"Komponen": "Informasi gambar", "Status": "Ready"},
            {"Komponen": "Grayscale dan Otsu thresholding", "Status": "Ready"},
            {"Komponen": "Connected-component segmentation (+ margin otomatis)", "Status": "Ready"},
            {"Komponen": "Klasifikasi CNN", "Status": "Ready"},
            {"Komponen": "SVM / Random Forest", "Status": "Dibandingkan di notebook (tidak dipakai di aplikasi)"},
            {
                "Komponen": "Penggabungan hasil menjadi rangkaian angka",
                "Status": "Ready",
            },
            {"Komponen": "Riwayat dan export CSV", "Status": "Ready"},
        ]
    )
    st.dataframe(status_df, width="stretch", hide_index=True)

    st.markdown("### Scope and limitations")
    st.warning(
        "Segmentasi ditujukan untuk satu baris angka yang terpisah jelas. Angka yang saling "
        "menempel, tulisan bertumpuk, banyak baris, latar kompleks, atau goresan digit yang terputus "
        "dapat menghasilkan potongan yang keliru. Untuk kondisi tersebut mungkin diperlukan projection "
        "profile, watershed, deteksi objek, atau koreksi manual.",
        icon="⚠️",
    )
    st.info(
        "Preprocessing saat inferensi harus mengikuti preprocessing saat training. Tim perlu menyepakati "
        "ukuran input, grayscale/RGB, normalisasi, polaritas warna, padding/centering, dan format file model.",
        icon="💡",
    )
