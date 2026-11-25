
import streamlit as st
import textwrap


# Render HTML secara langsung, bukan sebagai Markdown
def show_html(content):
    st.html(textwrap.dedent(content))


# Callback tombol navigasi
def _go_to_recognizer():
    st.session_state["page_navigation"] = "Recognizer"


def render_page():

    # =====================================================
    # HERO SECTION
    # =====================================================

    show_html("""
        <div style="
            background:linear-gradient(125deg,#0F1515,#082829 60%,#1C5052);
            padding:48px 42px;
            border-radius:24px;
            margin-bottom:26px;
        ">
            <div style="
                color:#B7D7D0;
                font-size:12px;
                font-weight:700;
                letter-spacing:3px;
                margin-bottom:18px;
            ">
                COMPUTER VISION · MACHINE LEARNING
            </div>

            <h1 style="
                color:#FFFFFF;
                font-size:3.2rem;
                line-height:1.15;
                margin:0 0 16px 0;
            ">
                See the Numbers.<br>
                Understand the Digits.
            </h1>

            <p style="
                color:#D6E8E4;
                font-size:1rem;
                max-width:620px;
                line-height:1.8;
                margin-bottom:0;
            ">
                DigitLens membantu mengenali rangkaian angka dari gambar
                melalui pemrosesan citra, segmentasi, dan klasifikasi
                menggunakan Machine Learning.
            </p>
        </div>
    """)

    # =====================================================
    # BUTTON
    # =====================================================

    st.write("")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.button(
            "Mulai Mengenali Angka →",
            type="primary",
            use_container_width=True,
            on_click=_go_to_recognizer,
        )

    with col2:
        st.caption(
            "Unggah gambar angka dan lihat bagaimana DigitLens "
            "memproses serta mengenali setiap digit."
        )

    st.write("")

    # =====================================================
    # FEATURES
    # =====================================================

    st.markdown("## Mengenal DigitLens")

    st.caption(
        "Dua teknologi utama yang bekerja dalam proses pengenalan angka."
    )

    feature1, feature2 = st.columns(2, gap="large")

    with feature1:
        show_html("""
            <div style="
                background:#B7D7D0;
                border:1px solid #B7D7D0;
                border-radius:18px;
                padding:24px;
                min-height:190px;
            ">
                <div style="font-size:30px;">▧</div>

                <h3 style="color:#0F1515;margin-top:12px;">
                    Image Processing
                </h3>

                <p style="color:#1C5052;line-height:1.8;">
                    Mengolah gambar melalui grayscale, thresholding,
                    dan segmentasi untuk memisahkan kandidat digit.
                </p>
            </div>
        """)

    with feature2:
        show_html("""
            <div style="
                background:#1C5052;
                border:1px solid #1C5052;
                border-radius:18px;
                padding:24px;
                min-height:190px;
            ">
                <div style="font-size:30px;">🧠</div>

                <h3 style="color:#FFFFFF;margin-top:12px;">
                    Machine Learning
                </h3>

                <p style="color:#D6E8E4;line-height:1.8;">
                    Menggunakan model CNN untuk mengklasifikasikan
                    setiap potongan gambar menjadi digit 0 sampai 9.
                </p>
            </div>
        """)

    st.write("")

    # =====================================================
    # WORKFLOW
    # =====================================================

    st.markdown("## Bagaimana DigitLens bekerja?")

    st.caption(
        "Proses pengenalan angka dilakukan melalui lima tahapan."
    )

    steps = [
        ("01", "Input", "Pengguna mengunggah gambar berisi angka."),
        ("02", "Preprocessing", "Gambar disiapkan untuk diproses."),
        ("03", "Segmentation", "Kandidat angka dipisahkan menjadi potongan digit."),
        ("04", "Classification", "CNN memprediksi setiap digit dari 0 sampai 9."),
        ("05", "Sequence", "Hasil prediksi disusun dari kiri ke kanan."),
    ]

    for number, title, description in steps:

        show_html(f"""
            <div style="
                display:flex;
                align-items:center;
                gap:18px;
                background:#FFFFFF;
                border:1px solid #B7D7D0;
                border-radius:16px;
                margin-bottom:12px;
                padding:17px 20px;
            ">
                <div style="
                    background:#B7D7D0;
                    color:#0F1515;
                    min-width:44px;
                    height:44px;
                    border-radius:12px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-weight:800;
                ">
                    {number}
                </div>

                <div>
                    <h4 style="margin:0;color:#0F1515;">
                        {title}
                    </h4>

                    <p style="
                        margin:5px 0 0 0;
                        color:#526B69;
                        font-size:14px;
                    ">
                        {description}
                    </p>
                </div>
            </div>
        """)

    # =====================================================
    # PROJECT SCOPE
    # =====================================================

    show_html("""
        <div style="
            background:#0F1515;
            padding:26px;
            border-radius:20px;
            margin-top:24px;
        ">
            <div style="
                color:#B7D7D0;
                font-size:11px;
                font-weight:700;
                letter-spacing:2px;
            ">
                PROJECT SCOPE
            </div>

            <h3 style="color:#FFFFFF;margin:10px 0;">
                Built to recognize digit sequences.
            </h3>

            <p style="
                color:#D6E8E4;
                line-height:1.8;
                margin-bottom:0;
            ">
                DigitLens saat ini difokuskan pada gambar berisi
                satu baris angka yang memiliki jarak antar-digit
                cukup jelas. Kualitas gambar dan keberhasilan
                segmentasi dapat memengaruhi hasil pengenalan.
            </p>
        </div>
    """)

    st.write("")

    st.caption(
        "DigitLens · Image-based Digit Recognition · Local Prototype"
    )
