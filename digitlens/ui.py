from __future__ import annotations

import streamlit as st

from digitlens.predictor import is_model_available


APP_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.025em; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
  [data-testid="stSidebar"] { border-right: 1px solid rgba(120, 120, 120, .16); }
  [data-testid="stSidebar"] .block-container { padding-top: 1.7rem; }
  .brand-mark { display:inline-flex; align-items:center; justify-content:center; width:42px; height:42px; border-radius:13px; background:linear-gradient(135deg,#2563eb,#4f46e5); color:white; font-size:21px; margin-right:10px; }
  .brand-name { font-family:'Manrope',sans-serif; font-size:1.25rem; font-weight:800; }
  .eyebrow { color:#64748b; text-transform:uppercase; letter-spacing:.11em; font-size:.72rem; font-weight:700; }
  .muted { color:#64748b; }
  .soft-card { border:1px solid rgba(120,120,120,.22); border-radius:16px; padding:18px 20px; background:rgba(128,128,128,.035); }
  .result-card { border:1px solid rgba(37,99,235,.22); border-radius:18px; padding:22px; background:linear-gradient(145deg,rgba(37,99,235,.08),rgba(79,70,229,.035)); }
  .status-pill { display:inline-block; padding:4px 10px; border-radius:999px; background:rgba(217,119,6,.12); color:#b45309; font-size:.72rem; font-weight:700; letter-spacing:.04em; }
  .step-number { display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:9px; background:rgba(37,99,235,.12); color:#2563eb; font-weight:800; }
  div[data-testid="stMetric"] { border:1px solid rgba(120,120,120,.18); padding:12px 14px; border-radius:13px; background:rgba(128,128,128,.025); }
  .small-note { font-size:.82rem; color:#64748b; }
  .stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:600; }
</style>
"""


@st.cache_resource(show_spinner="Memuat model CNN...")
def _model_ready() -> bool:
    """Cek sekali per proses apakah model CNN dapat dimuat."""
    return is_model_available()


def inject_css() -> None:
    """Apply the shared application stylesheet."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def initialize_session_state() -> None:
    """Initialize session values used across pages."""
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []
    if "last_segmentation" not in st.session_state:
        st.session_state.last_segmentation = None
    if "last_uploaded_key" not in st.session_state:
        st.session_state.last_uploaded_key = None
    if "page_navigation" not in st.session_state:
        st.session_state.page_navigation = "Homepage"


def render_sidebar() -> str:
    """Render the common sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(
            "<div style='display:flex;align-items:center;margin-bottom:8px;'>"
            "<span class='brand-mark'>⌗</span><span class='brand-name'>DigitLens</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("Image-based Digit Recognition")
        st.divider()
        page = st.radio(
            "NAVIGATION",
            [
                "Homepage",
                "Recognizer",
                "History",
                "About Project",
            ],
            key="page_navigation",
        )
        st.divider()
        st.markdown("**Model status**")
        if _model_ready():
            st.success("Model CNN aktif", icon="🧠")
            st.caption(
                "Input 28×28 grayscale · output digit 0–9\n"
                "Klasifikasi dilakukan per potongan digit."
            )
        else:
            st.warning("Model CNN belum dapat dimuat", icon="🧠")
            st.caption(
                "Periksa models/model_cnn.keras dan instalasi Keras/TensorFlow. "
                "Detail ada di halaman Recognizer."
            )
        st.divider()
        st.caption("Local prototype · v0.2 · Multi-digit segmentation")

    return page


def render_header(title: str, subtitle: str) -> None:
    """Render a consistent title block on each page."""
    st.markdown(
        f"<div class='eyebrow'>DIGITLENS / {title.upper()}</div>",
        unsafe_allow_html=True,
    )
    st.title(title)
    st.caption(subtitle)
