from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st


HISTORY_COLUMNS = [
    "Waktu",
    "Nama file",
    "Kandidat digit",
    "Angka terbaca",
    "Confidence minimum",
    "Aktivitas",
    "Status",
    "Lokasi gambar",
]


def history_dataframe() -> pd.DataFrame:
    """Convert the current session history into a display/export DataFrame."""
    history = st.session_state.get("scan_history", [])

    if not history:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    rows = [
        {
            "Waktu": item["timestamp"],
            "Nama file": item["filename"],
            "Kandidat digit": item["segment_count"],
            "Angka terbaca": item.get("number", ""),
            "Confidence minimum": item.get("min_confidence"),
            "Aktivitas": item["activity"],
            "Status": item["status"],
            "Lokasi gambar": item.get("image_path", ""),
        }
        for item in history
    ]

    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def add_history(
    filename: str,
    segment_count: int,
    status: str,
    image_path: str,
    number: str = "",
    min_confidence: float | None = None,
) -> None:
    """Add one segmentation activity to the top of the session history."""
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []

    st.session_state.scan_history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "segment_count": segment_count,
            "activity": "Segmentasi digit",
            "status": status,
            "image_path": image_path,
            "number": number,
            "min_confidence": (
                round(min_confidence, 4) if min_confidence is not None else None
            ),
        },
    )
