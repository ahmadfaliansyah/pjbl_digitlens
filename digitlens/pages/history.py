from __future__ import annotations

import streamlit as st

from digitlens.history_store import history_dataframe
from digitlens.ui import render_header


def render_page() -> None:
    render_header(
        "Activity History",
        "Riwayat proses segmentasi yang dilakukan selama sesi aplikasi.",
    )
    st.caption(
        "Riwayat disimpan sementara di memori sesi. Data akan hilang ketika sesi "
        "aplikasi direset. File gambar yang sudah tersimpan di folder tidak ikut "
        "terhapus ketika riwayat dibersihkan."
    )

    history_df = history_dataframe()
    total = len(history_df)
    with_segments = (
        int((history_df["Kandidat digit"] > 0).sum())
        if total
        else 0
    )
    no_segments = total - with_segments

    m1, m2, m3 = st.columns(3)
    m1.metric("Total activity", total)
    m2.metric("Digit candidates found", with_segments)
    m3.metric("No candidate", no_segments)

    st.markdown("### Activity log")
    if history_df.empty:
        st.markdown(
            "<div class='soft-card' style='text-align:center;padding:28px;'>"
            "<div style='font-size:2rem;'>◷</div><strong>Belum ada aktivitas</strong>"
            "<p class='muted' style='margin:5px 0 0 0;'>Riwayat akan muncul setelah proses segmentasi dijalankan.</p></div>",
            unsafe_allow_html=True,
        )
        return

    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Confidence minimum": st.column_config.NumberColumn(format="percent"),
        },
    )
    csv_data = history_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download history (CSV)",
        data=csv_data,
        file_name="digitlens_history.csv",
        mime="text/csv",
    )

    if st.button("Clear history", type="secondary"):
        st.session_state.scan_history = []
        st.session_state.last_segmentation = None
        st.rerun()
