from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


st.set_page_config(
    page_title="Analisis Sentimen Shopee & Tokopedia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BASE_FROM = """
FROM dw.fact_review AS f
JOIN dw.dim_application AS a
    ON a.application_key = f.application_key
JOIN dw.dim_date AS d
    ON d.date_key = f.date_key
LEFT JOIN dw.dim_sentiment AS s
    ON s.sentiment_key = f.sentiment_key
LEFT JOIN dw.dim_topic AS t
    ON t.topic_key = f.topic_key
"""


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    """Create one reusable SQLAlchemy engine from Streamlit secrets."""
    try:
        connection_url = st.secrets["connections"]["dw"]["url"]
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "Konfigurasi database belum ditemukan. Isi [connections.dw] pada "
            ".streamlit/secrets.toml atau Streamlit Cloud Secrets."
        ) from exc

    # Supabase usually provides postgresql://. Use the psycopg 3 SQLAlchemy driver.
    if connection_url.startswith("postgresql://"):
        connection_url = connection_url.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    elif connection_url.startswith("postgres://"):
        connection_url = connection_url.replace(
            "postgres://", "postgresql+psycopg://", 1
        )

    if not connection_url.startswith("postgresql+psycopg://"):
        raise RuntimeError(
            "Format URL harus postgresql:// atau postgresql+psycopg://. "
            "Salin URL dari Supabase Connect dan jangan mengarang host/port."
        )

    return create_engine(
        connection_url,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
        pool_recycle=1800,
    )


@st.cache_data(ttl=180, show_spinner=False)
def read_sql_df(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a read-only query and cache the result briefly."""
    with get_engine().connect() as connection:
        return pd.read_sql_query(text(sql), connection, params=params or {})


def make_base_filter(
    applications: list[str], start_date: date, end_date: date
) -> tuple[str, dict[str, Any]]:
    clauses = ["d.full_date >= :start_date", "d.full_date <= :end_date"]
    params: dict[str, Any] = {"start_date": start_date, "end_date": end_date}

    if applications:
        placeholders = []
        for index, application in enumerate(applications):
            key = f"app_{index}"
            placeholders.append(f":{key}")
            params[key] = application
        clauses.append(f"a.application_name IN ({', '.join(placeholders)})")

    return " AND ".join(clauses), params


def make_named_filter(
    column_sql: str, values: list[str], prefix: str, params: dict[str, Any]
) -> str | None:
    if not values:
        return None
    placeholders = []
    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        placeholders.append(f":{key}")
        params[key] = value
    return f"{column_sql} IN ({', '.join(placeholders)})"


def show_connection_error(exc: Exception) -> None:
    LOGGER.exception("Gagal mengakses Supabase PostgreSQL")
    st.error(
        "Dashboard belum dapat membaca database. Periksa URL Session Pooler, "
        "username dashboard_reader, password, dan hak akses SELECT."
    )
    with st.expander("Detail teknis untuk pemeriksaan"):
        st.code(f"{type(exc).__name__}: {exc}")


def load_filter_options() -> tuple[list[str], date, date, list[str]]:
    apps_df = read_sql_df(
        "SELECT application_name FROM dw.dim_application ORDER BY application_name"
    )
    dates_df = read_sql_df(
        """SELECT MIN(full_date) AS min_date, MAX(full_date) AS max_date
           FROM dw.dim_date WHERE full_date IS NOT NULL"""
    )
    topics_df = read_sql_df(
        """SELECT DISTINCT topic_name FROM dw.dim_topic
           ORDER BY topic_name"""
    )

    if dates_df.empty or pd.isna(dates_df.loc[0, "min_date"]):
        raise RuntimeError("Dimensi tanggal belum memiliki rentang tanggal yang valid.")

    min_date = pd.Timestamp(dates_df.loc[0, "min_date"]).date()
    max_date = pd.Timestamp(dates_df.loc[0, "max_date"]).date()
    apps = apps_df["application_name"].dropna().astype(str).tolist()
    topics = topics_df["topic_name"].dropna().astype(str).tolist()
    return apps, min_date, max_date, topics


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "0"


def fmt_pct(numerator: Any, denominator: Any) -> str:
    denominator = int(denominator or 0)
    if denominator == 0:
        return "0,00%"
    value = 100 * float(numerator or 0) / denominator
    return f"{value:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")


st.title("📊 Analisis Sentimen Shopee dan Tokopedia")
st.caption(
    "Dashboard analisis ulasan Google Play Store menggunakan LinearSVC dan "
    "pengelompokan topik keluhan dengan K-Means."
)

try:
    engine = get_engine()
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    application_options, min_date, max_date, topic_options = load_filter_options()
except Exception as exc:  # pragma: no cover - depends on user's Supabase
    show_connection_error(exc)
    st.stop()

with st.sidebar:
    st.header("Filter Dashboard")
    selected_apps = st.multiselect(
        "Aplikasi",
        options=application_options,
        default=application_options,
        help="Pilih satu atau kedua aplikasi.",
    )
    picked_dates = st.date_input(
        "Rentang tanggal review",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
    )
    if isinstance(picked_dates, tuple) and len(picked_dates) == 2:
        start_date, end_date = picked_dates
    elif isinstance(picked_dates, tuple) and len(picked_dates) == 1:
        start_date = end_date = picked_dates[0]
    else:
        start_date = end_date = picked_dates

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    st.divider()
    st.caption("Sumber data: Supabase PostgreSQL Data Warehouse")
    st.caption("Akses database: read-only")

where_sql, base_params = make_base_filter(selected_apps, start_date, end_date)

# ----------------------------- Overview KPIs -----------------------------
KPI_SQL = f"""
SELECT
    COUNT(*) AS total_reviews,
    AVG(f.score)::numeric(10,2) AS average_rating,
    COUNT(*) FILTER (WHERE s.sentiment_label = 'negatif') AS negative_reviews,
    COUNT(*) FILTER (WHERE f.sentiment_key IS NOT NULL) AS predicted_reviews,
    COUNT(*) FILTER (WHERE f.sentiment_key IS NULL) AS unpredicted_reviews
{BASE_FROM}
WHERE {where_sql}
"""

try:
    kpi = read_sql_df(KPI_SQL, base_params).iloc[0]
except Exception as exc:
    show_connection_error(exc)
    st.stop()

total_reviews = int(kpi["total_reviews"] or 0)
average_rating = float(kpi["average_rating"]) if pd.notna(kpi["average_rating"]) else 0.0
negative_reviews = int(kpi["negative_reviews"] or 0)
predicted_reviews = int(kpi["predicted_reviews"] or 0)
unpredicted_reviews = int(kpi["unpredicted_reviews"] or 0)
negative_pct = fmt_pct(negative_reviews, predicted_reviews)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Review", fmt_int(total_reviews))
m2.metric("Rata-rata Rating", f"{average_rating:.2f} / 5")
m3.metric("Review Negatif", fmt_int(negative_reviews), f"{negative_pct} dari review terprediksi")
m4.metric("Tidak Dapat Diprediksi", fmt_int(unpredicted_reviews))

st.caption(
    "Catatan: persentase negatif menggunakan denominator review yang memiliki "
    "prediksi sentimen. Review tanpa sentimen tidak digabungkan ke kelas netral."
)

# ------------------------------- Page tabs --------------------------------
tab_overview, tab_sentiment, tab_topics, tab_explorer, tab_model = st.tabs(
    ["Ringkasan", "Analisis Sentimen", "Analisis Topik", "Detail Review", "Evaluasi Model"]
)

with tab_overview:
    st.subheader("Gambaran Umum Ulasan")
    left, right = st.columns(2)

    rating_df = read_sql_df(
        f"""SELECT f.score AS rating, COUNT(*) AS jumlah
            {BASE_FROM}
            WHERE {where_sql} AND f.score IS NOT NULL
            GROUP BY f.score ORDER BY f.score""",
        base_params,
    )
    with left:
        st.markdown("**Distribusi Rating Google Play**")
        if rating_df.empty:
            st.info("Tidak ada data rating untuk filter ini.")
        else:
            fig = px.bar(
                rating_df,
                x="rating",
                y="jumlah",
                text="jumlah",
                labels={"rating": "Rating", "jumlah": "Jumlah review"},
            )
            fig.update_layout(xaxis=dict(dtick=1), margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    sentiment_df = read_sql_df(
        f"""SELECT COALESCE(s.sentiment_label, 'Tidak dapat diprediksi') AS sentimen,
                   COUNT(*) AS jumlah
            {BASE_FROM}
            WHERE {where_sql}
            GROUP BY COALESCE(s.sentiment_label, 'Tidak dapat diprediksi')
            ORDER BY sentimen""",
        base_params,
    )
    with right:
        st.markdown("**Distribusi Sentimen**")
        if sentiment_df.empty:
            st.info("Tidak ada data sentimen untuk filter ini.")
        else:
            fig = px.pie(
                sentiment_df,
                names="sentimen",
                values="jumlah",
                hole=0.45,
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), legend_title="Sentimen")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Perbandingan Rating dan Sentimen per Aplikasi**")
    compare_df = read_sql_df(
        f"""SELECT a.application_name AS aplikasi,
                   COALESCE(s.sentiment_label, 'Tidak dapat diprediksi') AS sentimen,
                   COUNT(*) AS jumlah
            {BASE_FROM}
            WHERE {where_sql}
            GROUP BY a.application_name,
                     COALESCE(s.sentiment_label, 'Tidak dapat diprediksi')
            ORDER BY a.application_name, sentimen""",
        base_params,
    )
    if compare_df.empty:
        st.info("Tidak ada data untuk dibandingkan.")
    else:
        fig = px.bar(
            compare_df,
            x="aplikasi",
            y="jumlah",
            color="sentimen",
            barmode="stack",
            labels={"aplikasi": "Aplikasi", "jumlah": "Jumlah review", "sentimen": "Sentimen"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab_sentiment:
    st.subheader("Distribusi dan Tren Sentimen")
    trend_df = read_sql_df(
        f"""SELECT TO_CHAR(DATE_TRUNC('month', d.full_date), 'YYYY-MM') AS bulan,
                   s.sentiment_label AS sentimen,
                   COUNT(*) AS jumlah
            {BASE_FROM}
            WHERE {where_sql}
              AND d.full_date IS NOT NULL
              AND s.sentiment_label IS NOT NULL
            GROUP BY DATE_TRUNC('month', d.full_date), s.sentiment_label
            ORDER BY DATE_TRUNC('month', d.full_date), s.sentiment_label""",
        base_params,
    )
    if trend_df.empty:
        st.info("Belum ada data tren sentimen untuk filter ini.")
    else:
        fig = px.line(
            trend_df,
            x="bulan",
            y="jumlah",
            color="sentimen",
            markers=True,
            labels={"bulan": "Bulan", "jumlah": "Jumlah review", "sentimen": "Sentimen"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    app_sentiment_df = read_sql_df(
        f"""SELECT a.application_name AS aplikasi,
                   s.sentiment_label AS sentimen,
                   COUNT(*) AS jumlah
            {BASE_FROM}
            WHERE {where_sql}
              AND s.sentiment_label IS NOT NULL
            GROUP BY a.application_name, s.sentiment_label
            ORDER BY a.application_name, s.sentiment_label""",
        base_params,
    )
    st.markdown("**Perbandingan Sentimen Antar-Aplikasi**")
    if app_sentiment_df.empty:
        st.info("Tidak ada prediksi sentimen pada filter ini.")
    else:
        fig = px.bar(
            app_sentiment_df,
            x="aplikasi",
            y="jumlah",
            color="sentimen",
            barmode="group",
            labels={"aplikasi": "Aplikasi", "jumlah": "Jumlah review", "sentimen": "Sentimen"},
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_topics:
    st.subheader("Topik Keluhan dari K-Means")
    st.caption(
        "Topik K-Means berasal dari ulasan negatif yang diproses pada Notebook 07. "
        "Jumlah topik tidak sama dengan jumlah seluruh review."
    )
    topic_df = read_sql_df(
        f"""SELECT t.topic_id,
                   t.topic_name AS nama_topik,
                   t.interpretation_status AS status_interpretasi,
                   COUNT(*) AS jumlah_review
            {BASE_FROM}
            WHERE {where_sql}
              AND f.topic_key IS NOT NULL
            GROUP BY t.topic_id, t.topic_name, t.interpretation_status
            ORDER BY jumlah_review DESC""",
        base_params,
    )
    if topic_df.empty:
        st.info("Tidak ada assignment topik untuk filter ini.")
    else:
        fig = px.bar(
            topic_df.sort_values("jumlah_review"),
            x="jumlah_review",
            y="nama_topik",
            color="status_interpretasi",
            orientation="h",
            labels={
                "jumlah_review": "Jumlah review",
                "nama_topik": "Topik",
                "status_interpretasi": "Status interpretasi",
            },
        )
        fig.update_layout(yaxis_title=None, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(topic_df, use_container_width=True, hide_index=True)

    pending = read_sql_df(
        """SELECT topic_id, topic_name, interpretation_status
           FROM dw.dim_topic
           WHERE interpretation_status <> 'completed'
           ORDER BY topic_id"""
    )
    if not pending.empty:
        st.warning("Masih ada topik yang belum selesai diinterpretasikan.")
        st.dataframe(pending, use_container_width=True, hide_index=True)

with tab_explorer:
    st.subheader("Penelusuran Detail Review")
    with st.form("review_filter_form"):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            sentiment_choices = [
                "negatif", "netral", "positif", "Tidak dapat diprediksi"
            ]
            chosen_sentiments = st.multiselect(
                "Sentimen", sentiment_choices, default=sentiment_choices
            )
        with col2:
            chosen_topics = st.multiselect("Topik", topic_options, default=topic_options)
        with col3:
            search_text = st.text_input("Cari teks ulasan", placeholder="Masukkan kata/frasa...")
        submitted = st.form_submit_button("Terapkan filter review")

    # Persist the latest explorer selections between Streamlit reruns.
    if "explorer_filters" not in st.session_state or submitted:
        st.session_state["explorer_filters"] = {
            "sentiments": chosen_sentiments,
            "topics": chosen_topics,
            "search": search_text,
        }
    explorer_filters = st.session_state["explorer_filters"]

    explorer_params = dict(base_params)
    explorer_clauses = [where_sql]
    selected_sentiments = explorer_filters["sentiments"]
    mapped_sentiments = [x for x in selected_sentiments if x != "Tidak dapat diprediksi"]
    include_unpredicted = "Tidak dapat diprediksi" in selected_sentiments
    sentiment_parts = []
    sentiment_clause = make_named_filter(
        "s.sentiment_label", mapped_sentiments, "sentiment", explorer_params
    )
    if sentiment_clause:
        sentiment_parts.append(sentiment_clause)
    if include_unpredicted:
        sentiment_parts.append("f.sentiment_key IS NULL")
    if sentiment_parts:
        explorer_clauses.append("(" + " OR ".join(sentiment_parts) + ")")
    elif not selected_sentiments:
        explorer_clauses.append("1 = 0")

    chosen_topic_names = explorer_filters["topics"]
    topic_clause = make_named_filter(
        "t.topic_name", chosen_topic_names, "topic", explorer_params
    )
    if topic_clause:
        explorer_clauses.append(topic_clause)
    elif not chosen_topic_names:
        explorer_clauses.append("f.topic_key IS NULL")

    search_value = str(explorer_filters["search"]).strip()
    if search_value:
        explorer_clauses.append("f.review_text ILIKE :search_text")
        explorer_params["search_text"] = f"%{search_value}%"

    explorer_where = " AND ".join(explorer_clauses)
    count_df = read_sql_df(
        f"SELECT COUNT(*) AS total {BASE_FROM} WHERE {explorer_where}",
        explorer_params,
    )
    explorer_total = int(count_df.loc[0, "total"] or 0)
    page_size = st.selectbox("Jumlah baris per halaman", [25, 50, 100], index=1)
    max_pages = max(1, (explorer_total + page_size - 1) // page_size)
    page = st.number_input("Halaman", min_value=1, max_value=max_pages, value=1, step=1)
    offset = (int(page) - 1) * page_size

    explorer_params.update({"limit_rows": int(page_size), "offset_rows": int(offset)})
    review_df = read_sql_df(
        f"""SELECT
                f.review_id,
                a.application_name AS aplikasi,
                d.full_date AS tanggal,
                f.score AS rating,
                COALESCE(s.sentiment_label, 'Tidak dapat diprediksi') AS sentimen,
                t.topic_id,
                t.topic_name AS topik,
                f.prediction_status,
                f.review_text AS ulasan
            {BASE_FROM}
            WHERE {explorer_where}
            ORDER BY d.full_date DESC NULLS LAST, f.loaded_at_utc DESC
            LIMIT :limit_rows OFFSET :offset_rows""",
        explorer_params,
    )
    st.caption(f"Menampilkan {fmt_int(len(review_df))} dari {fmt_int(explorer_total)} review yang cocok.")
    st.dataframe(review_df, use_container_width=True, hide_index=True, height=520)
    if not review_df.empty:
        st.download_button(
            "Unduh halaman ini (CSV)",
            data=review_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="detail_review_halaman.csv",
            mime="text/csv",
        )

with tab_model:
    st.subheader("Informasi Model Klasifikasi")
    model_df = read_sql_df(
        """SELECT model_version, model_name, model_type,
                  validation_macro_f1, test_macro_f1, created_at
           FROM dw.model_registry
           ORDER BY created_at DESC"""
    )
    if model_df.empty:
        st.info("Belum ada metadata model di dw.model_registry.")
    else:
        latest_model = model_df.iloc[0]
        c1, c2 = st.columns(2)
        val_f1 = latest_model["validation_macro_f1"]
        test_f1 = latest_model["test_macro_f1"]
        c1.metric(
            "Validation Macro-F1",
            f"{float(val_f1) * 100:.2f}%" if pd.notna(val_f1) else "Belum tersedia",
        )
        c2.metric(
            "Test Macro-F1",
            f"{float(test_f1) * 100:.2f}%" if pd.notna(test_f1) else "Belum tersedia",
        )
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown("**Catatan metodologis**")
    st.markdown(
        "- Rating Google Play merupakan penilaian bintang, bukan label sentimen.\n"
        "- Sentimen pada dashboard berasal dari prediksi LinearSVC.\n"
        "- Review dengan `empty_after_preprocessing` ditampilkan sebagai "
        "**Tidak dapat diprediksi**, bukan netral.\n"
        "- Topik K-Means hanya tersedia untuk review yang mendapatkan assignment topik."
    )

st.divider()
st.caption(
    "Sumber: Data Warehouse Supabase PostgreSQL · "
    "Analisis Sentimen Pengguna Shopee dan Tokopedia Menggunakan Support Vector Machine"
)
