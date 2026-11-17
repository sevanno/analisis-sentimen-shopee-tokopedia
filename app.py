from __future__ import annotations

import hashlib
import html
import hmac
import io
import json
import logging
import re
from calendar import month_name
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np

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

# Warna sentimen dibuat konsisten pada seluruh grafik.
SENTIMENT_COLORS = {
    "positif": "#16A34A",
    "negatif": "#DC2626",
    "netral": "#2563EB",
    "Tidak dapat diprediksi": "#9CA3AF",
}
SENTIMENT_ORDER = ["negatif", "netral", "positif"]
EVALUATION_DIR = Path(__file__).resolve().parent / "evaluation"

st.markdown(
    """
    <style>
      .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; }
      .stTabs [data-baseweb="tab"] { padding: 0.55rem 0.85rem; }
      .stTabs [aria-selected="true"] { border-bottom-color: #2563EB; }
      [data-testid="stMetric"] { padding: 0.15rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
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



def load_evaluation_files(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved classification report and confusion matrix for one split."""
    if split not in {"validation", "test"}:
        raise ValueError("Split evaluasi tidak dikenal.")

    report_path = EVALUATION_DIR / f"classification_report_{split}.csv"
    matrix_path = EVALUATION_DIR / f"confusion_matrix_{split}.csv"
    if not report_path.is_file() or not matrix_path.is_file():
        raise FileNotFoundError(
            f"File evaluasi {split} belum lengkap di folder evaluation/."
        )

    report = pd.read_csv(report_path, index_col=0, encoding="utf-8-sig")
    matrix = pd.read_csv(matrix_path, index_col=0, encoding="utf-8-sig")
    report.index = report.index.astype(str).str.strip()
    matrix.index = matrix.index.astype(str).str.replace("aktual_", "", regex=False)
    matrix.columns = matrix.columns.astype(str).str.replace("prediksi_", "", regex=False)
    return report, matrix


def render_evaluation_split(split: str, title: str) -> None:
    report, matrix = load_evaluation_files(split)
    required_classes = ["negatif", "netral", "positif"]
    missing = [name for name in required_classes if name not in report.index]
    if missing:
        st.error("Laporan evaluasi tidak memuat kelas: " + ", ".join(missing))
        return

    accuracy = float(report.loc["accuracy", "f1-score"])
    macro = report.loc["macro avg"]
    support = int(report.loc[required_classes, "support"].sum())

    st.markdown(f"**{title}**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    k2.metric("Macro Precision", f"{float(macro['precision']) * 100:.2f}%")
    k3.metric("Macro Recall", f"{float(macro['recall']) * 100:.2f}%")
    k4.metric("Macro-F1", f"{float(macro['f1-score']) * 100:.2f}%")
    st.caption(f"Jumlah data berlabel manual: {support:,}".replace(",", "."))

    report_view = report.loc[required_classes + ["macro avg", "weighted avg"]].copy()
    report_view = report_view.rename(
        index={
            "negatif": "Negatif",
            "netral": "Netral",
            "positif": "Positif",
            "macro avg": "Macro average",
            "weighted avg": "Weighted average",
        },
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-score",
            "support": "Jumlah data",
        },
    )
    st.markdown("**Classification Report**")
    st.dataframe(
        report_view.style.format(
            {"Precision": "{:.2%}", "Recall": "{:.2%}", "F1-score": "{:.2%}", "Jumlah data": "{:.0f}"}
        ),
        use_container_width=True,
    )

    left, right = st.columns([1, 1.15])
    with left:
        f1_frame = report.loc[required_classes, ["f1-score"]].reset_index()
        f1_frame.columns = ["sentimen", "f1_score"]
        f1_frame["sentimen"] = f1_frame["sentimen"].map(
            {"negatif": "Negatif", "netral": "Netral", "positif": "Positif"}
        )
        f1_colors = {"Negatif": SENTIMENT_COLORS["negatif"], "Netral": SENTIMENT_COLORS["netral"], "Positif": SENTIMENT_COLORS["positif"]}
        fig = px.bar(
            f1_frame, x="sentimen", y="f1_score", color="sentimen",
            color_discrete_map=f1_colors, text=f1_frame["f1_score"].map(lambda v: f"{v:.1%}"),
            labels={"sentimen": "Sentimen", "f1_score": "F1-score"},
            category_orders={"sentimen": ["Negatif", "Netral", "Positif"]},
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=15, b=10))
        st.markdown("**F1-score per kelas**")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        matrix = matrix.reindex(index=required_classes, columns=required_classes, fill_value=0)
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        fig = px.imshow(
            matrix.to_numpy(),
            x=["Negatif", "Netral", "Positif"],
            y=["Negatif", "Netral", "Positif"],
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "Prediksi", "y": "Aktual", "color": "Jumlah"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=15, b=10))
        st.markdown("**Confusion Matrix**")
        st.plotly_chart(fig, use_container_width=True)


# -------------------------- Admin CSV upload helpers --------------------------
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_ROWS = 10_000
UPLOAD_BATCH_SIZE = 200


def _to_psycopg_url(connection_url: str) -> str:
    if connection_url.startswith("postgresql://"):
        return connection_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if connection_url.startswith("postgres://"):
        return connection_url.replace("postgres://", "postgresql+psycopg://", 1)
    if connection_url.startswith("postgresql+psycopg://"):
        return connection_url
    raise RuntimeError("URL ingest harus menggunakan PostgreSQL/psycopg dari Supabase Session Pooler.")


@st.cache_resource(show_spinner=False)
def get_ingest_engine() -> Engine:
    """Writer connection; database role is restricted to INSERT on two tables."""
    try:
        connection_url = st.secrets["connections"]["ingest"]["url"]
    except Exception as exc:
        raise RuntimeError(
            "Koneksi ingest belum diatur. Tambahkan [connections.ingest] pada Streamlit Secrets."
        ) from exc
    return create_engine(
        _to_psycopg_url(connection_url),
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=1,
        pool_recycle=1800,
    )


def get_admin_credentials() -> tuple[str, str]:
    try:
        username = str(st.secrets["admin"]["username"])
        password = str(st.secrets["admin"]["password"])
    except Exception as exc:
        raise RuntimeError(
            "Kredensial admin belum diatur. Tambahkan [admin] username dan password pada Streamlit Secrets."
        ) from exc
    if not username or not password:
        raise RuntimeError("Username/password admin tidak boleh kosong.")
    return username, password


def verify_admin(username: str, password: str) -> bool:
    expected_username, expected_password = get_admin_credentials()
    return hmac.compare_digest(username, expected_username) and hmac.compare_digest(
        password, expected_password
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@st.cache_resource(show_spinner="Memuat model klasifikasi dan topik...")
def load_upload_artifacts() -> dict[str, Any]:
    """Load the exact Notebook 05/06 and Notebook 07 artifacts."""
    artifact_paths = {
        "svm_model": ARTIFACT_DIR / "linear_svc_sentiment.joblib",
        "svm_vectorizer": ARTIFACT_DIR / "tfidf_vectorizer.joblib",
        "topic_vectorizer": ARTIFACT_DIR / "kmeans_topic_tfidf_vectorizer.joblib",
        "topic_model": ARTIFACT_DIR / "kmeans_negative_topics.joblib",
        "topic_metadata": ARTIFACT_DIR / "topic_analysis_metadata.json",
    }
    missing = [f"{name}: {path}" for name, path in artifact_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Artefak model belum lengkap. Salin file berikut ke folder artifacts/:\n- "
            + "\n- ".join(missing)
        )

    svm_model = joblib.load(artifact_paths["svm_model"])
    svm_vectorizer = joblib.load(artifact_paths["svm_vectorizer"])
    topic_vectorizer = joblib.load(artifact_paths["topic_vectorizer"])
    topic_model = joblib.load(artifact_paths["topic_model"])
    topic_metadata = json.loads(artifact_paths["topic_metadata"].read_text(encoding="utf-8"))

    model_classes = [str(value) for value in svm_model.classes_]
    if set(model_classes) != {"negatif", "netral", "positif"} or len(model_classes) != 3:
        raise RuntimeError(f"Kelas model SVM tidak sesuai: {model_classes}")
    svm_features = getattr(svm_model, "n_features_in_", None)
    if svm_features is not None and int(svm_features) != len(svm_vectorizer.vocabulary_):
        raise RuntimeError("Jumlah fitur LinearSVC tidak cocok dengan TF-IDF vectorizer.")
    topic_features = getattr(topic_model, "n_features_in_", None)
    if topic_features is not None and int(topic_features) != len(topic_vectorizer.vocabulary_):
        raise RuntimeError("Jumlah fitur K-Means tidak cocok dengan topic TF-IDF vectorizer.")

    created = str(topic_metadata.get("created_at_utc", ""))[:10].replace("-", "")
    if not created:
        raise RuntimeError("topic_analysis_metadata.json tidak memiliki created_at_utc yang valid.")
    selected_k = topic_metadata.get("selected_k")
    if selected_k is None:
        raise RuntimeError("topic_analysis_metadata.json tidak memiliki selected_k.")
    topic_version = f"kmeans-k{selected_k}-{created}"

    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    except ImportError as exc:
        raise RuntimeError("Library Sastrawi belum terpasang. Tambahkan Sastrawi ke requirements.txt.") from exc

    stemmer = StemmerFactory().create_stemmer()
    stopwords = set(StopWordRemoverFactory().get_stop_words())
    keep_words = {
        "tidak", "bukan", "belum", "jangan", "kurang", "tanpa", "enggan",
        "sangat", "banget", "terlalu", "sekali", "cukup", "amat", "lebih", "paling",
        "tapi", "tetapi", "namun", "melainkan", "padahal",
    }
    stopwords -= keep_words

    return {
        "svm_model": svm_model,
        "svm_vectorizer": svm_vectorizer,
        "topic_vectorizer": topic_vectorizer,
        "topic_model": topic_model,
        "model_classes": model_classes,
        "stemmer": stemmer,
        "stopwords": stopwords,
        "model_version": f"linear-svc-{_sha256_file(artifact_paths['svm_model'])[:12]}",
        "topic_version": topic_version,
    }


_NORMALIZATION_MAP = {
    "gak": "tidak", "gk": "tidak", "ga": "tidak", "ngga": "tidak",
    "nggak": "tidak", "enggak": "tidak", "tdk": "tidak", "ndak": "tidak",
    "blm": "belum", "belom": "belum", "udh": "sudah", "udah": "sudah",
    "bgt": "banget", "bgtt": "banget", "aja": "saja", "aj": "saja",
    "tp": "tapi", "ttp": "tetap", "krn": "karena", "karna": "karena",
    "dr": "dari", "dgn": "dengan", "utk": "untuk", "yg": "yang",
    "sm": "sama", "sy": "saya", "gw": "saya", "gue": "saya",
    "org": "orang", "bbrp": "beberapa", "apk": "aplikasi",
}
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_NON_LETTER_RE = re.compile(r"[^a-z\s]")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


def preprocess_review_text(value: Any, stemmer: Any, stopwords: set[str]) -> str:
    """Exact preprocessing logic mirrored from Notebook 06."""
    if pd.isna(value):
        return ""
    value = html.unescape(str(value)).lower()
    value = _URL_RE.sub(" ", value)
    value = _EMAIL_RE.sub(" ", value)
    value = _REPEATED_CHAR_RE.sub(r"\1\1", value)
    value = _NON_LETTER_RE.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""

    tokens = []
    for token in value.split():
        token = _NORMALIZATION_MAP.get(token, token)
        if token and token not in stopwords and len(token) > 1:
            tokens.append(token)
    stemmed = [stemmer.stem(token) for token in tokens]
    stemmed = [token for token in stemmed if token and token not in stopwords]
    return " ".join(stemmed)


def read_and_validate_upload(uploaded_file: Any) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Read a raw Google Play CSV and reject structurally invalid rows."""
    payload = uploaded_file.getvalue()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Ukuran file melebihi 20 MB. Pecah CSV menjadi beberapa file yang lebih kecil.")
    try:
        frame = pd.read_csv(
            io.BytesIO(payload),
            encoding="utf-8-sig",
            dtype={"review_id": "string"},
            low_memory=False,
        )
    except UnicodeDecodeError:
        frame = pd.read_csv(
            io.BytesIO(payload),
            encoding="latin-1",
            dtype={"review_id": "string"},
            low_memory=False,
        )
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    required = {"review_id", "review_text", "score", "at"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Kolom wajib belum ditemukan: " + ", ".join(missing))
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f"File berisi {len(frame):,} baris. Batas saat ini {MAX_UPLOAD_ROWS:,} baris per upload.")

    frame = frame.copy()
    frame["review_id"] = frame["review_id"].fillna("").astype(str).str.strip()
    frame["review_text"] = frame["review_text"].fillna("").astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["at"] = pd.to_datetime(frame["at"], errors="coerce", utc=True)
    if "thumbs_up_count" not in frame.columns:
        frame["thumbs_up_count"] = pd.NA
    frame["thumbs_up_count"] = pd.to_numeric(frame["thumbs_up_count"], errors="coerce")

    reasons = []
    for idx, row in frame.iterrows():
        row_reasons = []
        if not row["review_id"]:
            row_reasons.append("review_id kosong")
        if not row["review_text"]:
            row_reasons.append("review_text kosong")
        score = row["score"]
        if pd.isna(score) or score not in [1, 2, 3, 4, 5]:
            row_reasons.append("score harus bilangan bulat 1-5")
        if pd.isna(row["at"]):
            row_reasons.append("at kosong/tidak dapat dibaca")
        if row_reasons:
            reasons.append({"baris_csv": int(idx) + 2, "review_id": row["review_id"], "alasan": "; ".join(row_reasons)})

    invalid_df = pd.DataFrame(reasons)
    if not invalid_df.empty:
        return pd.DataFrame(), 0, invalid_df

    frame["score"] = frame["score"].astype(int)
    duplicate_count = int(frame["review_id"].duplicated(keep="first").sum())
    frame = frame.drop_duplicates(subset=["review_id"], keep="first").copy()
    frame["at"] = frame["at"].map(lambda value: value.isoformat())
    frame = frame[["review_id", "review_text", "score", "at", "thumbs_up_count"]]
    return frame.reset_index(drop=True), duplicate_count, pd.DataFrame()


def predict_uploaded_reviews(frame: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    out["text_preprocessed"] = out["review_text"].map(
        lambda value: preprocess_review_text(value, artifacts["stemmer"], artifacts["stopwords"])
    )
    out["predicted_sentiment"] = None
    out["prediction_status"] = "empty_after_preprocessing"
    out["decision_score_negatif"] = np.nan
    out["decision_score_netral"] = np.nan
    out["decision_score_positif"] = np.nan
    out["svm_decision_margin_gap"] = np.nan
    out["topic_id"] = pd.Series([None] * len(out), dtype="object")

    valid_mask = out["text_preprocessed"].str.strip().ne("")
    if valid_mask.any():
        texts = out.loc[valid_mask, "text_preprocessed"].tolist()
        x_svm = artifacts["svm_vectorizer"].transform(texts)
        predictions = artifacts["svm_model"].predict(x_svm)
        scores = np.asarray(artifacts["svm_model"].decision_function(x_svm))
        if scores.ndim != 2 or scores.shape[1] != 3:
            raise RuntimeError("Output decision_function LinearSVC bukan matriks tiga kelas.")
        out.loc[valid_mask, "predicted_sentiment"] = predictions
        out.loc[valid_mask, "prediction_status"] = "predicted"
        for index, label in enumerate(artifacts["model_classes"]):
            out.loc[valid_mask, f"decision_score_{label}"] = scores[:, index]
        sorted_scores = np.sort(scores, axis=1)
        out.loc[valid_mask, "svm_decision_margin_gap"] = sorted_scores[:, -1] - sorted_scores[:, -2]

        negative_mask = valid_mask & out["predicted_sentiment"].eq("negatif")
        if negative_mask.any():
            topic_texts = out.loc[negative_mask, "text_preprocessed"].tolist()
            x_topics = artifacts["topic_vectorizer"].transform(topic_texts)
            topic_ids = artifacts["topic_model"].predict(x_topics)
            out.loc[negative_mask, "topic_id"] = [int(value) for value in topic_ids]

    return out


def get_upload_dimension_maps(topic_version: str, model_version: str) -> dict[str, Any]:
    """Read the registered app/sentiment/topic/model keys using read-only credentials."""
    app_df = read_sql_df(
        "SELECT application_key, application_name FROM dw.dim_application"
    )
    sentiment_df = read_sql_df(
        "SELECT sentiment_key, sentiment_label FROM dw.dim_sentiment"
    )
    topic_df = read_sql_df(
        "SELECT topic_key, topic_id FROM dw.dim_topic WHERE topic_version = :v ORDER BY topic_id",
        {"v": topic_version},
    )
    model_df = read_sql_df(
        "SELECT model_version FROM dw.model_registry WHERE model_version = :v",
        {"v": model_version},
    )
    if model_df.empty:
        raise RuntimeError(
            "Model LinearSVC yang tersedia di artifacts tidak cocok dengan dw.model_registry. "
            "Pastikan file linear_svc_sentiment.joblib sama dengan model dari Notebook 05/08."
        )
    if topic_df.empty:
        raise RuntimeError(
            f"Versi topik {topic_version} tidak ditemukan di dw.dim_topic. "
            "Pastikan model K-Means dan topic_analysis_metadata.json berasal dari Notebook 07 yang sama dengan Notebook 08."
        )
    if topic_df["topic_id"].duplicated().any():
        raise RuntimeError("Ada topic_id duplikat pada versi topik di dw.dim_topic.")
    return {
        "application_keys": dict(zip(app_df["application_name"], app_df["application_key"].astype(int))),
        "sentiment_keys": dict(zip(sentiment_df["sentiment_label"], sentiment_df["sentiment_key"].astype(int))),
        "topic_keys": dict(zip(topic_df["topic_id"].astype(int), topic_df["topic_key"].astype(int))),
    }


def _nullable_number(value: Any, cast_type: Any = float) -> Any:
    if value is None or pd.isna(value):
        return None
    return cast_type(value)


def build_date_dimension_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    dates = pd.to_datetime(frame["at"], errors="coerce", utc=True).dt.tz_convert("Asia/Jakarta").dt.date
    rows: dict[int, dict[str, Any]] = {}
    for value in dates.dropna().unique():
        date_value = pd.Timestamp(value).date()
        key = int(date_value.strftime("%Y%m%d"))
        dow = date_value.isoweekday()
        rows[key] = {
            "date_key": key,
            "full_date": date_value,
            "day": date_value.day,
            "month": date_value.month,
            "month_name": month_name[date_value.month],
            "quarter": (date_value.month - 1) // 3 + 1,
            "year": date_value.year,
            "day_of_week": dow,
            "is_weekend": dow >= 6,
        }
    return list(rows.values())


def build_fact_rows(
    frame: pd.DataFrame,
    application_name: str,
    maps: dict[str, Any],
    model_version: str,
) -> list[dict[str, Any]]:
    if application_name not in maps["application_keys"]:
        raise RuntimeError(f"Aplikasi {application_name} belum terdaftar pada dw.dim_application.")
    app_key = maps["application_keys"][application_name]
    dates = pd.to_datetime(frame["at"], errors="coerce", utc=True).dt.tz_convert("Asia/Jakarta").dt.date
    records = []
    for position, (_, row) in enumerate(frame.iterrows()):
        date_value = dates.iloc[position]
        date_key = int(date_value.strftime("%Y%m%d")) if pd.notna(date_value) else 0
        sentiment = row["predicted_sentiment"]
        sentiment_key = maps["sentiment_keys"].get(str(sentiment)) if sentiment is not None and not pd.isna(sentiment) else None
        topic_value = row["topic_id"]
        topic_key = None
        if topic_value is not None and not pd.isna(topic_value):
            topic_id = int(topic_value)
            if topic_id not in maps["topic_keys"]:
                raise RuntimeError(
                    f"topic_id {topic_id} tidak ditemukan pada versi topik yang terdaftar. "
                    "Periksa kesesuaian artefak K-Means dengan dw.dim_topic."
                )
            topic_key = maps["topic_keys"][topic_id]
        status = str(row["prediction_status"])
        if status == "predicted" and sentiment_key is None:
            raise RuntimeError("Ada baris berstatus predicted tetapi sentiment_key tidak ditemukan.")
        thumbs = _nullable_number(row.get("thumbs_up_count"), int)
        if thumbs is not None and thumbs < 0:
            thumbs = None
        records.append({
            "review_id": str(row["review_id"]),
            "application_key": int(app_key),
            "date_key": date_key,
            "sentiment_key": sentiment_key,
            "topic_key": topic_key,
            "score": int(row["score"]),
            "thumbs_up_count": thumbs,
            "review_text": str(row["review_text"]),
            "prediction_status": status,
            "svm_decision_margin_gap": _nullable_number(row["svm_decision_margin_gap"]),
            "decision_score_negatif": _nullable_number(row["decision_score_negatif"]),
            "decision_score_netral": _nullable_number(row["decision_score_netral"]),
            "decision_score_positif": _nullable_number(row["decision_score_positif"]),
            "model_version": model_version,
        })
    return records


def insert_fact_rows(conn: Any, records: list[dict[str, Any]]) -> int:
    columns = [
        "review_id", "application_key", "date_key", "sentiment_key", "topic_key",
        "score", "thumbs_up_count", "review_text", "prediction_status",
        "svm_decision_margin_gap", "decision_score_negatif", "decision_score_netral",
        "decision_score_positif", "model_version",
    ]
    inserted_count = 0
    for start in range(0, len(records), UPLOAD_BATCH_SIZE):
        batch = records[start:start + UPLOAD_BATCH_SIZE]
        values_sql = []
        params: dict[str, Any] = {}
        for row_idx, record in enumerate(batch):
            placeholders = []
            for col in columns:
                key = f"v{row_idx}_{col}"
                placeholders.append(f":{key}")
                params[key] = record[col]
            values_sql.append("(" + ", ".join(placeholders) + ")")
        sql = text(
            "INSERT INTO dw.fact_review (" + ", ".join(columns) + ") VALUES "
            + ", ".join(values_sql)
            + " ON CONFLICT (review_id) DO NOTHING"
        )
        result = conn.execute(sql, params)
        if result.rowcount is not None and result.rowcount >= 0:
            inserted_count += int(result.rowcount)
        else:
            raise RuntimeError("Database driver tidak mengembalikan rowcount yang valid untuk proses upload.")
    return inserted_count


def save_uploaded_reviews(frame: pd.DataFrame, application_name: str, artifacts: dict[str, Any], maps: dict[str, Any]) -> tuple[int, int]:
    records = build_fact_rows(frame, application_name, maps, artifacts["model_version"])
    date_rows = build_date_dimension_rows(frame)
    date_sql = text("""
        INSERT INTO dw.dim_date
            (date_key, full_date, day, month, month_name, quarter, year, day_of_week, is_weekend)
        VALUES
            (:date_key, :full_date, :day, :month, :month_name, :quarter, :year, :day_of_week, :is_weekend)
        ON CONFLICT (date_key) DO NOTHING
    """)
    with get_ingest_engine().begin() as conn:
        if date_rows:
            conn.execute(date_sql, date_rows)
        inserted_count = insert_fact_rows(conn, records)
    return inserted_count, len(records) - inserted_count


def render_admin_upload_tab(application_options: list[str]) -> None:
    st.subheader("Kelola Data Review")
    st.caption("Tambahkan review baru melalui file CSV.")

    try:
        get_admin_credentials()
    except Exception as exc:
        st.error(str(exc))
        return

    if not st.session_state.get("admin_authenticated", False):
        with st.form("admin_login_form"):
            st.markdown("**Login admin**")
            username = st.text_input("Username admin")
            password = st.text_input("Password admin", type="password")
            login_clicked = st.form_submit_button("Login", type="primary")
        if login_clicked:
            try:
                if verify_admin(username.strip(), password):
                    st.session_state["admin_authenticated"] = True
                    st.session_state["admin_username"] = username.strip()
                    st.rerun()
                else:
                    st.error("Username atau password administrator tidak sesuai.")
            except Exception as exc:
                st.error(str(exc))
        st.info("Login hanya untuk administrator.")
        return

    top_left, top_right = st.columns([4, 1])
    top_left.success(f"Login sebagai administrator: {st.session_state.get('admin_username', 'admin')}")
    if top_right.button("Logout", key="admin_logout"):
        st.session_state["admin_authenticated"] = False
        st.session_state.pop("admin_username", None)
        st.session_state.pop("admin_upload_preview", None)
        st.session_state.pop("admin_upload_hash", None)
        st.rerun()

    last_result = st.session_state.pop("admin_last_upload_result", None)
    if last_result:
        st.success(last_result)

    st.markdown("#### Pilih aplikasi dan file CSV")
    if not application_options:
        st.error("Dimensi aplikasi kosong. Periksa dw.dim_application.")
        return
    application_name = st.selectbox("Aplikasi untuk CSV ini", application_options, key="admin_app_select")
    uploaded_file = st.file_uploader(
        "Pilih CSV ulasan mentah",
        type=["csv"],
        key="admin_csv_upload",
        help="Kolom wajib: review_id, review_text, score, dan at.",
    )

    if uploaded_file is None:
        st.info("Unggah CSV review mentah sesuai template.")
        st.download_button(
            "Unduh template CSV",
            data="review_id,review_text,score,at,thumbs_up_count\n".encode("utf-8"),
            file_name="template_upload_review.csv",
            mime="text/csv",
        )
        return

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    current_preview = st.session_state.get("admin_upload_preview")
    if current_preview and (
        st.session_state.get("admin_upload_hash") != file_hash
        or st.session_state.get("admin_upload_app") != application_name
    ):
        st.session_state.pop("admin_upload_preview", None)
        st.session_state.pop("admin_upload_hash", None)
        st.session_state.pop("admin_upload_invalid", None)

    st.caption(f"File: {uploaded_file.name} · Ukuran: {uploaded_file.size / 1024:.1f} KB")
    st.markdown("#### Periksa dan prediksi data")
    if st.button("Validasi CSV dan prediksi", type="primary", key="admin_validate_csv"):
        try:
            clean_df, duplicate_file_count, invalid_df = read_and_validate_upload(uploaded_file)
            if not invalid_df.empty:
                st.session_state["admin_upload_invalid"] = invalid_df
                st.session_state.pop("admin_upload_preview", None)
                st.error(f"Ditemukan {len(invalid_df)} baris tidak valid. Perbaiki CSV lalu upload kembali.")
            elif clean_df.empty:
                st.error("Tidak ada baris valid pada CSV.")
            else:
                artifacts = load_upload_artifacts()
                maps = get_upload_dimension_maps(artifacts["topic_version"], artifacts["model_version"])
                if application_name not in maps["application_keys"]:
                    raise RuntimeError(f"Aplikasi {application_name} tidak terdaftar di dw.dim_application.")
                clean_df = predict_uploaded_reviews(clean_df, artifacts)
                st.session_state["admin_upload_preview"] = {
                    "data": clean_df,
                    "duplicate_file_count": duplicate_file_count,
                    "application": application_name,
                    "filename": uploaded_file.name,
                    "model_version": artifacts["model_version"],
                    "topic_version": artifacts["topic_version"],
                    "topic_key_count": len(maps["topic_keys"]),
                }
                st.session_state["admin_upload_hash"] = file_hash
                st.session_state["admin_upload_app"] = application_name
                st.session_state.pop("admin_upload_invalid", None)
        except Exception as exc:
            st.error(f"Validasi/prediksi tidak berhasil: {type(exc).__name__}: {exc}")

    invalid_previous = st.session_state.get("admin_upload_invalid")
    if invalid_previous is not None and not invalid_previous.empty:
        st.warning("Contoh baris CSV yang perlu diperbaiki:")
        st.dataframe(invalid_previous.head(30), use_container_width=True, hide_index=True)

    preview_state = st.session_state.get("admin_upload_preview")
    if preview_state and preview_state.get("application") == application_name:
        preview_df = preview_state["data"]
        st.markdown("#### Pratinjau hasil")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baris valid", fmt_int(len(preview_df)))
        c2.metric("Duplikat dalam CSV", fmt_int(preview_state["duplicate_file_count"]))
        c3.metric("Prediksi sentimen", fmt_int(int(preview_df["prediction_status"].eq("predicted").sum())))
        c4.metric("Tidak dapat diprediksi", fmt_int(int(preview_df["prediction_status"].eq("empty_after_preprocessing").sum())))

        sentiment_counts = preview_df["predicted_sentiment"].fillna("Tidak dapat diprediksi").value_counts(dropna=False).rename_axis("sentimen").reset_index(name="jumlah")
        st.write("Distribusi sentimen hasil prediksi")
        st.dataframe(sentiment_counts, use_container_width=True, hide_index=True)
        topic_counts = preview_df[preview_df["topic_id"].notna()]["topic_id"].astype(int).value_counts().sort_index().rename_axis("topic_id").reset_index(name="jumlah")
        if not topic_counts.empty:
            st.write("Distribusi topic_id untuk prediksi negatif")
            st.dataframe(topic_counts, use_container_width=True, hide_index=True)

        display_columns = ["review_id", "review_text", "score", "at", "predicted_sentiment", "prediction_status", "topic_id"]
        st.dataframe(preview_df[display_columns].head(30), use_container_width=True, hide_index=True)
        st.caption(
            f"Model: {preview_state['model_version']} · Versi topik: {preview_state['topic_version']} · "
            f"Dimensi topik ditemukan: {preview_state['topic_key_count']}"
        )

        confirm_key = f"admin_upload_confirm_{file_hash[:12]}"
        save_button_key = f"admin_save_upload_{file_hash[:12]}"
        confirmed = st.checkbox(
            "Saya sudah memeriksa preview dan menyetujui penyimpanan data ini ke Data Warehouse.",
            key=confirm_key,
        )
        if st.button(
            f"Simpan {len(preview_df):,} review ke Supabase",
            type="primary",
            disabled=not confirmed,
            key=save_button_key,
        ):
            try:
                with st.spinner("Menyimpan review baru ke Supabase..."):
                    artifacts = load_upload_artifacts()
                    maps = get_upload_dimension_maps(artifacts["topic_version"], artifacts["model_version"])
                    inserted, duplicates_db = save_uploaded_reviews(
                        preview_df, application_name, artifacts, maps
                    )
                st.cache_data.clear()
                st.session_state.pop("admin_upload_preview", None)
                st.session_state.pop("admin_upload_hash", None)
                st.session_state.pop("admin_upload_app", None)
                st.session_state["admin_last_upload_result"] = (
                    f"Proses selesai. {inserted:,} review baru berhasil ditambahkan; "
                    f"{duplicates_db:,} review dilewati karena review_id sudah ada di Data Warehouse. "
                    f"{preview_state['duplicate_file_count']:,} duplikat di dalam CSV juga telah diabaikan."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Penyimpanan gagal. Tidak ada perubahan parsial karena proses menggunakan satu transaksi. Detail: {type(exc).__name__}: {exc}")


st.title("Analisis Sentimen Shopee & Tokopedia")
st.caption("Analisis ulasan Google Play Store")

try:
    engine = get_engine()
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    application_options, min_date, max_date, topic_options = load_filter_options()
except Exception as exc:  # pragma: no cover - depends on user's Supabase
    show_connection_error(exc)
    st.stop()

with st.sidebar:
    st.header("Filter")
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

st.caption("Persentase negatif dihitung dari review yang berhasil diprediksi.")

# ------------------------------- Page tabs --------------------------------
tab_overview, tab_sentiment, tab_topics, tab_explorer, tab_model, tab_admin = st.tabs(
    ["Ringkasan", "Sentimen", "Topik", "Review", "Evaluasi Model", "Kelola Data"]
)

with tab_overview:
    st.subheader("Ringkasan Ulasan")
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
                color="sentimen",
                color_discrete_map=SENTIMENT_COLORS,
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
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            labels={"aplikasi": "Aplikasi", "jumlah": "Jumlah review", "sentimen": "Sentimen"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab_sentiment:
    st.subheader("Tren Sentimen")
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
            color_discrete_map=SENTIMENT_COLORS,
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
            color_discrete_map=SENTIMENT_COLORS,
            barmode="group",
            labels={"aplikasi": "Aplikasi", "jumlah": "Jumlah review", "sentimen": "Sentimen"},
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_topics:
    st.subheader("Topik Keluhan")
    st.caption("Topik dibentuk dari review yang diprediksi negatif.")

    with st.expander("Lihat cara penentuan jumlah cluster (K)"):
        st.markdown(
            """
            Jumlah cluster tidak ditentukan secara langsung oleh K-Means. Pada penelitian ini,
            beberapa kandidat nilai K diuji terlebih dahulu, kemudian dibandingkan menggunakan
            **Silhouette Score**.

            Tahap pemilihannya:
            1. Ulasan yang diprediksi negatif oleh LinearSVC dipilih sebagai data topic modeling.
            2. Data fitting diambil secara seimbang, maksimal 7.000 ulasan dari setiap aplikasi.
            3. Teks diubah menjadi fitur TF-IDF khusus untuk topic modeling dan dinormalisasi L2.
            4. K-Means dijalankan untuk kandidat K = 4, 5, 6, 7, dan 8.
            5. Silhouette Score dihitung pada sampel 2.000 ulasan dengan cosine distance.
            6. K dengan skor tertinggi dipilih sebagai kandidat, kemudian kata dominan dan contoh
               ulasan diperiksa secara manual untuk memberi nama topik.
            """
        )
        k_score_df = pd.DataFrame(
            {
                "Jumlah cluster (K)": [4, 5, 6, 7, 8],
                "Silhouette Score": [0.0145, 0.0153, 0.0125, 0.0176, 0.0111],
            }
        )
        st.dataframe(k_score_df, use_container_width=True, hide_index=True)
        st.success(
            "K = 7 dipilih karena memperoleh Silhouette Score tertinggi "
            "(0,0176) di antara kandidat yang diuji."
        )
        st.warning(
            "Seluruh skor berada dekat dengan 0. Ini menunjukkan pemisahan cluster "
            "masih lemah dan beberapa topik mungkin saling tumpang tindih. Karena itu, "
            "hasil K-Means ditafsirkan secara hati-hati berdasarkan kata dominan dan "
            "contoh ulasan, bukan dianggap sebagai kelompok yang terpisah sempurna."
        )
        st.caption(
            "Catatan teknis: K-Means pada TF-IDF yang dinormalisasi L2 menggunakan "
            "optimasi centroid berbasis Euclidean, sedangkan Silhouette Score dievaluasi "
            "menggunakan cosine distance."
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
    st.subheader("Detail Review")
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
    st.subheader("Evaluasi Model LinearSVC")
    st.caption("Evaluasi menggunakan review yang telah diberi label secara manual.")
    eval_validation, eval_test = st.tabs(["Validasi", "Pengujian"] )
    with eval_validation:
        try:
            render_evaluation_split("validation", "Hasil pada data validasi")
        except Exception as exc:
            st.error(f"File evaluasi validasi belum dapat dibaca: {type(exc).__name__}: {exc}")
    with eval_test:
        try:
            render_evaluation_split("test", "Hasil pada data pengujian")
        except Exception as exc:
            st.error(f"File evaluasi test belum dapat dibaca: {type(exc).__name__}: {exc}")

    with st.expander("Tentang evaluasi"):
        st.write(
            "Model menggunakan tiga kelas sentimen: negatif, netral, dan positif. "
            "‘Tidak dapat diprediksi’ merupakan status untuk review yang kosong setelah preprocessing, "
            "bukan kelas sentimen keempat. Macro-F1 menghitung rata-rata F1 ketiga kelas dengan bobot sama."
        )
        st.write(
            "Performa kelas netral masih lebih rendah dibanding kelas lainnya. "
            "Jumlah data evaluasi yang relatif terbatas juga perlu dipertimbangkan saat menafsirkan hasil."
        )

    model_df = read_sql_df(
        """SELECT model_version, model_name, model_type,
                  validation_macro_f1, test_macro_f1, created_at
           FROM dw.model_registry
           ORDER BY created_at DESC"""
    )
    if not model_df.empty:
        with st.expander("Versi model"):
            st.dataframe(model_df, use_container_width=True, hide_index=True)

with tab_admin:
    render_admin_upload_tab(application_options)

