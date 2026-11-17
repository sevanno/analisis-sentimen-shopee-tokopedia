from pathlib import Path
import tomllib

from sqlalchemy import create_engine, text

# Membaca konfigurasi rahasia
project_root = Path(__file__).resolve().parent.parent
secret_file = project_root / ".streamlit" / "secrets.toml"

if not secret_file.exists():
    raise FileNotFoundError("secrets.toml tidak ditemukan")

config = tomllib.loads(
    secret_file.read_text(encoding="utf-8")
)

url = config["connections"]["dw"]["url"]

# Menyesuaikan driver PostgreSQL
if url.startswith("postgresql://"):
    url = url.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1
    )
elif url.startswith("postgres://"):
    url = url.replace(
        "postgres://",
        "postgresql+psycopg://",
        1
    )

engine = create_engine(url, pool_pre_ping=True)

with engine.connect() as conn:

    # Identitas database dan user
    identity = conn.execute(
        text("SELECT current_database(), current_user")
    ).one()

    print("Database:", identity[0])
    print("User:", identity[1])

    # Memeriksa dim_date
    result = conn.execute(text("""
        SELECT
            COUNT(*) AS total_baris,
            COUNT(full_date) AS tanggal_valid,
            COUNT(*) FILTER (
                WHERE full_date IS NULL
            ) AS tanggal_null,
            MIN(full_date) AS tanggal_awal,
            MAX(full_date) AS tanggal_akhir
        FROM dw.dim_date
    """)).mappings().one()

    print("\nHASIL PEMERIKSAAN DIM_DATE")

    for key, value in result.items():
        print(f"{key}: {value}")

engine.dispose()