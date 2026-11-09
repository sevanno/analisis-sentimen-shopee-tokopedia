# Dashboard Analisis Sentimen Shopee & Tokopedia

Aplikasi Streamlit untuk membaca Data Warehouse Supabase PostgreSQL pada schema `dw`.
Dashboard menggunakan database role `dashboard_reader` (read-only).

## Fitur

- KPI jumlah review, rata-rata rating, jumlah/persentase review negatif, dan review tanpa prediksi.
- Distribusi rating dan sentimen.
- Perbandingan Shopee dan Tokopedia.
- Tren sentimen bulanan.
- Distribusi topik K-Means.
- Penelusuran detail review dengan filter dan pencarian teks.
- Informasi Macro-F1 dari `dw.model_registry`.

## Struktur file

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## 1. Persiapan koneksi Supabase

1. Di Supabase, buka **Connect** lalu pilih **Session pooler**.
2. Salin connection string yang ditampilkan Supabase.
3. Gunakan role `dashboard_reader`, bukan `postgres`.
4. Pada Shared Session Pooler, username custom role berbentuk `dashboard_reader.PROJECT_REF`.
5. Gunakan host dan port persis dari Supabase Connect. Jangan menebak host berdasarkan region.
6. Jika password berisi karakter khusus seperti `@`, `#`, `?`, `/`, atau spasi, URL-encode bagian password.

## 2. Menjalankan secara lokal di Windows

Buka terminal pada folder proyek ini, kemudian jalankan:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika versi Python 3.14 tidak tersedia di laptop, gunakan versi Python 3.12 atau versi lain yang sudah terpasang dan kompatibel.

Salin template secrets:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit/secrets.toml` dan ganti URL contoh dengan URL Session Pooler role `dashboard_reader` milik proyek Supabase Anda.

Jalankan aplikasi:

```powershell
streamlit run app.py
```

Browser akan membuka dashboard lokal. Jangan unggah `.streamlit/secrets.toml` ke GitHub.

## 3. Deploy ke Streamlit Community Cloud

1. Buat repository GitHub dan unggah file aplikasi ini.
2. Pastikan `.streamlit/secrets.toml` tidak masuk repository. `.gitignore` sudah mengecualikannya.
3. Buka Streamlit Community Cloud dan pilih **Create app**.
4. Pilih repository, branch, dan `app.py` sebagai entrypoint.
5. Pada **Advanced settings > Secrets**, tempel isi `secrets.toml` yang benar.
6. Deploy dan periksa log jika ada error koneksi.

## Catatan metodologis

- `score` adalah rating asli Google Play, bukan sentimen.
- Sentimen ditarik dari `dw.dim_sentiment` berdasarkan hasil prediksi LinearSVC.
- `sentiment_key IS NULL` ditampilkan sebagai **Tidak dapat diprediksi**, bukan kelas netral.
- Topik K-Means hanya tersedia untuk review yang memperoleh `topic_key`.
- Tren bulanan memakai tanggal pada `dw.dim_date`.
- Detail review dibaca dengan pagination; aplikasi tidak menarik seluruh 285 ribu review sekaligus.

## Troubleshooting

- **Authentication failed**: periksa password role `dashboard_reader`, username pooler `dashboard_reader.PROJECT_REF`, host, dan port dari menu Connect.
- **Permission denied for schema dw/table**: periksa `USAGE` pada schema `dw` dan `SELECT` pada tabel yang digunakan.
- **ModuleNotFoundError**: pastikan `pip install -r requirements.txt` berhasil.
- Jangan pernah menaruh password database owner, Supabase `service_role`, atau rahasia lain di source code maupun GitHub.
