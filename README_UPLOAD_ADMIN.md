# Fitur Upload CSV (Admin) — Dashboard Sentimen Shopee & Tokopedia

## Fitur

- Login administrator dikonfigurasi melalui Streamlit Secrets.
- Upload CSV ulasan mentah Google Play untuk satu aplikasi setiap proses.
- Validasi kolom wajib, rating, tanggal, ID kosong, dan duplikasi dalam file.
- Preprocessing disalin dari Notebook 06 (Sastrawi, stopword, normalisasi).
- Prediksi sentimen menggunakan model LinearSVC dan TF-IDF tersimpan.
- Untuk prediksi negatif, `topic_id` ditentukan menggunakan topic TF-IDF dan K-Means tersimpan dari Notebook 07.
- Preview ditampilkan sebelum data disimpan.
- Penulisan bersifat append-only: ID yang sudah ada tidak ditimpa; duplikat dilewati.
- Cache dashboard dibersihkan setelah upload sukses.

## 1. Format CSV

Kolom wajib:

- `review_id`: ID unik review Google Play, tidak kosong.
- `review_text`: teks review, tidak kosong.
- `score`: rating bilangan bulat 1 sampai 5.
- `at`: timestamp yang dapat dibaca pandas, misalnya `2026-11-01T08:30:00Z`.

Kolom opsional:

- `thumbs_up_count`: jumlah helpful votes. Jika tidak ada, nilainya disimpan sebagai NULL.

Template tersedia pada `template_upload_review.csv`. Pilih Shopee atau Tokopedia pada dropdown dashboard; kolom aplikasi tidak diperlukan di CSV.

Batas awal aplikasi: 20 MB dan maksimum 10.000 baris per upload. Upload satu aplikasi per file. Baris dengan ID, teks, rating, atau tanggal tidak valid akan ditolak pada tahap validasi.

## 2. Salin model artifacts

Buat folder `artifacts/` pada root dashboard dan salin file berikut dari folder project skripsi:

| Sumber project skripsi | Tujuan dashboard |
|---|---|
| `models/svm/linear_svc_sentiment.joblib` | `artifacts/linear_svc_sentiment.joblib` |
| `models/svm/tfidf_vectorizer.joblib` | `artifacts/tfidf_vectorizer.joblib` |
| `models/topic_model/kmeans_topic_tfidf_vectorizer.joblib` | `artifacts/kmeans_topic_tfidf_vectorizer.joblib` |
| `models/topic_model/kmeans_negative_topics.joblib` | `artifacts/kmeans_negative_topics.joblib` |
| `data/processed/kmeans_topic_analysis/topic_analysis_metadata.json` | `artifacts/topic_analysis_metadata.json` |

Empat file joblib dan metadata harus berasal dari run Notebook 05–07 yang sama dengan model/version yang didaftarkan Notebook 08. Aplikasi memverifikasi versi hash LinearSVC dan versi topik terhadap `dw.model_registry` serta `dw.dim_topic`. Jika tidak cocok, upload ditolak agar tidak terjadi pencampuran model.

Jalankan pemeriksaan lokal:

```powershell
python check_model_artifacts.py
```

File `.joblib` bukan kredensial, tetapi file model harus diperlakukan sebagai artefak aplikasi. Periksa ukuran setiap file sebelum melakukan push. Jangan commit dataset mentah, `.env`, atau `secrets.toml`.

**Kompatibilitas model:** `joblib`/scikit-learn model pickle paling aman dibuka dengan versi scikit-learn dan joblib yang sama seperti saat model disimpan. Jalankan `python check_model_artifacts.py` pada environment yang digunakan untuk Notebook 05/06, lalu pin versi `scikit-learn` dan `joblib` tersebut secara tepat di `requirements.txt` sebelum deploy.

## 3. Buat role database khusus untuk upload

Jangan gunakan akun `postgres` atau `dashboard_reader` untuk menulis data.

1. Buka Supabase SQL Editor sebagai administrator.
2. Buka `sql/03_dashboard_ingest_role_template.sql`.
3. Ganti `REPLACE_WITH_LONG_RANDOM_PASSWORD` dengan password acak kuat untuk role `dashboard_ingest`.
4. Jalankan seluruh SQL.
5. Periksa hasil verifikasi. Target: `can_use_dw=true`, `can_insert_date=true`, `can_insert_fact=true`; sedangkan `can_select_fact`, `can_update_fact`, dan `can_delete_fact` harus `false`.

Role ini tidak diberi izin SELECT/UPDATE/DELETE. Dashboard tetap membaca data menggunakan role `dashboard_reader`; role `dashboard_ingest` hanya untuk INSERT pada `dw.dim_date` dan `dw.fact_review`.

## 4. Atur Streamlit Secrets

Pertahankan `[connections.dw]` yang sudah digunakan dashboard. Tambahkan bagian berikut ke file lokal `.streamlit/secrets.toml` dan Streamlit Cloud → App Settings → Secrets:

```toml
[connections.ingest]
url = "postgresql://dashboard_ingest.PROJECT_REF:INGEST_PASSWORD@POOLER_HOST:5432/postgres?sslmode=require"

[admin]
username = "admin"
password = "REPLACE_WITH_A_LONG_RANDOM_ADMIN_PASSWORD"
```

Salin HOST, port, dan format username pooler dari menu Connect Supabase. Untuk Shared Session Pooler username custom role umumnya `dashboard_ingest.PROJECT_REF`. Jangan menaruh password di GitHub. Gunakan password admin yang berbeda dari password database.

## 5. Upload ke GitHub dan deploy ulang

1. Ganti `app.py` lama dengan file `app.py` dari paket ini.
2. Ganti `requirements.txt` dengan versi paket ini, lalu pin `scikit-learn` dan `joblib` sesuai environment pembuat model.
3. Tambahkan folder `artifacts/` berisi lima artefak yang diminta.
4. Tambahkan `sql/03_dashboard_ingest_role_template.sql`, `README_UPLOAD_ADMIN.md`, `template_upload_review.csv`, dan `check_model_artifacts.py`.
5. Pastikan `.gitignore` memuat `.streamlit/secrets.toml`, `.env`, dan `.venv/`.
6. Periksa `git status`, lalu commit dan push ke GitHub.
7. Streamlit Community Cloud akan mendeteksi perubahan repository dan melakukan deployment ulang. Pastikan Secrets di Cloud sudah memiliki `[connections.ingest]` dan `[admin]`.

## 6. Alur penggunaan

1. Buka tab **Manajemen Data (Admin)**.
2. Login menggunakan username/password admin dari Secrets.
3. Pilih aplikasi untuk CSV tersebut.
4. Upload CSV mentah.
5. Klik **Validasi CSV dan prediksi**.
6. Periksa jumlah baris, distribusi sentimen/topik, dan preview review.
7. Centang persetujuan, lalu klik **Simpan ... review ke Supabase**.
8. Baris dengan `review_id` yang sudah ada tidak ditimpa; data tersebut dilaporkan sebagai duplikat dan dilewati.
9. Dashboard membersihkan cache setelah upload sukses.

## 7. Batasan penting

- Fitur ini hanya menerima CSV mentah, bukan CSV hasil prediksi Notebook 06.
- Satu file hanya untuk satu aplikasi yang dipilih pada dropdown.
- Klasifikasi menggunakan LinearSVC yang sudah ada, tidak melakukan training ulang.
- Topik hanya diberikan pada review yang diprediksi negatif dan diproses oleh model K-Means tersimpan. Topic ID harus cocok dengan versi yang ada di `dw.dim_topic`.
- `decision_function` LinearSVC disimpan sebagai skor/margin keputusan, bukan probabilitas.
- Upload append-only dan tidak mengubah review lama. Untuk memperbaiki data lama, gunakan proses ETL terkontrol terpisah.
- Fitur ini menambahkan jalur data baru ke dashboard. Uji dulu menggunakan CSV kecil (2–5 baris) dan verifikasi hasil di Supabase sebelum menggunakannya untuk file yang lebih besar.


## 8. Halaman Evaluasi Model (semua pengguna)

Halaman Evaluasi Model tetap dapat dilihat oleh pengunjung umum. Akses administrator hanya diperlukan untuk tab Manajemen Data (Admin). Empat CSV pada folder `evaluation/` menyediakan accuracy, macro precision, macro recall, Macro-F1, laporan metrik per kelas, dan confusion matrix untuk validation/test.

Angka pada file evaluasi yang disertakan berasal dari hasil Notebook 05 yang diberikan peneliti:
- Validation: accuracy 82,20%, macro precision 86,13%, macro recall 65,32%, Macro-F1 61,16% (118 sampel).
- Test: accuracy 89,02%, macro precision 86,12%, macro recall 71,79%, Macro-F1 74,85% (173 sampel).

Performa kelas netral relatif rendah (validation F1 11,76%; test F1 42,11%). Karena support kelas netral pada sampel juga kecil (16 validation dan 14 test), angka tersebut perlu ditafsirkan hati-hati. Sampel test pilot sebelumnya pernah digunakan dalam evaluasi kandidat teacher IndoBERT; meskipun test tidak digunakan untuk tuning LinearSVC di Notebook 05, keterbatasan ini perlu dijelaskan dalam laporan penelitian.
