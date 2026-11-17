# Dashboard Analisis Sentimen Shopee & Tokopedia

Aplikasi Streamlit untuk menyajikan hasil analisis sentimen review Google Play Store Shopee dan Tokopedia. Dashboard membaca Supabase PostgreSQL Data Warehouse menggunakan role read-only `dashboard_reader`.

## Fitur publik
- Ringkasan total review, rating, dan sentimen.
- Analisis distribusi serta tren sentimen.
- Perbandingan Shopee dan Tokopedia.
- Analisis tujuh kelompok topik keluhan K-Means.
- Pencarian dan penelusuran review.
- Evaluasi LinearSVC: accuracy, macro precision, macro recall, macro-F1, metrik per kelas, F1 per kelas, dan confusion matrix validation/test.

## Fitur khusus administrator
- Login administrator berbasis Streamlit Secrets.
- Upload CSV mentah, validasi, preview, prediksi LinearSVC, penetapan topik K-Means untuk prediksi negatif, dan penyimpanan append-only ke Supabase.
- Gunakan role DB `dashboard_ingest` terpisah yang hanya memiliki izin INSERT terbatas.

## File evaluasi
Empat laporan hasil Notebook 05 harus berada di folder `evaluation/`. File tersebut merupakan hasil aktual yang diberikan peneliti dan ditampilkan apa adanya pada halaman Evaluasi Model. Jangan mengedit angka metrik secara manual.

## Menyiapkan model untuk upload CSV
Salin artefak model dari proyek skripsi ke `artifacts/` sesuai tabel di `docs/README_UPLOAD_ADMIN.md`. Jangan melakukan training ulang saat setiap upload; aplikasi menggunakan model dan vectorizer yang sama dengan Notebook 05–07.

## Menjalankan lokal
```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Buat `.streamlit/secrets.toml` dari `.streamlit/secrets.toml.example`, isi koneksi database dan kredensial admin. Jangan commit `secrets.toml`.

## Keamanan
- Dashboard umum menggunakan `dashboard_reader`.
- Upload hanya tersedia setelah login administrator dan menggunakan `dashboard_ingest`.
- Jangan gunakan password postgres atau service-role key pada aplikasi.
- Jangan commit `.streamlit/secrets.toml`, `.env`, atau kredensial lain.
- Batasi akses admin pada orang yang dipercaya. Uji upload terlebih dahulu dengan CSV kecil.
