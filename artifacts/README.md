# Model artifacts untuk fitur Upload CSV

Folder ini harus diisi dari proyek skripsi sebelum menu upload admin dapat dipakai.

Salin file berikut dengan nama persis:
1. `models/svm/linear_svc_sentiment.joblib` -> `artifacts/linear_svc_sentiment.joblib`
2. `models/svm/tfidf_vectorizer.joblib` -> `artifacts/tfidf_vectorizer.joblib`
3. `models/topic_model/kmeans_topic_tfidf_vectorizer.joblib` -> `artifacts/kmeans_topic_tfidf_vectorizer.joblib`
4. `models/topic_model/kmeans_negative_topics.joblib` -> `artifacts/kmeans_negative_topics.joblib`
5. `data/processed/kmeans_topic_analysis/topic_analysis_metadata.json` -> `artifacts/topic_analysis_metadata.json`

Jangan membuat ulang model hanya untuk upload. Gunakan artefak yang sesuai dengan versi model yang tercatat pada Data Warehouse.

Catatan: file `.joblib` dibutuhkan saat deployment agar fitur upload CSV bisa melakukan prediksi. Periksa ukuran file dan batas GitHub/Streamlit sebelum push.
