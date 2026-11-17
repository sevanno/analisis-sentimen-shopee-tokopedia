from pathlib import Path
import json
import joblib

root = Path(__file__).resolve().parent.parent
artifact_dir = root / "artifacts"
required = [
    "linear_svc_sentiment.joblib",
    "tfidf_vectorizer.joblib",
    "kmeans_topic_tfidf_vectorizer.joblib",
    "kmeans_negative_topics.joblib",
    "topic_analysis_metadata.json",
]
missing = [name for name in required if not (artifact_dir / name).is_file()]
if missing:
    print("Artefak belum lengkap:")
    for name in missing:
        print(" -", artifact_dir / name)
    raise SystemExit(1)

svm = joblib.load(artifact_dir / "linear_svc_sentiment.joblib")
svm_vectorizer = joblib.load(artifact_dir / "tfidf_vectorizer.joblib")
topic_vectorizer = joblib.load(artifact_dir / "kmeans_topic_tfidf_vectorizer.joblib")
topic_model = joblib.load(artifact_dir / "kmeans_negative_topics.joblib")
metadata = json.loads((artifact_dir / "topic_analysis_metadata.json").read_text(encoding="utf-8"))

classes = [str(x) for x in svm.classes_]
if set(classes) != {"negatif", "netral", "positif"}:
    raise SystemExit(f"Kelas model SVM tidak sesuai: {classes}")
if getattr(svm, "n_features_in_", len(svm_vectorizer.vocabulary_)) != len(svm_vectorizer.vocabulary_):
    raise SystemExit("Jumlah fitur SVM dan TF-IDF tidak cocok.")
if getattr(topic_model, "n_features_in_", len(topic_vectorizer.vocabulary_)) != len(topic_vectorizer.vocabulary_):
    raise SystemExit("Jumlah fitur K-Means dan topic vectorizer tidak cocok.")
if metadata.get("selected_k") is None or not metadata.get("created_at_utc"):
    raise SystemExit("Metadata topik tidak memiliki selected_k/created_at_utc.")

print("Semua artefak tersedia dan dapat dibaca.")
print("Kelas LinearSVC:", classes)
print("Jumlah fitur SVM:", len(svm_vectorizer.vocabulary_))
print("Jumlah fitur K-Means:", len(topic_vectorizer.vocabulary_))
print("K terpilih:", metadata["selected_k"])
print("Tanggal metadata:", metadata["created_at_utc"])
