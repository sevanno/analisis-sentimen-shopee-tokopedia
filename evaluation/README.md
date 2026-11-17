# Hasil Evaluasi LinearSVC

Empat CSV pada folder ini berasal dari Notebook 05:
- `classification_report_validation.csv`
- `classification_report_test.csv`
- `confusion_matrix_validation.csv`
- `confusion_matrix_test.csv`

Aplikasi membaca data ini untuk menampilkan metrik agregat, metrik per kelas, dan confusion matrix. Nilai tidak boleh disunting secara manual. Jika model dilatih ulang, ekspor ulang keempat file dari Notebook 05 dan ganti file di folder ini.

Ringkasan angka sumber:
- Validation: accuracy 82.20%, macro precision 86.13%, macro recall 65.32%, macro-F1 61.16% (118 review).
- Test: accuracy 89.02%, macro precision 86.12%, macro recall 71.79%, macro-F1 74.85% (173 review).
