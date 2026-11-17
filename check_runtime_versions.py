import sys
import joblib
import sklearn
import streamlit

print("Python:", sys.version.split()[0])
print("scikit-learn:", sklearn.__version__)
print("joblib:", joblib.__version__)
print("Streamlit:", streamlit.__version__)
print("Catatan: pin versi scikit-learn dan joblib sesuai environment saat model .joblib dibuat.")
