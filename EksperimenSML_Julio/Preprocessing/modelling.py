import os
import argparse
import pandas as pd
import numpy as np
import joblib  # untuk menyimpan model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# —————————————————————————————
# Argumen input/output (untuk mlflow run -P)
# —————————————————————————————
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path dataset input")
parser.add_argument("--output", type=str, required=True, help="Path folder output model")
args = parser.parse_args()

dataset_path = args.input
output_model_path = args.output
os.makedirs(output_model_path, exist_ok=True)  # pastikan folder ada

# —————————————————————————————
# 1. Konfigurasi MLflow
# —————————————————————————————
experiment_name = "mpg_classification"
mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog()

# —————————————————————————————
# 2. Load dan Cek Dataset
# —————————————————————————————
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"File dataset tidak ditemukan di: {dataset_path}")

data = pd.read_csv(dataset_path)

# —————————————————————————————
# 3. Buat Target Multikelas dengan Quantile Binning
# —————————————————————————————
labels = ["Low", "Medium", "High"]
data['mpg_category'] = pd.qcut(data['mpg'], q=3, labels=labels)

print("Distribusi label mpg_category:")
print(data['mpg_category'].value_counts())

if data['mpg_category'].nunique() < 2:
    raise ValueError("Hanya ditemukan satu kelas di target 'mpg_category'. Cek kembali binning atau dataset.")

# —————————————————————————————
# 4. Siapkan Fitur dan Target
# —————————————————————————————
target_column = 'mpg_category'
X = data.drop(columns=[target_column, 'mpg'])
y = data[target_column]
X = pd.get_dummies(X, drop_first=True)

# —————————————————————————————
# 5. Split Data
# —————————————————————————————
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# —————————————————————————————
# 6. Training dan Logging ke MLflow
# —————————————————————————————
print("Mulai training dan logging ke MLflow...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {acc}")
print("\nLaporan klasifikasi:")
print(classification_report(y_test, y_pred))

mlflow.log_metric("accuracy_test", acc)

# —————————————————————————————
# 7. Simpan model sebagai artefak
# —————————————————————————————
model_path = os.path.join(output_model_path, "random_forest_model.pkl")
joblib.dump(model, model_path)
mlflow.log_artifact(model_path)

print(f"Model disimpan di: {model_path}")
print("Training dan logging selesai.")
