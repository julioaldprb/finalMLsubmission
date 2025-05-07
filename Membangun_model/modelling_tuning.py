import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# ————————————————
# 1. Konfigurasi MLflow
# ————————————————
experiment_name = "mpg_classification_tuning"
mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog()

# ————————————————
# 2. Load Dataset
# ————————————————
dataset_path = "namadataset_preprocessing/Automobile_clean.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"File tidak ditemukan: {dataset_path}")

data = pd.read_csv(dataset_path)
labels = ["Low", "Medium", "High"]
data['mpg_category'] = pd.qcut(data['mpg'], q=3, labels=labels)

# Pastikan setidaknya dua kelas
if data['mpg_category'].nunique() < 2:
    raise ValueError("Target memiliki kurang dari dua kelas setelah binning.")

# ————————————————
# 3. Siapkan fitur & target
# ————————————————
target = 'mpg_category'
X = data.drop(columns=[target, 'mpg'])
y = data[target]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ————————————————
# 4. Hyperparameter Tuning
# ————————————————
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Jalankan tuning
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

# ————————————————
# 5. Logging Manual ke MLflow
# ————————————————
with mlflow.start_run():
    # Log params
    for k, v in grid.best_params_.items():
        mlflow.log_param(k, v)
    mlflow.log_metric('cv_mean_accuracy', grid.best_score_)

    # Retrain & test
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    mlflow.log_metric('test_accuracy', acc)
    mlflow.log_metric('precision', prec)
    mlflow.log_metric('recall', rec)
    mlflow.log_metric('f1_score', f1)

    # Log model
    mlflow.sklearn.log_model(best_model, 'rf_best_model')

print("Tuning dan logging selesai.")
