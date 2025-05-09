# 📘 Panduan Lengkap Submission MSML Dicoding

Panduan ini disusun agar setiap kriteria (1–4) di kelas MSML Dicoding terpenuhi dengan struktur folder yang jelas, langkah demi langkah yang runtut, dan tanpa referensi ke DagsHub.

---

## 📂 Struktur Direktori Akhir
Berikut struktur repository yang direkomendasikan:

```
Membuat-Model-Machine-Learning/
├── EksperimenSML_Julio/
│   ├── Preprocessing/
│   │   └── automate_preprocessing.py      # (opsional: skrip otomatisasi)
│   └── Automobile.csv                     # dataset mentah
│
├── Membangun_model/
│   ├── mlruns/                            # direktori MLflow
│   ├── namadataset_preprocessing/
│   │   └── Automobile_clean.csv           # hasil preprocessing
│   ├── screenshoot_dashboard_artifact/    # screenshot dashboard MLflow
│   ├── modelling.py                       # training dasar & logging
│   ├── modelling_tuning.py                # hyperparameter tuning
│   └── requirements.txt                   # `pip install -r requirements.txt`
│
├── Monitoring_dan_Logging/
│   ├── bukti_alerting_Grafana/            # screenshot notifikasi alert
│   ├── bukti_monitoring_Grafana/          # screenshot dashboard Grafana
│   ├── bukti_monitoring_Prometheus/       # screenshot target Prometheus
│   ├── bukti_servik.png                   # screenshot service exporter
│   ├── inference.py                       # API & metrics exporter
│   ├── prometheus_exporter.py             # setup Prometheus client
│   └── prometheus.yml                     # konfigurasi Prometheus
│
├── Workflow-CI/Workflow-CI.txt
│   ├── .github/workflows/ci-mlflow-project.yml  # GitHub Actions CI
│   └── MLProject/
│       ├── MLproject                      # spec MLflow Projects
│       ├── conda.yaml                     # environment conda
│       ├── namadataset_preprocessing/
│       │   └── Automobile_clean.csv       # input preprocessing
│       └── modelling.py                   # entry-point script
│
└── README.md                              # file panduan ini
```

---

## 🔢 Daftar Isi
1. [Kriteria 1: Eksperimen Data & Preprocessing](#kriteria-1-eksperimen-data--preprocessing)  
2. [Kriteria 2: Membangun Model](#kriteria-2-membangun-model)  
3. [Kriteria 3: CI Workflow dengan MLflow Projects](#kriteria-3-ci-workflow-dengan-mlflow-projects)  
4. [Kriteria 4: Monitoring & Logging](#kriteria-4-monitoring--logging)  
5. [Tips & Checklist Akhir](#tips--checklist-akhir)  

---

## Kriteria 1: Eksperimen Data & Preprocessing

### 1.1 Setup & Struktur
1. Buat folder `EksperimenSML_Julio/` di root repo.  
2. Taruh data mentah `Automobile.csv` di dalamnya.  
3. (Opsional) Buat folder `Preprocessing/` dengan skrip `automate_preprocessing.py`.

### 1.2 Eksplorasi & Manual Preprocessing (Google Colab)
1. Buka `EksperimenSML_Julio/Automobile.csv` di Colab.  
2. Lakukan EDA: distribusi, korelasi, identifikasi missing.  
3. Terapkan cleaning (hapus duplikat), imputasi nilai hilang, encoding kategori.  
4. Simpan hasil bersih:
   ```python
   df.to_csv('namadataset_preprocessing/Automobile_clean.csv', index=False)
   ```

### 1.3 Automatisasi Preprocessing
1. Pindahkan `Automobile_clean.csv` ke `Membangun_model/namadataset_preprocessing/`.  
2. (Jika belum) buat `EksperimenSML_Julio/Preprocessing/automate_preprocessing.py`:
   ```python
   import pandas as pd

   def preprocess(input_path, output_path):
       df = pd.read_csv(input_path)
       # ... cleaning & encoding ...
       df.to_csv(output_path, index=False)

   if __name__ == '__main__':
       preprocess(
           '../EksperimenSML_Julio/Automobile.csv',
           '../Membangun_model/namadataset_preprocessing/Automobile_clean.csv'
       )
   ```
3. Jalankan untuk verifikasi:
   ```bash
   python EksperimenSML_Julio/Preprocessing/automate_preprocessing.py
   ```

---

## Kriteria 2: Membangun Model

### 2.1 Persiapan
1. Masuk ke folder `Membangun_model/`.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2.2 Training Dasar & MLflow Autolog (`modelling.py`)
1. Jalankan:
   ```bash
   python modelling.py \
     --input namadataset_preprocessing/Automobile_clean.csv \
     --output output_model
   ```
2. Hasil:
   - Model tersimpan di `output_model/random_forest_model.pkl`.  
   - Metrics dan artifact tercatat di `mlruns/` (MLflow UI).

### 2.3 Hyperparameter Tuning (`modelling_tuning.py`)
1. Jalankan:
   ```bash
   python modelling_tuning.py
   ```
2. MLflow mencatat param terbaik, CV score, dan metrik test.  
3. Cek hasil di MLflow UI:
   ```bash
   mlflow ui
   ```

---

## Kriteria 3: CI Workflow dengan MLflow Projects

### 3.1 Struktur MLProject
```
Workflow-CI/
└── MLProject/
    ├── MLproject      # definisi entry-points & params
    ├── conda.yaml     # environment (scikit-learn, mlflow, pandas)
    ├── namadataset_preprocessing/
    │   └── Automobile_clean.csv
    └── modelling.py   # skrip dengan `mlflow.set_experiment` + autolog
```

### 3.2 GitHub Actions
1. File: `.github/workflows/ci-mlflow-project.yml`.  
2. Setiap push ke `main`/`master` akan:
   - Checkout code  
   - Setup conda env  
   - `mlflow run MLProject -P input_data=... -P output_model_path=artifacts/model`  
   - Upload artifact model  
3. Pastikan status Action **passed** sebelum submission.

---

## Kriteria 4: Monitoring & Logging

### 4.1 Model Serving & Metrics Exporter (`inference.py`)
1. Load model & expose endpoint Flask (`/predict`).  
2. Jalankan Prometheus client di port `8000`.

```bash
python inference.py
# → API: http://localhost:5000/predict
# → Metrics: http://localhost:8000/metrics
```

### 4.2 Konfigurasi Prometheus
1. File: `Monitoring_dan_Logging/prometheus.yml`:
   ```yaml
   global:
     scrape_interval: 5s
   scrape_configs:
     - job_name: ml_model
       static_configs:
         - targets: ['localhost:8000']

     - job_name: "system_exporter"
       scrape_interval: 5s
       static_configs:
         - targets: ["localhost:8001"]
   ```
2. Jalankan:
   
**Untuk Dashboard Monitoring**
   ```
   - cd /d "D:\File Ku\Laskar AI\Membuat Model Machine Learning\Monitoring dan Logging"
   - python inference.py
   ```

**Untuk Dashboard Monitoring**
   ```
   - cd /d "D:\File Ku\Laskar AI\Membuat Model Machine Learning\Monitoring dan Logging"
   - prometheus_exporter.py
   ```

   **Nb: Jalankan di 2 terminal yang berbeda**
   ```bash
   cd prometheus-3.4.0-rc.0.windows-amd64
   prometheus.exe --config.file="D:/File Ku/Laskar AI/Membuat Model Machine Learning/Monitoring dan Logging/prometheus.yml"
   ```
3. - Buka `http://localhost:9090` → **Status → Targets** → pastikan job "ml_model_monitoring" dan "system_exporter" UP.  
   - Buka tab **Alerts** di Prometheus UI untuk melihat rule files.


### 4.3 Dashboard Grafana
1. Buka http://localhost:3000 (admin/admin).  
2. Tambah Data Source → Prometheus (http://localhost:9090).  
3. Buat dashboard dengan panel:
   - `api_requests_total` (Time Series)  
   - `inference_latency_seconds` (Histogram/Summary)  
   - `system_cpu_usage_percent` (Gauge)  
   - `system_memory_usage_bytes` (Gauge)  
4. Simpan dashboard → nama akun Dicoding.  
5. Screenshot → simpan ke `bukti_monitoring_Grafana/`.

### 4.4 Alerting
1. Atur Contact Point (Email/Webhook).  
2. Buat Rule:
   - High inference latency  
   - Error rate HTTP 5xx  
3. Definisikan folder & labels → simpan di `bukti_alerting_Grafana/`.  
4. Test → capture notifikasi.

---

## Tips & Checklist Akhir
- [✅] Struktur folder sesuai rekomendasi.  
- [✅] Semua script (`.py`, `.csv`, `.yml`) berada di tempatnya.  
- [✅] MLflow UI menampilkan run & artifacts.  
- [✅] CI pipeline passing di GitHub Actions.  
- [✅] Prometheus targets UP, Grafana dashboard lengkap, alert tested.  
- [✅] Penamaan file konsisten: `1.nama.png/jpg`, dst.  
- [✅] Repository di-zip dan push ke GitHub sebelum tenggat.

