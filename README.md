# Telco Churn MLOps

## TR

Telekomünikasyon müşteri kaybı (churn) tahmini için uçtan uca MLOps projesi. Model eğitimi, MLflow Model Registry, Prefect pipeline, FastAPI servisi ve CI/CD ile production-ready bir yapı sunar.

---

## Özellikler

- **Prefect pipeline**: Veri hazırlama → Eğitim → Değerlendirme → Model promotion (Staging → Production)
- **MLflow**: Deney takibi, model versiyonlama, artifact ve metrik loglama
- **FastAPI servisi**: `/health`, `/model-info`, `/predict`, `/reload-model` endpoint’leri
- **Docker**: Eğitim ve API için ayrı image’lar
- **CI/CD**: Lint → Unit test → Build → MLflow + train + smoke test → Deploy
- **Monitoring**: PSI tabanlı basit drift detection modülü
- **Yapılandırma**: YAML ile eşikler, model parametreleri ve registry ayarları

---

## Proje Yapısı

```
mlops-telco-churn/
├── api/
│   ├── predict_service.py   # FastAPI churn prediction servisi
│   ├── test_api_real.py     # API test scripti
│   └── Dockerfile           # API container
├── config/
│   ├── training_config.yaml # Pipeline eğitim config (sentetik data)
│   └── cleaned_data_config.yaml
├── data/
│   └── telco_cleaned.csv    # Temizlenmiş veri (opsiyonel)
├── monitoring/
│   └── drift_detector.py    # PSI tabanlı drift tespiti
├── pipelines/
│   └── prefect_pipeline.py  # Train → Evaluate → Promote flow
├── scripts/
│   ├── train_improved.py    # ChurnTrainer (MLflow + staging)
│   ├── train_cleaned_data.py
│   └── promote_model.py    # Staging → Production promotion
├── tests/
│   ├── smoke_test.py        # API deployment smoke test
│   └── test_feature_engineering.py
├── .github/workflows/
│   └── ci-cd-pipeline.yml   # GitHub Actions CI/CD
├── requirements.txt
├── Dockerfile               # Eğitim container (train_cleaned_data)
└── README.md
```

---

## Kurulum

### Gereksinimler

- Python 3.10+
- (Opsiyonel) Docker, MLflow tracking server

### Sanal ortam ve bağımlılıklar

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

### MLflow (yerel eğitim ve API için)

API’nin production modeli yükleyebilmesi için bir MLflow tracking server gereklidir.

```bash
# Terminal 1: MLflow server (artifact'ları da sunar)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow_artifacts \
  --serve-artifacts \
  --host 0.0.0.0 \
  -p 5000
```

Ortam değişkeni:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

---

## Pipeline (Prefect)

Tek komutla eğitim → değerlendirme → (şartlı) promotion.

```bash
# Varsayılan config ile, threshold’lar geçerse Staging’e alır
python -m pipelines.prefect_pipeline

# Belirli config
python -m pipelines.prefect_pipeline --config config/training_config.yaml

# Threshold’ları bypass edip doğrudan Production’a promote
python -m pipelines.prefect_pipeline --auto-promote

# Haftalık cron için (Prefect Cloud/Server ile)
python -m pipelines.prefect_pipeline --schedule
```

Akış:

1. **Data Preparation** – Veri hazırlığı (şu an dummy/sentetik)
2. **Model Training** – `ChurnTrainer` ile eğitim, MLflow’a log, Staging’e kayıt
3. **Model Evaluation** – ROC-AUC, PR-AUC, Recall ve production eşikleri kontrolü
4. **Model Promotion** – `--auto-promote` veya eşikler sağlanıyorsa Staging → Production


## Architecture

┌─────────────────────────────────────────────────────┐
│  DATA: telco_cleaned.csv (7043 customers)           │
│  - Contract type, payment method, tenure, charges   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING                                │
│  - Hash crosses: contract×payment, service×contract │
│  - Numerical features: tenure, charges              │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  MODEL TRAINING (XGBoost)                           │
│  - MLflow experiment tracking                       │
│  - Hyperparameter logging                           │
│  - Model registration                               │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  MODEL REGISTRY (MLflow)                            │
│  - Version control                                  │
│  - Stage management (None/Staging/Production)       │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  API DEPLOYMENT (FastAPI + Docker)                  │
│  - /predict: Real-time churn prediction             │
│  - /health: Service health check                    │
└─────────────────────────────────────────────────────┘


---

## Prediction API

### Yerel çalıştırma

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# Opsiyonel: farklı model adı (varsayılan: telco_churn_classifier_cleaned)
# export MLFLOW_MODEL_NAME=telco_churn_classifier

uvicorn api.predict_service:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI**: http://localhost:8000/docs  
- **ReDoc**: http://localhost:8000/redoc  

### Endpoint’ler

| Method | Endpoint        | Açıklama                          |
|--------|-----------------|-----------------------------------|
| GET    | `/`             | Servis bilgisi ve endpoint listesi |
| GET    | `/health`       | Sağlık kontrolü, `model_loaded`, `model_version` |
| GET    | `/model-info`   | Yüklü production model bilgisi  |
| POST   | `/predict`      | Churn tahmini (olasılık, risk seviyesi) |
| POST   | `/reload-model` | Modeli MLflow’dan yeniden yükle   |

### Örnek predict isteği

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "TEST-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "contract_type": "Month-to-month",
    "PaperlessBilling": "Yes",
    "payment_method": "Electronic check",
    "MonthlyCharges": 89.85,
    "TotalCharges": 1078.20,
    "service_combo_id": "Fiber optic_Yes_Yes_Yes",
    "geo_code": "G23"
  }'
```

---

## Docker

### API container

```bash
# Build
docker build -t churn-api:latest -f api/Dockerfile .

# Run (MLflow host makinede ise)
docker run -d -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_MODEL_NAME=telco_churn_classifier \
  --add-host=host.docker.internal:host-gateway \
  churn-api:latest
```

---

## CI/CD (GitHub Actions)

`.github/workflows/ci-cd-pipeline.yml` push/PR’da tetiklenir.

| Job         | Açıklama |
|------------|----------|
| **lint**   | Flake8 + Pylint (`scripts/`, `api/`) |
| **unit-test** | `pytest tests/test_feature_engineering.py` |
| **build**  | Training + API Docker image’larının build’i |
| **smoke-test** | MLflow server + pipeline (train + promote) + API container + smoke test |
| **deploy** | Tüm aşamalar geçerse “READY TO DEPLOY” çıktısı |

Smoke test: API’nin ayakta olması ve (CI’da MLflow ile eğitim yapıldığı için) `model_loaded: true` ile `/health` dönmesi.

---

##  CI/CD Pipeline

The project includes a 5-stage automated pipeline triggered on every push to \`main\` branch:

### Pipeline Stages

┌─────────────────────────────────────────┐
│  1.  LINT (Code Quality)                │
│     - Flake8 style checking             │
│     - Pylint quality score (≥6.0)       │
└──────────────┬──────────────────────────┘
               │ PASS 
               ▼
┌─────────────────────────────────────────┐
│  2.  UNIT TESTS                         │
│     - Feature engineering tests         │
│     - 4 tests, <1 second execution      │
└──────────────┬──────────────────────────┘
               │ PASS 
               ▼
┌─────────────────────────────────────────┐
│  3.  BUILD                              │
│     - Training Docker image             │
│     - API Docker image                  │
└──────────────┬──────────────────────────┘
               │ PASS 
               ▼
┌─────────────────────────────────────────┐
│  4.  SMOKE TEST                         │
│     - Start API container               │
│     - Health endpoint verification      │
│     - 200 OK response check             │
└──────────────┬──────────────────────────┘
               │ PASS 
               ▼
┌─────────────────────────────────────────┐
│  5.  DEPLOY                             │
│     - All stages passed                 │
│     - Ready for production              │
└─────────────────────────────────────────┘

**Toplam Süre:** ~6-9 minutes

---

## Yapılandırma

### `config/training_config.yaml`

- **experiment_name**: MLflow experiment adı  
- **data**: Örnek sayısı, test/val oranları, dengesizlik oranı  
- **model**: Algoritma (örn. LogisticRegression), `max_iter`, `class_weight`  
- **thresholds**: Production’a geçiş için min ROC-AUC, PR-AUC, Recall  
- **registry**: `model_name`, staging/production otomasyonu  

### API ortam değişkenleri

- **MLFLOW_TRACKING_URI**: MLflow server adresi (zorunlu, production model için)  
- **MLFLOW_MODEL_NAME**: Registry’deki model adı (varsayılan: `telco_churn_classifier_cleaned`)  

---

## Testler

```bash
# Unit testler
pytest tests/test_feature_engineering.py -v

# Smoke test (API’nin localhost:8000’de çalıştığı varsayılır)
python tests/smoke_test.py

# Farklı URL
SMOKE_TEST_API_URL=http://localhost:8000 python tests/smoke_test.py
```

---

## Monitoring – Drift Detection

`monitoring/drift_detector.py`: PSI (Population Stability Index) ile basit drift tespiti.

- PSI &lt; 0.1: Belirgin değişim yok  
- 0.1 ≤ PSI &lt; 0.25: Orta değişim (izle)  
- PSI ≥ 0.25: Güçlü değişim (alarm)  

Referans veri ile fit edilip, gelen veri ile karşılaştırma yapılabilir.

---

##  Model Performance

### Metrics (Test Set)

| Metric | Value |
|--------|-------|
| **PR-AUC** | 0.6344 |
| **ROC-AUC** | 0.8421 |
| **Accuracy** | 79.3% |
| **Precision** | 65.2% |
| **Recall** | 54.8% |
| **F1-Score** | 59.5% |

---

### Not: Modelin mlflow/experiments sonuçları plots klasöründe yer almaktadır.

---

## EN

An end-to-end MLOps project for predicting telecommunications customer churn. It provides a production-ready infrastructure featuring model training, the MLflow Model Registry, a Prefect pipeline, a FastAPI service, and CI/CD.

---

## Features

- **Prefect pipeline**: Data preparation → Training → Evaluation → Model promotion (Staging → Production)
- **MLflow**: Experiment tracking, model versioning, artifact and metric logging
- **FastAPI service**: `/health`, `/model-info`, `/predict`, `/reload-model` endpoints
- **Docker**: Separate images for training and the API
- **CI/CD**: Lint → Unit test → Build → MLflow + train + smoke test → Deploy
- **Monitoring**: Simple drift detection module based on PSI
- **Configuration**: Thresholds, model parameters, and registry settings via YAML

---

## Projet Structure

```
mlops-telco-churn/
├── api/
│   ├── predict_service.py   # FastAPI Churn Prediction Service
│   ├── test_api_real.py     # API test script
│   └── Dockerfile           # API container
├── config/
│   ├── training_config.yaml # Pipeline train config (synthetic data)
│   └── cleaned_data_config.yaml
├── data/
│   └── telco_cleaned.csv    # Cleaned data (optional)
├── monitoring/
│   └── drift_detector.py    # PSI based drift detection
├── pipelines/
│   └── prefect_pipeline.py  # Train → Evaluate → Promote flow
├── scripts/
│   ├── train_improved.py    # ChurnTrainer (MLflow + staging)
│   ├── train_cleaned_data.py
│   └── promote_model.py    # Staging → Production promotion
├── tests/
│   ├── smoke_test.py        # API deployment smoke test
│   └── test_feature_engineering.py
├── .github/workflows/
│   └── ci-cd-pipeline.yml   # GitHub Actions CI/CD
├── requirements.txt
├── Dockerfile               # Train container (train_cleaned_data)
└── README.md
```

---

## Installation

### Requirements
- Python 3.10+
- (Optional) Docker, MLflow tracking server

### Virtual environment and dependencies

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

### MLflow (for local training and API)

An MLflow tracking server is required for the API to load the production model.

```bash
# Terminal 1: MLflow server (serves artifacts)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow_artifacts \
  --serve-artifacts \
  --host 0.0.0.0 \
  -p 5000
```

Env variable:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

---

## Pipeline (Prefect)

One-touch training → evaluation → (conditional) promotion.

```bash
# With the default configuration, it enters Staging if the thresholds are exceeded.
python -m pipelines.prefect_pipeline

# Certain config
python -m pipelines.prefect_pipeline --config config/training_config.yaml

# Bypass thresholds and promote directly to Production.
python -m pipelines.prefect_pipeline --auto-promote

# For weekly cron (with Prefect Cloud/Server)
python -m pipelines.prefect_pipeline --schedule
```

Flow:

1. **Data Preparation** – (now dummy/synthetic)
2. **Model Training** – Training with `ChurnTrainer`, logging to MLflow, recording to Staging.
3. **Model Evaluation** – ROC-AUC, PR-AUC, Recall and production threshold control
4. **Model Promotion** – `--auto-promote` or if the thresholds are met Staging → Production


## Architecture

┌─────────────────────────────────────────────────────┐
│  DATA: telco_cleaned.csv (7043 customers)           │
│  - Contract type, payment method, tenure, charges   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING                                │
│  - Hash crosses: contract×payment, service×contract │
│  - Numerical features: tenure, charges              │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  MODEL TRAINING (XGBoost)                           │
│  - MLflow experiment tracking                       │
│  - Hyperparameter logging                           │
│  - Model registration                               │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  MODEL REGISTRY (MLflow)                            │
│  - Version control                                  │
│  - Stage management (None/Staging/Production)       │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  API DEPLOYMENT (FastAPI + Docker)                  │
│  - /predict: Real-time churn prediction             │
│  - /health: Service health check                    │
└─────────────────────────────────────────────────────┘

---

## Prediction API

### Local run

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# Optional: different model name (default: telco_churn_classifier_cleaned)
# export MLFLOW_MODEL_NAME=telco_churn_classifier

uvicorn api.predict_service:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI**: http://localhost:8000/docs  
- **ReDoc**: http://localhost:8000/redoc  

### Endpoints

| Method | Endpoint        | Description                          |
|--------|-----------------|-----------------------------------|
| GET    | `/`             | Service information and endpoint list |
| GET    | `/health`       | Health control, `model_loaded`, `model_version` |
| GET    | `/model-info`   | Loaded production model information  |
| POST   | `/predict`      | Churn prediction (probability, level of risk) |
| POST   | `/reload-model` | Reload the model from MLflow.   |

### Example predict request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "TEST-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "contract_type": "Month-to-month",
    "PaperlessBilling": "Yes",
    "payment_method": "Electronic check",
    "MonthlyCharges": 89.85,
    "TotalCharges": 1078.20,
    "service_combo_id": "Fiber optic_Yes_Yes_Yes",
    "geo_code": "G23"
  }'
```

---

## Docker

### API container

```bash
# Build
docker build -t churn-api:latest -f api/Dockerfile .

# Run (If MLflow is on the host machine)
docker run -d -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_MODEL_NAME=telco_churn_classifier \
  --add-host=host.docker.internal:host-gateway \
  churn-api:latest
```

---

## CI/CD (GitHub Actions)

`.github/workflows/ci-cd-pipeline.yml` is triggered in push/PR.

| Job         | Description |
|------------|----------|
| **lint**   | Flake8 + Pylint (`scripts/`, `api/`) |
| **unit-test** | `pytest tests/test_feature_engineering.py` |
| **build**  | Training + Build of API Docker image |
| **smoke-test** | MLflow server + pipeline (train + promote) + API container + smoke test |
| **deploy** | If all stages are completed, the output will be “READY TO DEPLOY”.|

Smoke test: The API is up and running and (since training is done with MLflow in CI) it returns `model_loaded: true` and `/health`.

---

## CI/CD Pipeline

The project includes a 5-stage automated pipeline triggered on every push to \`main\` branch:

### Pipeline Stages

┌─────────────────────────────────────────┐
│  1.  LINT (Code Quality)                │
│     - Flake8 style checking             │
│     - Pylint quality score (≥6.0)       │
└──────────────┬──────────────────────────┘
               │ PASS 
               ▼
┌─────────────────────────────────────────┐
│  2.  UNIT TESTS                         │
│     - Feature engineering tests         │
│     - 4 tests, <1 second execution      │
└──────────────┬──────────────────────────┘
               │ PASS 
               ▼
┌─────────────────────────────────────────┐
│  3.  BUILD                              │
│     - Training Docker image             │
│     - API Docker image                  │
└──────────────┬──────────────────────────┘
               │ PASS 
               ▼
┌─────────────────────────────────────────┐
│  4.  SMOKE TEST                         │
│     - Start API container               │
│     - Health endpoint verification      │
│     - 200 OK response check             │
└──────────────┬──────────────────────────┘
               │ PASS 
               ▼
┌─────────────────────────────────────────┐
│  5.  DEPLOY                             │
│     - All stages passed                 │
│     - Ready for production              │
└─────────────────────────────────────────┘

---

**Total Duration:** ~6-9 minutes

## Configuration

### `config/training_config.yaml`

- **experiment_name**: MLflow experiment name
- **data**: Number of samples, test/val ratios, imbalance ratio
- **model**: Algorithm (e.g., LogisticRegression), `max_iter`, `class_weight`
- **thresholds**: Minimum ROC-AUC, PR-AUC, Recall for production
- **registry**: `model_name`, staging/production automation

### API Environment Variables

- **MLFLOW_TRACKING_URI**: MLflow server address (required, for production models)
- **MLFLOW_MODEL_NAME**: Model name in the Registry (default: `telco_churn_classifier_cleaned`)

---

## Tests

```bash
# Unit tests
pytest tests/test_feature_engineering.py -v

# Smoke test (The API is assumed to be running on localhost:8000.)
python tests/smoke_test.py

# Different URL
SMOKE_TEST_API_URL=http://localhost:8000 python tests/smoke_test.py
```

---

## Monitoring – Drift Detection

`monitoring/drift_detector.py`: Simple drift detection using PSI (Population Stability Index).

- PSI &lt; 0.1: No certain change  
- 0.1 ≤ PSI &lt; 0.25: Medium change (watch)  
- PSI ≥ 0.25: Strong change (alarm)

The data can be fitted with a reference data and compared with the incoming data.

---

##  Model Performance

### Metrics (Test Set)

| Metric | Value |
|--------|-------|
| **PR-AUC** | 0.6344 |
| **ROC-AUC** | 0.8421 |
| **Accuracy** | 79.3% |
| **Precision** | 65.2% |
| **Recall** | 54.8% |
| **F1-Score** | 59.5% |

---

### Note: The model's mlflow/experiments results are located in the plots folder.

