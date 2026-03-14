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

Config’teki eşikler (`config/training_config.yaml`):

- `production_min_roc_auc`, `production_min_pr_auc`, `production_min_recall`
- Registry: `model_name: telco_churn_classifier` (API’de `MLFLOW_MODEL_NAME` ile override edilebilir)

---

## Model Promotion (Manuel)

Staging’deki modeli Production’a taşımak için:

```bash
# En son Staging modelini Production’a al
python scripts/promote_model.py --auto

# Belirli versiyonu promote et
python scripts/promote_model.py --version 3

# Model adı (API cleaned model kullanıyorsa)
python scripts/promote_model.py --auto --model-name telco_churn_classifier_cleaned
```

`MLFLOW_TRACKING_URI` ayarlı olmalıdır.

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

### Eğitim container (root Dockerfile)

```bash
docker build -t churn-train:latest .
docker run --rm -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  --add-host=host.docker.internal:host-gateway \
  churn-train:latest
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
### Not: Modelin mlflow/experiments sonuçları plots klasöründe yer almaktadır.
