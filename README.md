# 🚀 Bank Customer Churn Prediction 

This project provides a FastAPI application serving a `GradientBoostingClassifier` model for **Bank Customer Churn Prediction**. It features **Docker**, **Prometheus**, and **Grafana** integration for robust monitoring.

---

## 🛠️ Local Setup

### 1. Create & Activate Environment

```bash
conda create -n churn_serving python=3.12 -y
conda activate churn_serving
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
```txt
fastapi
uvicorn
scikit-learn
pydantic
pandas
joblib
prometheus-fastapi-instrumentator
```

### 3. Train and Save the Model

Example:
```python
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# ... your data loading and training code ...
joblib.dump(gb_model, "models/gb_model.pkl")
```

---

## 🚦 Run the API Locally

```bash
uvicorn api.app:app --reload
```
- API root: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔗 API Endpoints

| Endpoint   | Method | Description            |
|------------|--------|------------------------|
| `/`        | GET    | Welcome message        |
| `/health`  | GET    | API health check       |
| `/predict` | POST   | Churn prediction       |
| `/metrics` | GET    | Prometheus metrics     |

---

## 🧪 Sample JSON for `/predict`

```json
{
  "data": [
    {
      "CreditScore": 720,
      "Geography": "Germany",
      "Gender": "Male",
      "Age": 45,
      "Tenure": 7,
      "Balance": 150000.0,
      "NumOfProducts": 1,
      "HasCrCard": 0,
      "IsActiveMember": 0,
      "EstimatedSalary": 120000.0
    }
  ]
}
```

---

## 📬 Using Postman

1. POST to `http://127.0.0.1:8000/predict`
2. In **Body**, select `raw` and choose `JSON`.
3. Paste the sample input and click **Send**.

---

## 📄 Swagger UI

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ✅ Running Tests

```bash
pytest tests/test_api.py
```

---

## 🐳 Dockerized Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run with Docker Compose

```bash
docker-compose up --build
```

### Docker Compose

```yaml
version: "3.8"

services:
  churn-api:
    image: <your-ecr-url>/churn-prediction:your-tag
    ports:
      - "443:443"
    restart: always
```

---

## 🧪 GitHub Actions Workflow

GitHub Actions will:

1. Run unit tests.
2. Build Docker image.
3. Push it to ECR with tag `your-tag`.
4. SSH into EC2 instance using a base64-encoded SSH key.
5. Clone repo & run docker-compose.

Make sure these secrets are added:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `EC2_SSH_KEY` (base64 of PEM)
- `EC2_PUBLIC_IP`
- `ECR_REPOSITORY`
- `ECR_REGION`

---

## 📊 Monitoring with Prometheus & Grafana

### docker-compose.yml Example

```yaml
version: '3.8'

services:
  fastapi:
    build: .
    container_name: fastapi_app
    ports:
      - "8000:8000"
    depends_on:
      - prometheus

  prometheus:
    image: prom/prometheus
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### FastAPI Instrumentation

Add to `api/app.py`:
```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

---

## 📈 Grafana Dashboard Setup

1. Go to [http://localhost:3000](http://localhost:3000)
2. Login: `admin` / `admin`
3. Add Prometheus datasource (URL: `http://prometheus:9090`)
4. Create or import a dashboard (JSON).
5. Useful metrics:
   - Request duration: `http_request_duration_seconds_sum`
   - Total requests: `http_requests_total`

---

## 📁 Structure

```bash
MLOps-Course-Labs/
├── api/
│   ├── app.py
│   └── model.py
├── models/
│   └── gb_model.pkl
├── tests/
│   └── test_api.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .github/
    └── workflows/
        └── deploy.yml
```

---

## 🧠 Why These Metrics?

- **Request Duration**: Measures API performance and latency.
- **Request Count**: Tracks API usage and helps detect overload.

---

**Tip:**  
- Ensure `models/gb_model.pkl` exists before starting the API.
