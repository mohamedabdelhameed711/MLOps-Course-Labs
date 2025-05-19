# 🚀 Model Serving – Bank Customer Churn Prediction

This directory contains the FastAPI application used to serve the best trained model (`GradientBoostingClassifier`) for **Bank Customer Churn Prediction**.

---

## 🛠️ Setup

### Create & activate environment (if not already):

```bash
conda create -n churn_serving python=3.12 -y
conda activate churn_serving
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt` should include:**

```txt
fastapi
uvicorn
scikit-learn
pydantic
pandas
joblib
```

### Ensure model is saved in `models/gb_model.pkl`

You can train & save it using the following Python code:

```python
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# ... train your model as gb_model ...
joblib.dump(gb_model, "models/gb_model.pkl")
```

---

## 🚦 Run the API

```bash
uvicorn api.app:app --reload
```

API will be live at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔗 Endpoints

| Endpoint   | Method | Description                   |
|------------|--------|-------------------------------|
| `/`        | GET    | Welcome message               |
| `/health`  | GET    | Check API health              |
| `/predict` | POST   | Predict churn for customer(s) |

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

## 📄 Swagger UI

After running the API, go to:

📍 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

It provides an interactive interface to test the API.

---

## ✅ Tests

Basic test file is included (`test_api.py`).

```
