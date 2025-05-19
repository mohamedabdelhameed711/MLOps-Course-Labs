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

## 📬 Using Postman to Test `/predict`

1. Open **Postman** and click **New > HTTP Request**.
2. Set the request type to `POST`.
3. Enter the URL:

```
http://127.0.0.1:8000/predict
```

4. Click on the **Body** tab, choose **raw**, and set format to **JSON**.
5. Paste the sample JSON above.
6. Click **Send**. You will get a response like:

```json
{
  "predictions": [0]
}
```

> `0` = Not likely to churn, `1` = Likely to churn

---

## 📄 Swagger UI

After running the API, go to:

📍 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

It provides an interactive interface to test the API.

---

## ✅ Tests

Basic test file is included (`test_api.py`).

---

## 🐳 Docker Support

This project can be containerized using Docker to ensure consistent environments across different machines.

### 📄 Dockerfile

Make sure you have a `Dockerfile` in the root of your project like this:

```Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### ⚙️ Build the Docker Image
```bash
docker build -t myapi .
```

### ▶️ Run the Docker Container
```bash
docker run --name docker-myapi -p 8000:8000 myapi
```

### 🌍 Access the API

Open [http://localhost:8000/docs](http://localhost:8000/docs) or use Postman as described above.

### 🧾 View Logs from Running Container (Bonus)
```bash
docker logs docker-myapi
```

---

## 📁 Directory Structure

```bash
    MLOps-Course-Labs/
    ├── api/
    │   ├── model.py          
    │   └── app.py            
    ├── tests/
    │   └── test_api.py
    ├── models/
    │   ├── gb_best_model.py
    │   └── gb_model.pkl        
    ├── requirements.txt
    ├── dockerfile  
```
