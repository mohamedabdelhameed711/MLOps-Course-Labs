from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def sample_body():
    return {
        "data": [
            {
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Female",
                "Age": 45,
                "Tenure": 3,
                "Balance": 60000,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 50000,
            }
        ]
    }


# health endpoint 
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# schema validation
def test_predict_schema_error():
    bad = sample_body()
    bad["data"][0].pop("CreditScore")      # remove required field
    r = client.post("/predict", json=bad)
    assert r.status_code == 422            # Unprocessable Entity


# successful prediction 
def test_predict_success():
    r = client.post("/predict", json=sample_body())
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body and "probabilities" in body
    assert len(body["predictions"]) == 1
