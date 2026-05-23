import os

import requests


ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {RUNPOD_API_KEY}"}


def test_endpoint_health():
    assert ENDPOINT_URL, "ENDPOINT_URL environment variable is not set"
    response = requests.get(
        f"{ENDPOINT_URL}/health",
        headers=auth_headers(),
        timeout=30,
    )
    assert response.status_code == 200


def test_endpoint_prediction():
    assert ENDPOINT_URL, "ENDPOINT_URL environment variable is not set"
    payload = {
        "input": {
            "samples": [
                {
                    "Age": 22,
                    "Fare": 7.25,
                    "Pclass": 3,
                    "Sex": "male",
                    "Embarked": "S",
                    "Title": "Mr",
                    "Family_size_cat": "Small",
                }
            ]
        }
    }

    response = requests.post(
        f"{ENDPOINT_URL}/runsync",
        json=payload,
        headers=auth_headers(),
        timeout=90,
    )

    assert response.status_code == 200
    data = response.json()
    assert "output" in data
    assert "prediction" in data["output"]
    assert data["output"]["prediction"][0] in [0, 1]
