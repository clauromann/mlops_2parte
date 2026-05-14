import os

import joblib
import pandas as pd
import runpod


FEATURES = [
    "Age",
    "Fare",
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "Family_size_cat",
]


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.joblib")
    with open(model_path, "rb") as file:
        return joblib.load(file)


MODEL = load_model()


def handler(event):
    """RunPod serverless handler.

    Expected input:
    {
      "input": {
        "samples": [
          {
            "Age": 22,
            "Fare": 7.25,
            "Pclass": 3,
            "Sex": "male",
            "Embarked": "S",
            "Title": "Mr",
            "Family_size_cat": "Small"
          }
        ]
      }
    }
    """
    input_data = event.get("input", {})
    samples = input_data.get("samples", [])

    if not samples:
        return {"error": "No samples provided"}

    for index, sample in enumerate(samples):
        missing = [feature for feature in FEATURES if feature not in sample]
        if missing:
            return {"error": f"Sample {index} missing features: {missing}"}

    X = pd.DataFrame(samples)[FEATURES]
    predictions = MODEL.predict(X).tolist()
    return {"prediction": predictions}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
