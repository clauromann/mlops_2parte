from pathlib import Path

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
    model_path = Path(__file__).resolve().parent / "model" / "model.joblib"
    return joblib.load(model_path)


MODEL = load_model()


def handler(event):
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
