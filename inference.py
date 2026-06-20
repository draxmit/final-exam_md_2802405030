from pathlib import Path

import joblib
import numpy as np
import pandas as pd

class CreditScoreInference:
    def __init__(self, model_file=None):
        if model_file is None:
            model_file = Path(__file__).parent / "artifacts" / "credit_score_pipeline.pkl"
        self.bundle = joblib.load(model_file)
        self.classes = self.bundle["classes"]

    def predict(self, records):
        df = pd.DataFrame(records)
        proba = self.bundle["pipeline"].predict_proba(df)
        class_ids = np.argmax(proba, axis=1)
        return {
            "probabilities": proba.tolist(),
            "predictions": class_ids.tolist(),
            "labels": [self.classes[i] for i in class_ids],
        }
