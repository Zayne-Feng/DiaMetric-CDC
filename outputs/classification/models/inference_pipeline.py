import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

print("=" * 80)
print("Deployment Verification")
print("=" * 80)


class DiabetesRiskPredictorAtomic:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.wrapper = joblib.load(self.model_dir / "champion_model_calibrated.pkl")

        with open(self.model_dir / "champion_model_metadata.json", "r") as f:
            self.metadata = json.load(f)
        with open(self.model_dir / "feature_configuration.json", "r") as f:
            self.feature_config = json.load(f)

        self.optimal_threshold = self.metadata["optimal_thresholds"]["recommended"]
        self.target_features = self.feature_config["enhanced_features_ordered"]
        self.n_clusters = self.metadata["data_specs"]["n_clusters"]

    def _prepare_data(self, data):
        X = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
        X = X.reindex(columns=self.target_features)

        if "Cluster_ID" in X.columns:
            X["Cluster_ID"] = X["Cluster_ID"].astype(
                pd.CategoricalDtype(categories=list(range(self.n_clusters)))
            )

        num_cols = X.select_dtypes(exclude=["category"]).columns
        X[num_cols] = X[num_cols].astype("float32")
        return X

    def predict_risk(self, data):
        X_final = self._prepare_data(data)

        try:
            probs = self.wrapper.predict_proba(X_final)
            prob_pos = float(probs[0, 1])

            if np.isnan(prob_pos):
                print(
                    "⚠ Detection: Calibration layer failure. Falling back to Raw Booster..."
                )
                base_model = getattr(
                    self.wrapper,
                    "base_estimator",
                    getattr(self.wrapper, "estimator", self.wrapper),
                )
                raw_probs = base_model.predict_proba(X_final)
                prob_pos = float(raw_probs[0, 1])

            return {
                "probability": prob_pos,
                "prediction": prob_pos >= self.optimal_threshold,
                "threshold_used": self.optimal_threshold,
                "diabetes_risk": f"{prob_pos:.2%}",
                "status": "POSITIVE"
                if prob_pos >= self.optimal_threshold
                else "NEGATIVE",
                "risk_tier": "High"
                if prob_pos > 0.5
                else ("Moderate" if prob_pos > 0.2 else "Low"),
                "is_reliable": not np.isnan(prob_pos),
            }
        except Exception as e:
            return {"Error": str(e), "is_reliable": False}
