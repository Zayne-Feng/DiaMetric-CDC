
"""
Diabetes Risk Prediction Inference Pipeline
Generated: 2026-02-07 18:29:14
Model: XGBClassifier
"""
import json
import numpy as np
import pandas as pd
import joblib  # Fixed: Using joblib for efficient array handling
from pathlib import Path

class DiabetesRiskPredictor:
    """Production inference pipeline for diabetes risk prediction"""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

        # Load artifacts using joblib for optimized performance
        self.model = joblib.load(self.model_dir / "champion_model_calibrated.pkl")
        self.encoder = joblib.load(self.model_dir / "cluster_encoder.pkl")

        with open(self.model_dir / "champion_model_metadata.json", 'r') as f:
            self.metadata = json.load(f)

        with open(self.model_dir / "feature_configuration.json", 'r') as f:
            self.feature_config = json.load(f)

        # Use optimal clinical threshold (e.g., 0.147) from Phase 7.2
        self.optimal_threshold = self.metadata['optimal_thresholds']['recommended']

    def predict_single(self, patient_data):
        """Predict risk for a single patient input dictionary"""
        df = pd.DataFrame([patient_data])

        # 1. Encode Cluster_ID
        cluster_encoded = self.encoder.transform(df[['Cluster_ID']])
        cluster_df = pd.DataFrame(
            cluster_encoded,
            columns=self.feature_config['cluster_encoded_features']
        )

        # 2. Build feature matrix and align column order 
        X = pd.concat([
            df[self.feature_config['baseline_features']],
            cluster_df,
            df[['Risk_Index']]
        ], axis=1)

        X = X[self.feature_config['enhanced_features_ordered']]

        # 3. Probabilistic Inference
        probability = self.model.predict_proba(X)[0, 1]
        prediction = int(probability >= self.optimal_threshold)

        # Risk Stratification based on clinical thresholds
        if probability < 0.2:
            risk_level = 'Low'
        elif probability < 0.5:
            risk_level = 'Moderate'
        else:
            risk_level = 'High Risk'

        return {
            'probability': round(float(probability), 4),
            'prediction': prediction,
            'risk_level': risk_level,
            'threshold_used': self.optimal_threshold
        }

    def predict_batch(self, df):
        """Bulk prediction for multiple survey responses"""
        cluster_encoded = self.encoder.transform(df[['Cluster_ID']])
        cluster_df = pd.DataFrame(cluster_encoded, 
                                columns=self.feature_config['cluster_encoded_features'],
                                index=df.index)

        X = pd.concat([df[self.feature_config['baseline_features']], 
                      cluster_df, df[['Risk_Index']]], axis=1)
        X = X[self.feature_config['enhanced_features_ordered']]

        probs = self.model.predict_proba(X)[:, 1]
        results = df.copy()
        results['Risk_Probability'] = probs
        results['Risk_Level'] = pd.cut(probs, bins=[0, 0.2, 0.5, 1.0], 
                                      labels=['Low', 'Moderate', 'High Risk'])
        return results

if __name__ == "__main__":
    print("DiabetesRiskPredictor class ready for deployment.")
