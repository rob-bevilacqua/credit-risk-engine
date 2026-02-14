import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# they probably do it better than me
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

class RiskModel:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # standard scaler normalizes the range of each column
        # so large values aren't treated as more important
        self.scaler = StandardScaler()
        # Logistic regression is the standard model used for credit risk
        # Logically explainable compared to black box methods
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")

    def run_training_pipeline(self, data_path: Path):
        print("Loading processed data for training...")
        df = pd.read_parquet(data_path)

        # drop non numeric
        X = df.select_dtypes(include=['number']).drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
        y = df['TARGET']

        #  Train/Test Split (80/20)
        #  stratify ensures both sets have the same 8% Default ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # We fill NaNs with 0 for the baseline; the scaler then Z-scores everything.
        X_train_scaled = self.scaler.fit_transform(X_train.fillna(0))
        X_test_scaled = self.scaler.transform(X_test.fillna(0))

        # Fit Model (Finding the optimal beta coefficients)
        print(f"Training on {X_train.shape[1]} features...")
        self.clf.fit(X_train_scaled, y_train)

        # rank features
        self.save_feature_importance(X_train.columns.tolist())
        
        # Evaluation (ROC-AUC)
        # We want the probability (0 to 1), not just the class (0 or 1)
        probs = self.clf.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, probs)

        print("\n")
        print(f"BASELINE ROC-AUC: {auc_score:.4f}")
        print("\n")
        
        # Save Artifacts
        joblib.dump(self.clf, self.model_dir / "logistic_model.pkl")
        joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
        print(f"Model and Scaler saved to {self.model_dir}")

    def save_feature_importance(self, feature_names):
        """
        Extracts the weights (coefficients) from the model, maps them to feature 
        names, and saves a ranked CSV to the model directory.
        """
        # Extract coefficients 
        weights = self.clf.coef_[0]
        
        # Create a DataFrame for mapping and sorting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Weight': weights,
            'Abs_Weight': np.abs(weights) # We rank by magnitude, regardless of +/-
        })
        
        # Sort by Importance (Absolute Weight)
        importance_df = importance_df.sort_values(by='Abs_Weight', ascending=False)
        
        # Save to the models directory
        output_path = self.model_dir / "feature_importance.csv"
        importance_df.to_csv(output_path, index=False)
        
        print(f"Feature importance ranked and saved to: {output_path}")