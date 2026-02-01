import os
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch.nn as nn
import pandas as pd
import joblib
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

print("Loading dataset")
df = pd.read_parquet("MSK_MET_CURATED_FINAL.parquet")

print("Loading ensemble info, label encoder, test indices")

ensemble_info = joblib.load("ensemble_info.pkl")
label_encoder = joblib.load("label_encoder.pkl")
test_indices = np.load("test_indices.npy", allow_pickle=True)

feature_columns = ensemble_info["feature_columns"]

print(f"Got {len(feature_columns)} feature columns")
print(f"Got {len(test_indices)} test samples")

missing = sorted(set(feature_columns) - set(df.columns))
if missing:
    print(f"Missing {len(missing)} feature columns")
    for col in missing:
        df[col] = 0
else:
    print("All feature columns present")

X = df[feature_columns]

y = pd.Series(
    label_encoder.transform(df["CANCER_TYPE"]),
    index=df.index
)

X_test = X.loc[test_indices]
y_test = y.loc[test_indices].values

print(f"Test set: {X_test.shape}")

class MLP(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 903),
            nn.BatchNorm1d(903),
            nn.ReLU(),
            nn.Dropout(0.2637),
            nn.Linear(903, 1806),
            nn.BatchNorm1d(1806),
            nn.ReLU(),
            nn.Dropout(0.2637),
            nn.Linear(1806, c)
        )

    def forward(self, x):
        return self.net(x)

print("Loading MLP model")
checkpoint = torch.load("mlp_model.pt", map_location="cpu")

mlp_model = MLP(
    d=checkpoint["input_dim"],
    c=checkpoint["num_classes"]
)
mlp_model.load_state_dict(checkpoint["model_state_dict"])
mlp_model.eval()

print("Loading XGBoost")

booster = xgb.Booster()
booster.load_model("xgb_model.json")

dtest = xgb.DMatrix(X_test.values)

with torch.no_grad():
    mlp_logits = mlp_model(
        torch.tensor(X_test.values, dtype=torch.float32)
    )
    mlp_proba = torch.softmax(mlp_logits, 1).numpy()
    mlp_pred = mlp_proba.argmax(1)

xgb_proba = booster.predict(dtest)
xgb_pred = xgb_proba.argmax(1)

W_XGB = 0.6
ensemble_proba = W_XGB * xgb_proba + (1 - W_XGB) * mlp_proba
ensemble_pred = ensemble_proba.argmax(1)

from sklearn.metrics import accuracy_score, f1_score, classification_report

print("Results")

print("MLP")
print(f"Accuracy: {accuracy_score(y_test, mlp_pred):.4f}")

print("XGBoost")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")

print("Ensemble")
print(f"Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
print(f"F1: {f1_score(y_test, ensemble_pred, average='weighted'):.4f}")

print("Classification Report")
print(
    classification_report(
        y_test,
        ensemble_pred,
        target_names=label_encoder.classes_
    )
)

print("\nValidation completed successfully (no segfaults)")

ones_per_patient = (X_test == 1).sum(axis=1)
candidates = ones_per_patient[ones_per_patient == 12]

if len(candidates) == 0:
    raise ValueError("No test patient found with exactly 5 ones")

patient_idx = candidates.index[0]

true_label_enc = y.loc[patient_idx]
true_label = label_encoder.inverse_transform([true_label_enc])[0]

excluded_cols = {
    "SAMPLE_ID", "Sample_Id", "Tumor_Sample_Barcode", "PATIENT_ID",
    "CANCER_TYPE", "SEX", "AGE_AT_SEQUENCING",
    "TMB_NONSYNONYMOUS", "MSI_SCORE",
}

patient_row = df.loc[patient_idx]

patient_info = {
    "AGE_AT_SEQUENCING": patient_row.get("AGE_AT_SEQUENCING", "NA"),
    "SEX": patient_row.get("SEX", "NA"),
    "TMB_NONSYNONYMOUS": patient_row.get("TMB_NONSYNONYMOUS", "NA"),
    "MSI_SCORE": patient_row.get("MSI_SCORE", "NA"),
}

patient_features = X_test.loc[patient_idx]
active_features = patient_features[patient_features == 1].index.tolist()

row_pos = X_test.index.get_loc(patient_idx)

mlp_probs = mlp_proba[row_pos]
xgb_probs = xgb_proba[row_pos]
ens_probs = ensemble_proba[row_pos]

mlp_pred_label = label_encoder.inverse_transform([mlp_probs.argmax()])[0]
xgb_pred_label = label_encoder.inverse_transform([xgb_probs.argmax()])[0]
ens_pred_label = label_encoder.inverse_transform([ens_probs.argmax()])[0]

print("\nPatient Analysis")

print(f"\nPatient index: {patient_idx}")
print(f"True cancer type: {true_label}")

print("\nInfo:")
for k, v in patient_info.items():
    print(f"  {k}: {v}")

print(f"\nFeatures ({len(active_features)}):")
for f in active_features:
    print(f"  - {f}")

print("\nPredictions:")
print(f"  MLP: {mlp_pred_label} ({mlp_probs.max():.4f})")
print(f"  XGBoost: {xgb_pred_label} ({xgb_probs.max():.4f})")
print(f"  Ensemble: {ens_pred_label} ({ens_probs.max():.4f})")