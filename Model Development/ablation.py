import os
import time
import copy
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

FIXED_XGB_WEIGHT = 0.5
FIXED_MLP_WEIGHT = 0.5

COLORS = {
    'dark_blue':    '#1e5c8e',
    'blue_1':       '#2a6d9d',
    'blue_2':       '#367eac',
    'blue_3':       '#4490ba',
    'blue_4':       '#53a2c8',
    'light_blue_1': '#63b4d6',
    'light_blue_2': '#74c6e4',
    'light_blue_3': '#86d8f1',
    'lightest_blue':'#99ebff'
}

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "/content/saved_artifacts"

print("=" * 80)
print("ABLATION STUDY | 10x XGBoost + 10x MLP | FIXED 50/50 BLEND | EARLY STOP + RESTORE")
print(f"Execution Engine: {device.upper()}")
print(f"Artifact directory: {SAVE_DIR}")
print("=" * 80)

print("\nStep 1: Loading saved artifacts from the main pipeline...")

ensemble_info = pd.read_pickle(os.path.join(SAVE_DIR, "ensemble_info.pkl"))
label_encoder = pd.read_pickle(os.path.join(SAVE_DIR, "label_encoder.pkl"))

target_col      = ensemble_info["target_col"]
num_classes     = ensemble_info["num_classes"]
N_XGB           = ensemble_info["n_xgb"]
N_MLP           = ensemble_info["n_mlp"]
SAVED_XGB_WEIGHT = ensemble_info["xgb_weight"]
SAVED_MLP_WEIGHT = ensemble_info["mlp_weight"]
feature_columns = ensemble_info["feature_columns"]
input_dim       = ensemble_info["input_dim"]
xgb_best_iters  = ensemble_info.get("xgb_best_iters", None)

es = ensemble_info.get("early_stopping", {})
XGB_EARLY_STOPPING_ROUNDS = es.get("xgb_rounds", 50)
MLP_MAX_EPOCHS            = es.get("mlp_max_epochs", 120)
MLP_WARMUP                = es.get("mlp_warmup", 40)
MLP_PATIENCE              = es.get("mlp_patience", 25)
MLP_MIN_DELTA             = es.get("mlp_min_delta", 1e-4)

XGB_DIR = os.path.join(SAVE_DIR, "xgb_models")
MLP_DIR = os.path.join(SAVE_DIR, "mlp_models")

df = pd.read_parquet(os.path.join(SAVE_DIR, "MSKMET_Curated_Final.parquet"))
if df.index.name != "SAMPLE_ID" and "SAMPLE_ID" in df.columns:
    df = df.set_index("SAMPLE_ID")

train_ids = np.load(os.path.join(SAVE_DIR, "train_indices.npy"), allow_pickle=True)
val_ids   = np.load(os.path.join(SAVE_DIR, "val_indices.npy"),   allow_pickle=True)
test_ids  = np.load(os.path.join(SAVE_DIR, "test_indices.npy"),  allow_pickle=True)

X_all = df.drop(columns=[target_col])
for col in X_all.columns:
    if X_all[col].isna().any():
        X_all[col] = X_all[col].fillna(X_all[col].median())

y_all = pd.Series(label_encoder.transform(df[target_col].values), index=df.index)
all_cols = list(X_all.columns)

print(f"  ├─ Samples            : {X_all.shape[0]}")
print(f"  ├─ Total features     : {len(all_cols)}")
print(f"  ├─ Target classes     : {num_classes}")
print(f"  ├─ Ensemble members   : {N_XGB} XGB + {N_MLP} MLP (member #0 = anchor)")
print(f"  ├─ FIXED blend (XGB/MLP): {FIXED_XGB_WEIGHT:.2f} / {FIXED_MLP_WEIGHT:.2f}  (held constant -> attribution)")
print(f"  └─ Early stop          : XGB {XGB_EARLY_STOPPING_ROUNDS} rounds | "
      f"MLP warmup {MLP_WARMUP}, patience {MLP_PATIENCE}")

SOM_MAT_SUBCATS = [
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Ins", "Frame_Shift_Del",
    "Splice_Site", "Translation_Start_Site", "In_Frame_Ins", "In_Frame_Del",
    "Nonstop_Mutation"
]
mutation_suffixes = ["_ANY_SOM_MAT"] + [f"_{s}" for s in SOM_MAT_SUBCATS]

ablation_groups = {
    "No_AGE":       ["AGE_AT_SEQUENCING"],
    "No_SEX":       ["SEX"],
    "No_TMB":       ["TMB_NONSYNONYMOUS"],
    "No_MSI":       ["MSI_SCORE"],
    "No_DMETS":     [c for c in all_cols if c.startswith("DMETS_DX_")],
    "No_MUTATIONS": [c for c in all_cols
                     if any(c.endswith(s) for s in mutation_suffixes) or c == "IS_ANY_SOM_MAT"],
    "No_CNA":       [c for c in all_cols
                     if c.endswith("_AMP") or c.endswith("_DEL") or c.endswith("_ANY_CNA")],
    "No_FUSIONS":   [c for c in all_cols if c.endswith("_FUSION")],
    "No_HOTSPOTS":  [c for c in all_cols if c.endswith("_HOTSPOT")],
}
ablation_groups = {k: [c for c in v if c in all_cols] for k, v in ablation_groups.items()}

print("\n--- Ablation Group Sizes ---")
for name, cols in ablation_groups.items():
    flag = "  <-- WARNING: 0 features" if len(cols) == 0 else ""
    print(f"  {name:<14}: {len(cols)} features removed{flag}")

base_xgb_params = {
    "max_depth": 9, "learning_rate": 0.066, "n_estimators": 446,
    "subsample": 0.807, "colsample_bytree": 0.752, "min_child_weight": 2,
    "gamma": 0.293, "reg_lambda": 1.835, "reg_alpha": 0.147
}

base_lr = 1.0037e-05
base_wd = 2.5770e-05

base_h1      = 903
base_h2_mult = 2.0
base_p_drop  = 0.2637


class CancerDataset(Dataset):
    def __init__(self, X_mat, y_vec):
        self.X = torch.FloatTensor(X_mat.values)
        self.y = torch.LongTensor(y_vec)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class DynamicMLP(nn.Module):
    def __init__(self, d_in, h1, h2, c_out, p_drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(h1, h2),   nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(h2, c_out)
        )

    def forward(self, x):
        return self.net(x)


def extract_mlp_probabilities(model, loader):
    model.eval()
    prob_list = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            prob_list.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.vstack(prob_list)


def verify_saved_ensemble():
    print("\n" + "=" * 80)
    print("STEP 3b — VERIFY SAVED ENSEMBLE (reload + re-predict, no retraining)")
    print("=" * 80)

    missing = [c for c in feature_columns if c not in X_all.columns]
    if missing:
        raise KeyError(f"{len(missing)} saved feature(s) missing from parquet, e.g. {missing[:5]}")

    X_test_v = X_all.loc[test_ids, feature_columns]
    y_test_v = y_all.loc[test_ids].values
    if X_test_v.shape[1] != input_dim:
        raise ValueError(f"Feature count {X_test_v.shape[1]} != saved input_dim {input_dim}")
    X_test_vals = X_test_v.values
    print(f"  Test samples: {X_test_v.shape[0]}  |  Features: {X_test_v.shape[1]} (expected {input_dim})")
    print(f"  Applying deployed blend: XGB {SAVED_XGB_WEIGHT:.2f} / MLP {SAVED_MLP_WEIGHT:.2f}")

    print(f"\n  Loading {N_XGB} XGBoost models...")
    xgb_test_accum = np.zeros((X_test_v.shape[0], num_classes))
    for i in range(N_XGB):
        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(XGB_DIR, f"xgb_model_{i}.json"))
        try:
            clf.get_booster().set_param({"device": xgb_device})
        except Exception:
            pass
        if xgb_best_iters is not None:
            xgb_test_accum += clf.predict_proba(X_test_vals, iteration_range=(0, int(xgb_best_iters[i]) + 1))
        else:
            xgb_test_accum += clf.predict_proba(X_test_vals)
    mean_xgb_test = xgb_test_accum / float(N_XGB)

    print(f"  Loading {N_MLP} MLP models...")
    X_test_tensor = torch.FloatTensor(X_test_vals).to(device)
    mlp_test_accum = np.zeros((X_test_v.shape[0], num_classes))
    for i in range(N_MLP):
        ckpt = torch.load(os.path.join(MLP_DIR, f"mlp_model_{i}.pt"), map_location=device)
        net = DynamicMLP(ckpt["input_dim"], ckpt["h1"], ckpt["h2"],
                         ckpt["num_classes"], ckpt["p_drop"]).to(device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        with torch.no_grad():
            mlp_test_accum += torch.softmax(net(X_test_tensor), 1).cpu().numpy()
    mean_mlp_test = mlp_test_accum / float(N_MLP)

    final_test_proba = (SAVED_XGB_WEIGHT * mean_xgb_test) + (SAVED_MLP_WEIGHT * mean_mlp_test)
    xgb_pred = mean_xgb_test.argmax(1)
    mlp_pred = mean_mlp_test.argmax(1)
    ens_pred = final_test_proba.argmax(1)

    print("\n  REPRODUCED TEST RESULTS (from saved models):")
    print(f"    [XGBoost]  Top-1: {accuracy_score(y_test_v, xgb_pred):.4f}  "
          f"Top-3: {top_k_accuracy_score(y_test_v, mean_xgb_test, k=3):.4f}  "
          f"F1: {f1_score(y_test_v, xgb_pred, average='weighted'):.4f}")
    print(f"    [MLP]      Top-1: {accuracy_score(y_test_v, mlp_pred):.4f}  "
          f"Top-3: {top_k_accuracy_score(y_test_v, mean_mlp_test, k=3):.4f}  "
          f"F1: {f1_score(y_test_v, mlp_pred, average='weighted'):.4f}")
    print(f"    [Ensemble] Top-1: {accuracy_score(y_test_v, ens_pred):.4f}  "
          f"Top-3: {top_k_accuracy_score(y_test_v, final_test_proba, k=3):.4f}  "
          f"Top-5: {top_k_accuracy_score(y_test_v, final_test_proba, k=5):.4f}  "
          f"F1: {f1_score(y_test_v, ens_pred, average='weighted'):.4f}")
    print("=" * 80)
    print("  This should match the FIXED-blend All_Features baseline below.")
    print("=" * 80)


verify_saved_ensemble()


def train_ensemble(X_train, X_val, X_test, y_train, y_val, y_test):
    keep_cols = X_train.columns[X_train.sum(axis=0) != 0]
    X_train = X_train[keep_cols]
    X_val   = X_val[keep_cols]
    X_test  = X_test[keep_cols]
    input_dim_run = X_train.shape[1]

    train_loader = DataLoader(CancerDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader   = DataLoader(CancerDataset(X_val,   y_val),   batch_size=512, shuffle=False)
    test_loader  = DataLoader(CancerDataset(X_test,  y_test),  batch_size=512, shuffle=False)

    class_weights  = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[i] for i in y_train])

    xgb_val_accum  = np.zeros((X_val.shape[0],  num_classes))
    xgb_test_accum = np.zeros((X_test.shape[0], num_classes))
    mlp_val_accum  = np.zeros((X_val.shape[0],  num_classes))
    mlp_test_accum = np.zeros((X_test.shape[0], num_classes))

    for i in range(N_XGB):
        m_seed = 100 + i
        np.random.seed(m_seed)

        if i == 0:
            p_depth  = int(base_xgb_params["max_depth"])
            p_lr     = float(base_xgb_params["learning_rate"])
            p_sub    = float(base_xgb_params["subsample"])
            p_col    = float(base_xgb_params["colsample_bytree"])
            p_est    = int(base_xgb_params["n_estimators"])
            p_child  = int(base_xgb_params["min_child_weight"])
            p_gamma  = float(base_xgb_params["gamma"])
            p_lambda = float(base_xgb_params["reg_lambda"])
            p_alpha  = float(base_xgb_params["reg_alpha"])
        else:
            p_depth  = int(np.clip(base_xgb_params["max_depth"] + np.random.choice([-3, -2, -1, 0, 1, 2, 3]), 4, 13))
            p_lr     = float(np.clip(base_xgb_params["learning_rate"] * np.random.uniform(0.55, 1.65), 0.025, 0.14))
            p_sub    = float(np.clip(base_xgb_params["subsample"] * np.random.uniform(0.70, 1.15), 0.50, 1.00))
            p_col    = float(np.clip(base_xgb_params["colsample_bytree"] * np.random.uniform(0.70, 1.15), 0.50, 1.00))
            p_est    = int(np.clip(base_xgb_params["n_estimators"] + np.random.choice([-120, -90, -60, -30, 0, 30, 60, 90, 120]), 220, 700))
            p_child  = int(np.clip(base_xgb_params["min_child_weight"] + np.random.choice([-1, 0, 1, 2, 3]), 1, 6))
            p_gamma  = float(np.clip(base_xgb_params["gamma"] * np.random.uniform(0.35, 2.20), 0.00, 1.20))
            p_lambda = float(np.clip(base_xgb_params["reg_lambda"] * np.random.uniform(0.35, 2.50), 0.20, 6.00))
            p_alpha  = float(np.clip(base_xgb_params["reg_alpha"] * np.random.uniform(0.20, 3.00), 0.00, 1.50))

        clf = xgb.XGBClassifier(
            objective="multi:softprob", num_class=num_classes,
            eval_metric=["merror", "mlogloss"],
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            tree_method="hist", device=xgb_device,
            max_depth=p_depth, learning_rate=p_lr, n_estimators=p_est,
            subsample=p_sub, colsample_bytree=p_col, min_child_weight=p_child,
            gamma=p_gamma, reg_lambda=p_lambda, reg_alpha=p_alpha, random_state=m_seed
        )
        clf.fit(
            X_train.values, y_train, sample_weight=sample_weights,
            eval_set=[(X_train.values, y_train), (X_val.values, y_val)], verbose=False
        )
        best_it = int(clf.best_iteration)
        xgb_val_accum  += clf.predict_proba(X_val.values,  iteration_range=(0, best_it + 1))
        xgb_test_accum += clf.predict_proba(X_test.values, iteration_range=(0, best_it + 1))

    for i in range(N_MLP):
        m_seed = 200 + i
        torch.manual_seed(m_seed)
        np.random.seed(m_seed)

        if i == 0:
            h1_dims = int(np.clip(base_h1, 384, 1536))
            h2_dims = int(np.clip(base_h1 * base_h2_mult, 512, 3072))
            p_drop  = float(np.clip(base_p_drop, 0.10, 0.48))
            p_lr    = float(np.clip(base_lr, 3.0e-06, 3.0e-05))
            p_wd    = float(np.clip(base_wd, 5.0e-06, 1.0e-04))
        else:
            h1_dims = int(np.clip(base_h1 + np.random.choice([-384, -256, -128, 0, 128, 256, 384]), 384, 1536))
            h2_mult = float(np.random.uniform(1.35, 2.75))
            h2_dims = int(np.clip(h1_dims * h2_mult, 512, 3072))
            p_drop  = float(np.clip(base_p_drop * np.random.uniform(0.45, 1.75), 0.10, 0.48))
            p_lr    = float(np.clip(base_lr * np.random.uniform(0.35, 2.50), 3.0e-06, 3.0e-05))
            p_wd    = float(np.clip(base_wd * np.random.uniform(0.30, 3.00), 5.0e-06, 1.0e-04))

        net = DynamicMLP(input_dim_run, h1_dims, h2_dims, num_classes, p_drop).to(device)
        optimizer = optim.Adam(net.parameters(), lr=p_lr, weight_decay=p_wd)
        criterion = nn.CrossEntropyLoss()

        best_val_loss     = float("inf")
        best_state        = None
        epochs_no_improve = 0

        for epoch in range(MLP_MAX_EPOCHS):
            net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

            net.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    epoch_val_loss += criterion(net(xb), yb).item() * len(xb)
            val_loss_epoch = epoch_val_loss / len(val_loader.dataset)

            if val_loss_epoch < best_val_loss - MLP_MIN_DELTA:
                best_val_loss     = val_loss_epoch
                best_state        = copy.deepcopy(net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) >= MLP_WARMUP and epochs_no_improve >= MLP_PATIENCE:
                break

        if best_state is not None:
            net.load_state_dict(best_state)

        mlp_val_accum  += extract_mlp_probabilities(net, val_loader)
        mlp_test_accum += extract_mlp_probabilities(net, test_loader)

    mean_xgb_val  = xgb_val_accum  / float(N_XGB)
    mean_xgb_test = xgb_test_accum / float(N_XGB)
    mean_mlp_val  = mlp_val_accum  / float(N_MLP)
    mean_mlp_test = mlp_test_accum / float(N_MLP)

    final_val_proba  = (FIXED_XGB_WEIGHT * mean_xgb_val)  + (FIXED_MLP_WEIGHT * mean_mlp_val)
    final_test_proba = (FIXED_XGB_WEIGHT * mean_xgb_test) + (FIXED_MLP_WEIGHT * mean_mlp_test)
    fix_val_top1     = accuracy_score(y_val, final_val_proba.argmax(1))

    return {
        "xgb_test": mean_xgb_test, "mlp_test": mean_mlp_test,
        "ens_val": final_val_proba, "ens_test": final_test_proba,
        "n_features": input_dim_run,
        "w_xgb": FIXED_XGB_WEIGHT, "w_mlp": FIXED_MLP_WEIGHT, "blend_val_top1": fix_val_top1
    }


all_results = []


def run_ablation(run_name, drop_cols):
    print(f"\n{'=' * 80}")
    print(f"  RUN: {run_name}  |  Features dropped: {len(drop_cols)}")
    print(f"{'=' * 80}")
    run_start = time.time()

    keep_features = [c for c in all_cols if c not in set(drop_cols)]

    X_train = X_all.loc[train_ids, keep_features]
    X_val   = X_all.loc[val_ids,   keep_features]
    X_test  = X_all.loc[test_ids,  keep_features]
    y_train = y_all.loc[train_ids].values
    y_val   = y_all.loc[val_ids].values
    y_test  = y_all.loc[test_ids].values

    out = train_ensemble(X_train, X_val, X_test, y_train, y_val, y_test)

    xgb_pred = out["xgb_test"].argmax(1)
    mlp_pred = out["mlp_test"].argmax(1)
    ens_pred = out["ens_test"].argmax(1)

    elapsed = (time.time() - run_start) / 60.0

    metrics = {
        "Run":              run_name,
        "Features_Removed": len(drop_cols),
        "Features_Used":    out["n_features"],
        "Blend_XGB":        round(out["w_xgb"], 2),
        "Blend_MLP":        round(out["w_mlp"], 2),
        "Blend_Val_Top1":   round(out["blend_val_top1"], 4),
        "XGB_Top1": round(accuracy_score(y_test, xgb_pred), 4),
        "XGB_Top3": round(top_k_accuracy_score(y_test, out["xgb_test"], k=3), 4),
        "XGB_F1":   round(f1_score(y_test, xgb_pred, average="weighted"), 4),
        "MLP_Top1": round(accuracy_score(y_test, mlp_pred), 4),
        "MLP_Top3": round(top_k_accuracy_score(y_test, out["mlp_test"], k=3), 4),
        "MLP_F1":   round(f1_score(y_test, mlp_pred, average="weighted"), 4),
        "ENS_Top1": round(accuracy_score(y_test, ens_pred), 4),
        "ENS_Top3": round(top_k_accuracy_score(y_test, out["ens_test"], k=3), 4),
        "ENS_F1":   round(f1_score(y_test, ens_pred, average="weighted"), 4),
        "Runtime_min": round(elapsed, 2),
    }

    print(f"\n  Fixed blend: XGB {metrics['Blend_XGB']:.2f} / MLP {metrics['Blend_MLP']:.2f}  "
          f"(Val Top-1 at this blend: {metrics['Blend_Val_Top1']})")
    print(f"  [XGBoost]  Top-1: {metrics['XGB_Top1']}  Top-3: {metrics['XGB_Top3']}  F1: {metrics['XGB_F1']}")
    print(f"  [MLP]      Top-1: {metrics['MLP_Top1']}  Top-3: {metrics['MLP_Top3']}  F1: {metrics['MLP_F1']}")
    print(f"  [Ensemble] Top-1: {metrics['ENS_Top1']}  Top-3: {metrics['ENS_Top3']}  F1: {metrics['ENS_F1']}")
    print(f"  Runtime: {elapsed:.1f} min")

    pd.DataFrame(all_results + [metrics]).to_csv("/content/ablation_results.csv", index=False)
    return metrics


total_start = time.time()

print("\n" + "=" * 80)
print("  RUN 0 — BASELINE: All Features (accuracy confirmation)")
print("=" * 80)
baseline_metrics = run_ablation("All_Features", drop_cols=[])
all_results.append(baseline_metrics)

print("\n" + "=" * 80)
print("  BASELINE CONFIRMED")
print(f"  Fixed blend: XGB {baseline_metrics['Blend_XGB']:.2f} / MLP {baseline_metrics['Blend_MLP']:.2f}")
print(f"  Ensemble  Top-1: {baseline_metrics['ENS_Top1']}  "
      f"Top-3: {baseline_metrics['ENS_Top3']}  F1: {baseline_metrics['ENS_F1']}")
print(f"  XGBoost   Top-1: {baseline_metrics['XGB_Top1']}  "
      f"Top-3: {baseline_metrics['XGB_Top3']}  F1: {baseline_metrics['XGB_F1']}")
print(f"  MLP       Top-1: {baseline_metrics['MLP_Top1']}  "
      f"Top-3: {baseline_metrics['MLP_Top3']}  F1: {baseline_metrics['MLP_F1']}")
print("  (Ensemble baseline should equal the Step 3b reproduced ensemble.)")
print("=" * 80)

for run_name, drop_cols in ablation_groups.items():
    all_results.append(run_ablation(run_name, drop_cols))

results_df = pd.DataFrame(all_results)

baseline_ens_top1 = baseline_metrics["ENS_Top1"]
baseline_ens_top3 = baseline_metrics["ENS_Top3"]
baseline_ens_f1   = baseline_metrics["ENS_F1"]

results_df["ENS_Top1_Drop"] = (baseline_ens_top1 - results_df["ENS_Top1"]).round(4)
results_df["ENS_Top3_Drop"] = (baseline_ens_top3 - results_df["ENS_Top3"]).round(4)
results_df["ENS_F1_Drop"]   = (baseline_ens_f1   - results_df["ENS_F1"]).round(4)

results_df.drop(columns=["ENS_Top1_Drop", "ENS_Top3_Drop", "ENS_F1_Drop"]).to_csv(
    "/content/ablation_results.csv", index=False
)

print("\n\n" + "=" * 80)
print("ABLATION STUDY — FULL RESULTS SUMMARY (fixed 50/50 blend)")
print("=" * 80)
print(results_df.to_string(index=False))

print("\n\n--- Ensemble Drop vs Baseline (sorted by Top-1 drop) ---")
drop_summary = results_df[results_df["Run"] != "All_Features"][[
    "Run", "ENS_Top1", "ENS_Top1_Drop", "ENS_Top3", "ENS_Top3_Drop", "ENS_F1", "ENS_F1_Drop"
]].copy().sort_values("ENS_Top1_Drop", ascending=False)
print(drop_summary.to_string(index=False))

print(f"\nTotal runtime: {(time.time() - total_start) / 60:.1f} minutes")
print("Results saved to /content/ablation_results.csv")

print("\nGenerating ablation plots...")

results_pct = results_df.copy()
results_pct["Label"] = (
    results_pct["Run"].str.replace("All_Features", "All Features").str.replace("No_", "No ")
)
metric_cols = ["XGB_Top1", "XGB_Top3", "XGB_F1",
               "MLP_Top1", "MLP_Top3", "MLP_F1",
               "ENS_Top1", "ENS_Top3", "ENS_F1"]
for col in metric_cols:
    results_pct[col] = (results_pct[col] * 100).round(2)

baseline_pct = results_pct[results_pct["Run"] == "All_Features"].iloc[0]
ablation_pct = results_pct[results_pct["Run"] != "All_Features"].copy()

ablation_pct["ENS_Top1_Drop"] = (baseline_pct["ENS_Top1"] - ablation_pct["ENS_Top1"]).round(2)
ablation_pct["ENS_Top3_Drop"] = (baseline_pct["ENS_Top3"] - ablation_pct["ENS_Top3"]).round(2)
ablation_pct["ENS_F1_Drop"]   = (baseline_pct["ENS_F1"]   - ablation_pct["ENS_F1"]).round(2)

ablation_pct = ablation_pct.sort_values("ENS_Top1_Drop", ascending=True).reset_index(drop=True)
abl_labels   = ablation_pct["Label"].tolist()


def drop_color(d):
    if d <= 0:   return COLORS['lightest_blue']
    elif d < 2:  return COLORS['blue_3']
    elif d < 5:  return COLORS['blue_1']
    else:        return COLORS['dark_blue']


bar_colors_pct = [drop_color(d) for d in ablation_pct["ENS_Top1_Drop"]]

fig, ax = plt.subplots(figsize=(11, 6))
drops = ablation_pct["ENS_Top1_Drop"].values
bars = ax.barh(abl_labels, drops, color=bar_colors_pct, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, drops):
    sign = "+" if val < 0 else ""
    ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.2f}%", va='center', ha='left',
            fontsize=10, fontweight='bold', color=COLORS['dark_blue'])
ax.set_xlim(left=min(drops) - 0.3, right=drops.max() + 1.8)
ax.axvline(0, color='black', linewidth=1.0)
ax.set_xlabel("Top-1 Accuracy Drop vs Baseline (%)", fontsize=12, fontweight='bold')
ax.set_title("Ablation Study — Ensemble Top-1 Accuracy Impact per Feature Group\n(fixed 50/50 blend)",
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
legend_patches = [
    mpatches.Patch(color=COLORS['dark_blue'],    label='Large drop  (>= 5%)'),
    mpatches.Patch(color=COLORS['blue_1'],        label='Medium drop (2-5%)'),
    mpatches.Patch(color=COLORS['blue_3'],        label='Small drop  (< 2%)'),
    mpatches.Patch(color=COLORS['lightest_blue'], label='Improvement'),
]
ax.legend(handles=legend_patches, fontsize=10, frameon=True, shadow=True, loc='lower right')
plt.tight_layout()
plt.savefig("/content/ablation_drop_horizontal.png", dpi=300, bbox_inches='tight')
print("Saved: /content/ablation_drop_horizontal.png")
plt.close()

fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(len(ablation_pct))
width = 0.25
ax.bar(x - width, ablation_pct["ENS_Top1"], width, label="Top-1 Accuracy",
       color=COLORS['dark_blue'], alpha=0.92)
ax.bar(x, ablation_pct["ENS_Top3"], width, label="Top-3 Accuracy",
       color=COLORS['blue_3'], alpha=0.92)
ax.bar(x + width, ablation_pct["ENS_F1"], width, label="Weighted F1",
       color=COLORS['light_blue_2'], alpha=0.92)
ax.axhline(baseline_pct["ENS_Top1"], color=COLORS['dark_blue'],   linestyle='--', linewidth=1.3, alpha=0.55, label='Baseline Top-1')
ax.axhline(baseline_pct["ENS_Top3"], color=COLORS['blue_3'],       linestyle='--', linewidth=1.3, alpha=0.55, label='Baseline Top-3')
ax.axhline(baseline_pct["ENS_F1"],   color=COLORS['light_blue_2'], linestyle='--', linewidth=1.3, alpha=0.55, label='Baseline F1')
ax.set_xticks(x)
ax.set_xticklabels(abl_labels, rotation=30, ha='right', fontsize=11)
ax.set_ylabel("Score (%)", fontsize=13, fontweight='bold')
ax.set_title("Ablation Study — Ensemble Top-1, Top-3 & F1 per Feature Group\n(fixed 50/50 blend; dashed lines = All Features baseline)",
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, frameon=True, shadow=True, ncol=2)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.tick_params(labelsize=10)
all_vals = pd.concat([ablation_pct["ENS_Top1"], ablation_pct["ENS_Top3"], ablation_pct["ENS_F1"]])
ax.set_ylim(max(0, all_vals.min() - 5), min(100, all_vals.max() + 5))
plt.tight_layout()
plt.savefig("/content/ablation_grouped_bar.png", dpi=300, bbox_inches='tight')
print("Saved: /content/ablation_grouped_bar.png")
plt.close()

fig, ax = plt.subplots(figsize=(13, 6))
line_df = pd.concat([results_pct[results_pct["Run"] == "All_Features"], ablation_pct], ignore_index=True)
line_labels = line_df["Label"].tolist()
x_pos = np.arange(len(line_df))
ax.plot(x_pos, line_df["ENS_Top1"], marker='o', linewidth=2.5, markersize=7,
        color=COLORS['dark_blue'], label="Top-1 Accuracy")
ax.plot(x_pos, line_df["ENS_Top3"], marker='s', linewidth=2.5, markersize=7,
        color=COLORS['blue_3'], label="Top-3 Accuracy")
ax.plot(x_pos, line_df["ENS_F1"], marker='^', linewidth=2.5, markersize=7,
        color=COLORS['light_blue_2'], label="Weighted F1")
ax.axvspan(-0.5, 0.5, alpha=0.08, color=COLORS['dark_blue'])
y_inside = line_df[["ENS_Top1", "ENS_Top3", "ENS_F1"]].values.min() + 1
ax.text(0, y_inside, "Baseline", ha='center', va='bottom', fontsize=9,
        color=COLORS['dark_blue'], style='italic')
ax.set_xticks(x_pos)
ax.set_xticklabels(line_labels, rotation=30, ha='right', fontsize=10)
ax.set_ylabel("Score (%)", fontsize=13, fontweight='bold')
ax.set_title("Ablation Study — Ensemble Metric Trends Across Feature Groups (fixed 50/50 blend)",
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("/content/ablation_line_plot.png", dpi=300, bbox_inches='tight')
print("Saved: /content/ablation_line_plot.png")
plt.close()

cmap_heatmap = LinearSegmentedColormap.from_list(
    'custom_blues',
    ['#E8F4F8', '#B8D9E8', '#88BED8', '#5AA3C8',
     '#3D88B8', '#2A6D9D', '#1e5c8e', '#143D5E', '#0A1E2E']
)
heatmap_df = pd.concat([results_pct[results_pct["Run"] == "All_Features"], ablation_pct],
                       ignore_index=True).set_index("Label")[metric_cols]
fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(heatmap_df.values, cmap=cmap_heatmap, aspect='auto',
               vmin=heatmap_df.values.min() - 1, vmax=heatmap_df.values.max() + 0.5)
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Score (%)", fontsize=10)
ax.set_xticks(np.arange(len(metric_cols)))
ax.set_yticks(np.arange(len(heatmap_df)))
ax.set_xticklabels(metric_cols, rotation=40, ha='right', fontsize=10)
ax.set_yticklabels(heatmap_df.index, fontsize=10)
mid = (heatmap_df.values.min() + heatmap_df.values.max()) / 2
for i in range(len(heatmap_df)):
    for j in range(len(metric_cols)):
        val = heatmap_df.values[i, j]
        text_color = 'white' if val < mid else COLORS['dark_blue']
        ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=text_color)
ax.set_title("Ablation Study — All Metrics Heatmap (XGBoost / MLP / Ensemble, fixed 50/50 blend)",
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("/content/ablation_heatmap.png", dpi=300, bbox_inches='tight')
print("Saved: /content/ablation_heatmap.png")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
drop_metrics = [
    ("ENS_Top1_Drop", "Top-1 Accuracy Drop (%)", COLORS['dark_blue']),
    ("ENS_Top3_Drop", "Top-3 Accuracy Drop (%)", COLORS['blue_3']),
    ("ENS_F1_Drop",   "Weighted F1 Drop (%)",    COLORS['light_blue_2']),
]
for ax, (col, title, color) in zip(axes, drop_metrics):
    vals = ablation_pct[col].values
    bars = ax.barh(abl_labels, vals, color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        sign = "+" if val < 0 else ""
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.2f}%", va='center', ha='left',
                fontsize=9, fontweight='bold', color=COLORS['dark_blue'])
    ax.set_xlim(left=min(vals) - 0.3, right=vals.max() + 1.5)
    ax.axvline(0, color='black', linewidth=1.0)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Drop vs Baseline (%)", fontsize=10)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
fig.suptitle("Ablation Study — Ensemble Drop vs Baseline by Metric (fixed 50/50 blend)",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("/content/ablation_drop_comparison.png", dpi=300, bbox_inches='tight')
print("Saved: /content/ablation_drop_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("ABLATION COMPLETE")
print("=" * 80)
print("Outputs:")
print("  /content/ablation_results.csv")
print("  /content/ablation_drop_horizontal.png   — horizontal drop bar")
print("  /content/ablation_grouped_bar.png       — Top-1 / Top-3 / F1 grouped bar")
print("  /content/ablation_line_plot.png         — metric trends line plot")
print("  /content/ablation_heatmap.png           — full metrics heatmap")
print("  /content/ablation_drop_comparison.png   — side-by-side drop panels")
print("=" * 80)