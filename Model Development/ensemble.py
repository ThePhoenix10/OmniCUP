import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import time
import warnings
import os
import copy

warnings.filterwarnings("ignore")

COLORS = {
    'dark_blue':     '#1e5c8e',
    'blue_1':        '#2a6d9d',
    'blue_2':        '#367eac',
    'blue_3':        '#4490ba',
    'blue_4':        '#53a2c8',
    'light_blue_1':  '#63b4d6',
    'light_blue_2':  '#74c6e4',
    'light_blue_3':  '#86d8f1',
    'lightest_blue': '#99ebff'
}

XGB_EARLY_STOPPING_ROUNDS = 50

MLP_MAX_EPOCHS   = 120
MLP_WARMUP       = 40
MLP_PATIENCE     = 25
MLP_MIN_DELTA    = 1e-4

BLEND_STEP = 0.1

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
start_time = time.time()

SAVE_DIR = "/content/saved_artifacts"
XGB_DIR  = os.path.join(SAVE_DIR, "xgb_models")
MLP_DIR  = os.path.join(SAVE_DIR, "mlp_models")
os.makedirs(XGB_DIR, exist_ok=True)
os.makedirs(MLP_DIR, exist_ok=True)

print("="*80)
print("10x XGBoost + 10x MLP Deep Ensemble | BLEND GRID SEARCH (10% steps) | EARLY STOP + RESTORE")
print(f"Execution Engine: {device.upper()}")
print(f"Artifact directory: {SAVE_DIR}")
print("="*80)

print("\nStep 1: Ingesting curated target matrix...")

dataset_path = "MSKMET_Curated_Final.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Could not find '{dataset_path}'. Ensure the curation script ran successfully first.")

df = pd.read_csv(dataset_path)

if "SAMPLE_ID" in df.columns:
    df = df.set_index("SAMPLE_ID")
elif "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "SAMPLE_ID"}).set_index("SAMPLE_ID")

target_col = "Primary_Site_Target"

drop_cols = [target_col, "IS_MUTATED"]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)

for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target_col])

num_classes = len(np.unique(y))

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

train_ids = np.array(X_train.index)
val_ids   = np.array(X_val.index)
test_ids  = np.array(X_test.index)

keep_cols = X_train.columns[X_train.sum(axis=0) != 0]

X_train = X_train[keep_cols]
X_val = X_val[keep_cols]
X_test = X_test[keep_cols]

input_dim = X_train.shape[1]

print(f"  ├─ Retained Processing Features: {input_dim}")
print(f"  └─ Class Destinations Target   : {num_classes}")

class CancerDataset(Dataset):
    def __init__(self, X_mat, y_vec):
        self.X = torch.FloatTensor(X_mat.values)
        self.y = torch.LongTensor(y_vec)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(
    CancerDataset(X_train, y_train),
    batch_size=256,
    shuffle=True
)

val_loader = DataLoader(
    CancerDataset(X_val, y_val),
    batch_size=512,
    shuffle=False
)

test_loader = DataLoader(
    CancerDataset(X_test, y_test),
    batch_size=512,
    shuffle=False
)

class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(y_train),
    y=y_train
)

sample_weights = np.array([class_weights[i] for i in y_train])

xgb_val_accum = np.zeros((X_val.shape[0], num_classes))
xgb_test_accum = np.zeros((X_test.shape[0], num_classes))

mlp_val_accum = np.zeros((X_val.shape[0], num_classes))
mlp_test_accum = np.zeros((X_test.shape[0], num_classes))

xgb_train_loss_curves = []
xgb_val_loss_curves   = []
xgb_train_acc_curves  = []
xgb_val_acc_curves    = []

mlp_train_loss_curves = []
mlp_val_loss_curves   = []
mlp_train_acc_curves  = []
mlp_val_acc_curves    = []

xgb_best_iters = []
mlp_archs = []

print("\nStep 2: Training 10 strongly perturbed XGBoost classifiers (early stopping)...")

base_xgb_params = {
    "max_depth": 9,
    "learning_rate": 0.066,
    "n_estimators": 446,
    "subsample": 0.807,
    "colsample_bytree": 0.752,
    "min_child_weight": 2,
    "gamma": 0.293,
    "reg_lambda": 1.835,
    "reg_alpha": 0.147
}

for i in range(10):
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
        p_depth = int(np.clip(
            base_xgb_params["max_depth"] + np.random.choice([-3, -2, -1, 0, 1, 2, 3]),
            4,
            13
        ))

        p_lr = float(np.clip(
            base_xgb_params["learning_rate"] * np.random.uniform(0.55, 1.65),
            0.025,
            0.14
        ))

        p_sub = float(np.clip(
            base_xgb_params["subsample"] * np.random.uniform(0.70, 1.15),
            0.50,
            1.00
        ))

        p_col = float(np.clip(
            base_xgb_params["colsample_bytree"] * np.random.uniform(0.70, 1.15),
            0.50,
            1.00
        ))

        p_est = int(np.clip(
            base_xgb_params["n_estimators"] + np.random.choice([-120, -90, -60, -30, 0, 30, 60, 90, 120]),
            220,
            700
        ))

        p_child = int(np.clip(
            base_xgb_params["min_child_weight"] + np.random.choice([-1, 0, 1, 2, 3]),
            1,
            6
        ))

        p_gamma = float(np.clip(
            base_xgb_params["gamma"] * np.random.uniform(0.35, 2.20),
            0.00,
            1.20
        ))

        p_lambda = float(np.clip(
            base_xgb_params["reg_lambda"] * np.random.uniform(0.35, 2.50),
            0.20,
            6.00
        ))

        p_alpha = float(np.clip(
            base_xgb_params["reg_alpha"] * np.random.uniform(0.20, 3.00),
            0.00,
            1.50
        ))

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric=["merror", "mlogloss"],
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_depth=p_depth,
        learning_rate=p_lr,
        n_estimators=p_est,
        subsample=p_sub,
        colsample_bytree=p_col,
        min_child_weight=p_child,
        gamma=p_gamma,
        reg_lambda=p_lambda,
        reg_alpha=p_alpha,
        random_state=m_seed
    )

    clf.fit(
        X_train.values,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train.values, y_train), (X_val.values, y_val)],
        verbose=False
    )

    best_it = int(clf.best_iteration)
    xgb_best_iters.append(best_it)

    print(
        f"  ├─ Fitting XGB Model #{i+1} "
        f"[Trees(max): {p_est}, Best iter: {best_it}, Depth: {p_depth}, "
        f"LR: {p_lr:.4f}]"
    )

    res = clf.evals_result()
    xgb_train_loss_curves.append(res['validation_0']['mlogloss'])
    xgb_val_loss_curves.append(res['validation_1']['mlogloss'])
    xgb_train_acc_curves.append([1.0 - e for e in res['validation_0']['merror']])
    xgb_val_acc_curves.append([1.0 - e for e in res['validation_1']['merror']])

    xgb_val_accum  += clf.predict_proba(X_val.values,  iteration_range=(0, best_it + 1))
    xgb_test_accum += clf.predict_proba(X_test.values, iteration_range=(0, best_it + 1))

    clf.save_model(os.path.join(XGB_DIR, f"xgb_model_{i}.json"))

print("\nStep 3: Training 10 strongly perturbed MLPs (early stopping + best-weight restore)...")

base_lr = 1.0037e-05
base_wd = 2.5770e-05

base_h1         = 903
base_h2_mult    = 2.0
base_p_drop     = 0.2637

class DynamicMLP(nn.Module):
    def __init__(self, d_in, h1, h2, c_out, p_drop):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(p_drop),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(p_drop),

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
            probs = torch.softmax(model(x), 1).cpu().numpy()
            prob_list.append(probs)

    return np.vstack(prob_list)

for i in range(10):
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
        h1_dims = int(np.clip(
            base_h1 + np.random.choice([-384, -256, -128, 0, 128, 256, 384]),
            384,
            1536
        ))

        h2_multiplier = float(np.random.uniform(1.35, 2.75))
        h2_dims = int(np.clip(
            h1_dims * h2_multiplier,
            512,
            3072
        ))

        p_drop = float(np.clip(
            base_p_drop * np.random.uniform(0.45, 1.75),
            0.10,
            0.48
        ))

        p_lr = float(np.clip(
            base_lr * np.random.uniform(0.35, 2.50),
            3.0e-06,
            3.0e-05
        ))

        p_wd = float(np.clip(
            base_wd * np.random.uniform(0.30, 3.00),
            5.0e-06,
            1.0e-04
        ))

    net = DynamicMLP(
        input_dim,
        h1_dims,
        h2_dims,
        num_classes,
        p_drop
    ).to(device)

    optimizer = optim.Adam(
        net.parameters(),
        lr=p_lr,
        weight_decay=p_wd
    )

    criterion = nn.CrossEntropyLoss()

    this_train_loss, this_val_loss = [], []
    this_train_acc,  this_val_acc  = [], []

    best_val_loss     = float("inf")
    best_epoch        = -1
    best_state        = None
    epochs_no_improve = 0
    stopped_epoch     = MLP_MAX_EPOCHS

    for epoch in range(MLP_MAX_EPOCHS):
        net.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = net(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * len(xb)
            _, predicted = torch.max(outputs.data, 1)
            train_total += yb.size(0)
            train_correct += (predicted == yb).sum().item()

        this_train_loss.append(epoch_train_loss / len(train_loader.dataset))
        this_train_acc.append(train_correct / train_total)

        net.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                outputs = net(xb)
                loss = criterion(outputs, yb)

                epoch_val_loss += loss.item() * len(xb)
                _, predicted = torch.max(outputs.data, 1)
                val_total += yb.size(0)
                val_correct += (predicted == yb).sum().item()

        val_loss_epoch = epoch_val_loss / len(val_loader.dataset)
        this_val_loss.append(val_loss_epoch)
        this_val_acc.append(val_correct / val_total)

        if val_loss_epoch < best_val_loss - MLP_MIN_DELTA:
            best_val_loss     = val_loss_epoch
            best_epoch        = epoch
            best_state        = copy.deepcopy(net.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) >= MLP_WARMUP and epochs_no_improve >= MLP_PATIENCE:
            stopped_epoch = epoch + 1
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    print(
        f"  ├─ MLP #{i+1} "
        f"[Layers: ({h1_dims} -> {h2_dims}), Dropout: {p_drop:.3f}] "
        f"-> ran {stopped_epoch} epochs, best epoch {best_epoch + 1} "
        f"(val loss {best_val_loss:.4f}){' [early stopped]' if stopped_epoch < MLP_MAX_EPOCHS else ''}"
    )

    mlp_train_loss_curves.append(this_train_loss)
    mlp_val_loss_curves.append(this_val_loss)
    mlp_train_acc_curves.append(this_train_acc)
    mlp_val_acc_curves.append(this_val_acc)

    mlp_val_accum += extract_mlp_probabilities(net, val_loader)
    mlp_test_accum += extract_mlp_probabilities(net, test_loader)

    arch = {
        "index":         i,
        "seed":          m_seed,
        "h1":            h1_dims,
        "h2":            h2_dims,
        "p_drop":        p_drop,
        "input_dim":     input_dim,
        "num_classes":   num_classes,
        "best_epoch":    best_epoch,
        "epochs_run":    stopped_epoch,
        "best_val_loss": best_val_loss
    }
    mlp_archs.append(arch)

    torch.save(
        {
            "model_state_dict": net.state_dict(),
            **arch
        },
        os.path.join(MLP_DIR, f"mlp_model_{i}.pt")
    )

print("\nStep 4: Selecting XGB/MLP blend weight via validation grid search (10% steps)...")

mean_xgb_val  = xgb_val_accum / 10.0
mean_xgb_test = xgb_test_accum / 10.0

mean_mlp_val  = mlp_val_accum / 10.0
mean_mlp_test = mlp_test_accum / 10.0

blend_weights = np.round(np.arange(0.0, 1.0 + 1e-9, BLEND_STEP), 4)
blend_sweep   = []

for w in blend_weights:
    val_proba = (w * mean_xgb_val) + ((1.0 - w) * mean_mlp_val)
    val_top1  = accuracy_score(y_val, val_proba.argmax(1))
    blend_sweep.append((float(round(w, 4)), float(round(1.0 - w, 4)), float(val_top1)))
    print(f"  ├─ XGB {w:.1f} / MLP {1.0 - w:.1f}  ->  Val Top-1: {val_top1:.4f}")

best_idx      = int(np.argmax([s[2] for s in blend_sweep]))
XGB_WEIGHT    = blend_sweep[best_idx][0]
MLP_WEIGHT    = blend_sweep[best_idx][1]
best_val_top1 = blend_sweep[best_idx][2]

print(f"  └─ Selected blend: XGB {XGB_WEIGHT:.1f} / MLP {MLP_WEIGHT:.1f} "
      f"(Val Top-1: {best_val_top1:.4f})")

final_val_proba  = (XGB_WEIGHT * mean_xgb_val)  + (MLP_WEIGHT * mean_mlp_val)
final_test_proba = (XGB_WEIGHT * mean_xgb_test) + (MLP_WEIGHT * mean_mlp_test)

final_val_pred  = final_val_proba.argmax(1)
final_test_pred = final_test_proba.argmax(1)

xgb_test_pred = mean_xgb_test.argmax(1)
mlp_test_pred = mean_mlp_test.argmax(1)

val_accuracy  = accuracy_score(y_val,  final_val_pred)
test_accuracy = accuracy_score(y_test, final_test_pred)

print("\nStep 5: Saving artifacts for downstream ablation + SHAP...")

saved_df = X.copy()
saved_df.insert(0, target_col, df.loc[saved_df.index, target_col].values)
saved_df.to_parquet(os.path.join(SAVE_DIR, "MSKMET_Curated_Final.parquet"))

np.save(os.path.join(SAVE_DIR, "train_indices.npy"), train_ids)
np.save(os.path.join(SAVE_DIR, "val_indices.npy"),   val_ids)
np.save(os.path.join(SAVE_DIR, "test_indices.npy"),  test_ids)

ensemble_info = {
    "xgb_weight":       XGB_WEIGHT,
    "mlp_weight":       MLP_WEIGHT,
    "best_weight_xgb":  XGB_WEIGHT,
    "best_weight_mlp":  MLP_WEIGHT,
    "blend_sweep":      blend_sweep,
    "blend_step":       BLEND_STEP,
    "blend_val_top1":   best_val_top1,
    "blend_selection":  "val_top1",
    "feature_columns":  list(keep_cols),
    "n_xgb":            10,
    "n_mlp":            10,
    "mlp_archs":        mlp_archs,
    "xgb_best_iters":   xgb_best_iters,
    "num_classes":      num_classes,
    "target_col":       target_col,
    "input_dim":        input_dim,
    "early_stopping": {
        "xgb_rounds":   XGB_EARLY_STOPPING_ROUNDS,
        "mlp_warmup":   MLP_WARMUP,
        "mlp_patience": MLP_PATIENCE,
        "mlp_min_delta": MLP_MIN_DELTA,
        "mlp_max_epochs": MLP_MAX_EPOCHS
    }
}
pd.to_pickle(ensemble_info, os.path.join(SAVE_DIR, "ensemble_info.pkl"))
pd.to_pickle(label_encoder, os.path.join(SAVE_DIR, "label_encoder.pkl"))

print(f"  Split indices   : train/val/test .npy")
print(f"  Ensemble info   : {SAVE_DIR}/ensemble_info.pkl")
print(f"  Label encoder   : {SAVE_DIR}/label_encoder.pkl")
print(f"  XGB models      : {XGB_DIR}/xgb_model_0..9.json")
print(f"  MLP models      : {MLP_DIR}/mlp_model_0..9.pt")

def plot_member_curves(curves, xlabel, ylabel, title, outpath):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    for i, curve in enumerate(curves):
        ax.plot(
            range(1, len(curve) + 1), curve,
            color=cmap(i % 10), linewidth=1.8, alpha=0.9,
            label=f'Model {i + 1}'
        )
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=9, frameon=True, shadow=True, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()

print("\nGenerating XGBoost training curves (one line per model)...")
plot_member_curves(
    xgb_val_loss_curves, 'Boosting Round', 'Validation Log Loss',
    'XGBoost Validation Loss per Model', '/content/xgb_loss_curve.png'
)
plot_member_curves(
    xgb_val_acc_curves, 'Boosting Round', 'Validation Accuracy',
    'XGBoost Validation Accuracy per Model', '/content/xgb_accuracy_curve.png'
)

print("\nGenerating MLP training curves (one line per model)...")
plot_member_curves(
    mlp_val_loss_curves, 'Epoch', 'Validation Cross-Entropy Loss',
    'MLP Validation Loss per Model', '/content/mlp_loss_curve.png'
)
plot_member_curves(
    mlp_val_acc_curves, 'Epoch', 'Validation Accuracy',
    'MLP Validation Accuracy per Model', '/content/mlp_accuracy_curve.png'
)

print("\nGenerating blend-weight grid-search figure...")

sweep_w   = [s[0] for s in blend_sweep]
sweep_t1  = [s[2] * 100 for s in blend_sweep]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sweep_w, sweep_t1, marker='o', linewidth=2.5, markersize=8,
        color=COLORS['dark_blue'], label='Validation Top-1')
ax.scatter([XGB_WEIGHT], [best_val_top1 * 100], s=160, zorder=5,
           color=COLORS['light_blue_2'], edgecolor='black', linewidth=1.4,
           label=f'Selected (XGB {XGB_WEIGHT:.1f} / MLP {MLP_WEIGHT:.1f})')
ax.axvline(XGB_WEIGHT, color=COLORS['light_blue_2'], linestyle='--', linewidth=1.8, alpha=0.7)
ax.set_xlabel('XGBoost Blend Weight  (MLP weight = 1 - w)', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Ensemble Blend-Weight Grid Search (10% Steps)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(sweep_w)
ax.legend(fontsize=12, frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/content/blend_sweep.png', dpi=300, bbox_inches='tight')
print("Saved: /content/blend_sweep.png")
plt.close()

print("\nGenerating confusion matrix...")

colors_list = [
    '#E8F4F8',
    '#B8D9E8',
    '#88BED8',
    '#5AA3C8',
    '#3D88B8',
    '#2A6D9D',
    '#1e5c8e',
    '#143D5E',
    '#0A1E2E'
]
custom_cmap = LinearSegmentedColormap.from_list('custom_blues', colors_list)

cm = confusion_matrix(y_test, final_test_pred)

fig, ax = plt.subplots(figsize=(14, 12))

im = ax.imshow(cm, interpolation='nearest', cmap=custom_cmap, norm=LogNorm(vmin=max(cm.min(), 0.5), vmax=cm.max()))

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

cancer_types = label_encoder.classes_
tick_marks = np.arange(len(cancer_types))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(cancer_types, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(cancer_types, fontsize=9)

ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix - Ensemble Model', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: /content/confusion_matrix.png")
plt.close()

print("\n" + "="*80)
print("FINAL TEST RESULTS")
print("="*80)

print("\n[XGBOOST] (mean of 10 models, best iteration each)")
print(f"Top-1 Accuracy: {accuracy_score(y_test, xgb_test_pred):.4f}")
print(f"Top-3 Accuracy: {top_k_accuracy_score(y_test, mean_xgb_test, k=3):.4f}")
print(f"Weighted F1:    {f1_score(y_test, xgb_test_pred, average='weighted'):.4f}")

print("\n[MLP] (mean of 10 models, best epoch each)")
print(f"Top-1 Accuracy: {accuracy_score(y_test, mlp_test_pred):.4f}")
print(f"Top-3 Accuracy: {top_k_accuracy_score(y_test, mean_mlp_test, k=3):.4f}")
print(f"Weighted F1:    {f1_score(y_test, mlp_test_pred, average='weighted'):.4f}")

print("\n[ENSEMBLE] (blend weight selected on validation)")
print(f"Selected Weight (XGB): {XGB_WEIGHT:.2f}")
print(f"Selected Weight (MLP): {MLP_WEIGHT:.2f}")
print(f"Validation Top-1 Accuracy : {val_accuracy:.4f}")
print(f"Top-1 Accuracy            : {test_accuracy:.4f}")
print(f"Top-3 Accuracy            : {top_k_accuracy_score(y_test, final_test_proba, k=3):.4f}")
print(f"Top-5 Accuracy            : {top_k_accuracy_score(y_test, final_test_proba, k=5):.4f}")
print(f"Weighted F1-Score         : {f1_score(y_test, final_test_pred, average='weighted'):.4f}")

print("\nGenerating performance metrics comparison chart...")

models = ['XGBoost', 'MLP', 'Ensemble']
top1_acc = [
    accuracy_score(y_test, xgb_test_pred) * 100,
    accuracy_score(y_test, mlp_test_pred) * 100,
    accuracy_score(y_test, final_test_pred) * 100
]
top3_acc = [
    top_k_accuracy_score(y_test, mean_xgb_test, k=3) * 100,
    top_k_accuracy_score(y_test, mean_mlp_test, k=3) * 100,
    top_k_accuracy_score(y_test, final_test_proba, k=3) * 100
]
f1_scores = [
    f1_score(y_test, xgb_test_pred, average='weighted') * 100,
    f1_score(y_test, mlp_test_pred, average='weighted') * 100,
    f1_score(y_test, final_test_pred, average='weighted') * 100
]

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, top1_acc, width, label='Top-1 Accuracy',
               color=COLORS['dark_blue'], edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, top3_acc, width, label='Top-3 Accuracy',
               color=COLORS['light_blue_1'], edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, f1_scores, width, label='Weighted F1 Score',
               color=COLORS['light_blue_3'], edgecolor='black', linewidth=1.2)

def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.legend(fontsize=12, frameon=True, shadow=True, loc='lower right')
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('/content/model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: /content/model_performance_comparison.png")
plt.close()

print(f"\nPipeline Process Finished In: {(time.time() - start_time) / 60:.2f} minutes")

print("\n" + "="*80)
print("SAVED OUTPUTS")
print("="*80)
print("Figures:")
print("  /content/xgb_loss_curve.png")
print("  /content/xgb_accuracy_curve.png")
print("  /content/mlp_loss_curve.png")
print("  /content/mlp_accuracy_curve.png")
print("  /content/blend_sweep.png")
print("  /content/confusion_matrix.png")
print("  /content/model_performance_comparison.png")
print("\nArtifacts (for ablation + SHAP):")
print(f"  {SAVE_DIR}/train_indices.npy | val_indices.npy | test_indices.npy")
print(f"  {SAVE_DIR}/ensemble_info.pkl")
print(f"  {SAVE_DIR}/label_encoder.pkl")
print(f"  {XGB_DIR}/xgb_model_0..9.json")
print(f"  {MLP_DIR}/mlp_model_0..9.pt")
print("="*80)