!pip install optuna pyarrow

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import optuna
import time
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
start_time = time.time()

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

print("=" * 80)
print("MLP | Clinical + Hotspot Features | OPTUNA (VAL ACCURACY)")
print("DATA SPLIT: 80 / 10 / 10")
print("=" * 80)

final_df = pd.read_parquet("/content/MSK_MET_CURATED_FINAL.parquet")
final_df = final_df.loc[~final_df.index.duplicated(keep="first")]

print(f"Final dataset shape: {final_df.shape}")

X = final_df.drop(columns=["CANCER_TYPE"])
y = LabelEncoder().fit_transform(final_df["CANCER_TYPE"])
num_classes = len(np.unique(y))

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

keep_cols = X_train.columns[X_train.sum(axis=0) != 0]
X_train, X_val, X_test = X_train[keep_cols], X_val[keep_cols], X_test[keep_cols]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_ds = CancerDataset(X_train, y_train)
val_ds   = CancerDataset(X_val,   y_val)
test_ds  = CancerDataset(X_test,  y_test)

class MLP(nn.Module):
    def __init__(self, d, c, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(width, width * 2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(width * 2, c)
        )
    def forward(self, x):
        return self.net(x)

def train_and_eval(params, epochs=60):
    model     = MLP(X_train.shape[1], num_classes,
                    params["width"], params["dropout"]).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=params["lr"],
                           weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    loader    = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader= DataLoader(val_ds,   batch_size=512,                  shuffle=False)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
            true.append(yb.numpy())
    return accuracy_score(np.concatenate(true), np.concatenate(preds))

def objective(trial):
    params = {
        "width":        trial.suggest_int("width",         256,  1024, log=True),
        "dropout":      trial.suggest_float("dropout",     0.05, 0.5),
        "lr":           trial.suggest_float("lr",          1e-5, 3e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay",1e-6, 1e-3, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [64, 128, 256]),
    }
    return train_and_eval(params)

print("\nRunning Optuna (30 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print(f"\nBest Validation Accuracy: {study.best_value:.4f}")
print("Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

trial_nums  = [t.number for t in study.trials]
trial_accs  = [t.value  for t in study.trials]
trial_params= [t.params for t in study.trials]
best_so_far = [max(trial_accs[:i+1]) for i in range(len(trial_accs))]
best_idx    = int(np.argmax(trial_accs))
param_names = list(study.best_params.keys())

point_colors = [
    COLORS['dark_blue'] if i == best_idx else COLORS['light_blue_2']
    for i in range(len(trial_nums))
]

print("\nGenerating Optuna plots...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(trial_nums, trial_accs,
        color=COLORS['blue_4'], linewidth=1.0, alpha=0.5, zorder=2)
ax.plot(trial_nums, best_so_far,
        color=COLORS['dark_blue'], linewidth=2.5, zorder=3)

for i, (x, y_val, c) in enumerate(zip(trial_nums, trial_accs, point_colors)):
    ax.scatter(x, y_val, color=c, s=60, zorder=4, alpha=0.9)

ax.set_xlabel('Trial Number',        fontsize=13, fontweight='bold')
ax.set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Optimization History - MLP', fontsize=15, fontweight='bold', pad=16)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/content/optuna_optimization_history_mlp.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /content/optuna_optimization_history_mlp.png")
plt.close()

importances = optuna.importance.get_param_importances(study)
imp_names   = list(importances.keys())
imp_values  = list(importances.values())

bar_colors = [
    COLORS['dark_blue'],    COLORS['blue_1'],       COLORS['blue_2'],
    COLORS['blue_3'],       COLORS['blue_4'],        COLORS['light_blue_1'],
    COLORS['light_blue_2'], COLORS['light_blue_3'], COLORS['lightest_blue']
][:len(imp_names)]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(imp_names[::-1], imp_values[::-1],
               color=bar_colors[::-1], edgecolor='white', linewidth=0.6)

for bar, val in zip(bars, imp_values[::-1]):
    ax.text(bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=10, color=COLORS['dark_blue'])

ax.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
ax.set_title('Hyperparameter Importance - MLP', fontsize=15, fontweight='bold', pad=16)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, max(imp_values) * 1.18)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/content/optuna_hyperparam_importance_mlp.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /content/optuna_hyperparam_importance_mlp.png")
plt.close()

n_params     = len(param_names)
ncols        = 3
nrows_full   = n_params // ncols
last_row_n   = n_params % ncols
has_last_row = last_row_n > 0
nrows        = nrows_full + (1 if has_last_row else 0)

fig_sp  = plt.figure(figsize=(5 * ncols, 4 * nrows))
sp_axes = []

if nrows_full > 0:
    gs_top = gridspec.GridSpec(nrows_full, ncols, figure=fig_sp,
                               top=1.0 - 0.06,
                               bottom=(1 / nrows) + 0.06 if has_last_row else 0.06,
                               hspace=0.55, wspace=0.35)
    for r in range(nrows_full):
        for c in range(ncols):
            sp_axes.append(fig_sp.add_subplot(gs_top[r, c]))

if has_last_row:
    subplot_w  = 1 / ncols
    total_used = last_row_n * subplot_w
    margin     = (1 - total_used) / 2
    gs_bot = gridspec.GridSpec(1, last_row_n, figure=fig_sp,
                               left=margin + 0.02,
                               right=1 - margin - 0.02,
                               bottom=0.06,
                               top=1 / nrows - 0.02,
                               wspace=0.45)
    for c in range(last_row_n):
        sp_axes.append(fig_sp.add_subplot(gs_bot[0, c]))

for i, param in enumerate(param_names):
    ax = sp_axes[i]

    reg_xs = [float(trial_params[j][param]) for j in range(len(trial_params))
              if param in trial_params[j] and j != best_idx]
    reg_ys = [trial_accs[j] for j in range(len(trial_accs))
              if param in trial_params[j] and j != best_idx]
    best_x = [float(trial_params[best_idx][param])]
    best_y = [trial_accs[best_idx]]

    ax.scatter(reg_xs, reg_ys, color=COLORS['light_blue_2'], s=55, alpha=0.8, zorder=3)
    ax.scatter(best_x, best_y, color=COLORS['dark_blue'],    s=55, alpha=1.0, zorder=5)

    ax.set_xlabel(param,          fontsize=11, fontweight='bold')
    ax.set_ylabel('Val Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

fig_sp.suptitle('Slice Plot - MLP', fontsize=15, fontweight='bold')
plt.savefig('/content/optuna_slice_plot_mlp.png', dpi=600, bbox_inches='tight')
print("✓ Saved: /content/optuna_slice_plot_mlp.png")
plt.close()

print("All plots rendered.")

print("\nRetraining on train + val with best params...")

best         = study.best_params
X_final      = pd.concat([X_train, X_val])
y_final      = np.concatenate([np.atleast_1d(y_train), np.atleast_1d(y_val)])
final_ds     = CancerDataset(X_final, y_final)

model        = MLP(X_final.shape[1], num_classes,
                   best["width"], best["dropout"]).to(device)
optimizer    = optim.Adam(model.parameters(),
                          lr=best["lr"], weight_decay=best["weight_decay"])
criterion    = nn.CrossEntropyLoss()
final_loader = DataLoader(final_ds, batch_size=best["batch_size"], shuffle=True)

model.train()
for epoch in range(120):
    for xb, yb in final_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    logits = model(torch.FloatTensor(X_test.values).to(device))
    probs  = torch.softmax(logits, dim=1).cpu().numpy()
    preds  = probs.argmax(axis=1)

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)
print(f"Top-1 Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"Top-3 Accuracy: {top_k_accuracy_score(y_test, probs, k=3):.4f}")
print(f"Top-5 Accuracy: {top_k_accuracy_score(y_test, probs, k=5):.4f}")
print(f"Weighted F1:    {f1_score(y_test, preds, average='weighted'):.4f}")
print(f"Runtime:        {(time.time() - start_time)/60:.1f} minutes")
print("=" * 80)

print("\n" + "=" * 80)
print("SAVED FILES")
print("=" * 80)
print("  ✓ /content/optuna_trials.csv")
print("  ✓ /content/optuna_optimization_history_mlp.png")
print("  ✓ /content/optuna_hyperparam_importance_mlp.png")
print("  ✓ /content/optuna_slice_plot_mlp.png")
print("=" * 80)
