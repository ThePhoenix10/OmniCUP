import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import optuna
import time
import json
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(42)

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
print("XGBoost | Clinical + Hotspot Features | OPTUNA (VAL ACCURACY)")
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

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)


def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
        'max_depth': trial.suggest_int('max_depth', 7, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 400, 700),
        'subsample': trial.suggest_float('subsample', 0.75, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 4),
        'gamma': trial.suggest_float('gamma', 0, 0.3),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.8, 2.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.8),
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train.values, y_train, sample_weight=sample_weights, verbose=False)
    y_val_pred = model.predict(X_val.values)
    return accuracy_score(y_val, y_val_pred)


print("\nRunning Optuna (30 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print(f"\nBest Validation Accuracy: {study.best_value:.4f}")
print("Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

trial_nums   = [t.number for t in study.trials]
trial_accs   = [t.value  for t in study.trials]
trial_params = [t.params for t in study.trials]
best_so_far  = [max(trial_accs[:i+1]) for i in range(len(trial_accs))]
best_idx     = int(np.argmax(trial_accs))
param_names  = list(study.best_params.keys())

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
for i, (x, y_val_pt, c) in enumerate(zip(trial_nums, trial_accs, point_colors)):
    ax.scatter(x, y_val_pt, color=c, s=60, zorder=4, alpha=0.9)
ax.set_xlabel('Trial Number',        fontsize=13, fontweight='bold')
ax.set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Optimization History - XGBoost', fontsize=15, fontweight='bold', pad=16)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/content/optuna_optimization_history_xgb.png', dpi=300, bbox_inches='tight')
print("Saved: /content/optuna_optimization_history_xgb.png")
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
ax.set_title('Hyperparameter Importance - XGBoost', fontsize=15, fontweight='bold', pad=16)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, max(imp_values) * 1.18)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/content/optuna_hyperparam_importance_xgb.png', dpi=300, bbox_inches='tight')
print("Saved: /content/optuna_hyperparam_importance_xgb.png")
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

fig_sp.suptitle('Slice Plot - XGBoost', fontsize=15, fontweight='bold')
plt.savefig('/content/optuna_slice_plot_xgb.png', dpi=600, bbox_inches='tight')
print("Saved: /content/optuna_slice_plot_xgb.png")
plt.close()

print("All plots rendered.")

with open("/content/best_xgb_params.json", "w") as f:
    json.dump(dict(study.best_params), f, indent=2)

print(f"\nRuntime: {(time.time() - start_time)/60:.1f} minutes")

print("\n" + "=" * 80)
print("SAVED FILES")
print("=" * 80)
print("  /content/optuna_optimization_history_xgb.png")
print("  /content/optuna_hyperparam_importance_xgb.png")
print("  /content/optuna_slice_plot_xgb.png")
print("  /content/best_xgb_params.json")
print("=" * 80)