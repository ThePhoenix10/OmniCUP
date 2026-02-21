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

warnings.filterwarnings("ignore")

COLORS = {
    'dark_blue': '#1e5c8e',
    'blue_1': '#2a6d9d',
    'blue_2': '#367eac',
    'blue_3': '#4490ba',
    'blue_4': '#53a2c8',
    'light_blue_1': '#63b4d6',
    'light_blue_2': '#74c6e4',
    'light_blue_3': '#86d8f1',
    'lightest_blue': '#99ebff'
}

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
start_time = time.time()

print("="*80)
print("XGBoost + MLP Ensemble | FIXED HYPERPARAMETERS")
print("DATA SPLIT: 80 / 10 / 10")
print("="*80)

final_df = pd.read_parquet("/content/MSK_MET_CURATED_FINAL.parquet")
final_df = final_df.loc[~final_df.index.duplicated(keep="first")]

print(f"Final dataset shape: {final_df.shape}")

X = final_df.drop(columns=["CANCER_TYPE"])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(final_df["CANCER_TYPE"])
num_classes = len(np.unique(y))

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

keep_cols = X_train.columns[X_train.sum(axis=0) != 0]
X_train, X_val, X_test = X_train[keep_cols], X_val[keep_cols], X_test[keep_cols]

print("=" * 80)
print(f"Number of features after zero-variance filtering: {len(keep_cols)}")
print("=" * 80)

aa_hotspot_cols = [c for c in keep_cols if c.endswith("_AA_HOTSPOT")]
splice_hotspot_cols = [c for c in keep_cols if c.endswith("_SPLICE_HOTSPOT")]

gene_hotspot_cols = [
    c for c in keep_cols
    if c.endswith("_HOTSPOT")
    and not c.endswith("_AA_HOTSPOT")
    and not c.endswith("_SPLICE_HOTSPOT")
]

total_hotspot_cols = (
    len(gene_hotspot_cols)
    + len(aa_hotspot_cols)
    + len(splice_hotspot_cols)
)

print("=" * 80)
print("HOTSPOT FEATURE SUMMARY (AFTER ZERO-VARIANCE FILTERING)")
print("=" * 80)
print(f"Total hotspot features        : {total_hotspot_cols}")
print(f"  ├─ Gene-level hotspots      : {len(gene_hotspot_cols)}")
print(f"  ├─ Amino-acid (AA) hotspots : {len(aa_hotspot_cols)}")
print(f"  └─ Splice-site hotspots     : {len(splice_hotspot_cols)}")
print("=" * 80)

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
sample_weights = np.array([class_weights[i] for i in y_train])

print("\nTraining XGBoost...")

xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda",
    max_depth=9,
    learning_rate=0.066,
    n_estimators=446,
    subsample=0.807,
    colsample_bytree=0.752,
    min_child_weight=2,
    gamma=0.293,
    reg_lambda=1.835,
    reg_alpha=0.147,
    random_state=42,
)

eval_set = [(X_train.values, y_train), (X_val.values, y_val)]
xgb_model.fit(
    X_train.values,
    y_train,
    sample_weight=sample_weights,
    eval_set=eval_set,
    verbose=False
)

xgb_results = xgb_model.evals_result()
xgb_train_loss = xgb_results['validation_0']['mlogloss']
xgb_val_loss = xgb_results['validation_1']['mlogloss']

print("\nCalculating XGBoost accuracy per round...")
xgb_train_acc = []
xgb_val_acc = []

for i in range(1, len(xgb_train_loss) + 1):
    train_pred = xgb_model.predict(X_train.values, iteration_range=(0, i))
    val_pred = xgb_model.predict(X_val.values, iteration_range=(0, i))

    xgb_train_acc.append(accuracy_score(y_train, train_pred))
    xgb_val_acc.append(accuracy_score(y_val, val_pred))

    if i % 50 == 0:
        print(f"  XGBoost Round {i}/{len(xgb_train_loss)} - Train Acc: {xgb_train_acc[-1]:.4f}, Val Acc: {xgb_val_acc[-1]:.4f}")

xgb_val_proba = xgb_model.predict_proba(X_val.values)
xgb_test_proba = xgb_model.predict_proba(X_test.values)

class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

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
    def forward(self, x): return self.net(x)

mlp = MLP(X_train.shape[1], num_classes).to(device)

optimizer = optim.Adam(
    mlp.parameters(),
    lr=1.0037217774773265e-05,
    weight_decay=2.577022348345711e-05
)

criterion = nn.CrossEntropyLoss()

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

print("\nTraining MLP with tracking...")

mlp_train_losses = []
mlp_val_losses = []
mlp_train_acc = []
mlp_val_acc = []

num_epochs = 120

for epoch in range(num_epochs):
    mlp.train()
    epoch_train_loss = 0
    train_correct = 0
    train_total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = mlp(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * len(xb)
        _, predicted = torch.max(outputs.data, 1)
        train_total += yb.size(0)
        train_correct += (predicted == yb).sum().item()

    mlp_train_losses.append(epoch_train_loss / len(train_loader.dataset))
    mlp_train_acc.append(train_correct / train_total)

    mlp.eval()
    epoch_val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = mlp(xb)
            loss = criterion(outputs, yb)

            epoch_val_loss += loss.item() * len(xb)
            _, predicted = torch.max(outputs.data, 1)
            val_total += yb.size(0)
            val_correct += (predicted == yb).sum().item()

    mlp_val_losses.append(epoch_val_loss / len(val_loader.dataset))
    mlp_val_acc.append(val_correct / val_total)

    if (epoch + 1) % 20 == 0:
        print(f"  MLP Epoch {epoch+1}/{num_epochs} - Train Acc: {mlp_train_acc[-1]:.4f}, Val Acc: {mlp_val_acc[-1]:.4f}")

def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    return np.vstack(out)

mlp_val_proba = predict(mlp, val_loader)
mlp_test_proba = predict(
    mlp,
    DataLoader(CancerDataset(X_test, y_test), batch_size=512)
)

best_w = 0.6

xgb_pred = xgb_test_proba.argmax(1)
mlp_pred = mlp_test_proba.argmax(1)

final_proba = best_w * xgb_test_proba + (1 - best_w) * mlp_test_proba
final_pred = final_proba.argmax(1)

print("\nGenerating XGBoost training curves...")

epochs_xgb = range(1, len(xgb_train_loss) + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_xgb, xgb_train_loss,
        color=COLORS['dark_blue'], linewidth=2.5, label='Training Loss')
ax.plot(epochs_xgb, xgb_val_loss,
        color=COLORS['light_blue_2'], linewidth=2.5, label='Validation Loss')
ax.set_xlabel('Boosting Round', fontsize=14, fontweight='bold')
ax.set_ylabel('Log Loss', fontsize=14, fontweight='bold')
ax.set_title('XGBoost Training & Validation Loss', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/content/xgb_loss_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /content/xgb_loss_curve.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_xgb, xgb_train_acc,
        color=COLORS['dark_blue'], linewidth=2.5, label='Training Accuracy')
ax.plot(epochs_xgb, xgb_val_acc,
        color=COLORS['light_blue_2'], linewidth=2.5, label='Validation Accuracy')
ax.set_xlabel('Boosting Round', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('XGBoost Training & Validation Accuracy', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/content/xgb_accuracy_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /content/xgb_accuracy_curve.png")
plt.close()

print("\nGenerating MLP training curves...")

epochs_mlp = range(1, len(mlp_train_losses) + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_mlp, mlp_train_losses,
        color=COLORS['dark_blue'], linewidth=2.5, label='Training Loss')
ax.plot(epochs_mlp, mlp_val_losses,
        color=COLORS['light_blue_2'], linewidth=2.5, label='Validation Loss')
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Cross-Entropy Loss', fontsize=14, fontweight='bold')
ax.set_title('MLP Training & Validation Loss', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/content/mlp_loss_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /content/mlp_loss_curve.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_mlp, mlp_train_acc,
        color=COLORS['dark_blue'], linewidth=2.5, label='Training Accuracy')
ax.plot(epochs_mlp, mlp_val_acc,
        color=COLORS['light_blue_2'], linewidth=2.5, label='Validation Accuracy')
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('MLP Training & Validation Accuracy', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/content/mlp_accuracy_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /content/mlp_accuracy_curve.png")
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

cm = confusion_matrix(y_test, final_pred)

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
print("✓ Saved: /content/confusion_matrix.png")
plt.close()

print("\n" + "="*80)
print("FINAL TEST RESULTS")
print("="*80)

print("\n[XGBOOST]")
print(f"Top-1 Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"Top-3 Accuracy: {top_k_accuracy_score(y_test, xgb_test_proba, k=3):.4f}")
print(f"Weighted F1:    {f1_score(y_test, xgb_pred, average='weighted'):.4f}")

print("\n[MLP]")
print(f"Top-1 Accuracy: {accuracy_score(y_test, mlp_pred):.4f}")
print(f"Top-3 Accuracy: {top_k_accuracy_score(y_test, mlp_test_proba, k=3):.4f}")
print(f"Weighted F1:    {f1_score(y_test, mlp_pred, average='weighted'):.4f}")

print("\n[ENSEMBLE]")
print(f"Ensemble Weight (XGB): {best_w:.2f}")
print(f"Ensemble Weight (MLP): {1-best_w:.2f}")
print(f"Top-1 Accuracy: {accuracy_score(y_test, final_pred):.4f}")
print(f"Top-3 Accuracy: {top_k_accuracy_score(y_test, final_proba, k=3):.4f}")
print(f"Weighted F1:    {f1_score(y_test, final_pred, average='weighted'):.4f}")

print(f"\nRuntime: {(time.time()-start_time)/60:.1f} minutes")
print("="*80)

print("\nGenerating performance metrics comparison chart...")

models = ['XGBoost', 'MLP', 'Ensemble']
top1_acc = [
    accuracy_score(y_test, xgb_pred) * 100,
    accuracy_score(y_test, mlp_pred) * 100,
    accuracy_score(y_test, final_pred) * 100
]
top3_acc = [
    top_k_accuracy_score(y_test, xgb_test_proba, k=3) * 100,
    top_k_accuracy_score(y_test, mlp_test_proba, k=3) * 100,
    top_k_accuracy_score(y_test, final_proba, k=3) * 100
]
f1_scores = [
    f1_score(y_test, xgb_pred, average='weighted') * 100,
    f1_score(y_test, mlp_pred, average='weighted') * 100,
    f1_score(y_test, final_pred, average='weighted') * 100
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
print("✓ Saved: /content/model_performance_comparison.png")
plt.close()

print("\nSaving models...")

xgb_model.save_model("/content/xgb_model.json")

torch.save(
    {
        "model_state_dict": mlp.state_dict(),
        "input_dim": X_train.shape[1],
        "num_classes": num_classes
    },
    "/content/mlp_model.pt"
)

ensemble_info = {
    "best_weight_xgb": best_w,
    "best_weight_mlp": 1 - best_w,
    "feature_columns": list(keep_cols)
}

pd.to_pickle(ensemble_info, "/content/ensemble_info.pkl")
pd.to_pickle(label_encoder, "/content/label_encoder.pkl")

print("\n" + "="*80)
print("SAVED FILES")
print("="*80)
print("Models:")
print("  ✓ /content/xgb_model.json")
print("  ✓ /content/mlp_model.pt")
print("  ✓ /content/ensemble_info.pkl")
print("  ✓ /content/label_encoder.pkl")
print("\nPlots:")
print("  ✓ /content/xgb_loss_curve.png")
print("  ✓ /content/xgb_accuracy_curve.png")
print("  ✓ /content/mlp_loss_curve.png")
print("  ✓ /content/mlp_accuracy_curve.png")
print("  ✓ /content/confusion_matrix.png")
print("  ✓ /content/model_performance_comparison.png")
print("="*80)
