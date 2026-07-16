import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import os
import glob
import warnings
warnings.filterwarnings("ignore")

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

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_blues',
    ['#74c6e4','#5AA3C8','#3D88B8',
     '#2A6D9D','#1e5c8e','#143D5E','#0A1E2E']
)
device = "cuda" if torch.cuda.is_available() else "cpu"

ART_DIR = "/content/saved_artifacts"
XGB_DIR = os.path.join(ART_DIR, "xgb_models")
MLP_DIR = os.path.join(ART_DIR, "mlp_models")

N_XGB_SHAP = None
N_MLP_SHAP = None

print("Loading ensemble metadata, models, and data...")

ensemble_info = pd.read_pickle(os.path.join(ART_DIR, "ensemble_info.pkl"))
label_encoder = pd.read_pickle(os.path.join(ART_DIR, "label_encoder.pkl"))

feature_columns = ensemble_info["feature_columns"]

xgb_w = ensemble_info.get("xgb_weight", ensemble_info.get("best_weight_xgb", 0.6))
mlp_w = ensemble_info.get("mlp_weight", ensemble_info.get("best_weight_mlp", 0.4))
num_classes = ensemble_info["num_classes"]
TARGET_COL  = ensemble_info.get("target_col", "Primary_Site_Target")
n_xgb_total = ensemble_info.get("n_xgb", 10)
n_mlp_total = ensemble_info.get("n_mlp", 10)

xgb_best_iters = ensemble_info.get("xgb_best_iters", None)

n_xgb_use = n_xgb_total if N_XGB_SHAP is None else min(N_XGB_SHAP, n_xgb_total)
n_mlp_use = n_mlp_total if N_MLP_SHAP is None else min(N_MLP_SHAP, n_mlp_total)

cancer_names = label_encoder.classes_

xgb_models = []
for i in range(n_xgb_use):
    m = xgb.XGBClassifier()
    m.load_model(os.path.join(XGB_DIR, f"xgb_model_{i}.json"))
    xgb_models.append(m)
print(f"    Loaded {len(xgb_models)} XGBoost members")

def _xgb_proba(member_idx, model, X_values):
    if xgb_best_iters is not None and member_idx < len(xgb_best_iters):
        bi = int(xgb_best_iters[member_idx])
        return model.predict_proba(X_values, iteration_range=(0, bi + 1))
    return model.predict_proba(X_values)

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

mlp_models = []
for i in range(n_mlp_use):
    ckpt = torch.load(os.path.join(MLP_DIR, f"mlp_model_{i}.pt"), map_location=device)
    net = DynamicMLP(
        ckpt["input_dim"],
        ckpt["h1"],
        ckpt["h2"],
        ckpt["num_classes"],
        ckpt["p_drop"]
    ).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    mlp_models.append(net)
print(f"    Loaded {len(mlp_models)} MLP members")

print("\n[1/8] Loading processed dataset...")

final_df = pd.read_parquet(os.path.join(ART_DIR, "MSKMET_Curated_Final.parquet"))

train_idx = np.load(os.path.join(ART_DIR, "train_indices.npy"), allow_pickle=True)
val_idx   = np.load(os.path.join(ART_DIR, "val_indices.npy"),   allow_pickle=True)
test_idx  = np.load(os.path.join(ART_DIR, "test_indices.npy"),  allow_pickle=True)

y_all = label_encoder.transform(final_df[TARGET_COL])
X_all = final_df.drop(columns=[TARGET_COL])

X_train = X_all.loc[train_idx].copy()
X_val   = X_all.loc[val_idx].copy()
X_test  = X_all.loc[test_idx].copy()

id_to_pos = {sid: i for i, sid in enumerate(final_df.index)}
train_pos = np.array([id_to_pos[s] for s in train_idx])
val_pos   = np.array([id_to_pos[s] for s in val_idx])
test_pos  = np.array([id_to_pos[s] for s in test_idx])

y_train = y_all[train_pos]
y_val   = y_all[val_pos]
y_test  = y_all[test_pos]

X_train = X_train.reindex(columns=feature_columns, fill_value=0)
X_test  = X_test.reindex(columns=feature_columns, fill_value=0)

n_features = len(feature_columns)
n_test     = X_test.shape[0]

print(f"    Test set : {X_test.shape[0]} samples | {X_test.shape[1]} features | {num_classes} classes")
print(f"    Train set: {X_train.shape[0]} samples")
print(f"    Ensemble blend (grid-searched): XGB {xgb_w:.2f} / MLP {mlp_w:.2f}")

X_test_tensor = torch.FloatTensor(X_test.values).to(device)

def ensemble_proba(X_values, X_tensor):
    xgb_p = np.mean(
        [_xgb_proba(i, m, X_values) for i, m in enumerate(xgb_models)],
        axis=0
    )
    mlp_p_list = []
    with torch.no_grad():
        for net in mlp_models:
            mlp_p_list.append(
                torch.softmax(net(X_tensor), dim=1).cpu().detach().numpy()
            )
    mlp_p = np.mean(mlp_p_list, axis=0)
    return xgb_w * xgb_p + mlp_w * mlp_p, xgb_p, mlp_p

print("\n    Running sanity check...")
from sklearn.metrics import accuracy_score
_ens_proba, _, _ = ensemble_proba(X_test.values, X_test_tensor)
_ens_pred  = _ens_proba.argmax(axis=1)
_sanity = accuracy_score(y_test, _ens_pred)
print(f"    Sanity Top-1 accuracy: {_sanity:.4f}")
print(f"    (Should match the Top-1 test accuracy your ensemble pipeline reported.)")

MUT_SUBTYPE_MAP = {
    "_MISSENSE_MUTATION":      "Missense",
    "_NONSENSE_MUTATION":      "Nonsense",
    "_FRAME_SHIFT_INS":        "Frame Shift Ins",
    "_FRAME_SHIFT_DEL":        "Frame Shift Del",
    "_SPLICE_SITE":            "Splice Site",
    "_TRANSLATION_START_SITE": "Translation Start",
    "_IN_FRAME_INS":           "In Frame Ins",
    "_IN_FRAME_DEL":           "In Frame Del",
    "_NONSTOP_MUTATION":       "Nonstop",
}

MUT_SUBGROUP_ORDER = ["Any Mutation"] + list(MUT_SUBTYPE_MAP.values())

def assign_subgroup(col):
    n = col.upper()

    if n.endswith("_AA_HOTSPOT"):
        return "Amino Acid Hotspot"
    if n.endswith("_SPLICE_HOTSPOT"):
        return "Splice Hotspot"
    if n.endswith("_HOTSPOT"):
        return "Gene-Level Hotspot"

    for suf, name in MUT_SUBTYPE_MAP.items():
        if n.endswith(suf):
            return name
    if n.endswith("_ANY_SOM_MAT") or n == "IS_ANY_SOM_MAT":
        return "Any Mutation"

    if n.endswith("_ANY_CNA"):
        return "Any CNA"
    if n.endswith("_AMP"):
        return "Amplification"
    if n.endswith("_DEL"):
        return "Deletion"

    if n.endswith("_FUSION"):
        return "Fusion"

    if col == "AGE_AT_SEQUENCING":
        return "Age"
    if col == "SEX":
        return "Sex"
    if col == "TMB_NONSYNONYMOUS":
        return "TMB"
    if col == "MSI_SCORE":
        return "MSI"
    if n.startswith("DMETS"):
        return "Metastasis Sites"

    return None

SUBGROUP_TO_CATEGORY = {sg: "Mutation" for sg in MUT_SUBGROUP_ORDER}
SUBGROUP_TO_CATEGORY.update({
    "Amplification":       "Copy Number",
    "Deletion":            "Copy Number",
    "Any CNA":             "Copy Number",
    "Fusion":              "Fusion",
    "Gene-Level Hotspot":  "Hotspot Mutation",
    "Amino Acid Hotspot":  "Hotspot Mutation",
    "Splice Hotspot":      "Hotspot Mutation",
    "Metastasis Sites":    "Clinical",
    "Age":                 "Clinical",
    "Sex":                 "Clinical",
    "TMB":                 "Clinical",
    "MSI":                 "Clinical",
})

def feature_category(col):
    sg = assign_subgroup(col)
    if sg is None:
        return "Other"
    return SUBGROUP_TO_CATEGORY.get(sg, "Other")

print("\n" + "="*60)
print("FEATURE SPLIT DIAGNOSTIC")
print("="*60)

_counts = {}
for col in feature_columns:
    sg = assign_subgroup(col)
    key = sg if sg is not None else "(ignored)"
    _counts[key] = _counts.get(key, 0) + 1

_diag_order = (
    MUT_SUBGROUP_ORDER
    + ["Amplification", "Deletion", "Any CNA",
       "Fusion",
       "Gene-Level Hotspot", "Amino Acid Hotspot", "Splice Hotspot",
       "Metastasis Sites", "Age", "Sex", "TMB", "MSI",
       "(ignored)"]
)
for sg in _diag_order:
    flag = "" if _counts.get(sg, 0) > 0 else "   (none — auto-hidden in plots)"
    print(f"  {sg:<22}: {_counts.get(sg, 0):>5} features{flag}")

print(f"  {'─'*36}")
print(f"  {'TOTAL':<22}: {len(feature_columns):>5} features")

_cna_cols = [c for c in feature_columns if assign_subgroup(c) in ("Amplification", "Deletion", "Any CNA")]
_leaked   = [c for c in _cna_cols
             if any(c.upper().endswith(s) for s in MUT_SUBTYPE_MAP)]
if _leaked:
    print(f"\n  WARNING: {len(_leaked)} mutation-subtype columns mis-classified as CNA:")
    for c in _leaked[:10]:
        print(f"       {c}")
else:
    print(f"\n  No mutation subtype leaked into CNA branch — _DEL split is clean")

_present_subtypes = [s for s in MUT_SUBTYPE_MAP.values() if _counts.get(s, 0) > 0]
_absent_subtypes  = [s for s in MUT_SUBTYPE_MAP.values() if _counts.get(s, 0) == 0]
print(f"\n  Mutation subtypes present ({len(_present_subtypes)}): {', '.join(_present_subtypes)}")
if _absent_subtypes:
    print(f"  Mutation subtypes absent  ({len(_absent_subtypes)}): {', '.join(_absent_subtypes)}")

print("\n  Example columns per mutation subgroup:")
for sg in MUT_SUBGROUP_ORDER:
    ex = [c for c in feature_columns if assign_subgroup(c) == sg][:4]
    if ex:
        print(f"    {sg:<20}: {ex}")
print("="*60 + "\n")

def to_snc(arr, ns, nf, nc):
    arr = np.array(arr)
    if arr.ndim == 2:
        return arr[:, :, None]
    dims = list(arr.shape)
    targets = [ns, nf, nc]
    perm = []
    for t in targets:
        for ax, d in enumerate(dims):
            if d == t and ax not in perm:
                perm.append(ax)
                break
    if len(perm) != 3:

        return arr
    return np.transpose(arr, perm)

print("[2/8] Computing XGBoost SHAP values (TreeExplainer, averaged over members)...")
xgb_shap_sum = None
for i, m in enumerate(xgb_models):
    expl = shap.TreeExplainer(m)
    sv = to_snc(expl.shap_values(X_test), n_test, n_features, num_classes).astype(np.float32)
    xgb_shap_sum = sv if xgb_shap_sum is None else xgb_shap_sum + sv
    print(f"    XGB member {i+1}/{len(xgb_models)} done")
mean_xgb_shap = xgb_shap_sum / float(len(xgb_models))
del xgb_shap_sum
print(f"    XGBoost SHAP averaged — shape: {mean_xgb_shap.shape}")

print("\n[3/8] Computing MLP SHAP values (DeepExplainer, averaged over members)...")
BG_PER_CLASS = 20
print(f"    Building stratified background dataset ({BG_PER_CLASS} patients per cancer type)...")
np.random.seed(42)
bg_rows = []
for c in range(num_classes):
    class_idx = np.where(y_train == c)[0]
    n = min(BG_PER_CLASS, len(class_idx))
    if n == 0:
        continue
    chosen = np.random.choice(class_idx, size=n, replace=False)
    bg_rows.append(X_train.iloc[chosen].values)
bg_matrix = np.vstack(bg_rows)
bg_data   = torch.FloatTensor(bg_matrix).to(device)
print(f"    Background size: {bg_data.shape[0]} samples")
print(f"    NOTE: running DeepExplainer on {len(mlp_models)} MLPs — this can take a while.")

mlp_shap_sum = None
for i, net in enumerate(mlp_models):
    expl = shap.DeepExplainer(net, bg_data)
    sv = to_snc(expl.shap_values(X_test_tensor), n_test, n_features, num_classes).astype(np.float32)
    mlp_shap_sum = sv if mlp_shap_sum is None else mlp_shap_sum + sv
    print(f"    MLP member {i+1}/{len(mlp_models)} done")
mean_mlp_shap = mlp_shap_sum / float(len(mlp_models))
del mlp_shap_sum
print(f"    MLP SHAP averaged — shape: {mean_mlp_shap.shape}")

print(f"\n[4/8] Blending SHAP with ensemble weights ({xgb_w:.2f} XGB / {mlp_w:.2f} MLP)...")

ensemble_shap = [
    xgb_w * mean_xgb_shap[:, :, c] + mlp_w * mean_mlp_shap[:, :, c]
    for c in range(num_classes)
]
ensemble_shap_arr = np.array(ensemble_shap)
mean_abs_shap     = np.mean(np.abs(ensemble_shap_arr), axis=0)
del mean_xgb_shap, mean_mlp_shap
print(f"    Ensemble SHAP — {num_classes} classes x {mean_abs_shap.shape[0]} samples x {mean_abs_shap.shape[1]} features")

feature_importance = pd.Series(
    mean_abs_shap.mean(axis=0),
    index=feature_columns
).sort_values(ascending=False)

TOP_N_GLOBAL = 30

top_features_global = feature_importance.head(TOP_N_GLOBAL).index.tolist()

print(f"\nTop 5 global features by mean |SHAP|:")
for f, v in feature_importance.head(5).items():
    print(f"  {f}: {v:.4f}")

print("\n[5/8] Generating global plots (Beeswarm + Bar)...")

_col_pos = {c: j for j, c in enumerate(feature_columns)}
shap_explanation = shap.Explanation(
    values=mean_abs_shap[:, [_col_pos[x] for x in top_features_global]],
    base_values=np.zeros(len(X_test)),
    data=X_test[top_features_global].values,
    feature_names=top_features_global
)

fig, ax = plt.subplots(figsize=(11, 9))
shap.plots.beeswarm(shap_explanation, max_display=TOP_N_GLOBAL, show=False,
                    color_bar=True, color=custom_cmap)
plt.title("SHAP Analysis on All Features",
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("/content/shap_beeswarm_global.png", dpi=300, bbox_inches='tight')
print("    Saved: shap_beeswarm_global.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))

top_feats = feature_importance.head(TOP_N_GLOBAL)
colors = [COLORS['dark_blue'] if i < 5 else
          COLORS['blue_2']    if i < 15 else
          COLORS['light_blue_2']
          for i in range(len(top_feats))]

bars = ax.barh(top_feats.index[::-1], top_feats.values[::-1],
               color=colors[::-1], edgecolor='white', linewidth=0.4)

ax.set_xlabel("Mean |SHAP Value|", fontsize=12, fontweight='bold')
ax.set_title("Global Feature Importance",
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig("/content/shap_bar_global.png", dpi=300, bbox_inches='tight')
print("    Saved: shap_bar_global.png")
plt.close()

print("\nComputing ensemble probabilities for all test samples...")
ens_proba, xgb_proba_full, mlp_proba_full = ensemble_proba(X_test.values, X_test_tensor)
ens_pred     = ens_proba.argmax(axis=1)
correct_mask = (ens_pred == y_test)
print(f"    Correct predictions: {correct_mask.sum()} / {len(correct_mask)}")

print(f"\n[6/8] Generating Plot F: Waterfall plots (highest-confidence + random patient)...")

import shap as _shap_mod, inspect as _inspect, os as _os
_wf_path = _os.path.join(
    _os.path.dirname(_inspect.getfile(_shap_mod)), "plots", "_waterfall.py"
)
try:
    with open(_wf_path, 'r') as _f:
        _wf_src = _f.read()
    _wf_src = _wf_src.replace("colors.red_rgb",  "(0.118, 0.361, 0.557)")
    _wf_src = _wf_src.replace("colors.blue_rgb", "(0.455, 0.776, 0.894)")
    with open(_wf_path, 'w') as _f:
        _f.write(_wf_src)
    import importlib, shap.plots._waterfall as _wf_mod
    importlib.reload(_wf_mod)
    print("    Waterfall colours patched to project palette")
except Exception as _e:
    print(f"    Could not patch waterfall colours: {_e}")
    print("       Waterfall will use SHAP default red/blue")

np.random.seed(0)

def _waterfall_for_sample(sample_row_idx, cancer_class_idx, cancer_label,
                           shap_arr, X_df, feature_cols, title):
    sv      = shap_arr[cancer_class_idx][sample_row_idx]
    top_idx = np.argsort(np.abs(sv))[::-1][:8]
    exp = shap.Explanation(
        values        = sv[top_idx],
        base_values   = 0,
        data          = X_df.iloc[sample_row_idx, top_idx].values,
        feature_names = [feature_cols[i] for i in top_idx]
    )

    shap.plots.waterfall(exp, max_display=8, show=False)
    fig = plt.gcf()
    ax  = plt.gca()

    import matplotlib.patches as mpatches
    _pos_color = COLORS['dark_blue']
    _neg_color = COLORS['light_blue_2']
    for artist in ax.get_children():
        if isinstance(artist, (mpatches.FancyBboxPatch, mpatches.Rectangle)):
            try:
                w = artist.get_width()
                artist.set_facecolor(_pos_color if w >= 0 else _neg_color)
                artist.set_edgecolor('white')
                artist.set_linewidth(0.5)
            except Exception:
                pass

    fig.set_size_inches(10, 5)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig

for c, cancer in enumerate(cancer_names):
    safe_name = cancer.replace("/", "_").replace(" ", "_")

    class_mask    = y_test == c
    class_indices = np.where(class_mask)[0]
    correct_idx   = class_indices[ens_pred[class_indices] == c]

    if len(correct_idx) == 0:
        print(f"    [{c+1:>2}/{num_classes}] {cancer} — no correct predictions, skipping")
        continue

    confidences  = ens_proba[correct_idx, c]
    hc_row       = correct_idx[np.argmax(confidences)]
    hc_sample_id = X_test.index[hc_row]
    hc_title     = f"{hc_sample_id} — {cancer}  [Confidence: {confidences.max():.3f}]"

    fig = _waterfall_for_sample(hc_row, c, cancer, ensemble_shap,
                                 X_test, list(feature_columns), hc_title)
    fig.savefig(f"/content/shap_waterfall_hc_{safe_name}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    rand_row       = correct_idx[np.random.randint(len(correct_idx))]
    rand_sample_id = X_test.index[rand_row]
    rand_conf      = ens_proba[rand_row, c]
    rand_title     = f"{rand_sample_id} — {cancer}  [Confidence: {rand_conf:.3f}]"

    fig = _waterfall_for_sample(rand_row, c, cancer, ensemble_shap,
                                 X_test, list(feature_columns), rand_title)
    fig.savefig(f"/content/shap_waterfall_rand_{safe_name}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"    [{c+1:>2}/{num_classes}] {cancer}  "
          f"| HC: {hc_sample_id} ({confidences.max():.3f})  "
          f"| Rand: {rand_sample_id} ({rand_conf:.3f})")

print(f"    Plot F complete")

print("\n[7/8] Generating Plot G: Global Feature Category Importance (stacked bars)...")

import matplotlib.colors as _mcolors
_mut_ramp = LinearSegmentedColormap.from_list(
    "mut_blues", ["#10335A", "#1A4A8A", "#2E74C0", "#5E9BD8", "#A8C8EC"]
)
_n_mut = max(len(MUT_SUBGROUP_ORDER), 2)
_mut_colors = {
    sg: _mcolors.to_hex(_mut_ramp(j / (_n_mut - 1)))
    for j, sg in enumerate(MUT_SUBGROUP_ORDER)
}

SUBGROUP_COLORS = dict(_mut_colors)
SUBGROUP_COLORS.update({

    "Amplification":      "#6B0000",
    "Deletion":           "#A82828",
    "Any CNA":            "#D26A6A",

    "Fusion":             "#6A2A9A",

    "Gene-Level Hotspot": "#0D5C63",
    "Amino Acid Hotspot": "#2A9BA8",
    "Splice Hotspot":     "#72CDD4",

    "Metastasis Sites":   "#1A5C2E",
    "Age":                "#2E8B57",
    "Sex":                "#4DAF7C",
    "TMB":                "#82CC9A",
    "MSI":                "#B8E8C8",
})

GROUPS_FULL = [
    ("Mutations",  list(MUT_SUBGROUP_ORDER)),
    ("CNAs",       ["Amplification", "Deletion", "Any CNA"]),
    ("Fusions",    ["Fusion"]),
    ("Hotspots",   ["Gene-Level Hotspot", "Amino Acid Hotspot", "Splice Hotspot"]),
    ("Clinical",   ["Metastasis Sites", "Age", "Sex", "TMB", "MSI"]),
]

_present_sg = {assign_subgroup(c) for c in feature_columns}
_present_sg.discard(None)
GROUPS = [
    (g, [sg for sg in sgs if sg in _present_sg])
    for g, sgs in GROUPS_FULL
]
GROUPS = [(g, sgs) for g, sgs in GROUPS if sgs]

all_correct_idx  = np.where(correct_mask)[0]
shap_all_correct = np.mean(
    np.abs(ensemble_shap_arr[:, all_correct_idx, :]), axis=1
).mean(axis=0)

subgroup_shap = {}
for col, val in zip(feature_columns, shap_all_correct):
    sg = assign_subgroup(col)
    if sg is None:
        continue
    subgroup_shap[sg] = subgroup_shap.get(sg, 0.0) + val

total_shap    = sum(subgroup_shap.values())
subgroup_norm = {sg: v / total_shap for sg, v in subgroup_shap.items()}

print("    Subgroup SHAP proportions:")
for group_name, subgroups in GROUPS:
    group_total = sum(subgroup_norm.get(sg, 0) for sg in subgroups)
    print(f"      {group_name:<12}: {group_total:.1%}")
    for sg in subgroups:
        print(f"        {sg:<22}: {subgroup_norm.get(sg, 0):.1%}")

group_names = [g[0] for g in GROUPS]
x_coords    = np.arange(len(GROUPS))

fig, ax = plt.subplots(figsize=(9, 5))

bottoms = np.zeros(len(GROUPS))
legend_handles = []
bar_tops = np.zeros(len(GROUPS))

top_sg_per_group = {i: sgs[-1] for i, (_, sgs) in enumerate(GROUPS)}

for sg_name in [sg for _, sgs in GROUPS for sg in sgs]:
    seg_vals = []
    for _, subgroups in GROUPS:
        seg_vals.append(subgroup_norm.get(sg_name, 0.0) if sg_name in subgroups else 0.0)
    seg_vals = np.array(seg_vals)
    color    = SUBGROUP_COLORS[sg_name]

    ax.bar(x_coords, seg_vals, bottom=bottoms,
           color=color, edgecolor='white', linewidth=0.5, width=0.55)

    for xi, (bot, val) in enumerate(zip(bottoms, seg_vals)):
        if val <= 0:
            continue
        is_top_segment = (top_sg_per_group[xi] == sg_name)
        if val > 0.025:
            ax.text(xi, bot + val / 2, f"{val:.1%}",
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold')
        elif is_top_segment:
            ax.text(xi, bottoms[xi] + val + 0.008, f"{val:.1%}",
                    ha='center', va='bottom', fontsize=8,
                    color='#222222', fontweight='bold')

    bottoms += seg_vals
    bar_tops = bottoms.copy()

    patch = plt.matplotlib.patches.Patch(color=color, label=sg_name)
    legend_handles.append(patch)

y_max_data = bar_tops.max()
for xi, (gname, subgroups) in enumerate(GROUPS):
    group_val = sum(subgroup_norm.get(sg, 0) for sg in subgroups)
    if group_val <= 0.025 and group_val > 0:
        ax.text(xi, bar_tops[xi] + 0.008, f"{group_val:.1%}",
                ha='center', va='bottom', fontsize=8,
                color='#222222', fontweight='bold')

ax.set_xticks(x_coords)
ax.set_xticklabels(group_names, fontsize=11, fontweight='bold')
ax.set_ylabel("Proportion of Mean |SHAP| Importance", fontsize=10, fontweight='bold')
ax.set_title("Global Feature Category Importance", fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, y_max_data * 1.08)
ax.grid(True, axis='y', alpha=0.25, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

ax.legend(
    handles=legend_handles,
    title=None,
    fontsize=7.5,
    loc='upper left',
    bbox_to_anchor=(1.01, 1.0),
    frameon=True,
    ncol=1,
    handlelength=1.0,
    handletextpad=0.4,
    columnspacing=0.8,
)

plt.tight_layout()
plt.savefig("/content/shap_grouped_category_bar.png", dpi=300, bbox_inches='tight')
print("    Saved: shap_grouped_category_bar.png")
plt.close()

print("\nGenerating Plot G2: Biological category contributions (per-cancer stacked)...")
print(f"    Using {correct_mask.sum()} correctly predicted samples")

per_class_contrib = {}
for c, cancer in enumerate(cancer_names):
    idx = np.where((y_test == c) & correct_mask)[0]
    if len(idx) < 5:
        print(f"    {cancer}: only {len(idx)} correct samples, skipping")
        continue
    shap_c       = np.abs(ensemble_shap_arr[c, idx, :])
    feature_sum  = shap_c.sum(axis=0)
    feature_prop = feature_sum / feature_sum.sum()
    per_class_contrib[cancer] = pd.Series(feature_prop, index=feature_columns)

category_results = {}
for cancer, series in per_class_contrib.items():
    cat = series.groupby(series.index.map(feature_category)).sum()
    cat = cat / cat.sum()
    category_results[cancer] = cat

plot_df = pd.DataFrame(category_results).T.fillna(0)
if "Clinical" in plot_df.columns:
    plot_df = plot_df.sort_values("Clinical", ascending=False)

category_colors = {
    "Clinical":         COLORS['dark_blue'],
    "Mutation":         COLORS['blue_3'],
    "Copy Number":      COLORS['light_blue_2'],
    "Hotspot Mutation": COLORS['blue_1'],
    "Fusion":           COLORS['lightest_blue'],
    "Other":            '#cccccc',
}
col_order  = [c for c in category_colors if c in plot_df.columns]
plot_df    = plot_df[col_order]
bar_colors = [category_colors[c] for c in col_order]

fig, ax = plt.subplots(figsize=(16, 7))
plot_df.plot(kind="bar", stacked=True, figsize=(16, 7),
             color=bar_colors, ax=ax, edgecolor='white', linewidth=0.3)

ax.set_ylabel("Proportion of SHAP Importance", fontsize=13, fontweight='bold')
ax.set_title("Feature Category Contribution per Cancer Type",
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel(f"n = {correct_mask.sum()} correct samples", fontsize=10, color='#555555')
ax.set_xticklabels(plot_df.index, rotation=45, ha='right', fontsize=9)
ax.legend(title="Feature Category", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("/content/shap_biological_contributions.png", dpi=300, bbox_inches='tight')
print("    Saved: shap_biological_contributions.png")
plt.close()

print(f"\n[8/8] Generating Plot H: Positive-only SHAP beeswarms — all {num_classes} cancer types...")

TOP_N_BEESWARM = 20
MIN_REQUIRED   = 5

skipped_h = []

for c, cancer in enumerate(cancer_names):
    safe_name  = cancer.replace("/", "_").replace(" ", "_")
    indices    = np.where(y_test == c)[0]

    top1_preds   = np.argmax(ens_proba[indices], axis=1)
    top1_correct = indices[top1_preds == c]

    if len(top1_correct) >= MIN_REQUIRED:
        selected  = top1_correct
        mode_used = "Top-1"
    else:
        top3_preds   = np.argsort(ens_proba[indices], axis=1)[:, -3:]
        top3_correct = np.array([
            indices[i] for i in range(len(indices))
            if c in top3_preds[i]
        ])
        selected  = top3_correct
        mode_used = "Top-3 fallback"

    n_sel = len(selected)
    if n_sel == 0:
        print(f"    [{c+1:>2}/{num_classes}] {cancer}  (0 samples even in Top-3 — skipping)")
        skipped_h.append(cancer)
        continue

    c_shap_sel       = ensemble_shap[c][selected]
    mean_shap_signed = c_shap_sel.mean(axis=0)

    pos_mask    = mean_shap_signed > 0
    pos_indices = np.where(pos_mask)[0]
    if len(pos_indices) == 0:
        print(f"    [{c+1:>2}/{num_classes}] {cancer}  (no positively contributing features — skipping)")
        skipped_h.append(cancer)
        continue

    pos_ranked  = pos_indices[np.argsort(mean_shap_signed[pos_indices])[::-1]]
    top_pos_idx = pos_ranked[:TOP_N_BEESWARM]

    shap_vals_pos   = c_shap_sel[:, top_pos_idx]
    feature_data    = X_test.iloc[selected, top_pos_idx].values
    feature_names_h = [feature_columns[i] for i in top_pos_idx]
    top_feature     = feature_names_h[0]
    top_val         = mean_shap_signed[top_pos_idx[0]]

    shap_exp = shap.Explanation(
        values        = shap_vals_pos,
        base_values   = np.zeros(n_sel),
        data          = feature_data,
        feature_names = feature_names_h
    )

    fig, ax = plt.subplots(figsize=(11, max(7, len(top_pos_idx) * 0.42 + 2)))
    shap.plots.beeswarm(shap_exp, max_display=TOP_N_BEESWARM, show=False,
                        color_bar=True, color=custom_cmap)

    ax = plt.gca()
    ax.set_title(f"SHAP Analysis: {cancer}", fontsize=12, fontweight='bold', pad=14)
    xlim = ax.get_xlim()
    ax.set_xlim(0, xlim[1])
    ax.axvline(0, color='#999999', linewidth=0.8, linestyle='--')
    ax.set_xlabel("SHAP Value", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"/content/shap_beeswarm_positive_{safe_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    [{c+1:>2}/{num_classes}] {cancer}  "
          f"({len(top1_correct)} Top-1"
          + (f" → {n_sel} Top-3 used" if mode_used == "Top-3 fallback" else " used")
          + f", top feature: {top_feature} {top_val:.4f})")

if skipped_h:
    print(f"\n    Skipped ({len(skipped_h)}): {', '.join(skipped_h)}")
print(f"    Plot H complete")

import zipfile
print("\nZipping all SHAP figures...")
zip_path  = "/content/shap_figures.zip"
shap_figs = glob.glob("/content/shap_*.png")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(shap_figs):
        zf.write(f, os.path.basename(f))
print(f"    {len(shap_figs)} figures zipped → {zip_path}")

print("\nTriggering auto-download of the figure bundle...")
try:
    from google.colab import files
    files.download(zip_path)
    print(f"    Auto-download started: {os.path.basename(zip_path)}")
except Exception as _e:
    print(f"    Auto-download unavailable (not in Colab?): {_e}")
    print(f"       Bundle is still saved at: {zip_path}")

print("\n" + "="*60)
print("SHAP ANALYSIS COMPLETE")
print("="*60)
print(f"  Models: {len(xgb_models)} XGB + {len(mlp_models)} MLP "
      f"(grid-searched blend {xgb_w:.2f}/{mlp_w:.2f})")
print(f"  [A] Global beeswarm")
print(f"  [B] Global bar")
print(f"  [F] Waterfall — HC + random patient per cancer")
print(f"  [G] Grouped feature-category bar (subgroups)")
print(f"  [G2] Per-cancer stacked category bar")
print(f"  [H] Positive-only SHAP beeswarm — one per cancer type")
print(f"\n  All figures zipped → /content/shap_figures.zip")
print("="*60)