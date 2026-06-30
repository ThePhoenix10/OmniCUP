import os
import logging
import sys
import json

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
import pandas as pd
import shap

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ==============================================================================
# ENSEMBLE LAYOUT
# ==============================================================================
# Models produced by the training pipeline live in these folders, next to app.py:
#   xgb_models/xgb_model_0.json ... xgb_model_9.json
#   mlp_models/mlp_model_0.pt   ... mlp_model_9.pt
# Blend weights (xgb_weight / mlp_weight) and per-MLP architectures come from
# ensemble_info.pkl, so nothing here is hardcoded to a fixed blend or arch.
XGB_DIR = "xgb_models"
MLP_DIR = "mlp_models"

# SHAP across every member is the most faithful but also the slowest part of a
# request (one GradientExplainer per MLP). Set this to an int (e.g. 3) to use
# only the first N members per family for SHAP; None = use all members.
SHAP_MEMBER_LIMIT = None


@app.route("/")
def index():
    return send_from_directory(".", "index.html")

def flush():
    try:
        sys.stdout.flush()
    except Exception:
        pass

print("App starting")
print(f"Working directory: {os.getcwd()}")
print("Directory listing (top-level):")
try:
    print(os.listdir("."))
except Exception as e:
    print(f"Could not list directory: {e}")
flush()

def load_hotspots_from_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        if 'ALLELE_HOTSPOT_FEATURE' in df.columns:
            col = 'ALLELE_HOTSPOT_FEATURE'
        else:
            col = df.columns[0]
        hotspots = df[col].dropna().unique().tolist()
        print(f"Loaded {len(hotspots)} hotspots from CSV")
        return sorted(hotspots)
    except Exception as e:
        print(f"Could not load hotspots CSV: {e}")
        return []

def load_genes_from_ensemble_info(filepath):
    try:
        ensemble_info_local = joblib.load(filepath)
        feature_cols = ensemble_info_local.get("feature_columns", [])
        # Gene-level suffixes in the new (LOF/Non-LOF-free) schema. Longest first
        # so e.g. "_Frame_Shift_Del" is matched before any shorter overlap. The
        # "." guard skips allele-level keys like "TP53.R175H_AA_HOTSPOT", which
        # belong to the separate hotspot picker, not the gene list.
        genomic_suffixes = [
            "_ANY_SOM_MAT",
            "_Missense_Mutation", "_Nonsense_Mutation",
            "_Frame_Shift_Ins", "_Frame_Shift_Del",
            "_Translation_Start_Site", "_Splice_Site",
            "_In_Frame_Ins", "_In_Frame_Del", "_Nonstop_Mutation",
            "_ANY_CNA", "_AMP", "_DEL", "_FUSION", "_HOTSPOT",
        ]
        genes = set()
        for col in feature_cols:
            for suffix in genomic_suffixes:
                if col.endswith(suffix):
                    gene = col[:-len(suffix)]
                    if gene and "." not in gene:
                        genes.add(gene)
                    break
        genes = sorted(list(genes))
        print(f"Loaded {len(genes)} genes from ensemble_info.pkl")
        return genes
    except Exception as e:
        print(f"Could not load ensemble_info.pkl: {e}")
        return []

print("Loading ensemble metadata")
ensemble_info = None
feature_columns = []
label_encoder = None
FEATURE_SET = set()

# Blend weights + member counts come straight from ensemble_info.pkl.
XGB_WEIGHT = 0.5
MLP_WEIGHT = 0.5
N_XGB = 10
N_MLP = 10

try:
    ensemble_info = joblib.load("ensemble_info.pkl")
    feature_columns = ensemble_info.get("feature_columns", [])
    FEATURE_SET = set(feature_columns)

    # Grid-search-selected blend (with backward-compatible aliases).
    XGB_WEIGHT = float(
        ensemble_info.get("xgb_weight",
                          ensemble_info.get("best_weight_xgb", 0.5))
    )
    MLP_WEIGHT = float(
        ensemble_info.get("mlp_weight",
                          ensemble_info.get("best_weight_mlp", 1.0 - XGB_WEIGHT))
    )
    N_XGB = int(ensemble_info.get("n_xgb", 10))
    N_MLP = int(ensemble_info.get("n_mlp", 10))

    print(f"Loaded ensemble_info.pkl with {len(feature_columns)} features")
    print(f"Blend weights -> XGB: {XGB_WEIGHT:.2f} | MLP: {MLP_WEIGHT:.2f}")
    print(f"Member counts -> XGB: {N_XGB} | MLP: {N_MLP}")
except Exception as e:
    print(f"ensemble_info.pkl not loaded: {e}")

try:
    label_encoder = joblib.load("label_encoder.pkl")
    print(f"Label encoder loaded with {len(label_encoder.classes_)} classes")
except Exception as e:
    print(f"label_encoder.pkl not loaded: {e}")
flush()

# ==============================================================================
# LOAD 10 XGBOOST BOOSTERS
# ==============================================================================
print(f"Loading {N_XGB} XGBoost boosters from '{XGB_DIR}/'")
xgb_models = []
try:
    for i in range(N_XGB):
        path = os.path.join(XGB_DIR, f"xgb_model_{i}.json")
        if not os.path.exists(path):
            print(f"  ! Missing {path} -- skipping")
            continue
        booster = xgb.Booster({"nthread": 1})
        booster.load_model(path)
        xgb_models.append(booster)
    print(f"XGBoost ensemble loaded: {len(xgb_models)} / {N_XGB} members")
except Exception as e:
    print(f"XGBoost ensemble not loaded: {e}")
    xgb_models = []
flush()

# ==============================================================================
# DYNAMIC MLP (matches the training pipeline; arch travels with each checkpoint)
# ==============================================================================
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

# ==============================================================================
# LOAD 10 MLP MODELS (each rebuilt from its own saved architecture)
# ==============================================================================
print(f"Loading {N_MLP} MLP checkpoints from '{MLP_DIR}/'")
mlp_models = []
try:
    for i in range(N_MLP):
        path = os.path.join(MLP_DIR, f"mlp_model_{i}.pt")
        if not os.path.exists(path):
            print(f"  ! Missing {path} -- skipping")
            continue
        ckpt = torch.load(path, map_location="cpu")
        net = DynamicMLP(
            d_in=ckpt["input_dim"],
            h1=ckpt["h1"],
            h2=ckpt["h2"],
            c_out=ckpt["num_classes"],
            p_drop=ckpt["p_drop"],
        )
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        mlp_models.append(net)
    print(f"MLP ensemble loaded: {len(mlp_models)} / {N_MLP} members")
except Exception as e:
    print(f"MLP ensemble not loaded: {e}")
    import traceback
    traceback.print_exc()
    mlp_models = []
flush()


@app.route("/api/model-status", methods=["GET"])
def model_status():
    status = {
        "xgb_loaded": len(xgb_models) > 0,
        "mlp_loaded": len(mlp_models) > 0,
        "xgb_members": len(xgb_models),
        "mlp_members": len(mlp_models),
        "xgb_weight": XGB_WEIGHT,
        "mlp_weight": MLP_WEIGHT,
        "label_encoder_loaded": label_encoder is not None,
        "feature_columns_count": len(feature_columns) if feature_columns is not None else 0,
        "cwd": os.getcwd(),
        "files_present": {
            "ensemble_info.pkl": os.path.exists("ensemble_info.pkl"),
            "label_encoder.pkl": os.path.exists("label_encoder.pkl"),
            "xgb_models/": os.path.isdir(XGB_DIR),
            "mlp_models/": os.path.isdir(MLP_DIR),
            "Allele_Level_Hotspot_Feature_Names.csv": os.path.exists("Allele_Level_Hotspot_Feature_Names.csv"),
        }
    }
    return jsonify(status)

@app.route("/api/data", methods=["GET"])
def get_data():
    hotspots = load_hotspots_from_csv("Allele_Level_Hotspot_Feature_Names.csv")
    genes = load_genes_from_ensemble_info("ensemble_info.pkl")

    return jsonify({
        "genes": genes,
        "hotspots": hotspots,
        "ensemble_weights": {
            "xgboost": XGB_WEIGHT,
            "mlp": MLP_WEIGHT
        }
    })

@app.route("/api/debug-patient", methods=["POST"])
def debug_patient():
    data = request.json
    print("\n" + "=" * 60)
    print("Patient Data Submitted")
    print("=" * 60)
    print("\nClinical Data:")
    c = data.get("Clinical", {})
    if c.get("Age") is not None:
        age_val = c['Age']
        age_str = f"NaN" if (age_val is None or age_val == "" or (isinstance(age_val, float) and np.isnan(age_val))) else str(age_val)
        print(f"AGE_AT_SEQUENCING: {age_str}")
    if c.get("Sex") is not None:
        sex_val = c['Sex']
        sex_str = f"NaN" if (sex_val is None or sex_val == "") else (f"1 ({sex_val})" if sex_val == 'Male' else f"0 ({sex_val})")
        print(f"SEX: {sex_str}")
    if c.get("TMB") is not None:
        tmb_val = c['TMB']
        tmb_str = f"NaN" if (tmb_val is None or tmb_val == "" or (isinstance(tmb_val, float) and np.isnan(tmb_val))) else str(tmb_val)
        print(f"TMB_NONSYNONYMOUS: {tmb_str}")
    if c.get("MSI") is not None:
        msi_val = c['MSI']
        msi_str = f"NaN" if (msi_val is None or msi_val == "" or (isinstance(msi_val, float) and np.isnan(msi_val))) else str(msi_val)
        print(f"MSI_SCORE: {msi_str}")
    print("\nGene Alterations:")
    genes = data.get("Genes", {})
    if genes:
        for gene, alts in genes.items():
            for alt_type, is_active in alts.items():
                if is_active:
                    print(f"{gene}_{alt_type}: 1")
    else:
        print("(none)")
    print("\nHotspots:")
    hotspots = data.get("Hotspots", [])
    if hotspots:
        for hotspot in hotspots:
            print(f"{hotspot}")
    else:
        print("(none)")
    print("\nDmets:")
    dmets = data.get("DMETS", {})
    if dmets:
        for dmet, value in dmets.items():
            if value:
                print(f"{dmet}: 1")
    else:
        print("(none)")
    print("=" * 60 + "\n")
    flush()
    return jsonify({"status": "logged"})

def build_feature_vector(data):
    """Turn the UI payload into the model feature vector (shared by predict)."""
    features = {}
    genes_with_mut = set()
    genes_with_hotspot = set()

    c = data.get("Clinical", {})

    if c.get("Age") is not None and c.get("Age") != "":
        try:
            age_val = float(c["Age"])
            features["AGE_AT_SEQUENCING"] = age_val if not np.isnan(age_val) else np.nan
        except (ValueError, TypeError):
            features["AGE_AT_SEQUENCING"] = np.nan
    else:
        features["AGE_AT_SEQUENCING"] = np.nan

    if c.get("Sex") is not None and c.get("Sex") != "":
        features["SEX"] = 1 if c["Sex"] == "Male" else 0
    else:
        features["SEX"] = np.nan

    if c.get("TMB") is not None and c.get("TMB") != "":
        try:
            tmb_val = float(c["TMB"])
            features["TMB_NONSYNONYMOUS"] = tmb_val if not np.isnan(tmb_val) else np.nan
        except (ValueError, TypeError):
            features["TMB_NONSYNONYMOUS"] = np.nan
    else:
        features["TMB_NONSYNONYMOUS"] = np.nan

    if c.get("MSI") is not None and c.get("MSI") != "":
        try:
            msi_val = float(c["MSI"])
            features["MSI_SCORE"] = msi_val if not np.isnan(msi_val) else np.nan
        except (ValueError, TypeError):
            features["MSI_SCORE"] = np.nan
    else:
        features["MSI_SCORE"] = np.nan

    for k, v in data.get("DMETS", {}).items():
        if v and k in FEATURE_SET:
            features[k] = 1

    # The nine somatic-mutation subcategories (plus the ANY rollup) that, when
    # present, imply the gene-level _ANY_SOM_MAT flag and the global
    # IS_ANY_SOM_MAT flag -- exactly how the curation script derives them.
    MUT_SUBCATS = {
        "ANY_SOM_MAT",
        "Missense_Mutation", "Nonsense_Mutation",
        "Frame_Shift_Ins", "Frame_Shift_Del",
        "Splice_Site", "Translation_Start_Site",
        "In_Frame_Ins", "In_Frame_Del", "Nonstop_Mutation",
    }

    genes_with_cna = set()

    for gene, alts in data.get("Genes", {}).items():
        gene = gene.strip()
        for alt, active in alts.items():
            if not active:
                continue
            col = f"{gene}_{alt}"
            if col in FEATURE_SET:
                features[col] = 1
            if alt in MUT_SUBCATS:
                genes_with_mut.add(gene)
            elif alt in ("AMP", "DEL"):
                genes_with_cna.add(gene)

    for label in data.get("Hotspots", []):
        if " - " not in label:
            continue
        base, kind = label.split(" - ", 1)
        gene = base.split(".")[0]
        if kind.lower().startswith("amino"):
            col = f"{base}_AA_HOTSPOT"
        elif kind.lower().startswith("splice"):
            col = f"{base}_splice_SPLICE_HOTSPOT"
        else:
            continue
        if col in FEATURE_SET:
            features[col] = 1
        genes_with_hotspot.add(gene)
        genes_with_mut.add(gene)   # a hotspot is a somatic mutation

    # --- Roll-ups so the vector matches the training data distribution ---
    # A specific mutation subcategory never appears in training without its
    # gene-level _ANY_SOM_MAT and the global IS_ANY_SOM_MAT also being set.
    if genes_with_mut and "IS_ANY_SOM_MAT" in FEATURE_SET:
        features["IS_ANY_SOM_MAT"] = 1
    for g in genes_with_mut:
        col = f"{g}_ANY_SOM_MAT"
        if col in FEATURE_SET:
            features[col] = 1
    # AMP or DEL implies the gene-level _ANY_CNA flag.
    for g in genes_with_cna:
        col = f"{g}_ANY_CNA"
        if col in FEATURE_SET:
            features[col] = 1
    for g in genes_with_hotspot:
        col = f"{g}_HOTSPOT"
        if col in FEATURE_SET:
            features[col] = 1

    X = np.zeros((1, len(feature_columns)), dtype=np.float32)
    col_idx = {cc: i for i, cc in enumerate(feature_columns)}
    for k, v in features.items():
        if k in col_idx:
            X[0, col_idx[k]] = v

    return X, features


@app.route("/predict", methods=["POST"])
def predict():
    try:
        missing = []
        if len(xgb_models) == 0:
            missing.append("xgb_models/ (no boosters loaded)")
        if len(mlp_models) == 0:
            missing.append("mlp_models/ (no MLPs loaded)")
        if label_encoder is None:
            missing.append("label_encoder.pkl")
        if feature_columns is None or len(feature_columns) == 0:
            missing.append("ensemble_info.pkl(feature_columns)")

        if len(missing) > 0:
            msg = "Models not loaded: " + ", ".join(missing)
            print(msg)
            flush()
            return jsonify({
                "status": "error",
                "message": msg
            }), 500

        data = request.json
        X, features = build_feature_vector(data)

        print("\n" + "=" * 60)
        print("SANITY CHECK: ALL FEATURES PASSED")
        print("=" * 60)
        def _fv(name):
            v = features.get(name)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A (no value -> treated as missing)"
            return v
        print(f"AGE_AT_SEQUENCING: {_fv('AGE_AT_SEQUENCING')}")
        print(f"SEX: {_fv('SEX')}")
        print(f"TMB_NONSYNONYMOUS: {_fv('TMB_NONSYNONYMOUS')}")
        print(f"MSI_SCORE: {_fv('MSI_SCORE')}")
        genes_added = [k for k in features if any(s in k for s in ['_Mutation', '_Frame_Shift_', '_Splice_Site', '_Translation_Start_Site', '_In_Frame_', '_Nonstop_', '_ANY_SOM_MAT', '_AMP', '_DEL', '_ANY_CNA', '_FUSION', '_HOTSPOT'])]
        if genes_added:
            for gf in sorted(genes_added):
                print(f"{gf}: {features[gf]}")
        dmets_added = [k for k in features if k.startswith('DMETS_')]
        if dmets_added:
            for df in sorted(dmets_added):
                print(f"{df}: {features[df]}")
        print(f"Blend -> XGB {XGB_WEIGHT:.2f} ({len(xgb_models)} members) / "
              f"MLP {MLP_WEIGHT:.2f} ({len(mlp_models)} members)")
        print("=" * 60 + "\n")
        flush()

        # Two views of the same patient vector:
        #   X_xgb -> blank clinical fields stay NaN; XGBoost reads NaN as "missing"
        #            and routes it down each split's default branch. Absent genes
        #            and DMETS are a genuine 0 (not missing), so they remain 0.
        #   X_mlp -> NaN replaced with 0, because a dense net cannot ingest NaN
        #            (it would propagate to NaN logits). Only the blank clinical
        #            fields differ between the two views; genomic 0s are identical.
        # ±inf (shouldn't occur) -> missing for XGB, 0 for MLP.
        X_xgb = np.where(np.isinf(X), np.nan, X).astype(np.float32)
        X_mlp = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # ----------------------------------------------------------------------
        # XGBoost: mean soft-vote over all boosters (best iteration baked into JSON)
        # ----------------------------------------------------------------------
        dmatrix = xgb.DMatrix(X_xgb, nthread=1, missing=np.nan)
        xgb_probs = np.stack([booster.predict(dmatrix) for booster in xgb_models], axis=0)
        mean_xgb_proba = xgb_probs.mean(axis=0)

        # ----------------------------------------------------------------------
        # MLP: mean soft-vote over all neural nets
        # ----------------------------------------------------------------------
        mlp_probs = []
        with torch.no_grad():
            x_tensor = torch.from_numpy(X_mlp)
            for net in mlp_models:
                logits = net(x_tensor)
                mlp_probs.append(torch.softmax(logits, 1).numpy())
        mean_mlp_proba = np.stack(mlp_probs, axis=0).mean(axis=0)

        # ----------------------------------------------------------------------
        # Weighted blend (weights selected on validation, from ensemble_info.pkl)
        # ----------------------------------------------------------------------
        ensemble = XGB_WEIGHT * mean_xgb_proba + MLP_WEIGHT * mean_mlp_proba
        row_sums = ensemble.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        ensemble = ensemble / row_sums
        ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=0.0, neginf=0.0)

        all_sorted = np.argsort(ensemble[0])[::-1]
        results = []
        for i, idx in enumerate(all_sorted):
            prob = float(ensemble[0, idx])
            if not np.isfinite(prob):
                prob = 0.0
            results.append({
                "rank": i + 1,
                "cancer_type": label_encoder.inverse_transform([idx])[0],
                "probability": prob,
                "confidence": f"{prob * 100:.1f}%"
            })

        # ----------------------------------------------------------------------
        # SHAP: average contributions across the ensemble, then blend
        # ----------------------------------------------------------------------
        shap_by_class = {}
        try:
            num_classes_local = len(label_encoder.classes_)

            xgb_for_shap = xgb_models if SHAP_MEMBER_LIMIT is None else xgb_models[:SHAP_MEMBER_LIMIT]
            mlp_for_shap = mlp_models if SHAP_MEMBER_LIMIT is None else mlp_models[:SHAP_MEMBER_LIMIT]

            # --- XGB SHAP: mean of pred_contribs over boosters ---
            xgb_shap_stack = []
            for booster in xgb_for_shap:
                contrib = booster.predict(
                    xgb.DMatrix(X_xgb, nthread=1, missing=np.nan),
                    pred_contribs=True
                )
                xgb_shap_stack.append(contrib)
            xgb_shap_raw = np.mean(xgb_shap_stack, axis=0)
            print(f"XGBoost SHAP raw shape: {xgb_shap_raw.shape} "
                  f"(avg over {len(xgb_for_shap)} boosters)")

            # --- MLP SHAP: mean of GradientExplainer values over nets ---
            background = torch.zeros(1, len(feature_columns))
            mlp_shap_accum = None
            for net in mlp_for_shap:
                explainer = shap.GradientExplainer(net, background)
                sv = explainer.shap_values(torch.from_numpy(X_mlp))
                if mlp_shap_accum is None:
                    mlp_shap_accum = [np.array(s) for s in sv]
                else:
                    for cc in range(len(sv)):
                        mlp_shap_accum[cc] = mlp_shap_accum[cc] + np.array(sv[cc])
            mlp_shap_all = [s / float(len(mlp_for_shap)) for s in mlp_shap_accum]
            print(f"MLP SHAP done -- {len(mlp_shap_all)} classes "
                  f"(avg over {len(mlp_for_shap)} nets)")

            # A blank clinical field is now missing (NaN in X_xgb, 0 in X_mlp), so
            # it is correctly excluded here -- only features the user actually set
            # (nonzero) show up in the per-class SHAP breakdown.
            inputted_mask = X_mlp[0] != 0
            inputted_indices = np.where(inputted_mask)[0]

            for c in range(num_classes_local):
                cancer_name = label_encoder.classes_[c]

                if xgb_shap_raw.ndim == 3:
                    if xgb_shap_raw.shape[1] == num_classes_local:
                        xgb_shap_c = xgb_shap_raw[0, c, :-1]
                    else:
                        xgb_shap_c = xgb_shap_raw[0, :-1, c]
                else:
                    n_feat_plus1 = xgb_shap_raw.shape[1] // num_classes_local
                    reshaped = xgb_shap_raw[0].reshape(num_classes_local, n_feat_plus1)
                    xgb_shap_c = reshaped[c, :-1]

                mlp_shap_c = np.array(mlp_shap_all[c][0])
                ensemble_shap = XGB_WEIGHT * xgb_shap_c + MLP_WEIGHT * mlp_shap_c

                if len(inputted_indices) == 0:
                    shap_by_class[cancer_name] = []
                    continue

                inputted_shap = ensemble_shap[inputted_indices]
                top_indices = np.argsort(np.abs(inputted_shap))[::-1]
                shap_features_list = []
                for i in top_indices:
                    fname = feature_columns[inputted_indices[i]]
                    shap_features_list.append({
                        "feature": fname,
                        "shap_value": float(inputted_shap[i])
                    })
                shap_by_class[cancer_name] = shap_features_list

            print(f"SHAP computed for all {num_classes_local} classes")

        except Exception as shap_error:
            print(f"SHAP computation failed: {shap_error}")
            import traceback
            traceback.print_exc()
            shap_by_class = {}
        flush()

        return jsonify({
            "status": "success",
            "predictions": results,
            "shap_by_class": shap_by_class
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        flush()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OmniCUP Ensemble Prediction Server")
    print("=" * 60)
    print(f"ensemble_info.pkl : {os.path.exists('ensemble_info.pkl')}")
    print(f"label_encoder.pkl : {os.path.exists('label_encoder.pkl')}")
    print(f"xgb_models/       : {os.path.isdir(XGB_DIR)} "
          f"({len(xgb_models)} loaded)")
    print(f"mlp_models/       : {os.path.isdir(MLP_DIR)} "
          f"({len(mlp_models)} loaded)")
    print(f"Blend             : XGB {XGB_WEIGHT:.2f} / MLP {MLP_WEIGHT:.2f}")
    print("=" * 60 + "\n")
    flush()

    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, port=port, host="0.0.0.0")