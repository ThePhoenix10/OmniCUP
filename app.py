import os
import logging
import sys
import json

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
import pandas as pd
import shap
from llama_index.experimental.query_engine import PandasQueryEngine

llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

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

print("Loading CIVIC database")
CIVIC_DF = None
query_engine = None

try:
    if os.path.exists("Civic.xlsx"):
        CIVIC_DF = pd.read_excel("Civic.xlsx")
        print(f"CIVIC database loaded from Excel with {len(CIVIC_DF)} rows")
    elif os.path.exists("Civic.csv"):
        CIVIC_DF = pd.read_csv("Civic.csv")
        print(f"CIVIC database loaded from CSV with {len(CIVIC_DF)} rows")
    else:
        print("CIVIC database file not found (Civic.xlsx or Civic.csv)")

    if CIVIC_DF is not None:
        print(f"Columns: {CIVIC_DF.columns.tolist()}")

        query_engine = PandasQueryEngine(
            df=CIVIC_DF,
            verbose=False,
            synthesize_response=False,
            instruction_str=f"""
You are working with a pandas dataframe named df. These are the EXACT column names:
{CIVIC_DF.columns.tolist()}

Rules:
- Use ONLY these column names.
- Do NOT invent or rename columns.
- Use exact string matching.
- Return valid pandas expressions only.
- Be precise and factual in your analysis.
- Display ALL rows without any truncation or ellipsis.
"""
        )
        print("PandasQueryEngine initialized successfully")
except Exception as e:
    print(f"Error loading CIVIC database: {e}")
    CIVIC_DF = None
    query_engine = None
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
        genomic_suffixes = ["_NON_LOF", "_LOF", "_MUT", "_AMP", "_DEL", "_FUSION"]
        genes = set()
        for col in feature_cols:
            for suffix in genomic_suffixes:
                if col.endswith(suffix):
                    gene = col[:-len(suffix)]
                    if gene:
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

try:
    ensemble_info = joblib.load("ensemble_info.pkl")
    feature_columns = ensemble_info.get("feature_columns", [])
    FEATURE_SET = set(feature_columns)
    print(f"Loaded ensemble_info.pkl with {len(feature_columns)} features")
except Exception as e:
    print(f"ensemble_info.pkl not loaded: {e}")

try:
    label_encoder = joblib.load("label_encoder.pkl")
    print(f"Label encoder loaded with {len(label_encoder.classes_)} classes")
except Exception as e:
    print(f"label_encoder.pkl not loaded: {e}")
flush()

print("Loading XGBoost booster")
xgb_model = None
try:
    xgb_model = xgb.Booster({"nthread": 1})
    xgb_model.load_model("xgb_model.json")
    print("XGBoost loaded successfully")
except Exception as e:
    print(f"XGBoost model not loaded: {e}")
    xgb_model = None
flush()

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

print("Loading MLP checkpoint")
mlp_model = None
try:
    checkpoint = torch.load("mlp_model.pt", map_location="cpu")
    mlp_model = MLP(
        d=checkpoint["input_dim"],
        c=checkpoint["num_classes"]
    )
    mlp_model.load_state_dict(checkpoint["model_state_dict"])
    mlp_model.eval()
    print("MLP loaded successfully")
except Exception as e:
    print(f"MLP model not loaded: {e}")
    mlp_model = None
flush()

@app.route("/api/model-status", methods=["GET"])
def model_status():
    status = {
        "xgb_loaded": xgb_model is not None,
        "mlp_loaded": mlp_model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "feature_columns_count": len(feature_columns) if feature_columns is not None else 0,
        "cwd": os.getcwd(),
        "files_present": {
            "ensemble_info.pkl": os.path.exists("ensemble_info.pkl"),
            "label_encoder.pkl": os.path.exists("label_encoder.pkl"),
            "xgb_model.json": os.path.exists("xgb_model.json"),
            "mlp_model.pt": os.path.exists("mlp_model.pt"),
            "Allele_Level_Hotspot_Feature_Names.csv": os.path.exists("Allele_Level_Hotspot_Feature_Names.csv"),
            "Civic.xlsx": os.path.exists("Civic.xlsx"),
            "Civic.csv": os.path.exists("Civic.csv"),
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
            "xgboost": 0.6,
            "mlp": 0.4
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

def parse_comma_delimited_genes(gene_input):
    if not gene_input:
        return []
    genes = [g.strip() for g in gene_input.split(",")]
    genes = [g for g in genes if g]
    return genes

@app.route("/api/therapy-lookup", methods=["POST"])
def therapy_lookup():
    try:
        if CIVIC_DF is None or query_engine is None:
            return jsonify({
                "status": "error",
                "message": "CIVIC database not loaded. Ensure Civic.xlsx or Civic.csv is in the application directory."
            }), 500

        data = request.json

        print("\n" + "=" * 60)
        print("Therapy Lookup Debug Request")
        print("=" * 60)
        print(f"Raw request data: {data}")
        print("=" * 60 + "\n")
        flush()

        cancer_type = data.get("cancer_type", "").strip() if isinstance(data.get("cancer_type"), str) else ""
        if not cancer_type:
            cancer_type = data.get("predicted_cancer_type", "").strip() if isinstance(data.get("predicted_cancer_type"), str) else ""

        gene_input = data.get("gene", "").strip() if isinstance(data.get("gene"), str) else ""
        molecular_alterations = parse_comma_delimited_genes(gene_input)

        print(f"Parsed cancer_type: {cancer_type}")
        print(f"Parsed molecular_alterations: {molecular_alterations}\n")
        flush()

        if not cancer_type:
            return jsonify({
                "status": "error",
                "message": "Please provide cancer_type or predicted_cancer_type."
            }), 400

        if not molecular_alterations:
            return jsonify({
                "status": "error",
                "message": "Please provide at least one molecular alteration (comma-delimited)."
            }), 400

        print("Querying CIVIC database for each gene and generating summaries\n")
        flush()

        all_summaries = ""
        all_results_text = ""

        for gene in molecular_alterations:
            print(f"\nProcessing: {gene} + {cancer_type}")
            flush()

            query = f"""
A patient has the following molecular alteration:
{gene}

The cancer type is:
{cancer_type}

Using ONLY the dataframe provided:

1. Identify rows where:
   - molecular_profile exactly matches '{gene}'
   - AND disease exactly matches '{cancer_type}'

2. Return ALL matching rows (do NOT merge, group, or deduplicate).

3. Display EVERY row with NO truncation or ellipsis (...).

4. For EACH returned row, output these columns:
   - therapies
   - evidence_level
   - rating
   - citation

5. Do NOT hide any rows. Show all matching rows completely.
6. Set pandas display options to show all rows: pd.set_option('display.max_rows', None) and pd.set_option('display.max_colwidth', None)
"""

            print("Query sent to PandasQueryEngine")
            flush()

            try:
                response = query_engine.query(query)
                raw_response_text = str(response)

                print(f"Query successful for {gene}")
                print(f"Response:\n{raw_response_text}\n")
                flush()

                all_results_text += f"--------------------------------------------------\n{gene}\n--------------------------------------------------\n{raw_response_text}\n\n"

                llm_summary_prompt = f"""You are an expert oncologist. Based on these CIVIC database results, provide a brief 2-sentence summary ONLY.

GENE: {gene}
CANCER TYPE: {cancer_type}

DATABASE RESULTS:
{raw_response_text}

Provide ONLY 2 sentences:
1. What is the most prevalent SINGLE therapy (not combinations) and what evidence level backs it up?
2. How strong is this evidence based on the clinical rating and outcomes?

Be specific. Keep it concise. Do NOT mention therapy combinations."""

                try:
                    print(f"Generating summary for {gene}")
                    flush()

                    llm_response = llm.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": llm_summary_prompt}
                        ],
                        temperature=0
                    )
                    gene_summary = llm_response.choices[0].message.content
                    print(f"Summary generated for {gene}")
                    flush()

                    all_summaries += f"SUMMARY: {gene}\n{gene_summary}\n"

                except Exception as llm_error:
                    print(f"Could not generate summary for {gene}: {str(llm_error)}")
                    flush()
                    all_summaries += f"SUMMARY: {gene}\n[Summary generation failed]\n\n"

            except Exception as query_error:
                print(f"Query failed for {gene}: {str(query_error)}")
                flush()
                all_results_text += f"--------------------------------------------------\n{gene}\n--------------------------------------------------\nNo results found or query error.\n\n"
                all_summaries += f"SUMMARY: {gene}\n[No therapies found]\n\n"

        final_output = f"""--------------------------------------------------
SUMMARIES (2 SENTENCES PER GENE)
--------------------------------------------------
{all_summaries}

--------------------------------------------------
THERAPIES FROM CIVIC DATABASE (BY GENE)
--------------------------------------------------
{all_results_text}
"""

        return jsonify({
            "status": "success",
            "clinical_analysis": final_output
        })

    except Exception as e:
        print(f"Error in therapy lookup: {str(e)}")
        import traceback
        traceback.print_exc()
        flush()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        missing = []
        if xgb_model is None:
            missing.append("xgb_model.json")
        if mlp_model is None:
            missing.append("mlp_model.pt")
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
        features = {}
        any_mutation = False
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

        for gene, alts in data.get("Genes", {}).items():
            gene = gene.strip()
            for alt, active in alts.items():
                if not active:
                    continue
                col = f"{gene}_{alt}"
                if col in FEATURE_SET:
                    features[col] = 1
                if alt in ["LOF", "NON_LOF"]:
                    genes_with_mut.add(gene)
                any_mutation = True

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
            genes_with_mut.add(gene)
            any_mutation = True

        if any_mutation and "IS_MUTATED" in FEATURE_SET:
            features["IS_MUTATED"] = 1
        for g in genes_with_mut:
            col = f"{g}_MUT"
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

        print("\n" + "=" * 60)
        print("SANITY CHECK: ALL FEATURES PASSED")
        print("=" * 60)
        print(f"AGE_AT_SEQUENCING: {features.get('AGE_AT_SEQUENCING')}")
        print(f"SEX: {features.get('SEX')}")
        print(f"TMB_NONSYNONYMOUS: {features.get('TMB_NONSYNONYMOUS')}")
        print(f"MSI_SCORE: {features.get('MSI_SCORE')}")
        genes_added = [k for k in features if any(s in k for s in ['_LOF','_NON_LOF','_AMP','_DEL','_FUSION','_MUT','_HOTSPOT'])]
        if genes_added:
            for gf in sorted(genes_added):
                print(f"{gf}: {features[gf]}")
        dmets_added = [k for k in features if k.startswith('DMETS_')]
        if dmets_added:
            for df in sorted(dmets_added):
                print(f"{df}: {features[df]}")
        print("=" * 60 + "\n")
        flush()

        # Replace NaN before model inference
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        dmatrix = xgb.DMatrix(X_clean, nthread=1)
        xgb_proba = xgb_model.predict(dmatrix)

        with torch.no_grad():
            logits = mlp_model(torch.from_numpy(X_clean))
            mlp_proba = torch.softmax(logits, 1).numpy()

        ensemble = 0.6 * xgb_proba + 0.4 * mlp_proba
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

        shap_by_class = {}
        try:
            # XGBoost native SHAP — (1, n_classes, n_features+1)
            xgb_shap_raw = xgb_model.predict(
                xgb.DMatrix(X_clean, nthread=1),
                pred_contribs=True
            )
            print(f"XGBoost SHAP raw shape: {xgb_shap_raw.shape}")

            num_classes_local = len(label_encoder.classes_)

            # MLP SHAP for all classes at once
            background   = torch.zeros(1, len(feature_columns))
            mlp_explainer = shap.GradientExplainer(mlp_model, background)
            mlp_shap_all  = mlp_explainer.shap_values(torch.from_numpy(X_clean))
            print(f"MLP SHAP done — {len(mlp_shap_all)} classes")
            # mlp_shap_all: list of n_classes arrays, each (1, n_features)

            # Only compute for features the user actually inputted
            inputted_mask    = X_clean[0] != 0
            inputted_indices = np.where(inputted_mask)[0]

            for c in range(num_classes_local):
                cancer_name = label_encoder.classes_[c]

                # XGB SHAP for this class
                if xgb_shap_raw.ndim == 3:
                    if xgb_shap_raw.shape[1] == num_classes_local:
                        xgb_shap_c = xgb_shap_raw[0, c, :-1]
                    else:
                        xgb_shap_c = xgb_shap_raw[0, :-1, c]
                else:
                    n_feat_plus1 = xgb_shap_raw.shape[1] // num_classes_local
                    reshaped     = xgb_shap_raw[0].reshape(num_classes_local, n_feat_plus1)
                    xgb_shap_c   = reshaped[c, :-1]

                mlp_shap_c    = np.array(mlp_shap_all[c][0])
                ensemble_shap = 0.6 * xgb_shap_c + 0.4 * mlp_shap_c

                if len(inputted_indices) == 0:
                    shap_by_class[cancer_name] = []
                    continue

                inputted_shap = ensemble_shap[inputted_indices]
                top_indices   = np.argsort(np.abs(inputted_shap))[::-1]
                shap_features_list = []
                for i in top_indices:
                    fname = feature_columns[inputted_indices[i]]
                    if fname == "IS_MUTATED":
                        continue
                    shap_features_list.append({
                        "feature":    fname,
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
    print("OmniCUP + CIVIC Therapy Lookup Server")
    print("=" * 60)
    print(f"ensemble_info.pkl: {os.path.exists('ensemble_info.pkl')}")
    print(f"label_encoder.pkl: {os.path.exists('label_encoder.pkl')}")
    print(f"xgb_model.json: {os.path.exists('xgb_model.json')}")
    print(f"mlp_model.pt: {os.path.exists('mlp_model.pt')}")
    print("=" * 60 + "\n")
    flush()

    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, port=port, host="0.0.0.0")