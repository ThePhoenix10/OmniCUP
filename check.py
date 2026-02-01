import pickle

with open("ensemble_info.pkl", "rb") as f:
    obj = pickle.load(f)

if isinstance(obj, list):
    feature_columns = obj
elif isinstance(obj, dict):
    if "feature_columns" in obj:
        feature_columns = obj["feature_columns"]
    else:
        feature_columns = next(
            v for v in obj.values()
            if isinstance(v, list) and all(isinstance(x, str) for x in v)
        )
else:
    raise TypeError("Could not find feature_columns")

suffixes = ["_MUT", "_LOF", "_NON_LOF", "_AMP", "_DEL", "_FUSION"]

base_features = set()
counts = {}

for suffix in suffixes:
    matching = [c for c in feature_columns if c.endswith(suffix)]
    counts[suffix[1:]] = len(matching)
    base_features.update(c[:-len(suffix)] for c in matching)

print("Counts by alteration type:")
for k, v in counts.items():
    print(f"{k}: {v}")

print("\nTotal unique genomic features:")
print(len(base_features))