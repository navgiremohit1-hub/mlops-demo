import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import accuracy_score

test_file = "exp5_pipeline/data/processed/test.csv"
df = pd.read_csv(test_file)

X = df.drop("species", axis=1)
y = df["species"]

with open("exp5_pipeline/models/model.pkl", "rb") as file:
    model = pickle.load(file)

predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

metrics_dir = Path("exp5_pipeline/metrics")
metrics_dir.mkdir(parents=True, exist_ok=True)

with open(metrics_dir / "metrics.json", "w") as file:
    json.dump({"accuracy": float(accuracy)}, file, indent=4)

print("Model Accuracy:", accuracy)
print("Evaluation completed.")
