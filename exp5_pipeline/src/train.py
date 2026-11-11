import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

train_file = "exp5_pipeline/data/processed/train.csv"
df = pd.read_csv(train_file)

X = df.drop("species", axis=1)
y = df["species"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X, y)

model_path = Path("exp5_pipeline/models/model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("Model training completed.")
