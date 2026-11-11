import json
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Original model accuracy:", round(accuracy, 4))

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "iris_random_forest_v1.joblib"
joblib.dump(model, model_path)
print("Model saved at:", model_path)

loaded_model = joblib.load(model_path)
print("Model loaded successfully.")

sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample)
print("Predicted class:", iris.target_names[prediction[0]])

loaded_accuracy = accuracy_score(y_test, loaded_model.predict(X_test))
print("Loaded model accuracy:", round(loaded_accuracy, 4))

metadata = {
    "model_name": "iris_random_forest",
    "model_version": "1",
    "algorithm": "RandomForestClassifier",
    "n_estimators": 100,
    "max_depth": 5,
    "test_accuracy": float(loaded_accuracy),
    "scikit_learn_version": sklearn.__version__
}

with open(model_dir / "metadata_v1.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("Model metadata saved.")
