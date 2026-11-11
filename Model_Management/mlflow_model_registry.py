import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Model_Management")

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
registered_name = "Iris_RandomForest_Classifier"

with mlflow.start_run(run_name="RandomForest_Version_1"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_random_forest",
        registered_model_name=registered_name
    )

print("Model registered:", registered_name)
print("Test accuracy:", round(accuracy, 4))

client = MlflowClient()
versions = client.search_model_versions(
    filter_string=f"name='{registered_name}'"
)

print("\nRegistered model versions:")
for version in versions:
    print(
        "Model:", version.name,
        "| Version:", version.version,
        "| Stage:", version.current_stage
    )
